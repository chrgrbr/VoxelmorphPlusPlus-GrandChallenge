import SimpleITK
from pathlib import Path
import torch
import torch.nn.functional as F

from foerstner import foerstner_kpts 
from thin_plate_spline import thin_plate_dense
from vxmplusplus_utils import get_vxmpp_models,return_crops, adam_mind, MINDSSC


class VoxelMorphPlusPlus():  
    def __init__(self):



        #self.in_path = Path('/input/images')
        #self.out_path = Path('/output/images/displacement-field')
        #self.model_path = Path('/opt/algorithm/voxelmorphplusplus.pth')


        self.in_path = Path('/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/test/images')
        self.out_path = Path('/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/output')
        self.model_path = Path('/share/data_abby1/grossbroehmer/L2R23_Algortihm_Submission/Algos/VoxelMorphPlusPlus/model_weights/voxelmorphplusplus.pth')

        self.grid_sp = 1
        self.do_MIND = True


        ##create displacement output folder 
        self.out_path.mkdir(parents=True, exist_ok=True)
        self.unet_model,self.heatmap,self.mesh = get_vxmpp_models()
        self.state_dicts = torch.load(str(self.model_path))  #model path 
        ##load model weights
        self.unet_model.load_state_dict(self.state_dicts[1])
        self.heatmap.load_state_dict(self.state_dicts[0])

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



    def read_input(self,do_MIND):
        fpath_fixed_image = list((self.in_path / 'fixed').glob('*.mha'))[0]
        fpath_moving_image = list((self.in_path / 'moving').glob('*.mha'))[0]
        fpath_fixed_mask = None
        fpath_moving_mask = None
        ##Check if Mask is provided
        if len(list((self.in_path / 'fixed-mask').glob('*.mha'))) == 1:
            fpath_fixed_mask = list((self.in_path / 'fixed-mask').glob('*.mha'))[0]
        if len(list((self.in_path / 'moving-mask').glob('*.mha'))) == 1:
            fpath_moving_mask = list((self.in_path / 'moving-mask').glob('*.mha'))[0]


        img_fixed = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_fixed_image))).permute(2,1,0)
        img_mov = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_moving_image))).permute(2,1,0)
        
        ##if no masks provided, use the whole image
        if fpath_fixed_mask is not None:
            mask_fixed = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_fixed_mask))).permute(2,1,0)
        else:
            mask_fixed = torch.ones_like(img_fixed)
        if fpath_moving_mask is not None:
            mask_mov = torch.from_numpy(SimpleITK.GetArrayFromImage(SimpleITK.ReadImage(fpath_moving_mask))).permute(2,1,0)
        else:
            mask_mov = torch.ones_like(img_mov)


        masked_fixed = F.interpolate(((img_fixed+1024)*mask_fixed).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()
        masked_mov = F.interpolate(((img_mov+1024)*mask_mov).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

        


        kpts_fix = foerstner_kpts(img_fixed.unsqueeze(0).unsqueeze(0).to(self.device), mask_fixed.unsqueeze(0).unsqueeze(0).to(self.device), 1.4, 3).cpu()
    
        if(do_MIND):
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    mind_fixed = F.avg_pool3d(mask_fixed.unsqueeze(0).unsqueeze(0).to(self.device).half()*\
                                MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).to(self.device),1,2).half(),2).cpu()
                    mind_mov = F.avg_pool3d(mask_mov.unsqueeze(0).unsqueeze(0).to(self.device).half()*\
                                MINDSSC(img_mov.unsqueeze(0).unsqueeze(0).to(self.device),1,2).half(),2).cpu()
                
        return img_fixed,masked_fixed,img_mov,masked_mov,kpts_fix,mind_fixed,mind_mov

    def get_warped_pair(self,_cf,_img_moving):
        H,W,D = _img_moving.shape[-3:]
        kpts_fixed = torch.flip((_cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
        kpts_moving = torch.flip((_cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
        with torch.no_grad():
            dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).to(self.device), (kpts_moving-kpts_fixed).unsqueeze(0).to(self.device), (H, W, D), 4, 0.01).cpu()
        warped_img = F.grid_sample(_img_moving.view(1,1,H,W,D).cpu(),dense_flow+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()
        return warped_img,dense_flow
    
    def predict(self, inputs):
        img_fixed,masked_fixed,img_mov,masked_mov,kpts_fix,mind_fixed,mind_mov = inputs
        predictions = []
        
        #MASKED INPUT IMAGES ARE HALF-RESOLUTION
        with torch.no_grad():
            keypts_fix = kpts_fix.squeeze().to(self.device)
            H,W,D = masked_fixed.shape[-3:]

            masked_fixed = masked_fixed.view(1,1,H,W,D).to(self.device)
            masked_mov = masked_mov.view(1,1,H,W,D).to(self.device)

            with torch.cuda.amp.autocast():
                #VoxelMorph requires some padding
                input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((masked_fixed,masked_mov),1).to(self.device))
                output = F.pad(F.interpolate(self.unet_model(input),scale_factor=2),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
                disp_est = torch.zeros_like(keypts_fix)
                for idx in torch.split(torch.arange(len(keypts_fix)),1024):
                    sample_xyz = keypts_fix[idx]
                    sampled = F.grid_sample(output,sample_xyz.to(self.device).view(1,-1,1,1,3),mode='bilinear')
                    disp_pred = self.heatmap(sampled.permute(2,1,0,3,4))
                    disp_est[idx] = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*self.mesh.view(1,11**3,3),1)


        #NOW EVERYTHING FULL-RESOLUTION
        H,W,D = img_fixed.shape[-3:]

        fixed_mind = mind_fixed.view(1,-1,H//2,W//2,D//2).to(self.device)
        moving_mind = mind_mov.view(1,-1,H//2,W//2,D//2).to(self.device)

        pred_xyz,_,dense_flow = adam_mind(keypts_fix,disp_est,fixed_mind,moving_mind,H,W,D)
        predictions = pred_xyz.cpu()+keypts_fix.cpu()

        keypts_fix = torch.flip(kpts_fix.squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
        keypts_moved = torch.flip(predictions.squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
                

        cf = torch.cat([keypts_fix, keypts_moved], dim=1)
        _,dense_flow = self.get_warped_pair(cf,img_mov)
        H,W,D = img_mov.shape
        dense_flow = dense_flow.flip(4).permute(0, 4, 1, 2, 3) * torch.tensor( [H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
        disp_lr = F.interpolate(dense_flow, size=(H // self.grid_sp, W // self.grid_sp, D // self.grid_sp), mode='trilinear',
                                                    align_corners=False)
        disp_lr = disp_lr.permute(0,2,3,4,1)[0].numpy()
        return disp_lr


    def write_outputs(self, displacement_field):
        out = SimpleITK.GetImageFromArray(displacement_field)
        ##You can give the output-mha file any name you want, but it must be saved to  "/output/displacement-field" 
        SimpleITK.WriteImage(out, str(self.out_path / 'thisIsAnArbitraryFilename.mha'))
        return
    

    def process(self):
        inputs = self.read_input(do_MIND=self.do_MIND) 
        outputs = self.predict(inputs)
        self.write_outputs(outputs)

if __name__ == "__main__":
    VoxelMorphPlusPlus().process()
