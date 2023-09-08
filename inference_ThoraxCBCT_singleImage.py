#!/usr/bin/env python
import torch
import torch.nn.functional as F
import os
import nibabel as nib
import SimpleITK as sitk
from scipy.ndimage import map_coordinates

#imports
from thin_plate_spline import thin_plate_dense
from foerstner import foerstner_kpts 
from vxmplusplus_utils import get_vxmpp_models,return_crops
from vxmplusplus_utils import adam_mind
from vxmplusplus_utils import MINDSSC


#os.environ['CUDA_DEVICE_ORDER']='PCI_BUS_ID'
#os.environ['CUDA_VISIBLE_DEVICES']='0'

def read_input(base_img_fixed,base_img_moving, base_mask_fixed, base_mask_moving,do_MIND = True):
    
    filename_fix = os.listdir(base_img_fixed)[0]  
    img_fixed = torch.from_numpy(nib.load(os.path.join(base_img_fixed, os.listdir(base_img_fixed)[0])).get_fdata()).float()
    mask_fixed = torch.from_numpy(nib.load(os.path.join(base_mask_fixed, os.listdir(base_mask_fixed)[0])).get_fdata()).float()
    masked_fixed = F.interpolate(((img_fixed+1024)*mask_fixed).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

    img_mov = torch.from_numpy(nib.load(os.path.join(base_img_moving, os.listdir(base_img_moving)[0])).get_fdata()).float()
    mask_mov = torch.from_numpy(nib.load(os.path.join(base_mask_moving, os.listdir(base_mask_moving)[0])).get_fdata()).float()
    masked_mov = F.interpolate(((img_mov+1024)*mask_mov).unsqueeze(0).unsqueeze(0),scale_factor=.5,mode='trilinear').squeeze()

    kpts_fix = foerstner_kpts(img_fixed.unsqueeze(0).unsqueeze(0).cuda(), mask_fixed.unsqueeze(0).unsqueeze(0).cuda(), 1.4, 3).cpu()
  
    if(do_MIND):
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                mind_fixed = F.avg_pool3d(mask_fixed.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_fixed.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
                mind_mov = F.avg_pool3d(mask_mov.unsqueeze(0).unsqueeze(0).cuda().half()*\
                            MINDSSC(img_mov.unsqueeze(0).unsqueeze(0).cuda(),1,2).half(),2).cpu()
            
    return img_fixed,masked_fixed,img_mov,masked_mov,kpts_fix,mind_fixed,mind_mov,filename_fix


def get_warped_pair(_cf,_img_moving):
    H,W,D = _img_moving.shape[-3:]
    
    kpts_fixed = torch.flip((_cf[:,:3]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))
    kpts_moving = torch.flip((_cf[:,3:]-torch.tensor([H/2,W/2,D/2]).view(1,-1)).div(torch.tensor([H/2,W/2,D/2]).view(1,-1)),(-1,))


    with torch.no_grad():
        dense_flow = thin_plate_dense(kpts_fixed.unsqueeze(0).cuda(), (kpts_moving-kpts_fixed).unsqueeze(0).cuda(), (H, W, D), 4, 0.01).cpu()
    warped_img = F.grid_sample(_img_moving.view(1,1,H,W,D).cpu(),dense_flow+F.affine_grid(torch.eye(3,4).unsqueeze(0),(1,1,H,W,D))).squeeze()

        
    return warped_img,dense_flow

def main():
    
    img_fixed,masked_fixed,img_mov,masked_mov,kpts_fix,mind_fixed,mind_mov,filename_fix = read_input('/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/input/img_fixed','/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/input/img_moving','/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/input/mask_fixed','/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/input/mask_moving',do_MIND=True) #hardcoded image and mask paths
    
    unet_model,heatmap,mesh = get_vxmpp_models()

    state_dicts = torch.load('/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/voxelmorphplusplus.pth')  #model path 
    unet_model.load_state_dict(state_dicts[1])
    heatmap.load_state_dict(state_dicts[0])

    predictions = []
    
    #MASKED INPUT IMAGES ARE HALF-RESOLUTION
    with torch.no_grad():
        keypts_fix = kpts_fix.squeeze().cuda()
        H,W,D = masked_fixed.shape[-3:]

        masked_fixed = masked_fixed.view(1,1,H,W,D).cuda()
        masked_mov = masked_mov.view(1,1,H,W,D).cuda()

        with torch.cuda.amp.autocast():
            #VoxelMorph requires some padding
            input,x_start,y_start,z_start,x_end,y_end,z_end = return_crops(torch.cat((masked_fixed,masked_mov),1).cuda())
            output = F.pad(F.interpolate(unet_model(input),scale_factor=2),(z_start,(-z_end+D),y_start,(-y_end+W),x_start,(-x_end+H)))
            disp_est = torch.zeros_like(keypts_fix)
            for idx in torch.split(torch.arange(len(keypts_fix)),1024):
                sample_xyz = keypts_fix[idx]
                sampled = F.grid_sample(output,sample_xyz.cuda().view(1,-1,1,1,3),mode='bilinear')
                disp_pred = heatmap(sampled.permute(2,1,0,3,4))
                disp_est[idx] = torch.sum(torch.softmax(disp_pred.view(-1,11**3,1),1)*mesh.view(1,11**3,3),1)


    #NOW EVERYTHING FULL-RESOLUTION
    H,W,D = img_fixed.shape[-3:]

    fixed_mind = mind_fixed.view(1,-1,H//2,W//2,D//2).cuda()
    moving_mind = mind_mov.view(1,-1,H//2,W//2,D//2).cuda()

    pred_xyz,disp_smooth,dense_flow = adam_mind(keypts_fix,disp_est,fixed_mind,moving_mind,H,W,D)
    predictions = pred_xyz.cpu()+keypts_fix.cpu()

    keypts_fix = torch.flip(kpts_fix.squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
    keypts_moved = torch.flip(predictions.squeeze(),(1,))*torch.tensor([H/2,W/2,D/2])+torch.tensor([H/2,W/2,D/2])
            

    cf = torch.cat([keypts_fix, keypts_moved], dim=1)
    img_warped,dense_flow = get_warped_pair(cf,img_mov)
    H,W,D = img_mov.shape
    dense_flow = dense_flow.flip(4).permute(0, 4, 1, 2, 3) * torch.tensor( [H - 1, W - 1, D - 1]).view(1, 3, 1, 1, 1) / 2
    grid_sp = 1
    disp_lr = F.interpolate(dense_flow, size=(H // grid_sp, W // grid_sp, D // grid_sp), mode='trilinear',
                                                align_corners=False)
    disp_lr = disp_lr.permute(0,2,3,4,1)
    disp_tmp = disp_lr[0].permute(3,0,1,2).numpy()
    disp_lr = disp_lr[0].numpy()
    displacement_field = nib.Nifti1Image(disp_lr, None)
    #nib.save(displacement_field, 'Learn2Reg23/vxmpp_algorithms/output/disp_field.nii.gz')
    #case_number = filename_fix.split('CT')[1][0:5]
    #acq_time = filename_fix.split('.nii.gz')[0][-5:]
    #nib.save(displacement_field, 'Learn2Reg23/vxmpp_algorithms/output/disp'+case_number+acq_time+case_number+'_0000.nii.gz')
    #print(displacement_field.shape)
    
    '''
    #just for testing
    A = nib.load('/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/input/img_moving/ThoraxCBCT_0011_0000.nii.gz').affine
    identity = np.meshgrid(np.arange(H), np.arange(W), np.arange(D), indexing='ij')
    moving_warped = map_coordinates(img_mov, identity + disp_tmp, order=0)
    moving_warped = nib.Nifti1Image(moving_warped, A)
    nib.save(moving_warped,'/share/data_supergrover3/heyer/Learn2Reg23/vxmpp_algorithms/results_30_08_23/11_0000_warped.nii.gz')    
         '''   
        
        

if __name__ == "__main__":
    main()






