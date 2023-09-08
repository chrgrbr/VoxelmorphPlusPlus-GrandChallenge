FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

RUN groupadd -r user && useradd -m --no-log-init -r -g user user

RUN mkdir -p /opt/app /input /output \
    && chown user:user /opt/app /input /output

USER user
WORKDIR /opt/app

ENV PATH="/home/user/.local/bin:${PATH}"

RUN python -m pip install --user -U pip && python -m pip install --user pip-tools



COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install -r requirements.txt


COPY --chown=user:user model_weights /opt/algorithm/
COPY --chown=user:user process.py filters.py foerstner.py thin_plate_spline.py vxmplusplus_utils.py /opt/app/

ENTRYPOINT [ "python", "-m", "process" ]
