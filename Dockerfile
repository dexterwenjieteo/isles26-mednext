# ISLES 2026 submission — MedNeXt-L (kernel 5) 5-fold ensemble
#
# Base image matches the official ISLES26 template:
#   cuda 12.6 is required for the T4 GPU instances used on Grand Challenge.
FROM --platform=linux/amd64 pytorch/pytorch:2.13.0-cuda12.6-cudnn9-runtime AS isles26_mednext_l

# Ensures Python output to stdout/stderr is not buffered.
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Non-root user (best practice, mirrors the official template).
RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user
WORKDIR /opt/app

# Virtualenv inheriting the system site-packages (torch + CUDA libraries).
RUN python -m venv --system-site-packages --without-pip /home/user/venv
ENV PATH="/home/user/venv/bin:$PATH"

# Python dependencies (torch/CUDA already come from the base image).
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install --no-cache-dir --no-color --requirement /opt/app/requirements.txt

# Vendored nnU-Net v2 + MedNeXt source — the EXACT pure-Python packages that
# produced the trained weights, including the custom MedNeXt trainer.
COPY --chown=user:user vendor/ /opt/app/vendor/
ENV PYTHONPATH="/opt/app/vendor:${PYTHONPATH}"

# Application code.
COPY --chown=user:user app.py /opt/app/
COPY --chown=user:user inference.py /opt/app/

# Required label — Grand Challenge uses it to detect the invoke API.
LABEL org.grand-challenge.api-method="invoke"
EXPOSE 4743
ENTRYPOINT ["python", "app.py"]
