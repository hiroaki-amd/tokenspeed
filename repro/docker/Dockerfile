# Reproduction environment for the skip-softmax MHA prefill measurements.
#
# Pinned deliberately: the numbers in the PR were produced on ROCm 7.2 with
# torch 2.11 and tokenspeed-triton 3.8.10.post20260721, and the whole point of
# this image is that a reader does not have to guess which toolchain was used.
# A ROCm 7.0 / torch 2.10 toolchain was also measured and agreed to within
# 0.006x on the synthetic sweep, so the results are not knife-edge dependent on
# these pins, but these are what was actually run.
#
# Note the kernel compiles through tokenspeed_triton, a separate top-level
# package, so whichever stock `triton` the base image carries is not on the
# path that matters here.
#
# Base image ships ROCm 7.2.4 and a matching torch build for gfx950 (MI350X).

FROM rocm/pytorch:rocm7.2.4_ubuntu24.04_py3.12_pytorch_release_2.10.0

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_PROGRESS_BARS=1

# torch 2.11+rocm7.2 over the image's 2.10, to match the measured toolchain.
RUN pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/rocm7.2 \
        torch==2.11.0+rocm7.2

# The Gluon dialect the kernel is written in lives in tokenspeed's Triton fork,
# which installs as its own `tokenspeed_triton` package and does not shadow the
# base image's `triton`. Installed after torch so torch's own dependency on
# pytorch-triton-rocm is resolved first and does not reorder this.
RUN pip install --no-cache-dir \
        tokenspeed-triton==3.8.10.post20260721

# The base image's torchvision and torchaudio are compiled against torch 2.10
# and abort at import time under 2.11 (a duplicate fake-impl registration for
# torchvision::nms). transformers imports torchvision opportunistically, so a
# text-only model load hits this. Nothing here needs either package.
RUN pip uninstall -y torchvision torchaudio

# Evaluation harness only. tokenspeed-kernel-amd itself deliberately does not
# depend on transformers, so these are installed here rather than declared as
# package dependencies.
RUN pip install --no-cache-dir \
        transformers==5.2.0 \
        accelerate==1.14.0 \
        tokenizers==0.22.2 \
        safetensors==0.8.0 \
        huggingface_hub==1.20.1 \
        numpy==2.4.6

WORKDIR /work

# Baked in from the repository this Dockerfile lives in, so the image
# reproduces exactly the kernel checked out at build time rather than
# whatever happens to be bind-mounted at run time. Build with the repository
# root as context (see run.sh), not this docker/ directory. The bundle
# (scripts/, results/) is still bind-mounted at run time, both so results
# land on the host and so a script edit does not need an image rebuild.
COPY tokenspeed-kernel-amd /work/tokenspeed-kernel-amd

ENV PYTHONPATH=/work/tokenspeed-kernel-amd/python:/work/bundle/scripts

CMD ["/bin/bash"]
