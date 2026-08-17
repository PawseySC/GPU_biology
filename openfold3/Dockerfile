FROM localhost/rocm-mpich-base-rocm7.2.3-mpich3.4.3-ubuntu24.04
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet \
    && apt-get install --no-install-recommends --yes --quiet \
        build-essential \
        cmake \
        git \
        tzdata \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get autoremove --yes \
    && apt-get clean

ENV PIP_NO_CACHE_DIR=true

# ROCm environment
ENV ROCM_RELEASE 7.2.3
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib

WORKDIR /opt

# Pin Miniforge installer (avoid "latest" busting cache every release).
ARG MINIFORGE_VERSION=25.3.0-3
RUN set -eux; \
    curl -fsSL -o /tmp/mf.sh \
      "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-$(uname -m).sh"; \
    bash /tmp/mf.sh -b -p /opt/miniforge3 -s; \
    rm -f /tmp/mf.sh
ENV PATH=/opt/miniforge3/bin:${PATH}

RUN git clone https://github.com/aqlaboratory/openfold-3.git --branch 0.4.5 /app/openfold-3 && \
    cd /app/openfold-3 && \
    mamba install python=3.13 -y && \
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.2 && \
    pip install openfold3 && \
    mamba clean -afy

WORKDIR /app/openfold-3

COPY Dockerfile /opt/docker-recipes/
LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="openfold-3 0.4.5 with ROCm7.2.3 support on Ubuntu 24.04 for AMD GPUs" \
      org.opencontainers.image.source="https://github.com/PawseySC/GPU_biology/tree/main/openfold-3/rocm7.2.3" \
      org.opencontainers.image.name="openfold-3_0.4.5_rocm7.2.3_ubuntu24.04" \
      Python_version="3.13"
