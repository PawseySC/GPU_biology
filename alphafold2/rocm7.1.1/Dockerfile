# ROCm MPICH base: pull or build locally — see alphafold2/rocm7.2.3/README.md (BASE_IMAGE).
ARG BASE_IMAGE=rocm-mpich-base:rocm7.1.1-mpich3.4.3-ubuntu24.04
FROM ${BASE_IMAGE}

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ENV ROCM_RELEASE=7.1.1
ENV ROCM_PATH /opt/rocm-$ROCM_RELEASE
ENV PATH $ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH
ENV LD_LIBRARY_PATH $LD_LIBRARY_PATH:$ROCM_PATH/lib:/opt/miniforge3/lib
ENV JAX_PLATFORMS "rocm,cpu"
ENV PIP_NO_CACHE_DIR=true

ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    apt-get update; \
    apt-get install --no-install-recommends --yes \
      build-essential \
      cmake \
      git \
      hmmer \
      kalign \
      tzdata \
      wget \
      clang \
      doxygen; \
    rm -rf /var/lib/apt/lists/*; \
    apt-get autoremove --yes; \
    apt-get clean

WORKDIR /opt

# Clone alphafold
ENV ALPHAFOLD_PATH=/app/alphafold
COPY triton-attention-v2.3.2.patch /opt/docker-recipes/triton-attention-v2.3.2.patch
COPY requirements.txt ${ALPHAFOLD_PATH}

RUN set -eux ; \
    git clone https://github.com/deepmind/alphafold.git ${ALPHAFOLD_PATH} && \
    cd ${ALPHAFOLD_PATH} && \
    git checkout e9b68483fc8764e7a1906e74a5d1f0ea2137311d && \
    git apply --verbose /opt/docker-recipes/triton-attention-v2.3.2.patch ; \
    wget -q -P ${ALPHAFOLD_PATH}/alphafold/common/ \
    https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt && \
    cd - 

# Install miniforge (mamba)
ARG MINIFORGE_VERSION=24.11.3-0
RUN set -eux; \
    curl -fsSL -O "https://github.com/conda-forge/miniforge/releases/download/${MINIFORGE_VERSION}/Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh"; \
    bash "Miniforge3-${MINIFORGE_VERSION}-Linux-x86_64.sh" -b -p /opt/miniforge3 -s; \
    rm -f Miniforge3-*.sh
ENV PATH /opt/miniforge3/bin:$PATH

RUN mamba install -y -c conda-forge -c bioconda \
    python=3.12 \
    bioconda::hhsuite=3.3.0 \
    pdbfixer \
    && mamba clean -afy

RUN pip install -r ${ALPHAFOLD_PATH}/requirements.txt 
ENV ROCM_JAX="https://github.com/ROCm/rocm-jax/releases/download/rocm-jax-v0.8.2"
RUN pip install jax==0.8.2
RUN pip install \
    "${ROCM_JAX}/jaxlib-0.8.2+rocm7-cp312-cp312-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" \
    "${ROCM_JAX}/jax_rocm7_pjrt-0.8.2+rocm7.1.1-py3-none-manylinux_2_28_x86_64.whl" \
    "${ROCM_JAX}/jax_rocm7_plugin-0.8.2+rocm7.1.1-cp312-cp312-manylinux_2_28_x86_64.whl"

# Triton 3.6.0 + jax-triton 0.3.1
RUN pip install triton==3.6.0
RUN pip install --no-deps jax-triton==0.3.1
RUN pip install --no-deps -e ${ALPHAFOLD_PATH}

COPY Dockerfile triton.patch /opt/docker-recipes/
LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \

ENV JAX_PLATFORMS=rocm,cpu
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_gpu_enable_command_buffer="
ENV TRITON_CACHE_DIR=/tmp/triton_cache
