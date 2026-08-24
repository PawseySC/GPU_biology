# AlphaFold2 v2.3.2 on ROCm 7.2.3 with the opt-in Triton Evoformer attention
# kernel (triton-attention-v2.3.2.patch).
#
# Derived from alphafold2/rocm7.2.3/Dockerfile. Differences, all forced by
# jax-triton 0.3.1 requiring jax>=0.8.2:
#   * JAX ROCm 0.7.1 -> 0.8.2 (wheels from ROCm/rocm-jax, not ROCm/jax)
#   * numpy 1.26.4 -> 2.0.2, scipy >=1.13 (jax 0.8.2 hard requirements)
#   * triton==3.6.0 + jax-triton==0.3.1
#   * the Triton patch applied to the AlphaFold checkout
#
# The kernel is inert unless AF2_USE_TRITON=1 is set at *run* time; with it
# unset this image behaves exactly like the stock ROCm 7.2.3 one.
#
# ROCm MPICH base: pull or build locally — see alphafold2/rocm7.2.3/README.md.
ARG BASE_IMAGE=rocm-mpich-base:rocm7.2.3-mpich3.4.3-ubuntu24.04
FROM ${BASE_IMAGE}

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ROCm environment (base image installs to /opt/rocm — not /opt/rocm-$ROCM_RELEASE).
ENV ROCM_RELEASE=7.2.3
ENV ROCM_PATH=/opt/rocm
ENV PATH=${ROCM_PATH}/bin:${ROCM_PATH}/lib/llvm/bin:$PATH
# ROCm 7 libs must precede conda's ROCm 6 stubs (libhsa-runtime64, libamdhip64)
# so jax_rocm7_plugin finds the correct symbol versions at runtime.
ENV LD_LIBRARY_PATH=${ROCM_PATH}/lib:/opt/mpich/lib:/opt/miniforge3/lib
ENV JAX_PLATFORMS=rocm,cpu
# JAX/XLA on ROCm 7.2.3: avoid "Too small divisible part of the contracting
# dimension" (disable XLA's *internal* Triton GEMM path). This is unrelated to
# the jax-triton custom call used by the Evoformer kernel below — that one is
# gated purely by AF2_USE_TRITON.
ENV XLA_PYTHON_CLIENT_PREALLOCATE=false
ENV XLA_PYTHON_CLIENT_ALLOCATOR=platform
ENV JAX_PLATFORM_NAME=gpu
ENV XLA_FLAGS="--xla_gpu_autotune_level=0 --xla_gpu_enable_triton_gemm=false"
# XLA ROCm: lld on PATH; LLVM for some jaxlib paths.
ENV LLVM_PATH=${ROCM_PATH}/lib/llvm
ENV HIP_DEVICE_LIB_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_BITCODE_PATH=${ROCM_PATH}/lib/llvm/amdgcn/bitcode
ENV JAX_ROCM_PLUGIN_INTERNAL_LLD_PATH=${ROCM_PATH}/lib/llvm/bin

# Triton JIT cache. Default is ~/.triton, which is not writable when the
# container runs as a non-root uid (Setonix / Slurm). /tmp always is.
ENV TRITON_CACHE_DIR=/tmp/triton_cache
# Opt-in switch read by alphafold/model/triton/__init__.py. Left at 0 so the
# image is a drop-in replacement; set -e AF2_USE_TRITON=1 to enable.
ENV AF2_USE_TRITON=0

ARG DEBIAN_FRONTEND=noninteractive
RUN set -eux; \
    printf '%s\n' \
      'Acquire::Retries "8";' \
      'Acquire::http::Timeout "30";' \
      'Acquire::https::Timeout "30";' \
      'Acquire::ForceIPv4 "true";' \
      > /etc/apt/apt.conf.d/99-network-retries; \
    find /etc/apt -type f \( -name '*.list' -o -name '*.sources' \) -print0 \
      | xargs -0 sed -ri 's|http://archive.ubuntu.com/ubuntu|https://security.ubuntu.com/ubuntu|g' || true; \
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
RUN set -eux ; \
    git clone --branch v2.3.2 https://github.com/google-deepmind/alphafold.git $ALPHAFOLD_PATH ; \
    cd $ALPHAFOLD_PATH ; \
    sed -i 's#CUDA#OpenCL#g' alphafold/relax/amber_minimize.py ; \
    sed -i 's#HIP#OpenCL#g' alphafold/relax/amber_minimize.py ; \
    cd $ALPHAFOLD_PATH/alphafold/common ; \
    curl -LO https://git.scicore.unibas.ch/schwede/openstructure/-/raw/7102c63615b64735c4941278d92b554ec94415f8/modules/mol/alg/src/stereo_chemical_props.txt

# --- Triton Evoformer attention -------------------------------------------
# triton-attention-v2.3.2.patch is the upstream triton-attention.patch rebased
# onto the v2.3.2 tag. The original was cut against alphafold `main`, whose
# modules.py is reformatted relative to v2.3.2 (`key_dim ** (-0.5)` vs
# `key_dim**(-0.5)`, multi-line `inference_subbatch(...)` calls), so it does not
# apply here. The rebase is textual only — the five hunks are byte-identical in
# meaning to the original.
#
# Applied *before* the compatibility seds below so those also rewrite the files
# the patch adds.
COPY triton-attention-v2.3.2.patch /opt/docker-recipes/triton-attention-v2.3.2.patch
RUN set -eux ; \
    cd $ALPHAFOLD_PATH ; \
    git apply --verbose /opt/docker-recipes/triton-attention-v2.3.2.patch ; \
    test -f alphafold/model/triton/evoformer_attn.py ; \
    python3 -c "import ast,sys; [ast.parse(open(f).read()) for f in ('alphafold/model/modules.py','alphafold/model/triton/evoformer_attn.py')]; print('Triton patch applied OK')"

COPY config.py ${ALPHAFOLD_PATH}/alphafold/model
COPY amber_minimize.py ${ALPHAFOLD_PATH}/alphafold/relax/amber_minimize.py

# Install miniforge (mamba)
RUN set -eux ; \
    curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" ; \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/miniforge3 -s ; \
    rm -rf ./Miniforge3-*
ENV PATH=/opt/miniforge3/bin:$PATH

# streamhpc::openmm-hip==8.0.0 is gone from repodata; conda-forge openmm-hip 8.1.1 solves with pdbfixer.
RUN mamba install -y -c conda-forge -c bioconda --strict-channel-priority \
    python=3.11 \
    'setuptools<82' \
    pdbfixer==1.9 \
    bioconda::hhsuite \
    openmm-hip=8.1.1 \
    && mamba clean -afy

# Install pip packages.
COPY requirements.txt /app/alphafold/requirements.txt
# JAX ROCm 7: AMD order pjrt -> plugin -> jaxlib -> jax (after clearing PyPI jax
# from requirements). Note the repo moved from ROCm/jax to ROCm/rocm-jax for the
# 0.8.x line.
#
# ROCM_JAX_VARIANT selects which ROCm the wheels were built against:
#   rocm7.2.0 (default) matches this 7.2.3 base image
#   rocm7.1.1           is the combination the Triton patch author validated
ARG ROCM_JAX_TAG=rocm-jax-v0.8.2
ARG JAX_ROCM_VER=0.8.2
ARG ROCM_JAX_VARIANT=rocm7.2.0
ENV ROCM_JAX_URL=https://github.com/ROCm/rocm-jax/releases/download/${ROCM_JAX_TAG}
RUN pip3 install -r /app/alphafold/requirements.txt --no-cache-dir --root-user-action=ignore --break-system-packages && \
    pip uninstall -y jax jaxlib jax-rocm60-pjrt jax-rocm60-plugin jax-rocm7-pjrt jax-rocm7-plugin 2>/dev/null || true && \
    rm -rf /opt/miniforge3/lib/python3.11/site-packages/jax_plugins && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "${ROCM_JAX_URL}/jax_rocm7_pjrt-${JAX_ROCM_VER}+${ROCM_JAX_VARIANT}-py3-none-manylinux_2_28_x86_64.whl" && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "${ROCM_JAX_URL}/jax_rocm7_plugin-${JAX_ROCM_VER}+${ROCM_JAX_VARIANT}-cp311-cp311-manylinux_2_28_x86_64.whl" && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "${ROCM_JAX_URL}/jaxlib-${JAX_ROCM_VER}+${ROCM_JAX_VARIANT}-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl" && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "jax==${JAX_ROCM_VER}" --no-deps && \
    PY_SITE="$(python3 -c 'import site; print(site.getsitepackages()[0])')" && \
    mkdir -p "${PY_SITE}/rocm" && ln -sfn "${ROCM_PATH}/lib" "${PY_SITE}/rocm/lib" && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages 'setuptools<82' && \
    python3 -c "import setuptools; print('setuptools', setuptools.__version__, 'OK')" && \
    python3 -c "import importlib.metadata as m; [print(p, m.version(p)) for p in ('jax-rocm7-pjrt','jax-rocm7-plugin','jaxlib','jax')]; print('JAX ROCm pip packages OK')"

# Triton + jax-triton.
# The upstream PyPI `triton` wheel bundles the AMD (amdgcn) backend, so no
# pytorch-triton-rocm is needed. jax-triton is installed --no-deps: it is a pure
# Python wheel whose only real dependencies (jax, triton) are already pinned
# above, and letting pip resolve them would drag a PyPI CUDA jax back in.
#
# IMPORTANT (from the patch author): jax-triton 0.3.1 requires jax>=0.8.2. On
# jax 0.8.0 the TritonKernel constructor signature differs and compiled kernels
# are silently corrupted — do not downgrade JAX here.
ARG TRITON_VER=3.6.0
ARG JAX_TRITON_VER=0.3.1
RUN python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages \
        "triton==${TRITON_VER}" && \
    python3 -m pip install --no-cache-dir --root-user-action=ignore --break-system-packages --no-deps \
        "jax-triton==${JAX_TRITON_VER}" && \
    mkdir -p "${TRITON_CACHE_DIR}" && chmod 1777 "${TRITON_CACHE_DIR}" && \
    JAX_PLATFORMS=cpu python3 -c "import triton, jax_triton, importlib.metadata as m; print('triton', m.version('triton'), '/ jax-triton', m.version('jax-triton'), 'import OK')"

# Fix OpenMM imports in AlphaFold
RUN find /app/alphafold -type f -name "*.py" -exec sed -i \
    -e 's/from simtk\.openmm/from openmm/g' \
    -e 's/from simtk import openmm/import openmm/g' \
    -e 's/import simtk\.openmm/import openmm/g' \
    -e 's/simtk\.openmm\./openmm./g' \
    {} +

RUN sed -i 's/from simtk.openmm.app.internal.pdbstructure import PdbStructure/from openmm.app.internal.pdbstructure import PdbStructure/g' \
    /app/alphafold/alphafold/relax/amber_minimize.py && \
    find /app/alphafold -type f -name "*.py" -exec sed -i \
    's/simtk\.unit/openmm.unit/g' {} +

# JAX 0.6+ / 0.7 / 0.8 vs AlphaFold v2.3.2: tree_map and jax.util.wraps removed.
RUN find /app/alphafold -type f -name "*.py" -exec sed -i \
    's/jax\.tree_map(/jax.tree_util.tree_map(/g' \
    {} + && \
    sed -i 's/@jax\.util\.wraps(fun, docstr=docstr)/@functools.wraps(fun)/' \
    /app/alphafold/alphafold/model/mapping.py

# NumPy 2 (required by jax 0.8.2) removed np.string_ in favour of np.bytes_.
# AlphaFold v2.3.2 uses it in the multimer MSA pairing path, which would
# otherwise fail only once a multimer job reaches feature processing.
RUN find /app/alphafold -type f -name "*.py" -exec sed -i \
    's/np\.string_/np.bytes_/g' {} + && \
    ! grep -rn "np\.string_" /app/alphafold --include="*.py"

# Optional pymol-open-source (cp311 wheel): same OpenGL-related apt stack as
# colabfold/rocm7.2.3/ for scripts/split_and_fold_segments_alphafold2_single_container.py.
# Default 0 keeps the image smaller for inference-only / two-container workflows.
ARG INSTALL_PYMOL=0
RUN set -eux; \
    if [ "${INSTALL_PYMOL}" = "1" ]; then \
      apt-get update && \
      apt-get install --no-install-recommends --yes \
        libgl1 \
        libglu1-mesa \
        libglew2.2 \
        libpng16-16t64 \
        libfreetype6 \
        libxml2 \
        libglib2.0-0t64 \
      && rm -rf /var/lib/apt/lists/* \
      && pip3 install --no-cache-dir --root-user-action=ignore --break-system-packages \
        pymol-open-source==3.1.0a0 \
      && python3 -c "import pymol; print('PyMOL', getattr(pymol, '__version__', '3.1.0a0'), 'OK')"; \
    else \
      echo "INSTALL_PYMOL=${INSTALL_PYMOL}: skipping pymol-open-source"; \
    fi

# setuptools>=82 removed pkg_resources (still required by TensorFlow / AlphaFold).
RUN python3 -m pip install --force-reinstall --no-cache-dir --root-user-action=ignore --break-system-packages \
        'setuptools<82' \
    && python3 -c "import setuptools; print('setuptools', setuptools.__version__, 'OK')"

WORKDIR /app/alphafold

# Build-time sanity check. No GPU is visible during `docker build`, so this can
# only confirm the fallback path: AF2 imports, and with AF2_USE_TRITON unset the
# Triton branch stays off. The real GO/NO-GO is tests/triton_tests/test_smoke.py,
# which must be run on a node with an AMD GPU (see README).
RUN set -eux; \
    JAX_PLATFORMS=cpu AF2_USE_TRITON= python3 -c "\
from alphafold.model.triton import TRITON_AVAILABLE, USE_TRITON; \
assert TRITON_AVAILABLE, 'triton/jax_triton did not import'; \
assert not USE_TRITON, 'USE_TRITON should be off without AF2_USE_TRITON=1'; \
import alphafold.model.modules; \
print('AF2 imports OK; Triton available, opt-in via AF2_USE_TRITON=1')"

COPY Dockerfile requirements.txt config.py amber_minimize.py /opt/docker-recipes/
LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="AlphaFold v2.3.2 on ROCm 7.2.3 with opt-in Triton Evoformer flash-attention (AF2_USE_TRITON=1), JAX 0.8.2, triton 3.6.0, jax-triton 0.3.1" \
      org.opencontainers.image.source="https://github.com/PawseySC/GPU_biology/alphafold2/rocm7.2.3_triton" \
      org.opencontainers.image.name="alphafold2v2.3.2_rocm7.2.3_triton"

# ---------------------------------------------------------------------------
# Enabling the kernel at run time
#
#   docker run --rm -it ... -e AF2_USE_TRITON=1 <image> python3 run_alphafold.py ...
#
# With AF2_USE_TRITON unset (or not 1) every code path is the stock XLA one, so
# this image is a safe drop-in for alphafold2-amd-gpu:v2.3.2_rocm7.2.3.
#
# GO/NO-GO on real hardware, cheapest first (needs a GPU; run from /app/alphafold
# so `tests.triton_tests.kernels` is importable):
#
#   python3 -m pytest tests/triton_tests/test_smoke.py -v        # jax-triton works at all
#   python3 -m pytest tests/triton_tests/test_kernel.py -v       # kernel vs pure-JAX reference
#   AF2_USE_TRITON=1 python3 -m pytest tests/triton_tests/test_integration.py -v
#
# Persistence (edits vs container lifecycle)
#
# Everything under /app/alphafold is fixed at *image build* time (clone + patch
# + COPY above). Recreating the container from the same image does not lose that
# tree, but rebuilding the image or starting from a fresh clone does — unless you
# keep your fork on the host and bind-mount it at run time:
#
#   mkdir -p "$HOME/alphafold-rocm" && cd "$HOME/alphafold-rocm"
#   git clone --branch v2.3.2 https://github.com/google-deepmind/alphafold.git .
#   git apply /path/to/triton-attention-v2.3.2.patch
#   # plus the same seds as this Dockerfile (or git remote + pull your branch)
#   docker run --rm -it \
#     -v "$HOME/alphafold-rocm:/app/alphafold" \
#     ...your gpu/rocm flags... \
#     your-image:tag
#
# The bind mount replaces /app/alphafold in the container; all edits live on
# the host. Large databases are usually separate mounts (e.g. /work/databases).
# ---------------------------------------------------------------------------
