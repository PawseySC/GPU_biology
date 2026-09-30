FROM rocm/dev-ubuntu-24.04:7.0.2

LABEL org.opencontainers.image.authors="Sarah Beecroft <sarah.beecroft@csiro.au>" \
      org.opencontainers.image.description="Tiberius v2.0.7 with ROCm7.0.2 support on Ubuntu 24.04 for AMD GPUs" \
      org.opencontainers.image.source="https://github.com/PawseySC/GPU_biology/tree/main/tiberius/v2.0.7" \
      org.opencontainers.image.name="tiberius_v2.0.7_rocm7.0.2_ubuntu24.04"

# Use bash to support string substitution.
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

WORKDIR /opt

# ROCm environment
ENV ROCM_RELEASE=7.0.2
ENV ROCM_PATH=/opt/rocm-$ROCM_RELEASE
ENV PATH=$ROCM_PATH/bin:$ROCM_PATH/llvm/bin:$PATH \
    LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ROCM_PATH/lib \
    PIP_NO_CACHE_DIR=1

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update --quiet && \
    apt-get install --no-install-recommends --yes --quiet \
    build-essential \
    time \
    libgsl-dev \
    libboost-all-dev \
    libsuitesparse-dev \
    liblpsolve55-dev \
    libsqlite3-dev \
    libmysql++-dev \
    libbamtools-dev \
    samtools \
    libhts-dev \
    cdbfasta \
    libfile-which-perl \
    libparallel-forkmanager-perl \
    libyaml-perl \
    libdbd-mysql-perl \
    bamtools \
    bedtools \
    diamond-aligner \
    seqkit \
    transdecoder \
    minimap2 \
    perl \
    libdb-dev \
    zlib1g-dev \
    hisat2 \
    curl \
    ca-certificates \
    python3-venv \
    miopen-hip \
    hipblas \
    hipblaslt \
    hipfft \
    hiprand \
    hipsolver \
    hipsparse \
    rccl \
    && \
    apt-get autoremove --yes && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install cpanm + Perl modules (URI includes URI::Escape)
RUN curl -fsSL https://cpanmin.us -o /tmp/cpanm.pl && \
    perl /tmp/cpanm.pl App::cpanminus && \
    cpanm --notest DB_File URI && \
    rm -rf /tmp/cpanm.pl /root/.cpanm

COPY apply_keras_rocm_patch.py constraints.txt Dockerfile pre_build_script.sh /opt/docker-recipes/
COPY vendor /opt

# ---- Augustus ---------------------------------------------------------------
RUN cd Augustus && \
    make clean && \
    make -j"$(nproc)" && \
    make install

# ---- miniprot + helpers -----------------------------------------------------
RUN make -C miniprot -j"$(nproc)" && \
    make -C miniprot-boundary-scorer -j"$(nproc)"

# ---- isolated Python environment --------------------------------------------
ENV VIRTUAL_ENV=/opt/tiberius-venv
ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
RUN python3 -m venv ${VIRTUAL_ENV} && \
    pip install --upgrade pip

# Versions are pinned by constraints.txt (pip freeze of the working image).
RUN C=/opt/docker-recipes/constraints.txt && \
    pip install -c $C \
    setuptools \
    wheel \
    "hatchling>=1.26" \
    "packaging>=24.0" \
    "pyfamsa<0.6.0" \
    pyyaml \
    pyBigWig \
    bio \
    scikit-learn \
    biopython \
    bcbio-gff \
    requests \
    numpy \
    pydantic \
    hidten \
    pandas && \
    pip install -c $C "bricks2marble[tf]" --no-deps && \
    pip install -c $C tensorflow-rocm==2.19.1 keras==3.15.1 -f https://repo.radeon.com/rocm/manylinux/rocm-rel-7.0.2/ --upgrade && \
    pip install -c $C --no-deps learnMSA

# Keras 3 calls the cuDNN LSTM op batch-major, which ROCm MIOpen rejects
# ("ROCm MIOpen only supports packed input output"). Patch it to time-major.
RUN python /opt/docker-recipes/apply_keras_rocm_patch.py

# ---- Tiberius ---------------------------------------------------------------
RUN cd Tiberius && \
    pip install . --no-deps && \
    chmod +x tiberius.py tiberius/*py && \
    mkdir -p /opt/Tiberius/model_weights && \
    chmod -R 777 /opt/Tiberius/model_weights && \
    ln -s /opt/Tiberius/tiberius.py /usr/local/bin/tiberius && \
    rm -rf /opt/Tiberius/docs /opt/Tiberius/figures /opt/Tiberius/test_data

ENV PATH=/usr/local/bin/:${PATH}:/opt/Augustus/bin/:/opt/Tiberius/tiberius/:/opt/Tiberius/:/opt/Tiberius/tiberius/scripts:/opt/miniprothint:/opt/miniprot:/opt/miniprot-boundary-scorer:/opt/stringtie:/opt/sratoolkit/bin:/opt/gffread:/opt/minimap2
