# Shared setup for the OpenFold3 container tests. Sourced by the run scripts.
# Edit the three settings at the top; everything below should be left alone
# unless you are deliberately running an experiment.

# ---------------------------------------------------------------- edit these
CONTAINER="${CONTAINER:-openfold3_rocm.sif}"
# Weights directory. OF3 looks in $OPENFOLD_CACHE, default ~/.openfold3
export OPENFOLD_CACHE="${OPENFOLD_CACHE:-/scratch/references/openfold3}"
# Node-local scratch. Must be node-local, not Lustre - see below.
LOCAL_SCRATCH="${LOCAL_SCRATCH:-/tmp/${USER}/of3_$$}"
# ---------------------------------------------------------------------------

module load singularity/3.11.4-nompi

mkdir -p "${LOCAL_SCRATCH}"

# Triton JIT cache. The OF3 AMD path runs Triton kernels, which are compiled on
# first use and cached. Pointing this at a shared filesystem means every array
# task compiles into the same directory over Lustre: slow at best, corrupt
# cache entries and spurious failures at worst. Keep it node-local.
export TRITON_CACHE_DIR="${LOCAL_SCRATCH}/triton"

# Same argument for MIOpen's kernel database.
export MIOPEN_USER_DB_PATH="${LOCAL_SCRATCH}/miopen"
export MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_USER_DB_PATH}"
mkdir -p "${TRITON_CACHE_DIR}" "${MIOPEN_USER_DB_PATH}"

# BLAS backend. The OF3-preview2 technical report's errata records hipBLASLt on
# MI300A producing structures with reduced chemical validity, corrected by
# falling back to rocBLAS. Nothing in the OF3 source pins this, so it inherits
# whatever the PyTorch ROCm build defaults to. We pin it to the known-good
# backend and let run_correctness.sh flip it to A/B the two.
export TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT:-0}"

# Bind mounts: weights, plus node-local scratch for the caches above.
SINGULARITY_ARGS=(
  --bind "${OPENFOLD_CACHE}:${OPENFOLD_CACHE}"
  --bind "${LOCAL_SCRATCH}:${LOCAL_SCRATCH}"
)

# Environment that must cross the container boundary.
export SINGULARITYENV_TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export SINGULARITYENV_MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH}"
export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${MIOPEN_CUSTOM_CACHE_DIR}"
export SINGULARITYENV_OPENFOLD_CACHE="${OPENFOLD_CACHE}"
export SINGULARITYENV_TORCH_BLAS_PREFER_HIPBLASLT="${TORCH_BLAS_PREFER_HIPBLASLT}"

of3_predict() {
  # of3_predict <query_json> <output_dir>
  srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 \
    singularity exec "${SINGULARITY_ARGS[@]}" "${CONTAINER}" \
    run_openfold predict \
      --query_json="$1" \
      --output_dir="$2" \
      --runner_yaml="${RUNNER_YAML:-runner_bench.yaml}" \
      --num_model_seeds="${NUM_MODEL_SEEDS:-1}" \
      --num_diffusion_samples="${NUM_DIFFUSION_SAMPLES:-1}" \
      --use_msa_server=false \
      --use_templates=false
}

of3_python() {
  # score.py needs numpy, which a bare compute node will not necessarily have.
  # The container does, so run it there rather than depending on the host
  # python. Runs on the CPU only - no --gres needed.
  singularity exec "${SINGULARITY_ARGS[@]}" --bind "${PWD}:${PWD}" \
    --pwd "${PWD}" "${CONTAINER}" python3 "$@"
}
