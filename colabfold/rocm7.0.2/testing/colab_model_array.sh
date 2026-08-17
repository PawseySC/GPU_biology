#!/bin/bash -l
#SBATCH --job-name=CF-models
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=${PAWSEY_PROJECT}-gpu
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err

set -euo pipefail

# Load required module
module load singularity/3.11.4-nompi

# Set input and output paths
A3M=1IH7.2.a3m
OUT=$MYSCRATCH/colabfold/${SLURM_JOB_ID}
containerImage=docker://quay.io/pawsey/colabfold:1.6.1_rocm7.0.2

DATA_DIR=/scratch/references/colabfold_jun2026/database/

# Set JAX/XLA JIT compilation cache
export JAX_COMPILATION_CACHE_DIR=${MYSOFTWARE}/jax_cache


# ---- model list ------------------------------------------------------------
MODELS=(
  alphafold2
  alphafold2_multimer_v1
  alphafold2_multimer_v2
  alphafold2_multimer_v3
  alphafold2_ptm
)

# Guard against a mismatch between --array and the list above
if (( SLURM_ARRAY_TASK_ID >= ${#MODELS[@]} )); then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} out of range (list has ${#MODELS[@]} entries)" >&2
  exit 1
fi

MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

# ---- paths -----------------------------------------------------------------
OUTPUT_DIR="${OUTPUT_DIR:-$MYSCRATCH/colabfold/${SLURM_JOB_ID}}"
mkdir -p "$OUTPUT_DIR"

echo "host           : $(hostname)"
echo "array task     : ${SLURM_ARRAY_TASK_ID} / job ${SLURM_ARRAY_JOB_ID}"
echo "model          : ${MODEL}"

srun -N 1 -n 1 -c "$SLURM_CPUS_PER_TASK" --gres=gpu:1 \
  colabfold_batch \
    --model-type ${DATA_DIR}/${MODEL} \
    $A3M \
    "$OUTPUT_DIR" \
    --num-models 3 \
    --num-recycle 3 \
    --use-gpu-relax \
    --data $DATA_DIR