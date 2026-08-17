#!/bin/bash -l
#SBATCH --account=${PAWSEYPROJECT}-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=esmfold
#SBATCH --array=0-5

# Load required module
module load singularity/3.11.4-nompi 

# Set container
CONTAINER=docker://quay.io/pawsey/esmfold_openfold:rocm6.3.3

# Set reference directory
REF_DIR=/scratch/references/alphafold_feb2024/databases

# Set directory where models have been predownloaded for you
export TORCH_HOME=/scratch/references/esmfold/models


# ---- input list ------------------------------------------------------------
INPUTS=(
  10.fasta
  1280aa.fasta
  3013aa.fasta
  5005aa.fasta
  T1204.fasta
  T1206.fasta
)

# Guard against a mismatch between --array and the list above
if (( SLURM_ARRAY_TASK_ID >= ${#INPUTS[@]} )); then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} out of range (list has ${#INPUTS[@]} entries)" >&2
  exit 1
fi

INPUT="${INPUTS[$SLURM_ARRAY_TASK_ID]}"

# Create results directory
RESULTS_DIR=results/${INPUT}/${SLURM_JOB_ID}
mkdir -p $RESULTS_DIR

# Run the container
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec \
  $CONTAINER \
  esm-fold -i $INPUT -o ${RESULTS_DIR} --chunk-size 16