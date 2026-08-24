#!/bin/bash -l
#SBATCH --account=<pawsey_project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name=of3_correct
#SBATCH --array=0-7
#SBATCH --output=logs/correct_%A_%a.out

# One modality per array task. Targets are small and easy on purpose: if a
# result is bad, that is the container, not a hard target.
#
#   sbatch run_correctness.sh
#   TORCH_BLAS_PREFER_HIPBLASLT=1 sbatch run_correctness.sh   # the A/B arm

set -euo pipefail
mkdir -p logs
source ./env.sh

INPUTS=(
  inputs/correctness/01_protein_monomer.json
  inputs/correctness/02_protein_multimer.json
  inputs/correctness/03_protein_ligand_ccd.json
  inputs/correctness/04_protein_ligand_smiles.json
  inputs/correctness/05_rna_monomer.json
  inputs/correctness/06_dna_ptm.json
  inputs/correctness/07_protein_dna.json
  inputs/correctness/08_protein_rna.json
)

if (( SLURM_ARRAY_TASK_ID >= ${#INPUTS[@]} )); then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} out of range (${#INPUTS[@]} inputs)" >&2
  exit 1
fi

INPUT="${INPUTS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(basename "${INPUT}" .json)

# Results are keyed by BLAS backend so the two arms of the A/B never collide.
ARM="hipblaslt${TORCH_BLAS_PREFER_HIPBLASLT}"
OUT="results/correctness/${ARM}/${NAME}"
mkdir -p "${OUT}"

echo "=== ${NAME} | TORCH_BLAS_PREFER_HIPBLASLT=${TORCH_BLAS_PREFER_HIPBLASLT} ==="

# Environment check first - cheap, and it fails loudly if the Triton/HIP stack
# is not wired up, which otherwise shows up as a confusing runtime error.
singularity exec "${SINGULARITY_ARGS[@]}" "${CONTAINER}" \
  validate-openfold3-rocm 2>&1 | tee "${OUT}/rocm_check.txt"

of3_predict "${INPUT}" "${OUT}" 2>&1 | tee "${OUT}/predict.log"

echo "done: ${OUT}"
