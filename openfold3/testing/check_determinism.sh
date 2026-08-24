#!/bin/bash -l
#SBATCH --account=<pawsey_project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name=of3_determ
#SBATCH --output=logs/determ_%j.out

# Same input, same seed, twice. Run this before anything else: if the model is
# nondeterministic then golden-output comparison is meaningless and every other
# test in this directory becomes noise.
#
# The queries pin `seeds: [42]`, so a divergence here is the stack, not the
# config - typically a nondeterministic reduction or an autotuner picking a
# different kernel between runs.

set -euo pipefail
mkdir -p logs
source ./env.sh

INPUT="${1:-inputs/correctness/01_protein_monomer.json}"
NAME=$(basename "${INPUT}" .json)
BASE="results/determinism/${NAME}"
rm -rf "${BASE}"
mkdir -p "${BASE}/run_a" "${BASE}/run_b"

echo "=== run A ==="
of3_predict "${INPUT}" "${BASE}/run_a" 2>&1 | tee "${BASE}/run_a.log"
echo "=== run B ==="
of3_predict "${INPUT}" "${BASE}/run_b" 2>&1 | tee "${BASE}/run_b.log"

echo
echo "=== comparison ==="
of3_python score.py determinism "${BASE}/run_a" "${BASE}/run_b"
