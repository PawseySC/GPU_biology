#!/bin/bash -l
#SBATCH --account=<pawsey_project>-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --job-name=of3_perf
#SBATCH --array=0-7
#SBATCH --output=logs/perf_%A_%a.out

# Token ladder. MSAs are off in these queries, so what is timed is the model,
# not the alignment pipeline - see README for measuring that separately.
#
# Expect the top of the ladder to OOM. That is the point: the ceiling is the
# number users ask for. A failed task here is a result, not a broken run.

set -uo pipefail
mkdir -p logs results/perf
source ./env.sh

INPUTS=(
  inputs/perf/ladder_01x_76tok.json
  inputs/perf/ladder_02x_152tok.json
  inputs/perf/ladder_04x_304tok.json
  inputs/perf/ladder_08x_608tok.json
  inputs/perf/ladder_16x_1216tok.json
  inputs/perf/ladder_32x_2432tok.json
  inputs/perf/mixed_protein_ligand.json
  inputs/perf/mixed_protein_rna.json
)

if (( SLURM_ARRAY_TASK_ID >= ${#INPUTS[@]} )); then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} out of range (${#INPUTS[@]} inputs)" >&2
  exit 1
fi

INPUT="${INPUTS[$SLURM_ARRAY_TASK_ID]}"
NAME=$(basename "${INPUT}" .json)
OUT="results/perf/${NAME}"
mkdir -p "${OUT}"

# Sample VRAM alongside the run. 2s, because diffusion sampling spikes over a
# few seconds and a coarse poll silently misses the peak.
./monitor_vram.sh "${OUT}/vram.csv" 2 &
MON=$!
trap 'kill ${MON} 2>/dev/null || true' EXIT

START=$(date +%s.%N)
of3_predict "${INPUT}" "${OUT}" 2>&1 | tee "${OUT}/predict.log"
RC=${PIPESTATUS[0]}
END=$(date +%s.%N)

kill ${MON} 2>/dev/null || true
wait ${MON} 2>/dev/null || true

# awk rather than bc: bc is not guaranteed to be installed on a compute node.
WALL=$(awk -v a="${START}" -v b="${END}" 'BEGIN { printf "%.3f", b - a }')
# Strip commas and quotes so an odd rocm-smi string cannot break the JSON.
GPU=$(rocm-smi --showproductname --csv 2>/dev/null | sed -n 2p | cut -d, -f2 \
      | tr -d '",' | sed 's/^ *//; s/ *$//')

# One row per run. report_perf.py turns these into the table you show people.
cat > "${OUT}/timing.json" <<EOF
{
  "name": "${NAME}",
  "input": "${INPUT}",
  "exit_code": ${RC},
  "wall_seconds": ${WALL},
  "gpu": "${GPU}",
  "blas_hipblaslt": ${TORCH_BLAS_PREFER_HIPBLASLT},
  "slurm_job": "${SLURM_JOB_ID}"
}
EOF

echo "done: ${OUT} (rc=${RC})"
exit ${RC}
