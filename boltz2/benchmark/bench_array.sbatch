#!/bin/bash --login
# SLURM array driver: one array task per manifest row.
#
# Do not sbatch this directly -- resource requests (--gres, --nodes) have to match the
# rows being run, so submit.sh splits the manifest by resource shape and passes the
# right options on the command line. Run:  ./submit.sh <manifest.tsv>
#
# There are deliberately no #SBATCH directives here. SLURM does not expand shell
# variables in directives, so an `#SBATCH --account=${PAWSEY_PROJECT}-gpu` line is
# taken literally and the job is rejected. submit.sh supplies --account, --partition,
# --nodes, --gres, --cpus-per-task, --time, --array and the log paths.

set -uo pipefail

MANIFEST="${1:?usage: bench_array.sbatch <manifest.tsv>}"
BENCH_ROOT="${BENCH_ROOT:?BENCH_ROOT must be exported}"
export BENCH_CACHE="${BENCH_CACHE:-/scratch/references/boltz}"
export BENCH_INPUTS_ROOT="${BENCH_INPUTS_ROOT:?BENCH_INPUTS_ROOT must be exported}"

# Non-MPI: each boltz process is independent, nothing here uses MPI.
module load "${SINGULARITY_MODULE:-singularity/3.11.4-nompi}"

# Row 1 of the file is the header, so array task N maps to file line N+1.
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "${MANIFEST}")
if [[ -z "${LINE}" ]]; then
    echo "No manifest row for array task ${SLURM_ARRAY_TASK_ID}" >&2
    exit 1
fi

IFS=$'\t' read -r RUN_ID TIER AXIS AXIS_VALUE INPUT BATCH_N TOKENS \
    RECYCLING_STEPS SAMPLING_STEPS DIFFUSION_SAMPLES MAX_PARALLEL_SAMPLES \
    AFFINITY USE_POTENTIALS DEVICES NODES LAUNCH_MODE NUM_WORKERS PREPROC_THREADS \
    IMAGE REPEAT \
    <<< "${LINE}"

export BENCH_ROOT BENCH_RUN_ID="${RUN_ID}" BENCH_IMAGE="${IMAGE}" \
       BENCH_INPUT="${INPUT}" BENCH_DEVICES="${DEVICES}" BENCH_NODES="${NODES}" \
       BENCH_LAUNCH_MODE="${LAUNCH_MODE}" \
       BENCH_RECYCLING_STEPS="${RECYCLING_STEPS}" \
       BENCH_SAMPLING_STEPS="${SAMPLING_STEPS}" \
       BENCH_DIFFUSION_SAMPLES="${DIFFUSION_SAMPLES}" \
       BENCH_MAX_PARALLEL_SAMPLES="${MAX_PARALLEL_SAMPLES}" \
       BENCH_USE_POTENTIALS="${USE_POTENTIALS}" \
       BENCH_NUM_WORKERS="${NUM_WORKERS}" \
       BENCH_PREPROC_THREADS="${PREPROC_THREADS}"

echo "=== ${RUN_ID} ==="
echo "tier=${TIER} axis=${AXIS}=${AXIS_VALUE} tokens=${TOKENS} devices=${DEVICES} nodes=${NODES} mode=${LAUNCH_MODE}"
echo "image=${IMAGE}"
echo "input=${INPUT}"

# One task per node; Lightning spawns the per-GCD workers inside it. 8 CPU cores per
# GCD matches the Setonix GPU node layout (64 cores / 8 GCDs).
srun --nodes="${NODES}" --ntasks="${NODES}" --ntasks-per-node=1 \
     --cpus-per-task=$(( DEVICES * 8 )) \
     --gres=gpu:"${DEVICES}" \
     "${BENCH_ROOT}/run_one.sh"
