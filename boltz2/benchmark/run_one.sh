#!/bin/bash
# Execute a single benchmark run. Invoked by bench_array.sbatch under srun, once per
# node. Reads its configuration from the environment (exported by the sbatch script).
#
# Required: BENCH_ROOT BENCH_RUN_ID BENCH_IMAGE BENCH_INPUT BENCH_CACHE
#           BENCH_DEVICES BENCH_NODES BENCH_LAUNCH_MODE and the sampling parameters.
set -uo pipefail

RANK="${SLURM_PROCID:-0}"
RUN_DIR="${BENCH_ROOT}/results/${BENCH_RUN_ID}"
[[ "${BENCH_NODES}" -gt 1 ]] && RUN_DIR="${RUN_DIR}/rank${RANK}"
mkdir -p "${RUN_DIR}"

MODE="${BENCH_LAUNCH_MODE:-ddp}"

# ---------------------------------------------------------------------------
# Preflight: Setonix compute nodes have no outbound internet, so an unpopulated
# weights cache does not "download slowly", it hangs or dies mid-sweep. Fail fast
# and loudly rather than burning an array of jobs.
# ---------------------------------------------------------------------------
if ! find "${BENCH_CACHE}" -maxdepth 4 -iname '*.ckpt' -size +0c -print -quit 2>/dev/null | grep -q .; then
    echo "ERROR: no Boltz checkpoint (*.ckpt) under BENCH_CACHE=${BENCH_CACHE}." >&2
    echo "       Compute nodes have no outbound internet -- pre-stage the cache first" >&2
    echo "       (see boltz2/v2.2.1/testing.MD, 'Model weights cache')." >&2
    exit 3
fi

# ---------------------------------------------------------------------------
# Per-process scratch for JIT / kernel-tuning / library caches.
#
# MIOpen, Triton and numba default to $HOME (or a read-only path inside the .sif).
# On Lustre with hundreds of array tasks that is slow and a metadata-server hazard,
# and it silently pollutes timings -- the first run pays for the tuning DB and the
# rest free-ride. Pin them to node-local storage so every run pays the same cost.
#
# WANDB_MODE=disabled matters more than it looks: boltz depends on wandb, and an
# accidental network call on an internet-less compute node stalls the run and
# corrupts the measurement.
# ---------------------------------------------------------------------------
NODE_TMP="${TMPDIR:-/tmp}/boltzbench-${SLURM_JOB_ID:-$$}-${RANK}"

setup_env() {
    # $1 = per-process scratch dir, $2 = HIP_VISIBLE_DEVICES value ("" = inherit all)
    local tmp="$1" gcd="$2"
    mkdir -p "${tmp}"/{miopen,triton,inductor,numba,mpl,home,tmp}
    export SINGULARITYENV_MIOPEN_USER_DB_PATH="${tmp}/miopen"
    export SINGULARITYENV_MIOPEN_CUSTOM_CACHE_DIR="${tmp}/miopen"
    export SINGULARITYENV_TRITON_CACHE_DIR="${tmp}/triton"
    export SINGULARITYENV_TORCHINDUCTOR_CACHE_DIR="${tmp}/inductor"
    export SINGULARITYENV_NUMBA_CACHE_DIR="${tmp}/numba"
    export SINGULARITYENV_MPLCONFIGDIR="${tmp}/mpl"
    export SINGULARITYENV_XDG_CACHE_HOME="${tmp}/home"
    export SINGULARITYENV_TMPDIR="${tmp}/tmp"
    export SINGULARITYENV_WANDB_MODE="disabled"
    export SINGULARITYENV_TQDM_DISABLE="1"
    export SINGULARITYENV_PYTORCH_HIP_ALLOC_CONF="${BENCH_ALLOC_CONF:-expandable_segments:True}"
    export SINGULARITYENV_OMP_NUM_THREADS="${BENCH_PREPROC_THREADS:-8}"
    export SINGULARITYENV_BENCH_RUN_ID="${BENCH_RUN_ID}"
    if [[ -n "${gcd}" ]]; then
        export SINGULARITYENV_HIP_VISIBLE_DEVICES="${gcd}"
    else
        unset SINGULARITYENV_HIP_VISIBLE_DEVICES
    fi
}

# ---------------------------------------------------------------------------
# Shard a directory of YAMLs round-robin into $2 slices, emitting slice $3 as a
# directory of symlinks under $4.
# ---------------------------------------------------------------------------
make_shard() {
    local src="$1" n="$2" idx="$3" dst="$4" i=0 f
    mkdir -p "${dst}"
    for f in "${src}"/*.yaml; do
        [[ -e "${f}" ]] || continue
        if (( i % n == idx )); then ln -sf "${f}" "${dst}/$(basename "${f}")"; fi
        ((i++))
    done
    printf '%s\n' "${dst}"
}

count_inputs() {
    if [[ -d "$1" ]]; then find "$1" -name '*.yaml' | wc -l | tr -d '[:space:]'; else echo 1; fi
}

# Multi-node: each node takes a disjoint slice so aggregate throughput is real.
INPUT="${BENCH_INPUT}"
if [[ "${BENCH_NODES}" -gt 1 && -d "${BENCH_INPUT}" ]]; then
    INPUT=$(make_shard "${BENCH_INPUT}" "${BENCH_NODES}" "${RANK}" "${RUN_DIR}/shard")
fi

# ---------------------------------------------------------------------------
# Build the boltz argument list. --no_kernels is unconditional: trifast triangular
# kernels are not usable on this build.
# ---------------------------------------------------------------------------
# Populates the global BOLTZ_ARGV. Built directly as an array rather than round-tripped
# through text: `mapfile` is bash 4+ and this has to stay portable.
set_boltz_args() {
    # $1 = input path, $2 = --devices value, $3 = out_dir
    BOLTZ_ARGV=(
        predict "$1"
        --cache "${BENCH_CACHE}"
        --out_dir "$3"
        --no_kernels
        --accelerator gpu
        --devices "$2"
        --recycling_steps "${BENCH_RECYCLING_STEPS}"
        --sampling_steps "${BENCH_SAMPLING_STEPS}"
        --diffusion_samples "${BENCH_DIFFUSION_SAMPLES}"
        --max_parallel_samples "${BENCH_MAX_PARALLEL_SAMPLES}"
        --num_workers "${BENCH_NUM_WORKERS}"
        --preprocessing-threads "${BENCH_PREPROC_THREADS}"
        --output_format mmcif
    )
    [[ "${BENCH_USE_POTENTIALS}" == "1" ]] && BOLTZ_ARGV+=(--use_potentials)
    return 0
}

launch() {
    # $1 = workdir for results, $2 = input, $3 = devices, $4 = HIP_VISIBLE_DEVICES
    local wd="$1" inp="$2" dev="$3" gcd="$4"
    mkdir -p "${wd}/out"
    setup_env "${NODE_TMP}/$(basename "${wd}")" "${gcd}"
    set_boltz_args "${inp}" "${dev}" "${wd}/out"
    singularity exec \
        -B "${BENCH_ROOT}" -B "${BENCH_INPUTS_ROOT}" -B "${BENCH_CACHE}" -B "${NODE_TMP}" \
        "${BENCH_IMAGE}" \
        python3 "${BENCH_ROOT}/bench_wrap.py" \
            --bench-json "${wd}/bench.json" --run-id "${BENCH_RUN_ID}" \
            -- "${BOLTZ_ARGV[@]}" \
        > "${wd}/stdout.log" 2> "${wd}/stderr.log"
}

# ---------------------------------------------------------------------------
# Power / utilisation sampler (host side, outside the container). One per node.
# ---------------------------------------------------------------------------
MON_PID=""
if [[ "${BENCH_MONITOR:-1}" == "1" ]]; then
    python3 "${BENCH_ROOT}/gpu_monitor.py" \
        --out "${RUN_DIR}/gpu.csv" --interval "${BENCH_MONITOR_INTERVAL:-1.0}" &
    MON_PID=$!
fi

T_START=$(date +%s.%N)

if [[ "${MODE}" == "indep" && "${BENCH_DEVICES}" -gt 1 ]]; then
    # N independent single-GCD processes, each pinned with HIP_VISIBLE_DEVICES.
    # This is the launch model the container's own GCD test uses, and it avoids DDP
    # collectives entirely -- worth measuring against ddp rather than assuming.
    PIDS=()
    for (( g=0; g<BENCH_DEVICES; g++ )); do
        (
            WD="${RUN_DIR}/gcd${g}"
            SH="${INPUT}"
            [[ -d "${INPUT}" ]] && SH=$(make_shard "${INPUT}" "${BENCH_DEVICES}" "${g}" "${WD}/shard")
            WT0=$(date +%s.%N)
            launch "${WD}" "${SH}" 1 "${g}"
            wrc=$?
            WT1=$(date +%s.%N)
            # Each pinned worker gets its own meta.json: collect_results treats gcd*
            # dirs as independent workers, so it needs this worker's shard size and
            # wall time, not the run-level totals.
            cat > "${WD}/meta.json" <<M
{
  "run_id": "${BENCH_RUN_ID}", "rank": ${g}, "nodes": ${BENCH_NODES},
  "devices": 1, "launch_mode": "indep", "gcd": ${g},
  "n_inputs": $(count_inputs "${SH}"), "exit_code": ${wrc},
  "wall_s": $(awk -v a="${WT0}" -v b="${WT1}" 'BEGIN{printf "%.3f", b-a}'),
  "image": "${BENCH_IMAGE}", "input": "${BENCH_INPUT}", "hostname": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-}", "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
  "recycling_steps": ${BENCH_RECYCLING_STEPS}, "sampling_steps": ${BENCH_SAMPLING_STEPS},
  "diffusion_samples": ${BENCH_DIFFUSION_SAMPLES},
  "max_parallel_samples": ${BENCH_MAX_PARALLEL_SAMPLES},
  "use_potentials": ${BENCH_USE_POTENTIALS}, "num_workers": ${BENCH_NUM_WORKERS},
  "preproc_threads": ${BENCH_PREPROC_THREADS}
}
M
            exit ${wrc}
        ) &
        PIDS+=($!)
    done
    RC=0
    for p in "${PIDS[@]}"; do wait "${p}" || RC=$?; done
    N_INPUTS=$(count_inputs "${INPUT}")
else
    # Single process; Lightning DDP shards the input list across BENCH_DEVICES GCDs.
    #
    # Lightning auto-detects SLURM and then expects one task per device. We run one
    # task per NODE and let it spawn the per-GCD workers, so we opt out of its SLURM
    # environment plugin -- Lightning treats a job named "bash" as non-SLURM, which is
    # the documented escape hatch.
    [[ "${BENCH_DEVICES}" -gt 1 ]] && export SINGULARITYENV_SLURM_JOB_NAME="bash"
    launch "${RUN_DIR}" "${INPUT}" "${BENCH_DEVICES}" ""
    RC=$?
    N_INPUTS=$(count_inputs "${INPUT}")
fi

T_END=$(date +%s.%N)
# awk rather than bc: bc is not guaranteed to be installed on compute nodes.
WALL_S=$(awk -v a="${T_START}" -v b="${T_END}" 'BEGIN{printf "%.3f", b-a}')

[[ -n "${MON_PID}" ]] && kill -TERM "${MON_PID}" 2>/dev/null && wait "${MON_PID}" 2>/dev/null

# Wall time here includes singularity start and image page-in, which bench.json's
# internal clock cannot see. Both numbers matter: one is the user's experience, the
# other is the model's.
cat > "${RUN_DIR}/meta.json" <<EOF
{
  "run_id": "${BENCH_RUN_ID}",
  "rank": ${RANK},
  "nodes": ${BENCH_NODES},
  "devices": ${BENCH_DEVICES},
  "launch_mode": "${MODE}",
  "n_inputs": ${N_INPUTS},
  "exit_code": ${RC},
  "wall_s": ${WALL_S},
  "image": "${BENCH_IMAGE}",
  "input": "${BENCH_INPUT}",
  "hostname": "$(hostname)",
  "slurm_job_id": "${SLURM_JOB_ID:-}",
  "slurm_array_task_id": "${SLURM_ARRAY_TASK_ID:-}",
  "rocr_visible_devices": "${ROCR_VISIBLE_DEVICES:-}",
  "recycling_steps": ${BENCH_RECYCLING_STEPS},
  "sampling_steps": ${BENCH_SAMPLING_STEPS},
  "diffusion_samples": ${BENCH_DIFFUSION_SAMPLES},
  "max_parallel_samples": ${BENCH_MAX_PARALLEL_SAMPLES},
  "use_potentials": ${BENCH_USE_POTENTIALS},
  "num_workers": ${BENCH_NUM_WORKERS},
  "preproc_threads": ${BENCH_PREPROC_THREADS}
}
EOF

# Keep structures only if asked -- a full sweep otherwise leaves tens of GB on
# /scratch for outputs nobody scores. Set BENCH_KEEP_STRUCTURES=1 for Tier V.
if [[ "${BENCH_KEEP_STRUCTURES:-0}" != "1" ]]; then
    find "${RUN_DIR}" \( -name '*.cif' -o -name '*.pdb' -o -name '*.npz' \) -delete 2>/dev/null
fi
rm -rf "${NODE_TMP}"

echo "[${BENCH_RUN_ID}] rank=${RANK} mode=${MODE} rc=${RC} wall=${WALL_S}s"
exit ${RC}
