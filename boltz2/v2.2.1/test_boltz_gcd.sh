#!/bin/bash -l
# =============================================================================
# test_boltz_gcd.sh
#
# Functional / smoke test for the Boltz2 ROCm container (see ./Dockerfile).
#
# Runs one independent `boltz predict` job per GCD (Graphics Compute Die), each
# pinned to its own GCD via HIP_VISIBLE_DEVICES, then verifies that every GCD
# produced a valid predicted structure. Use it to confirm a freshly built
# container works on all 1-8 GCDs of an AMD MI250X node (on Pawsey Setonix a
# GPU node has 4x MI250X = 8 GCDs, each exposed as a separate ROCm device).
#
# The bundled input (inputs/test.yaml) ships precomputed MSAs (the test_*.csv
# files), so NO MSA server / internet access is needed at predict time. Only
# the model weights must be present in the --cache directory. They are either
# pre-staged there, or downloaded once by the built-in warm-up step (which does
# need outbound internet; Setonix compute nodes usually do not have it, so
# pre-stage the cache there).
#
# Exit status: 0 if every GCD passed, non-zero if any GCD failed.
#
# Author: Pawsey Supercomputing Research Centre
# =============================================================================

set -uo pipefail

# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------
log()  { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARNING: %s\n' "$(date +%H:%M:%S)" "$*" >&2; }
die()  { printf '[%s] ERROR: %s\n' "$(date +%H:%M:%S)" "$*" >&2; exit 1; }

# Colour only when writing to a terminal.
if [[ -t 1 ]]; then
    c_grn=$'\033[32m'; c_red=$'\033[31m'; c_ylw=$'\033[33m'; c_rst=$'\033[0m'
else
    c_grn=''; c_red=''; c_ylw=''; c_rst=''
fi

# Absolute-path helpers (portable: no realpath/readlink -f dependency).
abs_dir()  { ( cd "$1" 2>/dev/null && pwd ) || die "No such directory: $1"; }
abs_file() {
    local d b
    d=$(dirname -- "$1"); b=$(basename -- "$1")
    printf '%s/%s\n' "$(abs_dir "$d")" "$b"
}

# ---------------------------------------------------------------------------
# Defaults (every one can be overridden by a CLI flag or the env var shown)
# ---------------------------------------------------------------------------
SCRIPT_DIR=$(abs_dir "$(dirname -- "${BASH_SOURCE[0]}")")

NUM_GCDS=8                                                    # --gcds  / -n
SWEEP=false                                                   # --sweep / -s
NO_KERNELS=true                                              # --kernels / --no-kernels
QUICK=false                                                  # --quick
DRY_RUN=false                                               # --dry-run

IMAGE=${BOLTZ_SIF:-}                                        # --image   / -i
YAML=${BOLTZ_YAML:-$SCRIPT_DIR/inputs/test.yaml}           # --yaml    / -y
WORKDIR=${BOLTZ_WORKDIR:-$SCRIPT_DIR}                      # --workdir / -w  (dir containing inputs/)
CACHE=${BOLTZ_CACHE:-${MYSCRATCH:-$PWD}/boltz/cache}       # --cache   / -c
OUTDIR=${BOLTZ_OUTDIR:-${MYSCRATCH:-$PWD}/boltz_gcd_test/${SLURM_JOB_ID:-$(date +%Y%m%d_%H%M%S)}}  # --outdir / -o
CE=${CONTAINER_EXEC:-}                                      # singularity | apptainer

usage() {
    cat <<EOF
Usage: $(basename "$0") --image IMAGE.sif [options]

Launches one 'boltz predict' per GCD, each pinned to its own GCD, and checks
that every GCD produced a valid predicted structure. Tests 1-8 GCDs on a node.

Required:
  -i, --image PATH     Container image (.sif). Env: BOLTZ_SIF

Options:
  -n, --gcds N         Number of GCDs to test, 1-8 (default: 8).
  -s, --sweep          Sweep 1,2,...,N GCDs (each level run concurrently) instead
                       of a single N-GCD run. Confirms scaling across the ladder.
  -c, --cache DIR      Boltz model-weights cache (default: \$MYSCRATCH/boltz/cache).
                       Env: BOLTZ_CACHE. On Setonix point at the shared pre-staged
                       cache, e.g. /scratch/references/boltz .
  -o, --outdir DIR     Output/log directory (default: \$MYSCRATCH/boltz_gcd_test/<jobid>).
                       Env: BOLTZ_OUTDIR
  -y, --yaml FILE      Input YAML (default: inputs/test.yaml). Env: BOLTZ_YAML
  -w, --workdir DIR    Directory that contains inputs/ ; used as the in-container
                       working dir so relative msa paths resolve (default: script dir).
      --quick          Reduce sampling for a faster functionality-only check
                       (--sampling_steps 25 --recycling_steps 1 --diffusion_samples 1).
      --kernels        Enable optimised kernels (default: --no-kernels, matching the
                       ROCm build which does not ship the CUDA-only kernels).
      --no-kernels     Force --no_kernels (default).
      --dry-run        Print the commands that would run, then exit.
  -h, --help           Show this help.

Environment: CONTAINER_EXEC lets you force 'singularity' or 'apptainer'.

Examples:
  # Whole node, all 8 GCDs at once (fast smoke test):
  $(basename "$0") --image boltz2_v2.2.1_rocm6.4.1.sif --cache /scratch/references/boltz

  # Full 1->8 GCD ladder:
  $(basename "$0") --image boltz2_v2.2.1_rocm6.4.1.sif --sweep --cache /scratch/references/boltz
EOF
}

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        -n|--gcds)     NUM_GCDS=${2:?}; shift 2;;
        -s|--sweep)    SWEEP=true; shift;;
        -i|--image)    IMAGE=${2:?}; shift 2;;
        -y|--yaml)     YAML=${2:?}; shift 2;;
        -w|--workdir)  WORKDIR=${2:?}; shift 2;;
        -c|--cache)    CACHE=${2:?}; shift 2;;
        -o|--outdir)   OUTDIR=${2:?}; shift 2;;
        --quick)       QUICK=true; shift;;
        --kernels)     NO_KERNELS=false; shift;;
        --no-kernels)  NO_KERNELS=true; shift;;
        --dry-run)     DRY_RUN=true; shift;;
        -h|--help)     usage; exit 0;;
        *)             die "Unknown argument: $1 (see --help)";;
    esac
done

# ---------------------------------------------------------------------------
# Validate & resolve
# ---------------------------------------------------------------------------
[[ $NUM_GCDS =~ ^[0-9]+$ && $NUM_GCDS -ge 1 ]] || die "--gcds must be a positive integer (1-8)."
[[ $NUM_GCDS -le 8 ]] || warn "You requested $NUM_GCDS GCDs; a Setonix MI250X node has 8."

# Container runtime.
if [[ -z $CE ]]; then
    if   command -v singularity >/dev/null 2>&1; then CE=singularity
    elif command -v apptainer   >/dev/null 2>&1; then CE=apptainer
    else die "Neither 'singularity' nor 'apptainer' is on PATH (load the module, or set CONTAINER_EXEC)."
    fi
fi

[[ -n $IMAGE ]] || { usage; die "No container image given. Pass --image PATH or set BOLTZ_SIF."; }
[[ -f $IMAGE ]] || die "Container image not found: $IMAGE"
IMAGE=$(abs_file "$IMAGE")

[[ -f $YAML ]] || die "Input YAML not found: $YAML"
YAML=$(abs_file "$YAML")
WORKDIR=$(abs_dir "$WORKDIR")

mkdir -p "$CACHE"  || die "Cannot create cache dir: $CACHE"
mkdir -p "$OUTDIR" || die "Cannot create output dir: $OUTDIR"
CACHE=$(abs_dir "$CACHE")
OUTDIR=$(abs_dir "$OUTDIR")

RESULTS="$OUTDIR/results.tsv"
: > "$RESULTS"

# ---------------------------------------------------------------------------
# Core routines
# ---------------------------------------------------------------------------

# Ask the container how many GCDs it can see (this is what boltz will see too).
probe_gcds() {
    local n
    n=$("$CE" exec "$IMAGE" python3 -c 'import torch; print(torch.cuda.device_count())' 2>/dev/null | tail -n1)
    [[ $n =~ ^[0-9]+$ ]] && printf '%s\n' "$n" || printf 'unknown\n'
}

# Did a boltz run produce a real structure? Echo the file and return 0 if so.
validate_output() {
    local d=$1 f
    f=$(find "$d" -type f \( -iname '*.cif' -o -iname '*.pdb' \) -size +0c 2>/dev/null | head -n1)
    [[ -n $f ]] && printf '%s\n' "$f"
    [[ -n $f ]]
}

# Launch one prediction pinned to a single GCD (background).
# Sets globals: LAST_PID, LAST_OUT, LAST_START.
launch_gcd() {
    local gcd=$1 tag=$2
    local out="$OUTDIR/$tag/gcd${gcd}"
    local numba="$out/.numba" xdg="$out/.xdg" mpl="$out/.mpl" tmp="$out/.tmp"
    mkdir -p "$out" "$numba" "$xdg" "$mpl" "$tmp"
    local log="$out/boltz.log"

    # HIP_VISIBLE_DEVICES=$gcd selects this one GCD out of the allocated set.
    # Per-GCD scratch dirs avoid concurrent writers clobbering shared caches.
    local -a cmd=(
        "$CE" exec
        --pwd "$WORKDIR"
        -B "$WORKDIR" -B "$CACHE" -B "$OUTDIR"
        --env "HIP_VISIBLE_DEVICES=$gcd"
        --env "NUMBA_CACHE_DIR=$numba"
        --env "XDG_CACHE_HOME=$xdg"
        --env "MPLCONFIGDIR=$mpl"
        --env "TMPDIR=$tmp"
        --env "WANDB_MODE=disabled"
        --env "TQDM_DISABLE=1"
        "$IMAGE"
        boltz predict "$YAML"
        --cache "$CACHE"
        --out_dir "$out"
        --accelerator gpu
        --devices 1
    )
    [[ $NO_KERNELS == true ]] && cmd+=( --no_kernels )
    [[ $QUICK == true ]]      && cmd+=( --sampling_steps 25 --recycling_steps 1 --diffusion_samples 1 )

    if [[ $DRY_RUN == true ]]; then
        printf 'DRY-RUN GCD %s ->\n  ' "$gcd"; printf '%q ' "${cmd[@]}"; printf '\n'
        LAST_PID=0; LAST_OUT=$out; LAST_START=$(date +%s)
        return 0
    fi

    ( "${cmd[@]}" ) >"$log" 2>&1 &
    LAST_PID=$!; LAST_OUT=$out; LAST_START=$(date +%s)
}

# Run one stage: N concurrent predictions on GCD 0..N-1, then validate each.
run_stage() {
    local n=$1 tag=$2
    log "=== Stage '$tag': ${n} concurrent prediction(s) on GCD 0..$((n-1)) ==="

    local -a PID=() OUT=() STA=() GCD=()
    local i
    for ((i=0; i<n; i++)); do
        launch_gcd "$i" "$tag"
        PID[i]=$LAST_PID; OUT[i]=$LAST_OUT; STA[i]=$LAST_START; GCD[i]=$i
        [[ $DRY_RUN == true ]] || log "  GCD $i -> pid ${PID[i]}  (log: ${OUT[i]}/boltz.log)"
    done

    if [[ $DRY_RUN == true ]]; then
        for ((i=0; i<n; i++)); do
            printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "${GCD[i]}" "DRY" "0" "-" >>"$RESULTS"
        done
        return 0
    fi

    local rc end secs struct res
    for ((i=0; i<n; i++)); do
        wait "${PID[i]}"; rc=$?
        end=$(date +%s); secs=$(( end - STA[i] ))
        if [[ $rc -eq 0 ]] && struct=$(validate_output "${OUT[i]}"); then
            res=PASS
            log "  ${c_grn}PASS${c_rst} GCD ${GCD[i]} in ${secs}s -> $struct"
        else
            res=FAIL; struct=${struct:-"-"}
            warn "  ${c_red}FAIL${c_rst} GCD ${GCD[i]} (rc=$rc, ${secs}s) -- see ${OUT[i]}/boltz.log"
        fi
        printf '%s\t%s\t%s\t%s\t%s\n' "$tag" "${GCD[i]}" "$res" "$secs" "$struct" >>"$RESULTS"
    done
}

# Populate the weights cache once (single process) if it looks empty, so the
# concurrent stages do not race to download the same files.
ensure_cache_warm() {
    if find "$CACHE" -maxdepth 4 -iname '*.ckpt' -size +0c -print -quit 2>/dev/null | grep -q .; then
        log "Model checkpoint found in cache: $CACHE (skipping warm-up)."
        return 0
    fi
    warn "No model checkpoint (*.ckpt) found in cache: $CACHE"
    warn "Running a one-time warm-up prediction on GCD 0 to download & cache the Boltz2 weights."
    warn "This step needs outbound internet. On Setonix compute nodes it will fail --"
    warn "pre-stage the cache instead (e.g. --cache /scratch/references/boltz)."
    [[ $DRY_RUN == true ]] && { log "(dry-run) would warm up the cache here."; return 0; }

    launch_gcd 0 "warmup"
    local pid=$LAST_PID out=$LAST_OUT start=$LAST_START rc end secs
    wait "$pid"; rc=$?
    end=$(date +%s); secs=$(( end - start ))
    if [[ $rc -eq 0 ]] && validate_output "$out" >/dev/null; then
        log "Warm-up succeeded in ${secs}s; cache is populated."
    else
        die "Warm-up prediction FAILED (rc=$rc). See $out/boltz.log
     Cannot continue. Either pre-stage the model cache, or fix GPU/container access first."
    fi
}

print_summary() {
    local total pass fail
    total=$(wc -l <"$RESULTS" | tr -d ' ')
    pass=$(awk -F'\t' '$3=="PASS"{c++} END{print c+0}' "$RESULTS")
    fail=$(awk -F'\t' '$3=="FAIL"{c++} END{print c+0}' "$RESULTS")

    echo
    echo "=============================== SUMMARY ==============================="
    printf '%-16s %-4s %-7s %-9s %s\n' "STAGE" "GCD" "RESULT" "TIME(s)" "STRUCTURE"
    awk -F'\t' '{printf "%-16s %-4s %-7s %-9s %s\n",$1,$2,$3,$4,$5}' "$RESULTS"
    echo "----------------------------------------------------------------------"
    printf 'Runs: %s   Passed: %s   Failed: %s\n' "$total" "$pass" "$fail"
    echo "Outputs & logs under: $OUTDIR"
    echo "======================================================================"

    [[ ${fail:-0} -eq 0 && ${total:-0} -gt 0 ]]
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
trap 'echo; warn "Interrupted -- killing background jobs."; kill $(jobs -p) 2>/dev/null; exit 130' INT TERM

log "Container runtime : $CE"
log "Container image   : $IMAGE"
log "Input YAML        : $YAML"
log "Work dir (--pwd)  : $WORKDIR"
log "Cache dir         : $CACHE"
log "Output dir        : $OUTDIR"
log "Mode              : $([[ $SWEEP == true ]] && echo "sweep 1..$NUM_GCDS" || echo "single run, $NUM_GCDS GCD(s)")  (quick=$QUICK, no_kernels=$NO_KERNELS)"

AVAIL=$(probe_gcds)
if [[ $AVAIL == unknown ]]; then
    warn "Could not read GCD count from the container (torch probe failed); continuing anyway."
elif [[ $AVAIL -eq 0 ]]; then
    die "The container sees 0 GPUs/GCDs. Check your allocation (e.g. --gres=gpu:8) and that /dev/kfd and /dev/dri are available. Try adding '--rocm' to the container command if needed."
else
    log "Container sees $AVAIL GCD(s)."
    if [[ $NUM_GCDS -gt $AVAIL ]]; then
        warn "Requested $NUM_GCDS GCD(s) but only $AVAIL visible; clamping to $AVAIL."
        NUM_GCDS=$AVAIL
    fi
fi

ensure_cache_warm

if [[ $SWEEP == true ]]; then
    for ((N=1; N<=NUM_GCDS; N++)); do
        run_stage "$N" "sweep_${N}gcd"
    done
else
    run_stage "$NUM_GCDS" "run_${NUM_GCDS}gcd"
fi

if print_summary; then
    log "${c_grn}ALL GCDS PASSED${c_rst}"
    exit 0
else
    warn "${c_red}ONE OR MORE GCDS FAILED${c_rst}"
    exit 1
fi
