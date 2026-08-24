#!/bin/bash
# Split a manifest by resource shape and submit one SLURM array per shape.
#
# Array tasks in a single job share one set of --gres/--nodes directives, but the
# manifest deliberately mixes 1-GCD and 8-GCD runs. So: group rows by (nodes, devices),
# write a sub-manifest per group, and submit each with matching directives.
#
# Usage:
#   export BENCH_ROOT=$MYSCRATCH/boltz-bench
#   export BENCH_INPUTS_ROOT=$MYSCRATCH/boltz-bench/inputs
#   export BENCH_CACHE=/scratch/references/boltz
#   ./submit.sh manifest.tsv [max_concurrent]
set -euo pipefail

MANIFEST="${1:?usage: submit.sh <manifest.tsv> [max_concurrent]}"
THROTTLE="${2:-8}"

: "${BENCH_ROOT:?export BENCH_ROOT - the working tree holding these scripts}"
: "${BENCH_INPUTS_ROOT:?export BENCH_INPUTS_ROOT}"
export BENCH_CACHE="${BENCH_CACHE:-/scratch/references/boltz}"

# SLURM does not expand shell variables inside #SBATCH directives -- an
# `#SBATCH --account=${PAWSEY_PROJECT}-gpu` line is taken literally and fails. The
# account has to be passed on the sbatch command line, where the shell expands it.
ACCOUNT="${BENCH_ACCOUNT:-${PAWSEY_PROJECT:-}-gpu}"
if [[ "${ACCOUNT}" == "-gpu" ]]; then
    echo "error: set BENCH_ACCOUNT or PAWSEY_PROJECT" >&2
    exit 2
fi
PARTITION="${BENCH_PARTITION:-gpu}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SPLIT_DIR="${BENCH_ROOT}/manifests"
mkdir -p "${SPLIT_DIR}" "${BENCH_ROOT}/results" "${BENCH_ROOT}/logs"

HEADER=$(head -1 "${MANIFEST}")
# Column indices (1-based) for nodes and devices, derived from the header so this
# survives a column being added to gen_manifest.py.
DEV_COL=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="devices") print i}' "${MANIFEST}")
NOD_COL=$(awk -F'\t' 'NR==1{for(i=1;i<=NF;i++) if($i=="nodes") print i}' "${MANIFEST}")

SHAPES=$(awk -F'\t' -v d="${DEV_COL}" -v n="${NOD_COL}" \
    'NR>1 {print $n"_"$d}' "${MANIFEST}" | sort -u)

echo "Manifest: ${MANIFEST}"
echo "Shapes:   $(echo "${SHAPES}" | tr '\n' ' ')"
echo

for shape in ${SHAPES}; do
    nodes="${shape%%_*}"
    devices="${shape##*_}"
    sub="${SPLIT_DIR}/shape_n${nodes}_d${devices}.tsv"

    { echo "${HEADER}"
      awk -F'\t' -v d="${DEV_COL}" -v n="${NOD_COL}" -v dv="${devices}" -v nv="${nodes}" \
          'NR>1 && $d==dv && $n==nv' "${MANIFEST}"
    } > "${sub}"

    count=$(( $(wc -l < "${sub}") - 1 ))
    [[ "${count}" -lt 1 ]] && continue

    # Big-token and 25-sample runs are slow; give the 1-GCD arrays generous walltime
    # rather than losing a sweep to a timeout at the top of the ladder.
    walltime="04:00:00"
    [[ "${nodes}" -gt 1 ]] && walltime="02:00:00"

    echo "  nodes=${nodes} devices=${devices}  ->  ${count} runs  (${sub})"
    jid=$(sbatch --parsable \
        --account="${ACCOUNT}" \
        --partition="${PARTITION}" \
        --array="1-${count}%${THROTTLE}" \
        --nodes="${nodes}" \
        --ntasks-per-node=1 \
        --cpus-per-task=$(( devices * 8 )) \
        --gres=gpu:"${devices}" \
        --time="${walltime}" \
        --job-name="boltzbench_n${nodes}d${devices}" \
        --output="${BENCH_ROOT}/logs/%x-%A_%a.out" \
        --error="${BENCH_ROOT}/logs/%x-%A_%a.out" \
        --export=ALL,BENCH_ROOT="${BENCH_ROOT}",BENCH_INPUTS_ROOT="${BENCH_INPUTS_ROOT}",BENCH_CACHE="${BENCH_CACHE}" \
        "${SCRIPT_DIR}/bench_array.sbatch" "${sub}")
    echo "    submitted job ${jid}"
done

echo
echo "Watch with:   squeue -u \$USER"
echo "Collect with: python3 ${SCRIPT_DIR}/collect_results.py --root ${BENCH_ROOT} --manifest ${MANIFEST} --out ${BENCH_ROOT}/results.csv"
