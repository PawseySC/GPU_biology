#!/bin/bash -l
# =============================================================================
# submit_setonix.sbatch
#
# Submit the Boltz2 container GCD test on a full Pawsey Setonix GPU node.
# A Setonix GPU node has 4x AMD MI250X = 8 GCDs; this requests the whole node
# and runs the 1->8 GCD test (see test_boltz_gcd.sh / testing.MD).
#
# Submit from this directory (boltz2/v2.2.1), for example:
#     sbatch --account=pawsey0001-gpu submit_setonix.sbatch
#
# EDIT before submitting:
#   * --account         : your <project>-gpu allocation (or pass --account on the CLI)
#   * BOLTZ_SIF         : path to the built container .sif
#   * BOLTZ_CACHE       : directory holding the pre-staged Boltz2 model weights
# =============================================================================

#SBATCH --job-name=boltz_gcd_test
#SBATCH --account=${PAWSEY_PROJECT}-gpu     # EDIT (SBATCH does NOT expand vars -- edit or use: sbatch --account=...)
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8                         # all 8 GCDs on the node
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=boltz_gcd_test-%j.out

# Non-MPI Singularity is all we need (each GCD runs an independent boltz process).
module load singularity/3.11.4-nompi

# --- site configuration (EDIT these two paths) -------------------------------
export BOLTZ_SIF="${BOLTZ_SIF:-$MYSCRATCH/boltz/boltz2_v2.2.1_rocm6.4.1.sif}"
export BOLTZ_CACHE="${BOLTZ_CACHE:-/scratch/references/boltz}"   # pre-staged model weights
export BOLTZ_OUTDIR="${BOLTZ_OUTDIR:-$MYSCRATCH/boltz_gcd_test/$SLURM_JOB_ID}"
# -----------------------------------------------------------------------------

mkdir -p "$BOLTZ_OUTDIR"

# The batch script already owns the whole node's 8 GCDs, so the test runner can
# pin one boltz process per GCD itself (via HIP_VISIBLE_DEVICES) -- no srun needed.
#
#   --sweep : run the full 1,2,3,...,8 GCD ladder (each level concurrent).
#             Drop --sweep for a single, faster all-8-GCDs run that still
#             exercises every GCD (change --time down to ~01:00:00 if you do).
bash "$SLURM_SUBMIT_DIR/test_boltz_gcd.sh" \
    --image  "$BOLTZ_SIF" \
    --cache  "$BOLTZ_CACHE" \
    --outdir "$BOLTZ_OUTDIR" \
    --gcds   8 \
    --sweep

echo "Test finished with exit code $? -- see $BOLTZ_OUTDIR/results.tsv"
