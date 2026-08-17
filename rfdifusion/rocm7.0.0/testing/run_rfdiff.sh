#!/bin/bash -l
#SBATCH --job-name=rfdiffusion
#SBATCH -A ${PAWSEY_PROJECT}-gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-20
# Load required modules
module load singularity/3.11.4-nompi

#Set up container variable
ContainerImage=docker://quay.io/pawsey/rfdiffusion:rocm7.0.0_dgl2.4.0

mkdir -p schedules

EXAMPLE_DIR="/app/RFdiffusion/examples/"

# Deterministic, sorted list of the 21 scripts
mapfile -t SCRIPTS < <(find "$EXAMPLE_DIR" -maxdepth 1 -type f -name '*.sh' | sort)

# Bail out if the array range and the file count disagree
if (( SLURM_ARRAY_TASK_ID >= ${#SCRIPTS[@]} )); then
    echo "Task ${SLURM_ARRAY_TASK_ID} has no matching script (found ${#SCRIPTS[@]})" >&2
    exit 1
fi
SCRIPT="${SCRIPTS[$SLURM_ARRAY_TASK_ID]}"
echo "Task ${SLURM_ARRAY_TASK_ID}: running $(basename "$SCRIPT")"

# Run RFdiffusion
srun N 1 -n 1 -c 8 --gres=gpu:1 \
singularity exec \
-B schedules:/app/RFdiffusion/rfdiffusion/inference/../../schedules \
${ContainerImage} \
bash -c "cd ${EXAMPLE_DIR} && bash ${SCRIPT}"