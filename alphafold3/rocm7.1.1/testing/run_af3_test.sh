#!/bin/bash -l
#SBATCH --nodes=1 
#SBATCH --gres=gpu:1
#SBATCH --account=<pawsey_project>-gpu
#SBATCH --partition=gpu
#SBATCH --time=1:00:00
#SBATCH --job-name=af3
#SBATCH --array=2-13

module load singularity/3.11.4-nompi 
singularity pull docker://quay.io/pawsey/alphafold3:rocm7.1.1

MODEL_DIR=/path/to/model/weights
CONTAINER=alphafold3_rocm7.1.1.sif

srun -N 1 -n 1 -c 8 --gres=gpu:1 --gpus-per-task=1 \
singularity exec ${CONTAINER} \
      python3 /app/alphafold3/run_alphafold.py \
      --model_dir=${MODEL_DIR} \
      --json_path=2pv7_data_${SLURM_ARRAY_TASK_ID}mer.json \
      --output_dir=af_output_${SLURM_ARRAY_TASK_ID} \
      --num_recycles=3 \
      --num_diffusion_samples=1 \
      --norun_data_pipeline
