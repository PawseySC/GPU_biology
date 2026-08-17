#!/bin/bash -l
#SBATCH --job-name=colabbatch
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --account=${PAWSEY_PROJECT}-gpu

# Load required module
module load singularity/3.11.4-nompi

# Set input and output paths
A3M=/path/to/your/msafile.a3m
OUT=$MYSCRATCH/colabfold/${SLURM_JOB_ID}
containerImage=docker://quay.io/pawsey/colabfold:1.6.1_rocm7.0.2

# Set JAX/XLA JIT compilation cache
export JAX_COMPILATION_CACHE_DIR=${MYSOFTWARE}/jax_cache

# Run ColabFold
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
        singularity exec $containerImage \
        colabfold_batch \
        --data /scratch/references/colabfold_jun2024/database \
        --num-recycle 3 \
        --model-type alphafold2_multimer_v3 \
        --num-models 3 $A3M $OUT
