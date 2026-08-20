#!/bin/bash -l
#SBATCH -A pawsey0000-gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --array=0-6
#SBATCH --output=monomerBENCH_%A.%a.log

# Load required module
module load singularity/3.11.4-nompi
#singularity pull docker://quay.io/pawsey/alphafold2:proteinfold
CONTAINER_IMAGE=alphafold2_proteinfold.sif

REF_DIR='/data/references/alphafold_feb2024/databases'
#export XLA_FLAGS='--xla_gpu_enable_command_buffer= --xla_gpu_autotune_level=0'
#export HIP_VISIBLE_DEVICES="0"
#export ROCR_VISIBLE_DEVICES="0"
# ---- input list ------------------------------------------------------------
INPUTS=(
  10.fasta
  1280aa.fasta
  2564aa.fasta
  3013aa.fasta
  3507aa.fasta
  4008aa.fasta
  5005aa.fasta
)

# Guard against a mismatch between --array and the list above
if (( SLURM_ARRAY_TASK_ID >= ${#INPUTS[@]} )); then
  echo "ERROR: task ${SLURM_ARRAY_TASK_ID} out of range (list has ${#INPUTS[@]} entries)" >&2
  exit 1
fi

INPUT="${INPUTS[$SLURM_ARRAY_TASK_ID]}"


# Run AlphaFold2
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec ${CONTAINER_IMAGE} \
  python /app/alphafold/run_alphafold.py \
  --fasta_paths=${INPUT} \
  --model_preset=monomer \
  --use_gpu_relax=True \
  --benchmark=True \
  --uniref90_database_path=${REF_DIR}/uniref90/uniref90.fasta \
  --mgnify_database_path=${REF_DIR}/mgnify/mgy_clusters_2022_05.fa \
  --pdb70_database_path=${REF_DIR}/pdb70/pdb70 \
  --data_dir=${REF_DIR} \
  --template_mmcif_dir=${REF_DIR}/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=${REF_DIR}/pdb_mmcif/obsolete.dat \
  --small_bfd_database_path=${REF_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --output_dir=monomer_bench_output/${INPUT} \
  --max_template_date=2023-05-14 \
  --db_preset=reduced_dbs \
  --logtostderr