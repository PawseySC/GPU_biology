#!/bin/bash -l
#SBATCH -A pawsey0000-gpu
#SBATCH --nodes=1
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:1
#SBATCH --output=multimer_%A.%a.log
#SBATCH --mail-type=END,FAIL

# Load required module
module load singularity/3.11.4-nompi
REF_DIR='/data/references/alphafold_feb2024/databases'
INPUT=10aa_dimer.fasta

#singularity pull docker://quay.io/pawsey/alphafold2:proteinfold
CONTAINER_IMAGE=alphafold2_proteinfold.sif

# Run AlphaFold2
srun -N 1 -n 1 -c 8 --gres=gpu:1 \
  singularity exec ${CONTAINER_IMAGE} \
  python /app/alphafold/run_alphafold.py \
  --fasta_paths=${INPUT} \
  --model_preset=multimer \
  --use_gpu_relax=True \
  --benchmark=True \
  --data_dir=${REF_DIR} \
  --uniref90_database_path=${REF_DIR}/uniref90/uniref90.fasta \
  --mgnify_database_path=${REF_DIR}/mgnify/mgy_clusters_2022_05.fa \
  --template_mmcif_dir=${REF_DIR}/pdb_mmcif/mmcif_files \
  --obsolete_pdbs_path=${REF_DIR}/pdb_mmcif/obsolete.dat \
  --small_bfd_database_path=${REF_DIR}/small_bfd/bfd-first_non_consensus_sequences.fasta \
  --pdb_seqres_database_path=${REF_DIR}/pdb_seqres/pdb_seqres.txt \
  --uniprot_database_path=${REF_DIR}/uniprot \
  --output_dir=multimer_output/${INPUT} \
  --max_template_date=2023-05-14 \
  --db_preset=reduced_dbs \
  --logtostderr