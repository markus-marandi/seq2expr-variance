#!/bin/bash
#SBATCH -A <your_project>
#SBATCH -p thin
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --array=0-14
#SBATCH --job-name=fz_%A_%a
#SBATCH -o logs/fz_%A_%a.out
#SBATCH -e logs/fz_%A_%a.err

module load Miniforge3/24.7.1-2-hpc1-bdist
source activate flashzoi

LIST=/proj/<proj>/gene_lists/chr20_genes.txt
GENE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$LIST")

ROOT=/proj/<proj>
python score_variants_flashzoi.py \
       --variants-dir $ROOT/intermediate/dataset1/flashzoi_inputs/variants \
       --onehot-dir   $ROOT/intermediate/dataset1/flashzoi_inputs \
       --pred-dir     $ROOT/output/dataset1/borzoi_preds \
       --gene         $GENE \
       --folds        4 \
       --batch        64 \
       --device       cuda