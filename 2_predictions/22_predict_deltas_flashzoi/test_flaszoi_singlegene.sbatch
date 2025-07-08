#!/bin/bash
###############################################################################
# SLURM resources – single DGX-A100 node, one GPU, 15-min wall-time
###############################################################################
#SBATCH -A berzelius-2025-176
#SBATCH -p berzelius           
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=0                    
#SBATCH --time=00:30:00
#SBATCH --job-name=fz_test
#SBATCH -o logs/fz_test_%j.out
#SBATCH -e logs/fz_test_%j.err

module purge
module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load Miniforge3/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flashzoi

###############################################################################
# Paths & single-gene choice
###############################################################################
BASE=/proj/berzelius-2025-176/users/x_mmara/seq2expr-variance
DATA=$BASE/data
TEST_GENE=ENSG00000020256

cd $BASE/22_predict_deltas_flashzoi

python score_variants_flashzoi.py \
       --variants-dir $DATA/intermediate/dataset1/flashzoi_inputs/variants \
       --onehot-dir   $DATA/intermediate/dataset1/flashzoi_inputs \
       --pred-dir     $DATA/output/dataset1/borzoi_preds \
       --gene         "$TEST_GENE" \
       --folds        1 \
       --batch        32\
       --device       cuda
