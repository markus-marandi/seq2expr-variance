#!/bin/bash
###############################################################################
# Usage:
#   sbatch --array=0-?? COHORT=ClinGen_gene_curation_list run_flashzoi.sh
#   sbatch --array=0-?? COHORT=nonessential_ensg         run_flashzoi.sh
# /proj/berzelius-2025-176/users/x_mmara/seq2expr-variance/data/intermediate/dataset3/flashzoi_outputs/nonessential_ensg/flashzoi_preds_meta.tsv
# sbatch --array=0-62 --export=ALL,COHORT=ClinGen_gene_curation_list flashzoi_array.sh
# sbatch --array=0-18 --export=ALL,COHORT=nonessential_ensg          flashzoi_array.sh
# Replace ?? with ((NG / 5) - 1) where NG is the #lines in the gene-list.
###############################################################################

#SBATCH -A berzelius-2025-176
#SBATCH -p berzelius                 # thin or fat are both fine
#SBATCH --ntasks=1                   # one task = one GPU job
#SBATCH --gpus=1                     # ≤40 GB VRAM fits Flashzoi batch 32
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G                    # leaves node RAM for neighbours
#SBATCH --time=00:45:00              # 5 genes × ≤20 min + cushion
#SBATCH --job-name=fz_${COHORT}
#SBATCH -o logs/fz_%x_%A_%a.out
#SBATCH -e logs/fz_%x_%A_%a.err


module purge
module load buildenv-gcccuda/12.1.1-gcc12.3.0
module load Miniforge3/25.3.0-3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate flashzoi


# ─── software ────────────────────────────────────────────────────────────
BASE=/proj/berzelius-2025-176/users/x_mmara/seq2expr-variance
DATASET=dataset3
COHORT=${COHORT:?You must pass COHORT=<ClinGen_gene_curation_list|nonessential_ensg>}

LIST=$BASE/data/gene_lists/gene_list_${COHORT}.txt 
DATA_ROOT=$BASE/data/intermediate/${DATASET}
PRED_DIR=$BASE/data/intermediate/${DATASET}/flashzoi_outputs/${COHORT}
IDX_DIR=$BASE/data/track_lists

if [[ $COHORT == "ClinGen_gene_curation_list" ]]; then
  IDX_FILE=$IDX_DIR/clingen_meta5_idx.txt
else
  IDX_FILE=$IDX_DIR/nonessential_GM12878_idx.txt
fi

# ─── 5 genes per SLURM task ──────────────────────────────────────────────
START=$((SLURM_ARRAY_TASK_ID * 5))
END=$((START + 4))
readarray -t GENES < <(sed -n "$((START+1)),$((END+1))p" "$LIST")

cd $BASE/22_predict_deltas_flashzoi   # folder containing score_variants_flashzoi.py

for GENE in "${GENES[@]}"; do
  echo ">>> Δ-scoring $GENE  (cohort $COHORT)"
  python score_variants_flashzoi.py \
         --dataset-root   "$DATA_ROOT" \
         --cohort         "$COHORT" \
         --pred-dir       "$PRED_DIR" \
         --track-idx-file "$IDX_FILE" \
         --folds          1 \
         --batch          32 \
         --gene           "$GENE" \
         --device         cuda
done