#!/bin/bash
################################################################################
# run_unified_flashzoi.sh - Examples of using the unified Flashzoi prediction script
#
# This script shows how to run predictions for different dataset types:
# - single_chr: for dataset1/dataset2 (chromosome 20 test datasets)  
# - genome: for dataset3 (genome-wide constraint-based cohorts)
################################################################################

set -euo pipefail

# Base paths (adjust as needed)
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="${BASE_DIR}/2_predictions/21_run_mean_ref"
DATA_DIR="${BASE_DIR}/data"

# Check if unified script exists
UNIFIED_SCRIPT="${SCRIPT_DIR}/predict_flashzoi_unified.py"
if [[ ! -f "$UNIFIED_SCRIPT" ]]; then
    echo "Error: Unified script not found at $UNIFIED_SCRIPT"
    exit 1
fi

# Function to run predictions for single chromosome datasets
run_single_chr() {
    local dataset_num="$1"
    echo "=== Running predictions for dataset${dataset_num} (single_chr) ==="
    
    python "$UNIFIED_SCRIPT" \
        --dataset-root "${DATA_DIR}/intermediate/dataset${dataset_num}" \
        --dataset-type single_chr \
        --track-idx-file "${DATA_DIR}/track_lists/muscle_idx.txt" \
        --output-dir "${DATA_DIR}/output/dataset${dataset_num}/flashzoi_preds" \
        --device cpu \
        --folds 1
    
    echo "✓ Dataset${dataset_num} predictions completed"
}

# Function to run predictions for genome-wide datasets  
run_genome() {
    local cohort="$1"
    local track_file="$2"
    echo "=== Running predictions for $cohort (genome-wide) ==="
    
    python "$UNIFIED_SCRIPT" \
        --dataset-root "${DATA_DIR}/intermediate/dataset3" \
        --dataset-type genome \
        --cohort "$cohort" \
        --track-idx-file "${DATA_DIR}/track_lists/${track_file}" \
        --output-dir "${DATA_DIR}/output/dataset3/flashzoi_outputs" \
        --device cuda \
        --autocast \
        --folds 1
    
    echo "✓ $cohort predictions completed"
}

# Parse command line arguments
case "${1:-help}" in
    "dataset1")
        run_single_chr 1
        ;;
    "dataset2") 
        run_single_chr 2
        ;;
    "clingen")
        run_genome "ClinGen_gene_curation_list" "clingen_meta5_idx.txt"
        ;;
    "nonessential")
        run_genome "nonessential_ensg" "nonessential_GM12878_idx.txt"
        ;;
    "all")
        echo "Running predictions for all datasets..."
        run_single_chr 1
        run_single_chr 2
        run_genome "ClinGen_gene_curation_list" "clingen_meta5_idx.txt"
        run_genome "nonessential_ensg" "nonessential_GM12878_idx.txt"
        echo "✓ All predictions completed"
        ;;
    "help"|*)
        echo "Usage: $0 {dataset1|dataset2|clingen|nonessential|all|help}"
        echo ""
        echo "Examples:"
        echo "  $0 dataset1      # Run predictions for dataset1 (single_chr)"
        echo "  $0 dataset2      # Run predictions for dataset2 (single_chr)" 
        echo "  $0 clingen       # Run predictions for ClinGen cohort (genome-wide)"
        echo "  $0 nonessential  # Run predictions for nonessential cohort (genome-wide)"
        echo "  $0 all           # Run predictions for all datasets"
        echo ""
        echo "Manual usage of unified script:"
        echo "  python predict_flashzoi_unified.py \\"
        echo "    --dataset-root data/intermediate/dataset1 \\"
        echo "    --dataset-type single_chr \\"
        echo "    --track-idx-file data/track_lists/muscle_idx.txt \\"
        echo "    --output-dir data/output/dataset1/flashzoi_preds \\"
        echo "    --device cpu"
        ;;
esac 