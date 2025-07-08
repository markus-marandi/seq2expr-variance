## Prediction Workflow

After preparing the one-hot inputs and variant annotations, run predictions using the unified Flashzoi script:

### Unified Prediction Script

The `predict_flashzoi_unified.py` script works for both single_chr and genome-wide datasets:

```bash
# Single chromosome datasets (dataset1/dataset2)
python 2_predictions/21_run_mean_ref/predict_flashzoi_unified.py \
    --dataset-root data/intermediate/dataset1 \
    --dataset-type single_chr \
    --track-idx-file data/track_lists/muscle_idx.txt \
    --output-dir data/output/dataset1/flashzoi_preds \
    --device cpu

# Genome-wide datasets (dataset3)
python 2_predictions/21_run_mean_ref/predict_flashzoi_unified.py \
    --dataset-root data/intermediate/dataset3 \
    --dataset-type genome \
    --cohort ClinGen_gene_curation_list \
    --track-idx-file data/track_lists/clingen_meta5_idx.txt \
    --output-dir data/output/dataset3/flashzoi_outputs \
    --device cuda --autocast
```

### Convenience Script

Use the shell script for easier execution:

```bash
# Individual datasets
./2_predictions/21_run_mean_ref/run_unified_flashzoi.sh dataset1
./2_predictions/21_run_mean_ref/run_unified_flashzoi.sh clingen

# All datasets  
./2_predictions/21_run_mean_ref/run_unified_flashzoi.sh all
```

### Key Features

- **Enhanced Models**: Uses flashzoi-replicate models (3x faster, 2.4x less memory than borzoi)
- **Flexible Devices**: Supports CPU, CUDA, and MPS backends
- **Unified Interface**: Same script for single_chr and genome workflows
- **Track Selection**: Configurable via track index files
- **Mixed Precision**: Optional autocast for GPU acceleration

### Output Structure

For each gene, the script saves:
- `{gene}_flashzoi_pred.npy` - Prediction array  
- `mean_ref.npy` - Mean reference value
- `{gene}_flashzoi_emb.npz` - Compressed embeddings
- `fold{N}_pred.npy` - Per-fold predictions (if using multiple folds)

Plus cohort-level metadata:
- `flashzoi_preds_meta.tsv` - Gene metadata with prediction paths
- `_diag/` - Diagnostic plots and correlations 