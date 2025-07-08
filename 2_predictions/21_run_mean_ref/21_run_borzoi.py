"""
DEPRECATED: This script is deprecated and uses older borzoi-replicate models.

Please use the new unified script instead:
    python predict_flashzoi_unified.py --dataset-type single_chr --device cpu

The new script:
- Uses enhanced flashzoi-replicate models (3x faster, 2.4x less memory)
- Works for both single_chr and genome datasets  
- Has flexible device selection (cpu/cuda/mps)
- Supports custom track index files

Examples:
    # For dataset1/dataset2:
    python predict_flashzoi_unified.py \
        --dataset-root data/intermediate/dataset1 \
        --dataset-type single_chr \
        --track-idx-file data/track_lists/muscle_idx.txt \
        --output-dir data/output/dataset1/flashzoi_preds \
        --device cpu
---

Runnning it in SLURM
python 21_run_borzoi.py \
    --meta-tsv /path/to/META_TSV.tsv \
    --targets-txt /path/to/targets_human.txt \
    --ckpt /path/to/checkpoint/dir \
    --out-dir /cfs/klemming/scratch/$USER/borzoi_outputs \
    --device cpu
"""
from __future__ import annotations
# Loading multiple Borzoi models, predicts muscle-related signal tracks,
# averages them, caches features, and saves results for each gene.
# reference implementation: https://github.com/johahi/borzoi-pytorch/blob/d79dc3c606389d3faeac3c903d25b3bbe02a42f5/notebooks/pytorch_borzoi_example_eqtl_chr10_116952944_T_C.ipynb

import sys
import hashlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from borzoi_pytorch import Borzoi
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, spearmanr
import warnings

warnings.warn(
    "21_run_borzoi.py is deprecated. Use predict_flashzoi_unified.py instead. "
    "See script header for migration instructions.",
    DeprecationWarning,
    stacklevel=2
)

DEVICE = torch.device('cpu')

# Use 16-bit floats for embeddings to save memory.
EMBEDDING_DTYPE = torch.float16

# Number of model folds to load and ensemble.
NUM_FOLDS = 4

# Window size of sequence the model sees.
WINDOW_SIZE = 524_288  # base pairs

PREDICTION_LENGTH = WINDOW_SIZE // 128  # = 6_144 bins
EMBEDDING_LENGTH  = WINDOW_SIZE // 64   # = 8_192 bins
EMBEDDING_CHANNELS = 1536                # feature depth from model

# Paths to our datasets: metadata TSV and output directories.
DATASETS = [
    {
        'name': 'dataset1',
        'meta_path': Path(
            '/mnt/sdb/markus_files/gene_exp/intermediate'
            '/dataset1/flashzoi_inputs/flashzoi_meta.tsv'
        ),
        'output_dir': Path(
            '/mnt/sdb/markus_files/gene_exp/output'
            '/dataset1/borzoi_preds'
        )
    },
    {
        'name': 'dataset2',
        'meta_path': Path(
            '/mnt/sdb/markus_files/gene_exp/intermediate'
            '/dataset2/flashzoi_inputs/flashzoi_meta.tsv'
        ),
        'output_dir': Path(
            '/mnt/sdb/markus_files/gene_exp/output'
            '/dataset2/borzoi_preds'
        )
    }
]

# File listing which tracks (currenly choosen muscle) to keep.
TRACKS_FILE = Path(
    '/mnt/sdb/markus_files/gene_exp/initial/targets_human.txt'
)


def calculate_md5(file_path: Path, chunk_size: int = 1 << 20) -> str:
    """
    Compute the MD5 checksum of a file in chunks to avoid using too much memory.
    Returns a hex string of the checksum.
    """
    md5_hasher = hashlib.md5()
    with open(file_path, 'rb') as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            md5_hasher.update(chunk)
    return md5_hasher.hexdigest()


def find_muscle_track_indices(txt_file: Path) -> List[int]: #TODO: remove the hardcoded muscle, make it dynamic
    """
    Read the targets text file (TSV). Find rows whose description mentions 'muscle'.
    Return a sorted list of their row indices.
    """
    df_targets = pd.read_csv(txt_file, sep='\t', comment='#')
    df_targets['idx'] = list(range(len(df_targets)))
    # Filter rows where the description contains "muscle" (case-insensitive)
    muscle_rows = df_targets[
        df_targets['description'].str.contains('muscle', case=False, na=False)
    ]
    muscle_indices = sorted(muscle_rows['idx'].tolist())

    if len(muscle_indices) == 0:
        raise RuntimeError('No muscle tracks found in targets file!')

    print(f"Found {len(muscle_indices)} muscle tracks; first 5 indices: {muscle_indices[:5]}")
    return muscle_indices

# Load which muscle tracks
MUSCLE_INDICES = find_muscle_track_indices(TRACKS_FILE)


def load_borzoi_models(num_folds: int) -> List[Borzoi]:
    """
    Download and prepare multiple Borzoi model folds for ensemble.
    Returns a list of loaded models in evaluation mode.
    """
    models: List[Borzoi] = []
    print(f"Loading {num_folds} Borzoi model folds...")

    for fold_idx in range(num_folds):
        # The pretrained model name includes the fold number
        model_name = f'johahi/borzoi-replicate-{fold_idx}'
        model = Borzoi.from_pretrained(model_name)
        model.to(DEVICE)
        model.eval()  # Turn off dropout, gradients, etc.
        models.append(model)
        print(f"  • Loaded fold {fold_idx}")

    print("All folds loaded and ready!\n")
    return models

# Load our model ensemble.
MODELS = load_borzoi_models(NUM_FOLDS)

# We will save the feature maps (embeddings) from the first model's backbone.
feature_cache: Dict[str, torch.Tensor] = {}

def save_backbone_features(_module, _input, output):
    """
    Hook function: called automatically during forward pass.
    Saves the output tensor (backbone features) into feature_cache.
    """
    # Detach from computation graph and store
    feature_cache['backbone'] = output.detach()

# Register our hook on the first model's convolution layer.
first_model = MODELS[0]
conv_layer = dict(first_model.named_modules())['horizontal_conv1.conv_layer']
conv_layer.register_forward_hook(save_backbone_features)

for dataset in DATASETS:
    name = dataset['name']
    meta_file = dataset['meta_path']
    output_base = dataset['output_dir']

    # Read metadata TSV: it lists each gene and the path to its one-hot file.
    df_meta = pd.read_csv(meta_file, sep='\t')
    # Drop rows where onehot_path is empty or missing
    df_meta = df_meta[
        df_meta['onehot_path'].notna() &
        (df_meta['onehot_path'].str.len() > 0)
    ].reset_index(drop=True)

    print(f"Processing dataset '{name}' with {len(df_meta)} valid one-hot files...")

    # Create output directories if they don't exist
    output_base.mkdir(parents=True, exist_ok=True)
    diag_dir = output_base / '_diag'
    diag_dir.mkdir(exist_ok=True)

    # We'll collect results and per-fold mean predictions
    all_results: List[Dict] = []
    fold_means: Dict[int, List[float]] = {k: [] for k in range(NUM_FOLDS)}

    # Loop over each row (gene) in metadata
    for idx, row in tqdm(df_meta.iterrows(), total=len(df_meta), desc=name):
        gene_name = row['gene']
        onehot_path = row['onehot_path']

        # 1. Load sequence one-hot encoding (numpy array)
        seq_array = np.load(onehot_path).astype(np.float32)
        seq_tensor = torch.from_numpy(seq_array)[None].to(DEVICE)

        # 2. Predict with each fold model, collect predictions
        fold_outputs = []
        with torch.no_grad():
            for fold_idx, model in enumerate(MODELS):
                output = model(seq_tensor)
                # Some models return dicts with a 'human' key
                if isinstance(output, dict):
                    output = output.get('human', output)
                fold_outputs.append(output)

        # Stack predictions: shape [num_folds, batch=1, time, channels]
        stacked = torch.stack(fold_outputs)
        # Average across folds: shape [1, time, channels]
        ensemble = stacked.mean(dim=0)
        # For each bin average only muscle channels
        muscle_bins = ensemble[0, MUSCLE_INDICES].mean(dim=0).cpu().numpy()

        # 3. Save backbone features from first fold
        embedding = feature_cache.pop('backbone')
        emb_numpy = embedding.cpu().to(EMBEDDING_DTYPE).numpy()

        # Make directory for this gene
        gene_dir = output_base / gene_name
        gene_dir.mkdir(exist_ok=True)

        # 4. Save files:
        # - ensemble prediction array
        # - overall mean value
        # - compressed embedding file
        np.save(gene_dir / f'{gene_name}_borzoi_pred.npy', muscle_bins)
        np.save(gene_dir / 'mean_ref.npy', muscle_bins.mean().item())
        np.savez_compressed(gene_dir / f'{gene_name}_borzoi_emb.npz', emb=emb_numpy)

        # 5. Save each fold's prediction if not already present
        for fold_idx in range(NUM_FOLDS):
            fold_file = gene_dir / f'fold{fold_idx}_pred.npy'
            if not fold_file.exists():
                single = stacked[fold_idx, 0, MUSCLE_INDICES].mean(dim=0).cpu().numpy()
                np.save(fold_file, single)

        # 6. Record mean of each fold's prediction for diagnostics
        for fold_idx in range(NUM_FOLDS):
            arr = np.load(gene_dir / f'fold{fold_idx}_pred.npy')
            fold_means[fold_idx].append(float(arr.mean()))

        # 7. Add result row for this gene to our results list
        result = {
            'gene': gene_name,
            'expr_value': row['expr_value'],
            'expr_rank': row['expr_rank'],
            'overlaps2kb': row['overlaps2kb'],
            'pred_npy': str(gene_dir / f'{gene_name}_borzoi_pred.npy'),
            'mean_ref': float(muscle_bins.mean()),
            'emb_npz': str(gene_dir / f'{gene_name}_borzoi_emb.npz')
        }
        all_results.append(result)

        print(f"[{idx+1}/{len(df_meta)}] {gene_name}: mean_ref = {muscle_bins.mean():.4f}")

    # After looping all genes, save combined metadata TSV for predictions
    meta_out = output_base / 'borzoi_preds_meta.tsv'
    pd.DataFrame(all_results).to_csv(meta_out, sep='\t', index=False)
    print(f"Ensemble predictions saved to {meta_out}\n")

    # Compare each fold versus the ensemble mean using correlation and plots.
    ensemble_means = np.array([r['mean_ref'] for r in all_results])

    for fold_idx in range(NUM_FOLDS):
        fold_vec = np.array(fold_means[fold_idx])
        pearson_corr = pearsonr(fold_vec, ensemble_means)[0]
        spearman_corr = spearmanr(fold_vec, ensemble_means)[0]
        print(
            f"Fold {fold_idx}: Pearson = {pearson_corr:.3f},"
            f" Spearman = {spearman_corr:.3f}"
        )

        # Save scatter plot comparing this fold to ensemble
        plt.figure()
        plt.scatter(ensemble_means, fold_vec)
        plt.xlabel('Ensemble mean_ref')
        plt.ylabel(f'Fold{fold_idx} mean_ref')
        plt.tight_layout()
        plt.savefig(diag_dir / f'fold{fold_idx}_vs_ensemble.png')
        plt.close()

    # Histogram of ensemble mean_ref
    plt.figure()
    plt.hist(ensemble_means, bins=15)
    plt.title('Distribution of ensemble mean_ref')
    plt.tight_layout()
    plt.savefig(diag_dir / 'hist_mean_ref.png')
    plt.close()

    # Scatter gene expression value vs predicted mean_ref
    expr_vals = df_meta['expr_value'].astype(float).to_numpy()
    plt.figure()
    plt.scatter(expr_vals, ensemble_means)
    plt.xlabel('Expression value')
    plt.ylabel('Ensemble mean_ref')
    plt.tight_layout()
    plt.savefig(diag_dir / 'scatter_exprValue_meanRef.png')
    plt.close()

    # Compute per-gene fold disagreement statistics
    stats_rows = []
    for res in all_results:
        gene_dir = Path(res['pred_npy']).parent
        fold_arrays = [np.load(gene_dir / f'fold{f}_pred.npy') for f in range(NUM_FOLDS)]
        stacked_folds = np.stack(fold_arrays)
        stats_rows.append({
            'gene': res['gene'],
            'within_std': float(stacked_folds.std(axis=0).mean()),
            'within_range': float(stacked_folds.ptp(axis=0).mean())
        })
    stats_df = pd.DataFrame(stats_rows).sort_values('within_range', ascending=False)
    stats_df.to_csv(diag_dir / 'per_gene_fold_stats.tsv', sep='\t', index=False)

    # Plot overlay of worst-disagreement gene
    worst_gene = stats_df.iloc[0]['gene']
    x_positions = np.arange(PREDICTION_LENGTH) * 128 / 1e3 - 262.144  # kb from TSS
    plt.figure(figsize=(10, 4))
    for fold_idx in range(NUM_FOLDS):
        y = np.load(output_base / worst_gene / f'fold{fold_idx}_pred.npy')
        plt.plot(x_positions, y, alpha=0.6, linewidth=0.8)
    # Plot ensemble prediction thicker and in black
    y_ens = np.load(output_base / worst_gene / f'{worst_gene}_borzoi_pred.npy')
    plt.plot(x_positions, y_ens, linewidth=2.2)
    plt.title(f"{name} - Worst disagreement: {worst_gene}")
    plt.xlabel('kb relative to TSS')
    plt.tight_layout()
    plt.savefig(diag_dir / f'overlay_{worst_gene}.png')
    plt.close()

    print(f"Diagnostics saved in {diag_dir}\n")

print("All datasets have been processed")