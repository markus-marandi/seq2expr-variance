#!/usr/bin/env python3
"""
predict_flashzoi_unified.py – Unified Flashzoi prediction script for all datasets

This script works for both single_chr (dataset1/dataset2) and genome-wide (dataset3) workflows.

Examples:
    # Single chromosome datasets (dataset1/dataset2)
    python predict_flashzoi_unified.py \
        --dataset-root data/intermediate/dataset1 \
        --dataset-type single_chr \
        --track-idx-file data/track_lists/muscle_idx.txt \
        --output-dir data/output/dataset1/flashzoi_preds \
        --device cpu

    # Genome-wide datasets (dataset3)  
    python predict_flashzoi_unified.py \
        --dataset-root data/intermediate/dataset3 \
        --dataset-type genome \
        --cohort ClinGen_gene_curation_list \
        --track-idx-file data/track_lists/clingen_meta5_idx.txt \
        --output-dir data/output/dataset3/flashzoi_outputs \
        --device cuda
"""
from __future__ import annotations
import argparse
import hashlib
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

from flashzoi_helpers import load_flashzoi_models

WINDOW_SIZE        = 524_288
PREDICTION_LENGTH  = WINDOW_SIZE // 128                 # 4,096 bins
EMBEDDING_LENGTH   = WINDOW_SIZE // 64                  # 8,192 bins
EMBEDDING_CHANNELS = 1_536
EMBED_DTYPE        = torch.float16

def md5(path: Path, chunk: int = 1 << 20) -> str:
    """Calculate MD5 hash of a file."""
    h = hashlib.md5()
    with path.open('rb') as f:
        for blk in iter(lambda: f.read(chunk), b''):
            h.update(blk)
    return h.hexdigest()

def load_track_indices(idx_file: Path) -> List[int]:
    """Read one integer track index per line from *idx_file*."""
    idx = [int(l.strip()) for l in idx_file.read_text().splitlines() if l.strip()]
    if not idx:
        raise ValueError(f"{idx_file} is empty or contains no valid integers.")
    print(f"Using {len(idx)} tracks → first 5: {idx[:5]}")
    return idx

def find_meta_file(dataset_root: Path, dataset_type: str, cohort: str = None) -> Path:
    """Find the appropriate metadata TSV file based on dataset type."""
    if dataset_type == "single_chr":
        # For dataset1/dataset2: look for flashzoi_meta.tsv in flashzoi_inputs/
        meta_file = dataset_root / "flashzoi_inputs" / "flashzoi_meta.tsv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        return meta_file
    elif dataset_type == "genome":
        if not cohort:
            raise ValueError("cohort parameter is required for genome-wide datasets")
        # For dataset3: look for {cohort}_flashzoi_meta.tsv in onehots/{cohort}/
        meta_file = dataset_root / "onehots" / cohort / f"{cohort}_flashzoi_meta.tsv"
        if not meta_file.exists():
            raise FileNotFoundError(f"Meta file not found: {meta_file}")
        return meta_file
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

def setup_output_directory(output_dir: Path, dataset_type: str, cohort: str = None) -> tuple[Path, Path]:
    """Setup output directory structure based on dataset type."""
    if dataset_type == "genome" and cohort:
        out_base = output_dir / cohort
    else:
        out_base = output_dir
    
    diag_dir = out_base / "_diag"
    out_base.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)
    return out_base, diag_dir

def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Unified Flashzoi prediction for single_chr and genome datasets"
    )
    
    # Required arguments
    ap.add_argument('--dataset-root', required=True, type=Path,
                    help='Path to dataset root (e.g., data/intermediate/dataset1)')
    ap.add_argument('--dataset-type', required=True, choices=['single_chr', 'genome'],
                    help='Type of dataset: single_chr (dataset1/2) or genome (dataset3)')
    ap.add_argument('--track-idx-file', required=True, type=Path,
                    help="Text file with one Borzoi track index per line")
    ap.add_argument('--output-dir', required=True, type=Path,
                    help='Where predictions & diagnostics are written')
    
    # Optional arguments
    ap.add_argument('--cohort', 
                    help='Cohort name (required for genome datasets)')
    ap.add_argument('--folds', type=int, default=1,
                    help='Flashzoi folds to ensemble; keep at 1 for speed')
    ap.add_argument('--device', default='cpu', choices=['cuda', 'cpu', 'mps'],
                    help='Device to run inference on')
    ap.add_argument('--autocast', action='store_true',
                    help='Mixed-precision inference (recommended for GPU)')
    
    args = ap.parse_args()
    
    # Validate arguments
    if args.dataset_type == "genome" and not args.cohort:
        ap.error("--cohort is required when --dataset-type is 'genome'")
    
    device = torch.device(args.device)
    track_idxs = torch.as_tensor(load_track_indices(args.track_idx_file),
                                 dtype=torch.long, device=device)
    
    # Find metadata file
    meta_tsv = find_meta_file(args.dataset_root, args.dataset_type, args.cohort)
    print(f"Using metadata file: {meta_tsv}")
    
    # Load and validate metadata
    df = pd.read_csv(meta_tsv, sep='\t').dropna(subset=['onehot_path'])
    print(f"Found {len(df)} genes with valid onehot files")
    
    # Setup output directories
    out_base, diag_dir = setup_output_directory(args.output_dir, args.dataset_type, args.cohort)
    print(f"Output directory: {out_base}")
    
    models = load_flashzoi_models(args.folds, device, use_autocast=args.autocast)
    feat_cache: Dict[str, torch.Tensor] = {}
    
    # Find a suitable Conv1d layer to hook for embeddings
    backbone_layer = None
    for name, module in models[0].named_modules():
        if 'horizontal_conv' in name and isinstance(module, torch.nn.Conv1d):
            backbone_layer = module
            print(f"Hooking backbone on {name}")
            break
    
    if backbone_layer is None:
        for module in models[0].modules():
            if isinstance(module, torch.nn.Conv1d):
                backbone_layer = module
                print("Hooking backbone on first Conv1d found (generic fallback)")
                break
    
    if backbone_layer is None:
        raise RuntimeError("Could not locate a Conv1d layer to hook for embeddings")
    
    backbone_layer.register_forward_hook(
        lambda _m, _i, o: feat_cache.__setitem__('backbone', o.detach())
    )
    
    # ─── prediction loop ─────────────────────────────────────────────────
    fold_means: Dict[int, List[float]] = {f: [] for f in range(args.folds)}
    combined_rows: List[Dict] = []
    
    for i, row in tqdm(df.iterrows(), total=len(df), desc=f"{args.dataset_type}"):
        gene = row['gene']
        seq_path = Path(row['onehot_path'])
        
        # Load sequence
        seq = torch.from_numpy(np.load(seq_path).astype(np.float32))[None].to(device)
        
        # Predict with autocast if enabled
        with torch.no_grad(), torch.autocast(
                device_type='cuda' if device.type == 'cuda' else 'cpu',
                dtype=torch.float16,
                enabled=(device.type == 'cuda' and args.autocast)):
            outs = [
                m(seq)['human'] if isinstance(m(seq), dict) else m(seq)
                for m in models
            ]
        
        # Ensemble (mean across folds)
        ens = torch.stack(outs).mean(0)             # [1, C, T]
        c = ens.shape[-1] // 2
        pred = ens[0, track_idxs, c-16:c+16].mean((0, 1)).cpu().numpy()
        
        # Capture embedding
        emb = feat_cache.pop('backbone').cpu().to(EMBED_DTYPE).numpy()
        
        # Make gene sub-folder
        gdir = out_base / gene
        gdir.mkdir(exist_ok=True)
        
        # Save arrays
        np.save(gdir / f"{gene}_flashzoi_pred.npy", pred)
        np.save(gdir / "mean_ref.npy", pred.mean().item())
        np.savez_compressed(gdir / f'{gene}_flashzoi_emb.npz', emb=emb)
        
        # Store per-fold predictions (saves space if folds==1)
        for f, out in enumerate(outs):
            ff = gdir / f'fold{f}_pred.npy'
            if not ff.exists():
                fold_pred = out[0, track_idxs, c-16:c+16].mean((0, 1)).cpu().numpy()
                np.save(ff, fold_pred)
            fold_means[f].append(float(np.load(ff).mean()))
        
        # Collect metadata row
        result_row = {
            'gene': gene,
            'pred_npy': str(gdir / f'{gene}_flashzoi_pred.npy'),
            'mean_ref': float(pred.mean()),
            'emb_npz': str(gdir / f'{gene}_flashzoi_emb.npz')
        }
        
        # Add expression info if available
        if 'expr_value' in row:
            result_row['expr_value'] = row['expr_value']
        if 'expr_rank' in row:
            result_row['expr_rank'] = row['expr_rank']
        if 'overlaps2kb' in row:
            result_row['overlaps2kb'] = row['overlaps2kb']
            
        combined_rows.append(result_row)
        print(f"[{i+1}/{len(df)}] {gene}: mean_ref = {pred.mean():.4f}")
    
    # ─── save cohort-level metadata ──────────────────────────────────────
    meta_out = out_base / 'flashzoi_preds_meta.tsv'
    pd.DataFrame(combined_rows).to_csv(meta_out, sep='\t', index=False)
    print(f"Predictions metadata saved → {meta_out}")
    
    # ─── diagnostics (correlations & plots) ──────────────────────────────
    ens_means = np.array([r['mean_ref'] for r in combined_rows])
    
    for f in range(args.folds):
        fm = np.array(fold_means[f])
        if len(fm) > 1:  # Need at least 2 points for correlation
            p, s = pearsonr(fm, ens_means)[0], spearmanr(fm, ens_means)[0]
            print(f"fold{f}: Pearson {p:.3f} | Spearman {s:.3f}")
            
            plt.figure(figsize=(6, 5))
            plt.scatter(ens_means, fm, s=8, alpha=0.7)
            plt.xlabel('ensemble mean_ref')
            plt.ylabel(f'fold{f} mean_ref')
            plt.title(f'Fold {f} vs Ensemble Correlation')
            plt.tight_layout()
            plt.savefig(diag_dir / f'fold{f}_vs_ensemble.png', dpi=150)
            plt.close()
    
    # Distribution plot
    plt.figure(figsize=(8, 5))
    plt.hist(ens_means, bins=min(15, len(ens_means)//2), alpha=0.7, edgecolor='black')
    plt.xlabel('Ensemble mean_ref')
    plt.ylabel('Count')
    plt.title(f'Distribution of mean_ref ({args.dataset_type})')
    plt.tight_layout()
    plt.savefig(diag_dir / 'hist_mean_ref.png', dpi=150)
    plt.close()
    
    print(f"Diagnostics saved → {diag_dir}")
    print("✓ Prediction completed successfully.")

if __name__ == '__main__':
    main() 