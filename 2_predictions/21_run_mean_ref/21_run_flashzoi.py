"""
run_flashzoi.py – Flashzoi ensemble inference with an arbitrary list of
Borzoi track indices supplied at run-time.

Create the list once, e.g.
    grep -n 'GTEX_Brain_Cortex.*RNA-seq' targets_gtex.txt | cut -d: -f1 > cortex_idx.txt

Then run:
    python run_flashzoi.py \
        --dataset-root  data/intermediate/dataset3 \
        --cohort        ClinGen_gene_curation_list \
        --pred-dir      data/output/dataset3/flashzoi_outputs \
        --track-idx-file cortex_idx.txt \
        --device        cuda
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
PREDICTION_LENGTH  = WINDOW_SIZE // 128                 # 6 144 bins
EMBEDDING_LENGTH   = WINDOW_SIZE // 64                  # 8 192 bins
EMBEDDING_CHANNELS = 1_536
EMBED_DTYPE        = torch.float16

def md5(path: Path, chunk: int = 1 << 20) -> str:
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


def main() -> None:
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--dataset-root', required=True,
                    help='…/data/intermediate/dataset3')
    ap.add_argument('--cohort',       required=True,
                    choices=['ClinGen_gene_curation_list', 'nonessential_ensg'])
    ap.add_argument('--pred-dir',     required=True,
                    help='where predictions & diagnostics are written')
    ap.add_argument('--folds',        type=int, default=1,
                    help='Flashzoi folds to ensemble; keep at 1 for speed')
    ap.add_argument('--device',       default='cuda', choices=['cuda', 'cpu', 'mps'])
    ap.add_argument('--autocast',     action='store_true',
                    help='mixed-precision inference (default on GPU)')
    ap.add_argument("--track-idx-file", required=True, type=Path,
                   help="text file with one Borzoi track index per line")
    args = ap.parse_args()

    device = torch.device(args.device)
    track_idxs = torch.as_tensor(load_track_indices(args.track_idx_file),
                                 dtype=torch.long, device=device)

    # TSV containing gene, onehot_path, etc.
    meta_tsv = (
        Path(args.dataset_root) /
        'onehots' / args.cohort /
        f'{args.cohort}_flashzoi_meta.tsv'
    )
    df = pd.read_csv(meta_tsv, sep='\t').dropna(subset=['onehot_path'])

    out_base = Path(args.pred_dir) / args.cohort
    diag_dir = out_base / '_diag'
    out_base.mkdir(parents=True, exist_ok=True)
    diag_dir.mkdir(exist_ok=True)

    models = load_flashzoi_models(args.folds, device, use_autocast=args.autocast)
    feat_cache: Dict[str, torch.Tensor] = {}

    backbone_layer = None
    for name, module in models[0].named_modules():
        if 'horizontal_conv' in name and isinstance(module, torch.nn.Conv1d):
            backbone_layer = module
            print(f"Hooking backbone on {name}")
            break

    # fall-back: first Conv1d in the whole model
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

    fold_means: Dict[int, List[float]] = {f: [] for f in range(args.folds)}
    combined_rows: List[Dict] = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc=args.cohort):
        gene = row['gene']
        seq_path = Path(row['onehot_path'])
        seq = torch.from_numpy(np.load(seq_path).astype(np.float32))[None].to(device)

        # predict
        with torch.no_grad(), torch.autocast(
                device_type='cuda',
                dtype=torch.float16,
                enabled=(device.type == 'cuda' and args.autocast)):
            outs = [
                m(seq)['human'] if isinstance(m(seq), dict) else m(seq)
                for m in models
            ]


        # ensemble (mean across folds) … but args.folds=1 by default
        ens = torch.stack(outs).mean(0)             # [1, C, T]
        c   = ens.shape[-1] // 2
        pred = ens[0, track_idxs, c-16:c+16].mean((0, 1)).cpu().numpy()

        # capture embedding
        emb = feat_cache.pop('backbone').cpu().to(EMBED_DTYPE).numpy()

        # make gene sub-folder
        gdir = out_base / gene
        gdir.mkdir(exist_ok=True)

        # save arrays
        np.save(gdir / f"{gene}_flashzoi_pred.npy", pred)
        np.save(gdir / "mean_ref.npy", pred.mean().item())
        np.savez_compressed(gdir / f'{gene}_flashzoi_emb.npz', emb=emb)

        # store per-fold preds (optional; saves space if folds==1)
        for f, out in enumerate(outs):
            ff = gdir / f'fold{f}_pred.npy'
            if not ff.exists():
                np.save(ff, out[0, track_idxs, c-16:c+16].mean((0, 1)).cpu().numpy())
            fold_means[f].append(float(np.load(ff).mean()))

        combined_rows.append({
            'gene': gene,
            'expr_value': row['expr_value'],
            'expr_rank':  row['expr_rank'],
            'overlaps2kb': row['overlaps2kb'],
            'pred_npy': str(gdir / f'{gene}_flashzoi_pred.npy'),
            'mean_ref':  float(pred.mean()),
            'emb_npz':   str(gdir / f'{gene}_flashzoi_emb.npz')
        })

        print(f"[{i+1}/{len(df)}] {gene}: mean_ref = {pred.mean():.4f}")

    # ─── save cohort-level metadata ──────────────────────────────────────
    meta_out = out_base / 'flashzoi_preds_meta.tsv'
    pd.DataFrame(combined_rows).to_csv(meta_out, sep='\t', index=False)
    print(f"Predictions saved → {meta_out}")

    # ─── quick diagnostics (correlations & plots) ────────────────────────
    ens_means = np.array([r['mean_ref'] for r in combined_rows])
    for f in range(args.folds):
        fm = np.array(fold_means[f])
        p, s = pearsonr(fm, ens_means)[0], spearmanr(fm, ens_means)[0]
        print(f"fold{f}: Pearson {p:.3f} | Spearman {s:.3f}")
        plt.figure()
        plt.scatter(ens_means, fm, s=8)
        plt.xlabel('ensemble mean_ref'); plt.ylabel(f'fold{f} mean_ref')
        plt.tight_layout()
        plt.savefig(diag_dir / f'fold{f}_vs_ensemble.png'); plt.close()

    plt.figure(); plt.hist(ens_means, bins=15)
    plt.title('Distribution of ensemble mean_ref'); plt.tight_layout()
    plt.savefig(diag_dir / 'hist_mean_ref.png'); plt.close()

    print(f"Diagnostics → {diag_dir}")
    print("✓ all done.")


if __name__ == '__main__':
    main()