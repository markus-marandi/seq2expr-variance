#!/usr/bin/env python3
import argparse, os, time
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from borzoi_pytorch import Borzoi
from flashzoi_helpers import load_flashzoi_models

BASE2ROW = {b: i for i, b in enumerate("ACGT")}


# ─── Flashzoi Forward ──────────────────────────────────────────────────────
def fwd(models: Sequence[Borzoi], x: torch.Tensor) -> torch.Tensor:
    y_sum = None
    with torch.autocast(x.device.type, dtype=torch.float16, enabled=x.device.type == "cuda"):
        for m in models:
            y = m(x)
            y = y["human"] if isinstance(y, dict) else y
            c = y.shape[-1] // 2
            sig = y[:, 1, c - 16:c + 16].mean(1)
            y_sum = sig if y_sum is None else y_sum + sig
    return y_sum / len(models)


# ─── Per-Gene Scoring ──────────────────────────────────────────────────────
def score_gene(gene_dir: Path, onehot_dir: Path, pred_dir: Path,
               out_dir: Path, models: Sequence[Borzoi],
               batch: int, device: torch.device):
    gene = gene_dir.name
    start_time = time.time()

    in_tsv = gene_dir / f"{gene}_variants.tsv"
    out_gene_dir = out_dir / gene
    out_gene_dir.mkdir(parents=True, exist_ok=True)
    tsv = out_gene_dir / f"{gene}_variants.tsv"

    if not in_tsv.exists():
        print(f"✘ {gene}: missing input TSV"); return

    df = pd.read_csv(in_tsv, sep="\t")
    df.rename(columns={c: c.upper() for c in df.columns}, inplace=True)

    if "REF" not in df.columns or "ALT" not in df.columns or "POS0" not in df.columns:
        print(f"✘ {gene}: missing required columns"); return

    snp_mask = (df["REF"].str.len() == 1) & (df["ALT"].str.len() == 1)
    df = df[snp_mask].reset_index(drop=True)

    if not df.index.is_unique:
        df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"✘ {gene}: no SNP records"); return

    df["DELTA"] = pd.to_numeric(df.get("DELTA", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["VAR_I"] = pd.to_numeric(df.get("VAR_I", pd.Series(np.nan, index=df.index)), errors="coerce")

    pending = df["DELTA"].isna().to_numpy()
    todo = np.flatnonzero(pending).tolist()
    todo = pd.Index(todo).drop_duplicates().tolist()

    if not todo:
        print(f"✔ {gene}: already done"); return

    try:
        ref_np = np.load(onehot_dir / f"{gene}.npy").astype(np.float16)
        ref_t = torch.from_numpy(ref_np).to(device)
        ref_avg = float(np.load(pred_dir / gene / "mean_ref.npy"))
    except Exception as e:
        print(f"✘ {gene}: missing onehot or mean_ref — {e}")
        return

    print(f"▶ {gene}: {len(todo)} variants to process")

    for s in tqdm(range(0, len(todo), batch), leave=False, desc=f"{gene} chunks"):
        idx = todo[s:s + batch]
        sub = df.loc[idx]

        N = len(sub)
        pos0 = torch.as_tensor(sub["POS0"].values, dtype=torch.long, device=device)
        alt = torch.as_tensor([BASE2ROW.get(a.upper(), -1) for a in sub["ALT"]],
                              dtype=torch.long, device=device)
        ref = torch.as_tensor([BASE2ROW[r] for r in sub["REF"]],
                              dtype=torch.long, device=device)

        xb = ref_t.expand(N, -1, -1).clone()
        xb[torch.arange(N, device=device), ref, pos0] = 0.
        xb[torch.arange(N, device=device), alt, pos0] = 1.

        bad = alt == -1
        valid_mask = ~bad.cpu().numpy()
        valid_idx = [idx[i] for i in range(len(idx)) if valid_mask[i]]
        invalid_idx = [idx[i] for i in range(len(idx)) if not valid_mask[i]]

        if valid_idx:
            xb_valid = xb[valid_mask]
            with torch.no_grad():
                delta_valid = (fwd(models, xb_valid).cpu().numpy() - ref_avg).astype(np.float32)
            df.loc[valid_idx, "DELTA"] = delta_valid

            if "AF" in sub.columns:
                af = pd.to_numeric(sub["AF"], errors="coerce").to_numpy(dtype=float)
            else:
                af = np.full(len(sub), np.nan, dtype=float)

            var_i_valid = (delta_valid ** 2) * (2 * af[valid_mask] * (1 - af[valid_mask]))
            df.loc[valid_idx, "VAR_I"] = var_i_valid

        df.loc[invalid_idx, "DELTA"] = np.nan
        df.loc[invalid_idx, "VAR_I"] = np.nan

        tmp = out_gene_dir / f"{gene}_variants.working"
        df.to_csv(tmp, sep="\t", index=False)
        os.replace(tmp, tsv)
        del xb
        torch.cuda.empty_cache()

    elapsed = time.time() - start_time
    vps = len(todo) / elapsed if elapsed > 0 else 0
    print(f"✓ {gene}: done in {elapsed:.2f}s | {len(todo)} variants | {vps:.1f} var/sec")

    log_path = Path("logs/variant_benchmark.tsv")
    log_path.parent.mkdir(exist_ok=True)
    with open(log_path, "a") as f:
        f.write(f"{gene}\t{len(todo)}\t{elapsed:.2f}\t{vps:.2f}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants-dir", required=True, type=Path)
    ap.add_argument("--onehot-dir", required=True, type=Path)
    ap.add_argument("--pred-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--folds", default=4, type=int)
    ap.add_argument("--batch", default=64, type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gene", help="Optional single gene to run")

    args = ap.parse_args()
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available()
                          else "cpu")
    models = load_flashzoi_models(args.folds, device)

    dirs = [args.variants_dir / args.gene] if args.gene else \
           sorted([d for d in args.variants_dir.iterdir() if d.is_dir()])

    for d in dirs:
        score_gene(d, args.onehot_dir, args.pred_dir,
                   args.out_dir, models, args.batch, device)


if __name__ == "__main__":
    main()
