#!/usr/bin/env python3

import os, sys, hashlib
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import pandas as pd
from pyfaidx import Fasta, FetchError
from tqdm.auto import tqdm

IN_BEDS = [
    "../../data/intermediate/dataset3/ClinGen_gene_curation_list_tss±2kb.bed",
    "../../data/intermediate/dataset3/nonessential_ensg_tss±2kb.bed",
]

REF_FA  = "../../data/initial/GRCh38.primary_assembly.genome.fa"

WINDOW      = 524_288
HALF_WIN    = WINDOW // 2
BASE2ROW    = {"A": 0, "C": 1, "G": 2, "T": 3}

def coerce_chrom(chrom: str, fasta: Fasta) -> str:
    if chrom in fasta:
        return chrom
    if chrom.startswith("chr") and chrom[3:] in fasta:
        return chrom[3:]
    if ("chr" + chrom) in fasta:
        return "chr" + chrom
    raise KeyError(f"{chrom} not found in FASTA")

def one_hot_encode(seq: str) -> np.ndarray:
    arr = np.zeros((4, len(seq)), dtype=np.float32)
    for i, b in enumerate(seq):
        j = BASE2ROW.get(b)
        if j is not None:
            arr[j, i] = 1.0
    return arr

def extract_window(chrom: str, tss: int, fasta: Fasta) -> np.ndarray:
    chrom = coerce_chrom(chrom, fasta)
    left  = max(0, tss - HALF_WIN)
    right = min(tss + HALF_WIN, len(fasta[chrom]))  # Clamp to chromosome length

    try:
        seq = fasta[chrom][left:right].seq
    except FetchError as e:
        raise RuntimeError(f"FASTA fetch failed for {chrom}:{left}-{right}") from e

    seq = seq.upper()
    if len(seq) < WINDOW:
        pad_len = WINDOW - len(seq)
        seq = seq + "N" * pad_len
        print(f"Padded {chrom}:{left}-{right} by {pad_len} bp")

    return one_hot_encode(seq)

def prepare_dataset(in_bed: str) -> Path:
    name_base = Path(in_bed).stem.replace("_tss±2kb", "")
    out_dir = Path("../../data/intermediate/dataset3/onehots") / name_base
    out_dir.mkdir(parents=True, exist_ok=True)

    cols = ["chrom","start","end","gene",
            "expr_value","expr_rank","overlaps","tss_pos"]
    df = pd.read_csv(in_bed, sep="\t", comment="#", names=cols,
                     dtype={"chrom":str,"gene":str,"overlaps":str})

    fasta = Fasta(REF_FA, strict_bounds=True)

    triplets = [(r.chrom, int(r.tss_pos) if pd.notna(r.tss_pos) else None) for _, r in df.iterrows()]
    counts = Counter(triplets)
    dup_loci = {k for k,v in counts.items() if v > 1}
    if dup_loci:
        print("same (chrom, tss) appears multiple times:")
        for chrom,tss in sorted(dup_loci):
            genes = df[(df.chrom == chrom) & (df.tss_pos == tss)].gene.tolist()
            print(f"  {chrom}:{tss} → {', '.join(genes)}")

    seen_genes = set()
    meta_records = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=Path(in_bed).name):
        gene = r.gene
        if gene in seen_genes:
            continue
        seen_genes.add(gene)

        chrom = r.chrom
        tss   = int(r.tss_pos) if pd.notna(r.tss_pos) else int(r.start) + 2000

        try:
            chrom_resolved = coerce_chrom(chrom, fasta)
            chrom_len = len(fasta[chrom_resolved])
        except KeyError:
            print(f"⚠ Skipping {gene}: unknown chromosome {chrom}")
            continue

        if tss + HALF_WIN > chrom_len:
            print(f"⚠ Skipping {gene}: TSS window exceeds {chrom} length")
            continue

        try:
            onehot = extract_window(chrom, tss, fasta)
        except RuntimeError as e:
            print(f"⚠ Skipping {gene}: {e}")
            continue

        out_npy = out_dir / f"{gene}.npy"
        np.save(out_npy, onehot)

        meta_records.append({
            "gene":        gene,
            "onehot_path": str(out_npy),
            "expr_value":  float(r.expr_value),
            "expr_rank":   float(r.expr_rank),
            "overlaps2kb": r.overlaps
        })

    meta_path = out_dir / f"{name_base}_flashzoi_meta.tsv"
    pd.DataFrame(meta_records).to_csv(meta_path, sep="\t", index=False)
    print(f"✓ wrote {len(meta_records)} arrays → {meta_path}")
    return meta_path


def md5(path: Path, chunk: int = 1<<20) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        while (b := f.read(chunk)):
            h.update(b)
    return h.hexdigest()

def sanity_check(meta_tsv: Path) -> None:
    meta = pd.read_csv(meta_tsv, sep="\t")
    seen = defaultdict(list)
    bad_shape = bad_onehot = 0

    for _, row in meta.iterrows():
        p = Path(row.onehot_path)
        h = md5(p)
        seen[h].append(row.gene)

        arr = np.load(p, mmap_mode="r")
        if arr.shape != (4, WINDOW):
            bad_shape += 1
            continue
        colsum = arr.sum(0)
        if not np.allclose(colsum[colsum > 0], 1.0):
            bad_onehot += 1

    dup_sets = [g for g in seen.values() if len(g) > 1]
    if dup_sets or bad_shape or bad_onehot:
        print("\nSanity-check warnings:")
        if dup_sets:
            print(f"  duplicate .npy files: {sum(len(s)-1 for s in dup_sets)}")
            for s in dup_sets:
                print("   ", ", ".join(s))
        if bad_shape:
            print(f"  arrays with wrong shape: {bad_shape}")
        if bad_onehot:
            print(f"  arrays violating one-hot: {bad_onehot}")
        sys.exit(1)

    print("✓ Sanity-check passed – all arrays unique, 4×524 288, valid one-hot.\n")

def main():
    for bed in IN_BEDS:
        try:
            meta = prepare_dataset(bed)
            sanity_check(meta)
        except Exception as e:
            print(f"Skipping dataset due to error: {e}")
    print("All datasets processed – ready for Borzoi.")

if __name__ == "__main__":
    main()