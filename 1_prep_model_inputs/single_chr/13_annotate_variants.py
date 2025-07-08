#!/usr/bin/env python3
import os
import csv
import pysam
import pandas as pd
from pathlib import Path

META_FILES = [
    "../../data/intermediate/dataset1/flashzoi_inputs/flashzoi_meta.tsv",
    "../../data/intermediate/dataset2/flashzoi_inputs/flashzoi_meta.tsv",
]
MANE_SUMMARY  = "../../data/initial/MANE.GRCh38.v1.4.summary.txt"
GNOMAD_VCF    = "../../data/initial/gnomad.genomes.v4.1.sites.chr20.vcf.bgz"

# Fixed half‐window for a 524 kb total (262 144 = 524 288/2)
HALF_WIN = 262_144
CHROM    = "chr20"


def load_mane_tss(mane_path):
    """
    Reads the MANE summary file, drops duplicate Ensembl_Gene entries,
    strips off ".N" from Ensembl IDs, normalizes NC_000020.## → chr20,
    and returns a dict { gene_base → TSS } for only chr20.
    """
    df = pd.read_csv(mane_path, sep="\t", dtype=str).drop_duplicates(subset="Ensembl_Gene")
    # Strip off version suffix (e.g. "ENSG000001234.5" → "ENSG000001234")
    df["gene_base"] = df["Ensembl_Gene"].str.replace(r"\.\d+$", "", regex=True)

    # Convert "NC_000020.11" → "chr20"
    df["chromosome"] = df["GRCh38_chr"].str.replace(
        r"^NC_0+(\d+)\.\d+$", r"chr\1", regex=True
    )

    def get_tss(row):
        return int(row["chr_start"]) if row["chr_strand"] == "+" else int(row["chr_end"])

    df["TSS"] = df.apply(get_tss, axis=1)

    # Keep only chr20
    df20 = df[df["chromosome"] == CHROM].copy()
    return dict(zip(df20["gene_base"], df20["TSS"]))


def write_variants_header(outfile):
    """
    Creates a TSV with columns:
       CHROM, POS, POS0, REF, ALT, AF
    """
    with open(outfile, "w", newline="") as fo:
        writer = csv.writer(fo, delimiter="\t")
        writer.writerow([
            "CHROM", "POS", "POS0",
            "REF", "ALT",
            "AF"
        ])


def annotate_variants(meta_tsv, mane_summary, vcf_path):
    # Extract dataset name from path
    dataset_name = Path(meta_tsv).parent.parent.name
    out_base = f"../../data/intermediate/{dataset_name}/flashzoi_inputs/variants"
    
    # 1) Load flashzoi_meta.tsv
    meta_df = pd.read_csv(meta_tsv, sep="\t")
    print(f"Processing {dataset_name}: Loaded metadata for {len(meta_df)} genes.")

    # 2) Build the gene→TSS map for chr20
    tss_dict = load_mane_tss(mane_summary)
    print(f"Found {len(tss_dict)} chr20 entries in MANE summary.")

    # 3) Open the gnomAD VCF via pysam (bgz + .tbi must exist side by side)
    vcf = pysam.VariantFile(vcf_path)
    print("Opened gnomAD VCF:", vcf_path)

    # 4) Iterate over each gene in meta_df
    for idx, row in meta_df.iterrows():
        gene = row["gene"]
        if gene not in tss_dict:
            print(f"[{idx+1}/{len(meta_df)}] SKIP {gene} (no chr20 TSS).")
            continue

        # Compute that gene's ±262 144 window (1-based coordinates)
        tss  = tss_dict[gene]
        left = max(1, tss - HALF_WIN)
        right= tss + HALF_WIN - 1
        print(f"[{idx+1}/{len(meta_df)}] {gene}: window = {left}-{right}")

        # Create an output folder for this gene
        gene_dir = os.path.join(out_base, gene)
        os.makedirs(gene_dir, exist_ok=True)

        var_tsv = os.path.join(gene_dir, f"{gene}_variants.tsv")
        write_variants_header(var_tsv)

        # 5) Fetch variants from chr20 between (left - 1, right) in pysam (0-based half-open)
        for rec in vcf.fetch(CHROM, left - 1, right):
            chrom = rec.chrom
            pos   = rec.pos     # 1-based position
            pos0  = pos - left  # zero‐based offset within the ±262 kb window
            ref   = rec.ref

            for alt in rec.alts:
                # AF is usually an array; take first element if present
                af_val = "NA"
                if "AF" in rec.info and rec.info["AF"] is not None:
                    af_entry = rec.info["AF"]
                    if isinstance(af_entry, (tuple, list)) and len(af_entry) > 0:
                        af_val = af_entry[0]
                    else:
                        af_val = af_entry

                # Write a line: CHROM, POS, POS0, REF, ALT, AF
                with open(var_tsv, "a", newline="") as fo:
                    writer = csv.writer(fo, delimiter="\t")
                    writer.writerow([
                        chrom, pos, pos0,
                        ref, alt,
                        af_val
                    ])

        print(f"Wrote variants to {var_tsv}")

    vcf.close()
    print(f"Done annotating variants for {dataset_name}.")


if __name__ == "__main__":
    print("Annotating 524 kb windows with gnomAD chr20 variants…")
    
    for meta_file in META_FILES:
        dataset_name = Path(meta_file).parent.parent.name
        out_base = f"../../data/intermediate/{dataset_name}/flashzoi_inputs/variants"
        os.makedirs(out_base, exist_ok=True)
        
        annotate_variants(meta_file, MANE_SUMMARY, GNOMAD_VCF)
        print(f"Variant lists for {dataset_name} are under: {out_base}")
    
    print("All datasets processed.")