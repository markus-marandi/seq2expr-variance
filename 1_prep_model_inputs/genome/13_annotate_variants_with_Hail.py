import hail as hl
import pandas as pd
from pathlib import Path
import re
import json
from tqdm import tqdm

# Load Spark config and inject Hail JAR path if needed
with open("../../data/initial/spark_conf.json") as f:
    spark_conf = json.load(f)

hail_home = "/mnt/sdb/markus_files/hail"

for k, v in spark_conf.items():
    if isinstance(v, str):
        spark_conf[k] = v.replace("{hail_home}", hail_home)

META_FILES = {
    "ClinGen_gene_curation_list": "../../data/intermediate/dataset3/ClinGen_gene_curation_list_flashzoi_meta.tsv",
    "nonessential_ensg_flashzoi_meta": "../../data/intermediate/dataset3/nonessential_ensg_flashzoi_meta.tsv",
}
MANE_PATH = "../../data/initial/MANE.GRCh38.v1.4.summary.txt"
GNOMAD_HT = "gs://gcp-public-data--gnomad/release/4.1/ht/genomes/gnomad.genomes.v4.1.sites.ht/"
OUT_BASE = Path("../../data/intermediate/dataset3/variants")
HALF_WIN = 262_144

def build_tss_dict(mane_path):
    print("Reading MANE summary")
    df = pd.read_csv(mane_path, sep="\t", dtype=str).drop_duplicates("Ensembl_Gene")
    df["gene_base"] = df["Ensembl_Gene"].str.replace(r"\.\d+$", "", regex=True)
    df["chrom"] = df["GRCh38_chr"].str.replace(r"^NC_0+(\d+)\.\d+$", r"chr\1", regex=True)
    df["TSS"] = df.apply(lambda r: int(r["chr_start"]) if r["chr_strand"] == "+" else int(r["chr_end"]), axis=1)
    return dict(zip(df["gene_base"], zip(df["chrom"], df["TSS"])))

def annotate_dataset(meta_path: str, dataset_name: str, tss_map: dict, ht: hl.Table):
    print(f"Processing dataset: {dataset_name}")
    meta_df = pd.read_csv(meta_path, sep="\t")
    dataset_out_base = OUT_BASE / dataset_name
    dataset_out_base.mkdir(parents=True, exist_ok=True)

    for _, row in tqdm(meta_df.iterrows(), total=len(meta_df), desc=f"{dataset_name}", unit="gene"):
        gene = row["gene"].strip()
        if gene not in tss_map:
            print(f"{gene} skipped: TSS not in MANE summary")
            continue

        chrom, tss = tss_map[gene]
        left = max(0, tss - HALF_WIN)
        right = tss + HALF_WIN

        try:
            gene_ht = ht.filter(
                (ht.locus.contig == chrom) &
                (ht.locus.position >= left) &
                (ht.locus.position <= right)
            )

            selected = gene_ht.select(
                CHROM = gene_ht.locus.contig,
                POS   = gene_ht.locus.position,
                REF   = gene_ht.alleles[0],
                ALT   = gene_ht.alleles[1],
                AF    = hl.if_else(hl.is_defined(gene_ht.info.AF), gene_ht.info.AF[0], hl.missing(hl.tfloat))
            )

            gene_out_dir = dataset_out_base / gene
            gene_out_dir.mkdir(parents=True, exist_ok=True)
            out_path = gene_out_dir / f"{gene}_variants.tsv"
            selected.export(str(out_path))

            count = sum(1 for _ in open(out_path)) - 1
            if count == 0:
                print(f"{gene} found in region {chrom}:{left}-{right}, but no variants.")
            else:
                print(f"{gene}: {chrom}:{left}-{right} → {out_path.name} ({count} variants)")

        except Exception as e:
            print(f"{gene} failed: {e}")
            continue

if __name__ == "__main__":
    hl.init(spark_conf=spark_conf, tmp_dir="../../data/cache", default_reference="GRCh38")
    ht = hl.read_table(GNOMAD_HT)
    tss_map = build_tss_dict(MANE_PATH)

    for dataset_name, meta_path in META_FILES.items():
        annotate_dataset(meta_path, dataset_name, tss_map, ht)

    hl.stop()
    print("\nDone processing all datasets.")