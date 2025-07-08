import pandas as pd
from tqdm import tqdm
import re

# --- Input paths ---
gene_list_path = "/Users/markus/seq2expr-variance/data/initial/dataset3/nonessential.ensg.tsv"
mane_path = "/Users/markus/seq2expr-variance/data/initial/MANE.GRCh38.v1.4.summary.txt"
out_bed_path = "/Users/markus/seq2expr-variance/data/intermediate/dataset3/nonessential_ensg_tss±2kb.bed"

# --- Load curated gene list ---
ensg_list = pd.read_csv(gene_list_path, header=None, names=["ensg"])
print(f"Loaded {len(ensg_list)} input gene IDs")

# --- Load and preprocess MANE summary ---
mane = pd.read_csv(mane_path, sep="\t", dtype=str)
mane["Ensembl_Gene"] = mane["Ensembl_Gene"].str.replace(r"\.\d+$", "", regex=True)

# --- find duplicates before filtering ---
dup_mane = mane[mane.duplicated("Ensembl_Gene", keep=False)]
dup_in_input = dup_mane[dup_mane["Ensembl_Gene"].isin(ensg_list["ensg"])]
if not dup_in_input.empty:
    print(f"{dup_in_input['Ensembl_Gene'].nunique()} duplicated genes from your list in MANE ({len(dup_in_input)} total rows)")
    print("Examples of duplicated genes:")
    print(dup_in_input["Ensembl_Gene"].value_counts().head())

# --- Assign chromosome names and TSS position ---
mane["chrom"] = mane["GRCh38_chr"].str.replace(r"^NC_0+(\d+)\.\d+$", r"chr\1", regex=True)
mane["TSS"] = mane.apply(lambda r: int(r["chr_start"]) if r["chr_strand"] == "+" else int(r["chr_end"]), axis=1)

# --- Filter to canonical chromosomes ---
mane = mane[mane["chrom"].str.match(r"^chr([1-9]|1[0-9]|2[0-2]|X|Y)$")]
print(f"Retained {len(mane)} MANE entries on canonical chromosomes")

# --- Drop duplicate Ensembl_Gene entries (keep first only) ---
mane = mane.drop_duplicates("Ensembl_Gene", keep="first")
print(f"Deduplicated to {len(mane)} unique Ensembl genes")

# --- Merge curated gene list with filtered MANE ---
genes = ensg_list.merge(mane, left_on="ensg", right_on="Ensembl_Gene")
print(f"Matched {len(genes)} entries from your list to MANE")

# --- Debug: which genes were not found in MANE ---
matched_ids = set(genes["Ensembl_Gene"])
input_ids = set(ensg_list["ensg"])
dropped_ids = input_ids - matched_ids
if dropped_ids:
    print(f"{len(dropped_ids)} gene(s) from your list were not found in MANE")
    print("Sample of missing genes:", list(dropped_ids)[:5])

# --- Create BED file: ±2kb around TSS ---
bed_records = []
for _, row in tqdm(genes.iterrows(), total=genes.shape[0], desc="Creating BED entries"):
    start = max(0, row["TSS"] - 2000)
    end = row["TSS"] + 2000
    bed_records.append([row["chrom"], start, end, row["Ensembl_Gene"]])

bed = pd.DataFrame(bed_records, columns=["chr", "start", "end", "name"])
bed.sort_values(["chr", "start"]).to_csv(out_bed_path, sep="\t", header=False, index=False)

print(f"BED file written with {len(bed)} entries → {out_bed_path}")