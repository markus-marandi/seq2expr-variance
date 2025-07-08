import pandas as pd

# 1) load GTEx TPM
tpm = pd.read_csv(
    "../../data/initial/GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_median_tpm.gct",
    sep="\t", skiprows=2
)
tpm = tpm.rename(columns={"Name": "gene_id", "Muscle_Skeletal": "TPM"})

# 2) strip versions from both sides
tpm["gene_base"] = tpm.gene_id.str.split(".").str[0]

# your MANE set for chr20, also version-stripped
mane = pd.read_csv(
    "../../data/initial/MANE.GRCh38.v1.4.summary.txt",
    sep="\t", dtype=str
)
mane["gene_base"] = mane.Ensembl_Gene.str.split(".").str[0]
genes_chr20 = set(
    mane.loc[
        mane.GRCh38_chr.str.contains(r"NC_0*20"), "gene_base"
    ]
)

# 3) rank & percentile
tpm["pct"] = tpm.TPM.rank(pct=True)

# 4) filter muscle TPM between 50–95th percentiles
tpm_sel = tpm[(tpm["pct"] >= 0.70) & (tpm["pct"] <= 0.95)].copy()

# 5) now intersect on gene_base
tpm_small = tpm_sel[tpm_sel["gene_base"].isin(genes_chr20)]

print("Found", len(tpm_small), "chr20–MANE genes in 70–95% TPM range")
print(tpm_small[["gene_id","gene_base","TPM","pct"]])

# 6) save
tpm_small.to_csv(
    "../data/intermediate/"
    "GTEx_chr20_mane_TPM.tsv", sep="\t", index=False
)