import pandas as pd, numpy as np, os

SUMMARY_PATH    = "../data/initial/MANE.GRCh38.v1.4.summary.txt"
MEDIAN_TPM_PATH = ("../data/initial/"
                   "GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_median_tpm.gct")

OUT1_BED = "../data/intermediate/dataset1/chr20_test15_tss±2kb.bed"
OUT2_BED = "../data/intermediate/dataset2/chr20_test15_tss±2kb.bed"

# ensure output dirs
for path in (os.path.dirname(OUT1_BED), os.path.dirname(OUT2_BED)):
    os.makedirs(path, exist_ok=True)


# load and filter GTEx TPM
def load_expression(path, pct_lo=0.70, pct_hi=0.95):
    df = (pd.read_csv(path, sep="\t", skiprows=2,
                      usecols=["Name","Muscle_Skeletal"])
            .rename(columns={"Name":"raw_id","Muscle_Skeletal":"expr_value"}))
    df["id_base"]   = df["raw_id"].str.replace(r"\.\d+$","",regex=True)
    df["expr_rank"] = df["expr_value"].rank(pct=True)
    return df[(df.expr_value>=1.0)&(df.expr_rank>=pct_lo)&(df.expr_rank<=pct_hi)].copy()


# loading mane summary
def load_summary(path):
    m = pd.read_csv(path, sep="\t", dtype=str).drop_duplicates("Ensembl_Gene")
    m["id_base"]      = m["Ensembl_Gene"].str.replace(r"\.\d+$","",regex=True)
    m["chromosome"]   = m["GRCh38_chr"].str.replace(r"^NC_0+(\d+)\.\d+$",r"chr\1",regex=True)
    m["tss_position"] = m.apply(
        lambda r: int(r.chr_start) if r.chr_strand=="+" else int(r.chr_end),
        axis=1)
    return m


# now annotatinig overlaps
def annotate_overlaps(chosen, allgenes):
    all_w = allgenes.assign(
        start=allgenes.tss_position.sub(2000).clip(lower=0),
        end=  allgenes.tss_position.add(2000)
    )[['chromosome','start','end','id_base']]
    all20 = all_w.query("chromosome=='chr20'")
    outs = []
    for _, r in chosen.iterrows():
        mask = (
            (all20.end   >= r.start) &
            (all20.start <= r.end) &
            (all20.id_base != r.id_base)
        )
        hits = all20.loc[mask,'id_base'].tolist()
        outs.append(",".join(hits) if hits else ".")
    return outs


# build a pool for genes - if anything is overlapping, then drop any ±2 kb window that overlaps another
def build_pool(n=60):
    expr   = load_expression(MEDIAN_TPM_PATH)
    summ   = load_summary(SUMMARY_PATH)
    chr20m = summ[summ.chromosome=="chr20"]
    common = set(expr.id_base)&set(chr20m.id_base)
    df     = chr20m[chr20m.id_base.isin(common)].copy()
    df     = (df.merge(expr[['id_base','expr_value','expr_rank']],on='id_base')
                .sort_values('expr_rank').reset_index(drop=True))
    if len(df)<n:
        raise RuntimeError(f"Need ≥{n} genes but got {len(df)}")
    idx  = np.linspace(0,len(df)-1,n,dtype=int)
    return df.loc[idx, ['chromosome','tss_position','id_base','expr_value','expr_rank']], df

# overlap filter
def filter_nonoverlapping(pool, full_df, needed=30):
    pool_win = pool.assign(
        start=pool.tss_position.sub(2000).clip(lower=0),
        end=  pool.tss_position.add(2000)
    )
    pool_win['overlaps'] = annotate_overlaps(pool_win, full_df)
    clean = pool_win[pool_win.overlaps=="."].reset_index(drop=True)
    if len(clean)<needed:
        raise RuntimeError(f"After filtering, only {len(clean)} non‐overlapping; need {needed}")
    return clean.iloc[:needed]


# write bed files
def write_bed(sel, out_bed):
    hdr = "#chromosome\tstart\tend\tid_base\texpr_value\texpr_rank\toverlaps\n"
    with open(out_bed,'w') as fo:
        fo.write(hdr)
        for _,r in sel.iterrows():
            fo.write("\t".join([
                r.chromosome,
                str(r.start),
                str(r.end),
                r.id_base,
                f"{r.expr_value:.3f}",
                f"{r.expr_rank:.3f}",
                r.overlaps
            ])+"\n")
    print(f"Wrote {len(sel)} → {out_bed}")


if __name__=="__main__":
    # 1) building 60‐gene pool
    pool, full_df = build_pool(60)
    # 2) keeping first 30 that are truly isolated
    clean30 = filter_nonoverlapping(pool, full_df, needed=30)
    # 3) splitting half
    half = 15
    set1 = clean30.iloc[:half].assign(
        start=lambda df: df.tss_position.sub(2000).clip(lower=0),
        end  =lambda df: df.tss_position.add(2000)
    )
    set1['overlaps'] = annotate_overlaps(set1, full_df)
    set2 = clean30.iloc[half:].assign(
        start=lambda df: df.tss_position.sub(2000).clip(lower=0),
        end  =lambda df: df.tss_position.add(2000)
    )
    set2['overlaps'] = annotate_overlaps(set2, full_df)
    # 4) write both
    write_bed(set1, OUT1_BED)
    write_bed(set2, OUT2_BED)
    print("Done")