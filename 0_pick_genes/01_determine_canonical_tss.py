import pandas as pd

# read the stuff
m = pd.read_csv('../data/initial/MANE.GRCh38.v1.4.summary.txt', sep='\t', dtype=str)

# remove dups, first one only
seen = set()
rows = []
for i, row in m.iterrows():
    gene = row['Ensembl_Gene']
    if gene not in seen:
        seen.add(gene)
        rows.append(row)
# make a new df
m2 = pd.DataFrame(rows)

# the original is like NC_000020.11
chr_list = []
import re
for x in m2['GRCh38_chr']:
    # here we try to match the numbers
    mobj = re.match(r"^NC_0+(\d+)\.\d+$", x)
    if mobj:
        c = 'chr' + mobj.group(1)
    else:
        c = x
    chr_list.append(c)
m2['chrom'] = chr_list

# compute transcriptome start sites, if + use start else end
tss_list = []
for idx, row in m2.iterrows():
    if row['chr_strand'] == '+':
        tss_list.append(int(row['chr_start']))
    else:
        tss_list.append(int(row['chr_end']))
m2['TSS'] = tss_list

# only chr20
m20 = m2[m2['chrom'] == 'chr20']

# make bed entries
bed_rows = []
for idx, row in m20.iterrows():
    name = re.sub(r"\.\d+$", "", row['Ensembl_Gene'])
    start = row['TSS'] - 2000
    end = row['TSS'] + 2000
    bed_rows.append([row['chrom'], start, end, name])

# write it out
out = pd.DataFrame(bed_rows)
out.to_csv('../data/intermediate/chr20_mane_promoters.bed', sep='\t', index=False, header=False)

print('Done writing BED for chr20')
