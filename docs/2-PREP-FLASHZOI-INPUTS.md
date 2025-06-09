Now we have two 15‐window BED files (each window = ±2 kb around the TSS, with expression and overlap annotations),
the next step is to turn those into the 524 kb “Flashzoi inputs” and attach expression labels.
So we will:
	1.	“Grow” each ±2 kb TSS window out to a full 524 kb sequence (so the TSS remains centered).
	2.	Pull out the corresponding 524 kb DNA from reference FASTA (e.g. hg38).
	3.	One-hot–encode those 524 kb sequences (A/C/G/T → 4-channel arrays) in the exact format Flashzoi expects.
	4.	Package each 524 kb array along with its measured TPM (or expr_rank) so you can compare model predictions back to real expression values.


Read 2 kb BED (the file you generated in 3-tss-centered-gene-set.py). That BED already has seven columns:
• chromosome (e.g. “chr20”)
• start (2 kb left of TSS)
• end (2 kb right of TSS)
• id_base (the Ensembl ID without “.###”)
• expr_value (the Muscle_Skeletal TPM you filtered on)
• expr_rank (the percentile rank in muscle TPM)
• overlaps (string of any other 2 kb windows it overlapped)
	2.	For each line, we computed the true TSS coordinate (it’s just start + 2000), then “grew” that TSS up/down by 262 144 bp on each side (so total = 524 288 bp).
	3.	We used pyfaidx to fetch exactly that 524 kb segment from hg38.fa file. If the TSS was close to chromosome ends, we silently padded with “N” so we always get exactly 524 kb.
	4.	That 524 kb string is one-hot encoded (A/C/G/T → 4 × 524 288 floats). We saved each gene’s 4×524 288 array as gene.npy (so Flashzoi can just do np.load("ENSG00000XXXXX.npy")).
	5.	We also wrote out a little flashzoi_meta.tsv (one per dataset) that has gene ID, the path to its .npy file, its expr_value, expr_rank, and any 2 kb overlaps. That metadata file is crucial later for “matching up” Flashzoi’s predictions with real TPM labels.