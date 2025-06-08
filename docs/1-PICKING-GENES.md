Data files & purpose
1. ENCFF150TGS.bed
- Open-chromatin peaks (skeletal-muscle ATAC-seq) to filter promoter windows to “open” vs “closed” in muscle.
2. gencode.v47.basic.annotation.gtf
- GENCODE v47 basic gene models (GRCh38), to list all gene TSSs; extract transcript exon/intron coordinates.
3. gnomad.genomes.v4.1.sites.chr20.vcf.bgz + .tbi to fetch common variants in each gene’s ±262 kb window.
4. GTEx_…gene_median_tpm.gct  - Median TPM by gene × tissue (GTEx v10).
• Use: select genes with moderate muscle‐skeletal expression (70–95 %).
   5.	MANE.GRCh38.v1.4.summary.txt
• MANE Select canonical transcripts.
• Use: pick a single, high‐confidence TSS per gene.
   6.	GRCh38.primary_assembly.genome.fa
• GRCh38 reference FASTA.
• Use: extract ±262 kb sequence windows for Flashzoi / Enformer input.
0.bed
