# Gene Selection Workflows

This directory contains two distinct workflows for selecting genes, depending on the dataset and analysis goals:

1. **Single Chromosome Path** (`single_chr/`) - For creating test datasets with expression-filtered genes
2. **Genome-wide Path** (`genome/`) - For processing predefined constraint-based gene cohorts

## Data Files & Dependencies

All workflows require these input files:

1. **gencode.v47.basic.annotation.gtf** - GENCODE v47 basic gene models (GRCh38) for transcript coordinates
2. **gnomad.genomes.v4.1.sites.chr20.vcf.bgz + .tbi** - Common variants for ±262 kb windows
3. **GTEx_Analysis_2022-06-06_v10_RNASeQCv2.4.2_gene_median_tpm.gct** - Median TPM by gene × tissue (GTEx v10)
4. **MANE.GRCh38.v1.4.summary.txt** - MANE Select canonical transcripts for TSS positions
5. **GRCh38.primary_assembly.genome.fa** - Reference FASTA for sequence extraction
6. **ENCFF150TGS.bed** - Open-chromatin peaks (skeletal-muscle ATAC-seq) for filtering promoter windows (for optional annotation)


## Workflow 1: Single Chromosome Path (Dataset 1 and Dataset 2)

**Purpose**: Create test datasets with muscle-expressed genes on chromosome 20

**Use case**: When you need smaller, curated datasets for model testing and validation

### Steps:

1. **`01_determine_canonical_tss.py`**
   - Extracts canonical TSS positions from MANE v1.4
   - Filters to chromosome 20 only
   - Creates ±2kb windows around each TSS
   - **Output**: `chr20_mane_promoters.bed`

2. **`02_filter_muscle_expression.py`**
   - Filters genes by GTEx skeletal muscle expression
   - Retains genes in 70-95th percentile of expression
   - Intersects with chr20 MANE genes
   - **Output**: `GTEx_chr20_mane_TPM.tsv`

3. **`03_tss_centered_gene_set.py`**
   - Selects 30 non-overlapping genes from expression-filtered set
   - Ensures no overlap between ±2kb windows
   - Splits into two groups of 15 genes each
   - **Outputs**: 
     - `dataset1/chr20_test15_tss±2kb.bed` (first 15 genes)
     - `dataset2/chr20_test15_tss±2kb.bed` (second 15 genes)

### Selection Criteria:
- 15 genes on chromosome 20
- 70-95th percentile for skeletal-muscle expression in GTEx
- Non-overlapping ±2kb windows
- Final output: 30 promoters split into two groups of 15 (dataset 1 and dataset 2)

## Workflow 2: Genome-wide Path (Dataset 3)

**Purpose**: Process constraint-based gene cohorts across the entire genome

**Use case**: When analyzing predefined gene lists (e.g., ClinGen HI, non-essential genes) without expression filtering

### Steps:

1. **`prep_bed_files_simple.py`**
   - Takes a predefined gene list as input (e.g., ClinGen HI genes, non-essential genes)
   - Maps each gene ID to its MANE TSS position
   - Creates ±2kb windows around TSS
   - Works genome-wide (all canonical chromosomes)
   - **Output**: `{gene_set_name}_tss±2kb.bed`

### Key Differences from Single Chromosome Path:
- **No expression filtering** - uses predefined gene lists
- **Genome-wide scope** - not limited to chromosome 20
- **External gene lists** - requires curated input gene sets
- **Larger scale** - typically processes hundreds of genes

### Selection Criteria:
- Genes from predefined constraint-based cohorts
- Present in MANE v1.4 canonical transcript set
- Located on canonical chromosomes (chr1-22, X, Y)
- No expression filtering applied

## Output File Formats

### BED Files
All workflows produce BED files with TSS-centered windows:
- **Columns**: chromosome, start, end, gene_id
- **Window size**: ±2kb around canonical TSS
- **Coordinate system**: 0-based, half-open intervals

### Downstream Usage
These BED files are used in subsequent steps to:
1. Extract ±262kb sequence windows for model input
2. Annotate variants within gene regions
3. Generate one-hot encoded sequence arrays
4. Create variant lists for expression variance prediction

## When to Use Each Workflow

**Choose Single Chromosome Path when**:
- Creating test/validation datasets
- Need expression-filtered genes
- Working with smaller gene sets
- Developing or debugging pipelines

**Choose Genome-wide Path when**:
- Analyzing constraint-based gene cohorts
- Working with predefined gene lists
- Need genome-wide coverage
- Expression filtering is not desired
