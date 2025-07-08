# Prepare Flashzoi Inputs

Here we describe the two distinct workflows for preparing model inputs from gene BED files to one-hot encoded sequences and variant annotations ready for Flashzoi/Borzoi prediction models.

## Overview

Both workflows transform gene TSS coordinates into:
1. **One-hot encoded sequence arrays** (4×524,288 bp) centered on TSS
2. **Metadata files** linking genes to their sequence arrays and expression data  
3. **Variant annotation files** with allele frequencies for model input

The key difference is scope and input source:
- **Single Chromosome**: Test datasets from expression-filtered chr20 genes
- **Genome-wide**: Production datasets from constraint-based gene cohorts

---

## Workflow 1: Single Chromosome Path (`single_chr/`)

**Purpose**: Process dataset1 and dataset2 (15 genes each) from chromosome 20 test sets

### Input Files

```
data/intermediate/dataset1/chr20_test15_tss±2kb.bed
data/intermediate/dataset2/chr20_test15_tss±2kb.bed
```

**BED Format** (8 columns with header):
```
#chromosome  start      end        id_base          expr_value  expr_rank  overlaps  tss_pos
chr20        24947269   24951269   ENSG00000077984  1.056       0.789      .         24949269
```

### Step 1: Create One-Hot Arrays (`12_grow_onehot.py`)

**Algorithm**:
1. **Read BED files** with pandas, skipping header line (`comment="#"`)
2. **Extract TSS position** from `tss_pos` column (fallback: `start + 2000`)
3. **Define 524kb window** around TSS: `[tss - 262144, tss + 262144)`
4. **Fetch sequence** using pyfaidx from reference FASTA
5. **Handle edge cases**: 
   - Clamp to chromosome boundaries
   - Pad with 'N' if sequence < 524kb
6. **One-hot encode**: A/C/G/T → [1,0,0,0]/[0,1,0,0]/[0,0,1,0]/[0,0,0,1]
7. **Save arrays**: `{gene}.npy` as 4×524288 float32 arrays

**Key Functions**:
- `coerce_chrom()`: Handle chr20 ↔ 20 naming differences in FASTA
- `extract_window()`: Fetch and pad sequences to exact 524kb length
- `one_hot_encode()`: Convert DNA string to 4-channel array

**Outputs**:
```
data/intermediate/dataset1/flashzoi_inputs/
├── flashzoi_meta.tsv
├── ENSG00000077984.npy
├── ENSG00000197122.npy
└── ... (13 more .npy files)

data/intermediate/dataset2/flashzoi_inputs/
├── flashzoi_meta.tsv  
├── ENSG00000171984.npy
└── ... (14 more .npy files)
```

**Metadata TSV Format**:
```
gene            onehot_path                                          expr_value  expr_rank  overlaps2kb
ENSG00000077984 ../../data/intermediate/dataset1/flashzoi_inputs/... 1.056       0.789      .
```

### Step 2: Annotate Variants (`13_annotate_variants.py`)

**Algorithm**:
1. **Read metadata** from `flashzoi_meta.tsv` files for both datasets
2. **Load MANE TSS map** for chr20 genes (strips version suffixes)
3. **Define variant windows**: ±262,144 bp around each TSS  
4. **Query gnomAD VCF** using pysam for chr20 variants in each window
5. **Extract variant info**: CHROM, POS, REF, ALT, AF (allele frequency)
6. **Calculate relative positions**: `POS0 = POS - window_start` (0-based offset)
7. **Write per-gene TSV files** with variant annotations

**Key Processing**:
- **Coordinate conversion**: 1-based VCF positions → 0-based window offsets
- **Multi-allelic handling**: Separate row for each ALT allele
- **AF extraction**: Handle both scalar and array AF values from gnomAD INFO
- **Window bounds**: `[tss-262144, tss+262144)` (1-based, inclusive)

**Outputs**:
```
data/intermediate/dataset1/flashzoi_inputs/variants/
├── ENSG00000077984/
│   └── ENSG00000077984_variants.tsv
├── ENSG00000197122/
│   └── ENSG00000197122_variants.tsv
└── ...

data/intermediate/dataset2/flashzoi_inputs/variants/
└── ... (similar structure)
```

**Variant TSV Format**:
```
CHROM  POS       POS0    REF  ALT  AF
chr20  24947300  31      C    T    0.001234
chr20  24947350  81      G    A    0.005678
```

---

## Workflow 2: Genome-wide Path (`genome/`)

**Purpose**: Process dataset3 constraint-based gene cohorts (ClinGen HI, non-essential)

### Input Files

```
data/intermediate/dataset3/ClinGen_gene_curation_list_tss±2kb.bed
data/intermediate/dataset3/nonessential_ensg_tss±2kb.bed
```

**BED Format** (same 8-column structure but no expression filtering)

### Step 1: Create One-Hot Arrays (`12_grow_onehot.py`)

**Identical algorithm** to single_chr but with key differences:

**Scale differences**:
- **Input scope**: All canonical chromosomes (chr1-22, X, Y)
- **Gene count**: Hundreds of genes vs. 15 per dataset
- **No expression filtering**: Uses predefined constraint-based gene lists

**Outputs**:
```
data/intermediate/dataset3/onehots/
├── ClinGen_gene_curation_list/
│   ├── ClinGen_gene_curation_list_flashzoi_meta.tsv
│   ├── ENSG00000000003.npy
│   └── ... (300+ .npy files)
└── nonessential_ensg/
    ├── nonessential_ensg_flashzoi_meta.tsv  
    └── ... (90+ .npy files)
```

### Step 2: Annotate Variants (`13_annotate_variants_with_Hail.py`)

**Technology difference**: Uses **Hail + Spark** instead of pysam for genome-wide scale

**Algorithm**:
1. **Initialize Hail** with Spark configuration for big data processing
2. **Load gnomAD Hail Table** from Google Cloud (genome-wide coverage)
3. **Build TSS dictionary** from MANE summary (all chromosomes)
4. **Process each gene cohort**:
   - Filter Hail table by chromosome and position range
   - Select relevant fields: CHROM, POS, REF, ALT, AF
   - Export to per-gene TSV files
5. **Parallel processing** via Spark for efficiency

**Key differences from single_chr**:
- **Data source**: Hail Table vs. local VCF file
- **Scale**: Genome-wide vs. chr20-only  
- **Technology**: Distributed processing vs. single-machine
- **Output location**: `data/intermediate/dataset3/variants/`

**Outputs**:
```
data/intermediate/dataset3/variants/
├── ClinGen_gene_curation_list/
│   ├── ENSG00000000003/
│   │   └── ENSG00000000003_variants.tsv
│   └── ...
└── nonessential_ensg/
    └── ... (similar structure)
```

---

## Common Technical Specifications

### Sequence Processing
- **Window size**: 524,288 bp (2^19, exactly what Borzoi expects)
- **TSS centering**: ±262,144 bp around canonical TSS from MANE v1.4
- **Coordinate system**: 0-based half-open intervals for sequence extraction
- **Edge handling**: Automatic padding with 'N' for chromosome boundaries
- **Reference genome**: GRCh38 primary assembly

### One-Hot Encoding
- **Array shape**: (4, 524288) as float32
- **Channel mapping**: A=0, C=1, G=2, T=3  
- **Unknown bases**: N and other non-ACGT → [0,0,0,0]
- **Case handling**: All sequences converted to uppercase
- **Validation**: Each position sums to ≤1.0 (exactly 1.0 for ACGT)

### Variant Annotation
- **Window definition**: Same ±262,144 bp as sequence windows
- **Position encoding**: Both absolute (POS) and relative (POS0) coordinates
- **Allele frequency**: Extracted from gnomAD INFO.AF field
- **Multi-allelic sites**: Separate row per alternate allele
- **Missing data**: AF="NA" when not available

### File Organization
```
data/intermediate/{dataset}/
├── {gene_set}_tss±2kb.bed              # Input BED files
├── flashzoi_inputs/ or onehots/{set}/  # One-hot arrays + metadata  
└── variants/{gene_set}/                # Variant annotations
```

### Quality Control
- **Duplicate detection**: MD5 hashing of .npy files
- **Shape validation**: All arrays must be (4, 524288)
- **One-hot validation**: Column sums ≤ 1.0
- **TSS overlap detection**: Reports genes with shared TSS positions
- **Error handling**: Graceful skipping of problematic genes with logging

---

## Usage Examples

### Single Chromosome Workflow
```bash
cd 1_prep_model_inputs/single_chr

# Create one-hot arrays for dataset1 and dataset2
python 12_grow_onehot.py

# Annotate variants for both datasets  
python 13_annotate_variants.py
```

### Genome-wide Workflow  
```bash
cd 1_prep_model_inputs/genome

# Create one-hot arrays for constraint-based gene sets
python 12_grow_onehot.py

# Annotate variants using Hail (requires Spark setup)
python 13_annotate_variants_with_Hail.py  
```

Both workflows produce identical output formats for downstream model prediction, differing only in scale and input gene selection methodology.