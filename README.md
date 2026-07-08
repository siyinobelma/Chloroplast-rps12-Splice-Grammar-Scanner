# rps12_fern: Deep learning reveals repeated degeneration of chloroplast trans-splicing grammar in ferns

[![DOI](https://zenodo.org/badge/xxxxx.svg)](https://doi.org/xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository contains code and data for the manuscript:

**"Relaxation of conserved rps12 trans-splicing syntax is associated with climatic niche divergence in ferns"**

We combine deep learning, chloroplast transcriptomics, phylogenetics, and macroecological analyses to investigate the evolutionary dynamics of rps12 trans-splicing across ferns. A three-channel convolutional neural network (CNN) trained on angiosperm plastomes reveals repeated lineage-specific degeneration of trans-splicing grammar in ferns, and transcriptomic data confirm functional impairment in low-scoring copies.

## Repository Structure
```bash
rps12_fern/
├── config.py # Global configuration
├── best_model.pt # Trained CNN model weights
├── environment.yml # Dependencies
├── scripts/
│ ├── model.py # CNN architecture definition
│ ├── train.py # Model training
│ ├── evaluate.py # Model evaluation
│ ├── predict_external.py # Cross-species prediction
│ ├── shap_analysis.py # Consensus SHAP attribution (seed plants)
│ ├── utils.py # Utility functions
│ ├── generate_negatives.py # Negative sample construction
│ └── preprocess.py # Data preprocessing
├── data/
│ ├── processed/ # Preprocessed training/val/test sets
├── analysis_results/ # Output figures and tables
├── shap_results/ # SHAP attribution results
└── README.md
```
## Dependencies

- Python ≥ 3.8
- PyTorch ≥ 1.10
- Biopython ≥ 1.79
- pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, captum
- ViennaRNA (for RNA secondary structure prediction)
- R (for ecological niche analysis)

### Install Python dependencies:
pip install -r requirements.txt
1. Data preparation
GenBank files for angiosperm, gymnosperm, and fern plastomes are downloaded from NCBI GenBank. Extract rps12 splice pairs using the extraction script (available upon request) to generate the input CSV files placed in data/.

2. Train the CNN model
```bash
python scripts/generate_negatives.py
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
```
## Quick Start: Predicting rps12 Trans-Splicing Scores for Your Own Species
If you only need to apply the pre-trained model to your own species, follow these steps.

### Step 1: Extract rps12 splice pairs from GenBank files
```bash
# Extract rps12 pairs (exon1-intron-exon2) from GB files
python extract_rps12_pairs.py -i /path/to/your/gb_files/ \
    --pairs your_species_pairs.csv \
    --exons your_species_exons.csv
-i: Directory containing GenBank (.gb or .gbk) files

--pairs: Output CSV file for extracted splice pairs

--exons: Output CSV file for individual exons and flanking introns
```
The output your_species_pairs.csv contains the following key columns:

Column	Description
species	Species name
accession	NCBI accession number
exon1_id	Identity of exon 1 (e.g., exon_1, exon_I)
exon1_seq	DNA sequence of exon 1
intron_seq	DNA sequence of the intron
exon2_id	Identity of exon 2 (e.g., exon_2, core)
exon2_seq	DNA sequence of exon 2
type	Splicing type (see below)
### Step 2: Predict trans-splicing competence
```bash
python predict_external.py \
    --pairs your_species_pairs.csv \
    --output your_species_predictions.csv \
    --model best_model.pt
```
### Step 3: Interpret the results
The output your_species_predictions.csv contains:

Column	Description
species	Species name
accession	NCBI accession number
exon1_id	Identity of exon 1
exon2_id	Identity of exon 2
type	Splicing type
positive_prob	CNN prediction score (0–1)
predicted_label	Binary label (1 if score ≥ 0.5, else 0)
How to interpret the prediction score (positive_prob):

> 0.95: Canonical trans-splicing grammar strongly conserved (typical of seed plants and high-scoring fern copies)

0.80–0.95: Minor deviations from canonical grammar, likely still functional

0.50–0.80: Intermediate state; partial degeneration of trans-splicing grammar (e.g., Vittaria graminifolia, score ≈ 0.66, retains ~90% splicing efficiency)

0.10–0.50: Substantial grammar degeneration; trans-splicing likely impaired

< 0.10: Near-complete loss of canonical trans-splicing grammar; experimentally validated as splicing failure in Haplopteris elongata (score = 0.033)

Understanding the type column:

Type	Splicing Mode	Description
cis	Cis-splicing	Exons are spliced in linear genomic order (exon_1–intron–exon_2, exon_2–intron–exon_3)
trans_exonI_core	Trans-splicing	Exon I (5′-rps12) pairs with the core fragment (exon 2 + intron + exon 3) from a separate transcript
trans_core_exonIII	Trans-splicing	Within the core fragment, exon 2 pairs with exon 3
In species with trans-splicing, the model evaluates each pairing independently. A single species may have multiple rows corresponding to different exon–intron–exon combinations (e.g., both trans_exonI_core and trans_core_exonIII). If a species has multiple copies (e.g., from the inverted repeat region), each copy receives its own score.

4. Run downstream analyses
```bash
# Cross-taxon comparison
python cross_taxon_analysis.py

# SHAP attribution
python shap_expansion_internal.py

# Intron length control
python intron_length_vs_score.py

# RNA secondary structure
python rna_structure_analysis_v2.py
```
Data Availability
Chloroplast genome sequences are publicly available from NCBI GenBank (accession numbers provided in Supplementary Table S2)

Transcriptomic data are available from NCBI SRA (SRR32213825)

GBIF occurrence data used for ecological niche analysis are available at https://doi.org/xxxxx

Citation
If you use this code or data, please cite:
