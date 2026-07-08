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
│ └──  preprocess.py # Data preprocessing
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

Install Python dependencies:
pip install -r requirements.txt
1. Data preparation
GenBank files for angiosperm, gymnosperm, and fern plastomes are downloaded from NCBI GenBank. Extract rps12 splice pairs using the extraction script (available upon request) to generate the input CSV files placed in data/.

2. Train the CNN model
bash
python scripts/generate_negatives.py
python scripts/preprocess.py
python scripts/train.py
python scripts/evaluate.py
3. Predict on external species
bash
python predict_external.py --pairs data/juelei_pairs.csv --output predict_result/juelei_predictions.csv
4. Run downstream analyses
bash
# Cross-taxon comparison
python cross_taxon_analysis.py

# SHAP attribution
python shap_expansion_internal.py

# Intron length control
python intron_length_vs_score.py

# RNA secondary structure
python rna_structure_analysis_v2.py
Data Availability
Chloroplast genome sequences are publicly available from NCBI GenBank (accession numbers provided in Supplementary Table S2)

Transcriptomic data are available from NCBI SRA (SRR32213825)

GBIF occurrence data used for ecological niche analysis are available at https://doi.org/xxxxx

Citation
If you use this code or data, please cite:
