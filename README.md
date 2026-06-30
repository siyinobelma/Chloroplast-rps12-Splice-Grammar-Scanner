🧬 rps12 Fern Trans-splicing Degeneration

This repository contains code and data for the study:

Deep learning reveals repeated degeneration of chloroplast rps12 trans-splicing architecture in ferns

We integrate deep learning, transcriptomics, phylogenetics, and environmental analysis to investigate the evolutionary dynamics of plastid rps12 trans-splicing systems.

📌 Overview

Our study demonstrates that:

rps12 trans-splicing architecture exhibits a bimodal distribution of sequence grammar states
CNN-based models recover biologically meaningful EBS/IBS interaction signatures
Multiple fern lineages independently evolve degenerate trans-splicing states
Low-scoring lineages are associated with climatically marginal environments
Transcriptomic evidence supports functional impairment of splicing in low-scoring copies
🧠 Model Architecture

We train a convolutional neural network (CNN) to classify rps12 trans-splicing competence based on plastid sequence features.

Input: rps12 exon-intron genomic sequences
Output: probability of canonical trans-splicing grammar
Framework: PyTorch
Key feature: automatic learning of EBS/IBS-like interaction patterns
📊 Key Analyses
1. CNN-based sequence grammar inference
Prediction of trans-splicing competence
Cross-taxon generalization
2. Bimodality analysis
Hartigan’s Dip Test confirms non-unimodal distribution
Suggests discrete grammar states
3. Transcriptomic validation
IGV-based RNA-seq evidence in Haplopteris elongata
Reduced exon–exon junction support in low-scoring copies
4. Phylogenetic reconstruction
Independent origins of degenerate states
Convergent evolution across fern lineages
5. Environmental association
PCA of climatic variables
Low-scoring species occupy environmental extremes
6. SHAP interpretation
Identification of key sequence regions driving predictions
📁 Repository Structure
scripts/        # CNN model and analysis pipelines
data/           # processed exon/intron datasets
analysis/       # downstream statistical analyses
models/         # trained CNN model (best_model.pt)
⚙️ Requirements

Install dependencies:

pip install -r requirements.txt

Key packages:

PyTorch
pandas
numpy
scikit-learn
matplotlib
seaborn
biopython
🚀 Usage
Train model
python scripts/train.py
Predict sequences
python scripts/predict.py
Run SHAP analysis
python scripts/shap_analysis.py
📈 Key Results
CNN scores show strong bimodal distribution
Degenerate rps12 architecture evolves repeatedly
Functional impairment confirmed by transcriptomics
Environmental association suggests ecological structuring
📦 Data Availability

Processed datasets used in this study are included in the data/ directory.
Raw sequencing data are available from public repositories cited in the manuscript.

📄 Citation

If you use this code, please cite:


📬 Contact

Corresponding author: [your email]
