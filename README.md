# Multimodal Age-Related Macular Degeneration Classification using OCT imaging and Demographic Tabular Data

# Multimodal Macular Degeneration Classification

## Overview

This term project for the **Projects in Biomedical AI (42-687)** course implements a multimodal deep learning framework that combines optical coherence tomography (OCT) images with structured clinical data to classify age-related macular degeneration (AMD) into six stages: Not AMD, Early AMD, Intermediate AMD, Geographic Atrophy (GA), Scarring, and Wet AMD. The image modality uses a pre-trained RETFound ViT-large encoder, the clinical (tabular) modality uses a gated TabTransformer, and both modalities are fused via a cross-attention module followed by a BERT-style decoder for final classification citeturn0file0.

## Table of Contents

* [Dataset](#dataset)
* [Method](#method)

  * [Image Encoder](#image-encoder)
  * [Tabular Encoder](#tabular-encoder)
  * [Multimodal Architecture](#multimodal-architecture)
* [Results](#results)
* [Installation](#installation)
* [Usage](#usage)
* [Project Structure](#project-structure)
* [Contributing](#contributing)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Authors](#authors)

---

## Dataset

* **Source**: University of Pittsburgh Medical Center (UPMC) AMD-OCT dataset, provided by the Choroidal Analysis Research Laboratory at UPMC.
* **Size**: 4,028 OCT volumes (≈637,408 B-scan slices) from 91 patients (2007–2023) citeturn0file0
* **Labels**: Not AMD, Early AMD, Intermediate AMD, GA, Scarring, Wet AMD

## Method

### Image Encoder

* **Backbone**: RETFound ViT-large (24-layer Transformer, 1024-dim embeddings), self-supervised pre-trained on 736,442 OCT B-scans.
* **Classification Head**: Frozen backbone + 128-dim projection + ReLU + Dropout → 6-way output.
* **Training**: 10 epochs, batch size 128, Adam lr=1e-4, ReduceLROnPlateau scheduler citeturn0file0

### Tabular Encoder

#### Preprocessing

* **Data Cleaning**: Removed records with >30% missing fields; imputed remaining missing values using median for continuous (age, visual acuity, ICD frequency) and mode for categorical features.
* **Normalization**: Continuous features (age, VA, ICD freq) standardized (zero mean, unit variance) based on training set statistics.
* **Encoding**: Categorical features (gender, smoking, drinking, drug abuse, ICD-10 code groups) label-encoded, then embedded via TabTransformer’s embedding layers.

#### Model & Training

* **Architecture**: Gated TabTransformer with 4 transformer blocks, each containing 4 attention heads and a gating mechanism to control information flow.
* **Hyperparameters**: 200 epochs, batch size 64, Adam optimizer with initial lr=1e-3, StepLR scheduler decaying lr by 0.5 every 40 epochs.
* **Regularization**: Dropout (0.2) applied after each feed-forward layer; early stopping on validation loss with patience of 15 epochs.
* **Implementation**: PyTorch, with training script available in `src/train_tabular.py` (also mirrored in my fork: `https://github.com/<your-github-username>/md-amp-classification`) citeturn0file0

### Multimodal Architecture

Below is the high-level architecture diagram illustrating feature extraction, cross-modal fusion, and classification stages (see `docs/architecture_diagram.png`).

![Multimodal Architecture](docs/architecture_diagram.png)

1. **Feature Extraction**

   * OCT features: RETFound ViT-large → 1024-dim vector
   * Clinical features: TabTransformer → 64-dim vector
2. **Cross-Attention Fusion**

   * Image features (Key/Value, projected to 768-dim)
   * Tabular features (Query, projected to 768-dim)
   * One multi-head cross-attention layer → fused 768-dim representation
3. **BERT Decoder & Classification**

   * Sequence \[Fused, Tabular] → BERT-Base (12 layers, hidden size 768)
   * \[CLS] token → 128-dim MLP → 6-way softmax

Training: Fusion block, BERT decoder, and MLP trained for 10 epochs (batch size 16) while keeping encoders frozen citeturn0file0

## Results

### Image-Only

* **B-scan Accuracy**: Train 84.59%, Val 80.05%
* **Volume-Level**: 92.45% (3724/4028) citeturn0file0

### Tabular-Only

* **Test Accuracy**: 46.3% (imbalance impacted recall on Not AMD) citeturn0file0

### Multimodal

* **B-scan Validation Accuracy**: 79.7% (peak at epoch 10) citeturn0file0

## Installation

```bash
 git clone https://github.com/<your-github-username>/md-amp-classification.git
 cd md-amp-classification
 conda create -n amd-classify python=3.10
 conda activate amd-classify
 pip install -r requirements.txt
```

## Usage

```bash
 python train_tabular.py --config configs/tabular.yaml  # Tabular only
 python train_image.py --config configs/image.yaml      # Image only
 python train_fusion.py --config configs/fusion.yaml    # Multimodal
 python predict.py --image path/to/image --clinical path/to/record
```

## Project Structure

```
├── data/
│   ├── images/
│   └── clinical.csv
├── docs/
│   └── architecture_diagram.png  # Multimodal architecture
├── src/
│   ├── train_tabular.py
│   ├── train_image.py
│   ├── train_fusion.py
│   └── predict.py
├── models/
├── configs/
├── requirements.txt
└── README.md
```

## Contributing

Contributions are welcome via issues or pull requests. Please check existing issues before opening a new one.

## License

MIT License. See [LICENSE](LICENSE).

## Acknowledgments

Thanks to Dr. Jay Chhablani, Sandeep Chandra Bollepalli, Sharat Chandra, the CAR Lab team, Dr. P. Sang Chalacheva, and TA Nishanth Arun.
Collaboration with [Choroidal Analysis Research Laboratory](https://ophthalmology.pitt.edu/research/basic-science-research/laboratories/choroidal-analysis-research-laboratory) citeturn0file0

## Authors

* Martin Goessweiner
* Jiayi Liu
* Jainam Modh
* Jonathan Tang



