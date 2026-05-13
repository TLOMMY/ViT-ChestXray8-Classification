# ViT-ChestXray8-Classification
A repository for Novo Nordisk PBL course in BlendED AI+X program. We're team in track 1 from the session 2026.3.9-2026.5.14

Track 1 is Vision Transformers (ViT) for Image Classification

A research-oriented medical imaging project exploring the application of Vision Transformers (ViTs) for automated chest X-ray classification using the NIH ChestX-ray8 dataset.


## Project Description

This project explores the application of Vision Transformers (ViTs) for automated chest X-ray classification using the NIH ChestX-ray8 dataset.

The project investigates how transformer-based deep learning architectures can be adapted for medical imaging tasks, with a particular focus on binary classification between normal and abnormal chest X-rays.

The project covers:

- medical image preprocessing
- patient-level dataset stratification
- Vision Transformer implementation
- patch embedding and self-attention mechanisms
- binary and multi-label classification
- overfitting and class imbalance analysis
- training efficiency on limited GPU environments such as Google Colab

The Vision Transformer divides chest X-ray images into image patches, embeds them into lower-dimensional representations, and applies transformer encoder blocks with self-attention to learn spatial relationships between image regions.

This repository documents the full workflow, preprocessing pipeline, model architecture, experimental setup, training observations, and future improvements for transformer-based medical image classification.

## Pipeline Overview

The complete workflow of the project is shown below:

Chest X-ray Image
↓
Data Preprocessing
↓
Patient Stratification
↓
Patch Extraction
↓
Patch Embedding
↓
Vision Transformer Encoder
↓
Classification Head
↓
Evaluation Metrics

## Dataset

This project uses the NIH ChestX-ray8 dataset.

The dataset contains:

- 112,120 frontal chest X-ray images
- 30,805 unique patients
- 14 thoracic disease labels
- "No Finding" normal category

The dataset is widely used for medical image classification research and weakly-supervised thoracic disease detection.

Main labels used in experiments:

- No Finding
- Infiltration

##Detailed Colab Workflow

### Preprocessing Examples

Original and preprocessed chest X-ray images:

### Pixel Value Distribution

Pixel histograms were analyzed to verify normalization quality and grayscale intensity distribution.

## Vision Transformer Architecture

The implemented Vision Transformer architecture contains:

- Patch Embedding
- Positional Encoding
- CLS Token
- Transformer Encoder Block
- Multi-Head Self-Attention
- Feed Forward Network (MLP)
- Classification Head

### Final Experimental Configuration

| Parameter | Value |
|---|---|
| Image Size | 64 × 64 |
| Patch Size | 16 × 16 |
| Number of Heads | 4 |
| Embedding Dimension | 32 |
| MLP Dimension | 64 |
| Batch Size | 64 |
| Learning Rate | 0.01 |
| Epochs | 150 |

## Results

Experimental results and evaluation metrics are currently being finalized.

The final repository update will include:

- Training and validation accuracy curves
- Loss curves
- Confusion matrix
- Recall and F1-score analysis
- Binary vs multi-label comparison
- Overfitting analysis

## Experimental Analysis

Several important observations were identified during experimentation:

- Validation accuracy alone can be misleading under class imbalance.
- Binary classification achieved significantly better performance than full multi-label classification.
- Overfitting occurred rapidly on small subsets.
- Training speed was heavily constrained by Google Colab GPU limitations.
- Smaller image sizes improved speed but reduced medical detail retention.

## Future Work

Potential future improvements include:

- weighted loss functions
- focal loss
- pretrained Vision Transformers
- improved class balancing
- higher-resolution image training
- larger GPU environments
- advanced data augmentation
- multi-label optimization
- Grad-CAM visualization
