# ViT Chest X-ray Classification

A repository for Novo Nordisk PBL course in BlendED AI+X program. We're team in track 1 from the session 2026.3.9-2026.5.14

Track 1 is Vision Transformers (ViT) for Image Classification

A research-oriented medical imaging project exploring the application of Vision Transformers (ViTs) for automated chest X-ray classification using the NIH ChestX-ray8 dataset.

<p align="center">
  <img src="figures/Cover.jpg" width="100%" alt="ViT Chest X-ray Binary Classification">
</p>

<h1 align="center">Vision Transformer for Chest X-ray Classification</h1>

<p align="center">
A PyTorch-based Vision Transformer (ViT) project for binary chest X-ray classification using the NIH ChestX-ray8 dataset.
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue.svg">
  <img src="https://img.shields.io/badge/PyTorch-2.0-red.svg">
  <img src="https://img.shields.io/badge/Model-VisionTransformer-green.svg">
  <img src="https://img.shields.io/badge/Dataset-NIHChestXray8-orange.svg">
  <img src="https://img.shields.io/badge/Platform-GoogleColab-yellow.svg">
</p>

---

# Project Overview

This project explores the application of Vision Transformers (ViTs) for automated chest X-ray classification using the NIH ChestXray8 dataset. The project focuses on understanding how transformer-based deep learning can be adapted for medical imaging tasks, particularly binary classification of normal versus abnormal chest X-rays.

The NIH ChestXray8 dataset contains over 112,000 chest X-ray images collected from more than 30,000 patients and includes labels for 14 thoracic diseases alongside a "No Finding" category. Due to the large volume of medical imaging data and the complexity of radiological interpretation, there is increasing interest in AI-assisted diagnostic systems that can support clinicians by improving speed, consistency, and scalability in medical image analysis.

Our project investigates:

* Patient-level dataset splitting to avoid data leakage
* Image preprocessing for medical imaging workflows
* Vision Transformer implementation from scratch in PyTorch
* Binary classification experiments on chest X-ray images
* Training behavior under limited computational resources
* Overfitting and class imbalance challenges in medical AI
* Evaluation using Accuracy, Recall, F1-score, and Confusion Matrix

This repository is designed as a reproducible educational research project for understanding Vision Transformers in healthcare imaging.

---

# Key Features

* Vision Transformer implementation from scratch
* Chest X-ray classification using NIH ChestXray8
* Patient-level stratified train/validation split
* Configurable binary classification pipeline
* PyTorch-based training workflow
* Google Colab compatible implementation
* Evaluation with Recall and F1-score
* Checkpoint saving and recovery system
* Visualization of training curves and metrics
* Experimental analysis of overfitting and class imbalance

---

# Quick Start

## 1. Clone Repository

```bash
git clone https://github.com/TLOMMY/ViT-ChestXray8-Classification.git
cd ViT-ChestXray8-Classification
```

---

## 2. Install Requirements

```bash
pip install -r requirements.txt
```

---

## 3. Download NIH ChestXray8 Dataset

This project uses the NIH ChestXray8 dataset from Kaggle.

```python
import kagglehub

path = kagglehub.dataset_download("nih-chest-xrays/data")
print(path)
```

Dataset Link:

[https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

## 4. Run Training

```bash
python scripts/train.py
```

---

## 5. Run Evaluation

```bash
python scripts/evaluate.py
```

---

# Project Structure

```text
ViT-ChestXray8-Classification/
# Project Structure
├── configs/             
│   └── config.py        # Main configuration file (paths, hyperparameters, training settings)
│
├── docs/                
│   ├── ViT Overview.pdf # ViT theoretical framework overview
│   └── setup.md         # Environment setup and running instructions
│
├── figures/             # Visualizations and experiment result plots
│   ├── Loss and Accuracy Curves on the Subdataset.png
│   ├── Metadata Example of Training and Validation Subsets.png
│   ├── Preprocessed Training Data Samples.png
│   ├── Training Loss and Metrics on the Full Dataset.png
│   ├── detailed engineering pipeline.png
│   ├── label_distribution.png
│   ├── label_distribution_data.png
│   ├── overall pipeline.png
│   ├── pixel_histogram.png
│   └── vit architecture.png
│
├── outputs/             
│   ├── confusion_matrix(Samp).png  
│   └── training_history(Samp).png          
│
├── scripts/             # Executable scripts
│   ├── train.py         # Training script
│   └── evaluate.py      # Evaluation script
│
├── src/                 # Core source code
│   ├── __init__.py      # Package initialization
│   ├── dataset.py       # Dataset loading and preprocessing
│   ├── evaluate.py      # Evaluation logic
│   ├── model.py         # ViT model definition
│   ├── train.py         # Training loop implementation
│   └── utils.py         # Utility functions (plotting, config parsing, etc.)
│
├── LICENSE              
├── README.md            
├── ViT_binary.ipynb     # Main Jupyter notebook
└── requirements.txt     # Python dependencies
```

---

# Pipeline Overview

## Overall Workflow

<p align="center">
  <img src="figures/overall pipeline.png" width="100%" alt="Pipeline Overview">
</p>

The project pipeline consists of:

1. Dataset Download and Loading
2. Patient-level Stratified Splitting
3. Image Preprocessing
4. Label Encoding
5. Vision Transformer Training
6. Validation and Evaluation
7. Metric Analysis and Visualization

---

## Detailed Engineering Pipeline

<p align="center">
  <img src="figures/detailed engineering pipeline.png" width="100%" alt="Pipeline Detials">
</p>

### Data Processing Steps

* Download NIH ChestXray8 dataset
* Group images by Patient ID
* Perform patient-level stratified splitting
* Prevent train/validation data leakage
* Filter target disease labels
* Convert labels into numerical format

### Preprocessing Operations

* Convert images to grayscale
* Resize images from 1024×1024 to 64×64
* Normalize pixel values to [0,1]
* Convert images to PyTorch tensors
* Load images using DataLoader

### ViT Components

* Patch Embedding
* CLS Token
* Positional Encoding
* Multi-Head Self Attention
* Transformer Encoder Blocks
* MLP Classification Head

### Training Features

* Adam Optimizer
* CrossEntropyLoss
* Validation Monitoring
* Checkpoint Saving
* Accuracy / Recall / F1-score Tracking
* Confusion Matrix Evaluation

---

## Vision Transformer Pipeline

<p align="center">
  <img src="figures/vit architecture.png" width="90%" alt="ViT Pipeline">
</p>

---

## Final Experimental Configuration

| Parameter           | Value            |
| ------------------- | ---------------- |
| Image Size          | 64 × 64          |
| Patch Size          | 16 × 16          |
| Batch Size          | 64               |
| Num Heads           | 4                |
| Embedding Dimension | 32               |
| MLP Dimension       | 64               |
| Transformer Layers  | 1                |
| Optimizer           | Adam             |
| Learning Rate       | 0.01             |
| Loss Function       | CrossEntropyLoss |
| Epochs              | 150              |
| Framework           | PyTorch          |
| Platform            | Google Colab     |

---

# Dataset

## NIH ChestXray8 Dataset

This project uses the NIH ChestXray8 dataset, a large-scale chest X-ray dataset released by the National Institutes of Health (NIH).

Dataset Statistics:

| Item           | Value                |
| -------------- | -------------------- |
| Total Images   | 112,120              |
| Patients       | 30,000+              |
| Disease Labels | 14                   |
| Imaging Type   | Frontal Chest X-rays |
| Dataset Source | NIH Clinical Center  |

The dataset includes thoracic disease labels such as:

* Atelectasis
* Cardiomegaly
* Effusion
* Infiltration
* Mass
* Nodule
* Pneumonia
* Pneumothorax
* No Finding

---

## Dataset Reference

Wang, X. et al.

"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases"

[https://arxiv.org/abs/1705.02315](https://arxiv.org/abs/1705.02315)

Kaggle Dataset:

[https://www.kaggle.com/datasets/nih-chest-xrays/data](https://www.kaggle.com/datasets/nih-chest-xrays/data)

---

# Data Preprocessing

## Patient-Level Stratified Split

To avoid data leakage, images were split at the patient level instead of image level.

This ensures:

* The same patient does not appear in both training and validation sets
* Better generalization evaluation
* More realistic medical AI experiments

---

## Label Encoding

Binary classification experiments were conducted using:

* No Finding
* Infiltration

Labels were converted into numerical format:

```python
{"No Finding": 0, "Infiltration": 1}
```

---

## Multi-hot Encoding

The repository also supports multi-label encoding for future multi-disease classification experiments.

---

## Preprocessing Visualization

### Sample Preprocessed Images

<p align="center">
  <img src="figures/Preprocessed Training Data Samples.png" width="90%">
</p>

---

### Label Distribution

<p align="center">
  <img src="figures/label_distribution.png" width="90%">
</p>

<p align="center">
  <img src="figures/label_distribution_data.png" width="40%">
</p>

---

### Pixel Histogram Analysis

<p align="center">
  <img src="figures/pixel_histogram.png" width="90%">
</p>

---

### Stratified Validation Sampling

<p align="center">
  <img src="figures/Metadata Example of Training and Validation Subsets.png" width="90%">
</p>

---

# Experimental Setup

## Small Dataset Experiment

Initial experiments were conducted using:

| Dataset Split  | Images |
| -------------- | ------ |
| Training Set   | 1,000  |
| Validation Set | 200    |

The purpose of this stage was:

* Faster iteration cycles
* Architecture debugging
* Hyperparameter testing
* Training behavior analysis

---

## Full Dataset Experiment

Larger experiments were later conducted using:

| Dataset Split  | Images |
| -------------- | ------ |
| Training Set   | 10,000 |
| Validation Set | 2,000  |

The full dataset experiments focused on:

* Reducing overfitting
* Improving generalization
* Evaluating Recall and F1-score
* Measuring scalability limitations in Google Colab

---

## Hyperparameter Experiments

Several configurations were tested:

| Num Heads | Embed Dim | MLP Dim | Result              |
| --------- | --------- | ------- | ------------------- |
| 1         | 16        | 16      | Poor convergence    |
| 4         | 32        | 64      | Best performance    |
| 8         | 32        | 64      | Slower and unstable |

Observations:

* 4 attention heads performed better than 8 heads
* Smaller images improved training speed
* High learning rate caused instability
* Validation accuracy plateaued due to overfitting and imbalance

---

# Results

## Training Curves

<p align="center">
  <img src="docs/results/training_curves.png" width="100%">
</p>

---

## Validation Metrics

<p align="center">
  <img src="docs/results/metrics_curve.png" width="100%">
</p>

---

## Final Metrics

| Metric              | Result |
| ------------------- | ------ |
| Training Accuracy   | TBD    |
| Validation Accuracy | TBD    |
| Recall              | TBD    |
| F1-score            | TBD    |

---

## Confusion Matrix

<p align="center">
  <img src="docs/results/confusion_matrix.png" width="60%">
</p>

---

# Discussion & Limitations

## Key Findings

* Vision Transformers can be adapted for medical imaging tasks
* Patient-level splitting is critical for realistic evaluation
* Validation performance is sensitive to dataset balance
* Smaller image sizes improve speed but reduce medical detail
* Google Colab hardware limitations significantly affect experimentation speed

---

## Main Challenges

### Overfitting

The model achieved very high training accuracy while validation performance stagnated, indicating overfitting.

---

### Class Imbalance

The NIH ChestXray8 dataset contains significantly more "No Finding" images than disease-positive images.

This can artificially inflate accuracy metrics.

For this reason, Recall and F1-score were introduced to better evaluate medical classification performance.

---

### Computational Constraints

Training Vision Transformers on high-resolution medical images requires significant GPU memory and training time.

Google Colab limitations affected:

* Epoch duration
* Batch size selection
* Input image resolution
* Number of experiments

---

# Future Work

Future improvements planned for this project include:

* Multi-label disease classification
* Transfer learning using pretrained ViT models
* Higher-resolution medical image training
* Data augmentation techniques
* Better class balancing strategies
* Grad-CAM explainability visualization
* Advanced transformer architectures
* Mixed precision training for speed optimization
* Comparison with CNN-based models
* Deployment as a lightweight medical AI demo


```

---

# References

## Vision Transformer

Dosovitskiy, A. et al.

"An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale"

[https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

---

## NIH ChestXray8

Wang, X. et al.

"ChestX-ray8: Hospital-scale Chest X-ray Database and Benchmarks on Weakly-Supervised Classification and Localization of Common Thorax Diseases"

[https://arxiv.org/abs/1705.02315](https://arxiv.org/abs/1705.02315)

---

# Team Contributions

| Team Member   | Contribution                                                                                                                 |
| ------------- | -----------------------------------------------------------------------------------------------------------------------------|
| Jessie B      | Background and motivation                                                                                                    |
| Beatriz Graça | Elaboration of methodology and collaborative training                                                                        |
| Mashiro Yasuda| The construction of the overall ViT code, the main model trainers and data collectors.                                       |
| Advik Iyer    | Experiment analysis                                                                                                          |
| Zhuoming Liang| Future Work                                                                                                                  |
| Bowen Liu     | The full setup of the GitHub repository and the refactoring of the pipeline.                                                 |

---

# Acknowledgements

We thank:

* Alex for his full - process guidance and help.
* NIH Clinical Center for releasing the ChestXray8 dataset
* PyTorch developers
* Google Colab for providing accessible GPU resources
* Open-source medical AI research communities

---

# License
MIT License
