#!/usr/bin/env python3
"""
Model evaluation script - load trained model and evaluate on test set

Usage:
    python scripts/evaluate.py --checkpoint checkpoints/best_model.pth
"""

import sys
import os
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kagglehub
import torch
from configs.config import Config
from src.dataset import load_and_split_data, get_dataloaders
from src.model import build_model
from src.evaluate import evaluate_model, plot_confusion_matrix
from src.utils import load_checkpoint


def main():
    parser = argparse.ArgumentParser(description='Evaluate ViT Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--img-size', type=int, default=64)
    parser.add_argument('--patch-size', type=int, default=16)

    args = parser.parse_args()

    config = Config()
    config.data.img_size = args.img_size
    config.model.img_size = args.img_size
    config.model.patch_size = args.patch_size

    # Download dataset
    path = kagglehub.dataset_download(config.data.dataset_name)
    config.data.dataset_name = path

    # Load data
    df_train, df_val, df_test = load_and_split_data(config)
    _, _, test_loader = get_dataloaders(df_train, df_val, df_test, config)

    # Load model
    model, device = build_model(config)
    load_checkpoint(args.checkpoint, model, device=device)

    # Evaluate
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device,
                                              class_names=config.data.class_names)
    plot_confusion_matrix(y_true, y_pred, class_names=config.data.class_names)


if __name__ == '__main__':
    main()
