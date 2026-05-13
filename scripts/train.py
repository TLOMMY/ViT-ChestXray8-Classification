#!/usr/bin/env python3
"""
ViT NIH Chest X-ray Binary Classification Training Main Script

Usage:
    python scripts/train.py

    # Quick debug with subset
    python scripts/train.py --subset

    # Resume from checkpoint
    python scripts/train.py --resume checkpoints/checkpoint_epoch_50.pth

    # Change classification task (example: Pneumonia vs No Finding)
    python scripts/train.py --target-labels Pneumonia "No Finding" --label-map "{\"No Finding\": 0, \"Pneumonia\": 1}"
"""

import sys
import os
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import kagglehub
from configs.config import Config
from src.dataset import load_and_split_data, get_dataloaders
from src.model import build_model
from src.train import Trainer
from src.evaluate import evaluate_model, plot_training_history, plot_confusion_matrix


def download_dataset(config):
    """Download/confirm dataset"""
    print("Checking dataset...")
    try:
        path = kagglehub.dataset_download(config.data.dataset_name)
        config.data.dataset_name = path
        print(f"Dataset path: {path}")
        return path
    except Exception as e:
        print(f"Dataset download failed: {e}")
        print("Please ensure Kaggle API is configured (~/kaggle/kaggle.json)")
        raise


def main():
    parser = argparse.ArgumentParser(description='ViT NIH Chest X-ray Training')
    parser.add_argument('--subset', action='store_true', 
                        help='Use subset for quick debug (1000 train / 200 val)')
    parser.add_argument('--resume', type=str, default='',
                        help='Path to checkpoint to resume training')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Total training epochs')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--img-size', type=int, default=64,
                        help='Input image size')
    parser.add_argument('--patch-size', type=int, default=16,
                        help='Patch size')
    parser.add_argument('--embed-dim', type=int, default=32,
                        help='Embedding dimension')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Attention head count')
    parser.add_argument('--transformer-units', type=int, default=1,
                        help='Transformer layer count')
    parser.add_argument('--target-labels', nargs='+', default=None,
                        help='Target labels for classification (e.g., Infiltration "No Finding")')
    parser.add_argument('--label-map', type=str, default=None,
                        help="JSON string for label mapping, e.g., '{\"No Finding\": 0, \"Infiltration\": 1}'")
    parser.add_argument('--early-stop-metric', type=str, default='f1',
                        choices=['f1', 'accuracy', 'recall'],
                        help='Metric to monitor for early stopping')

    args = parser.parse_args()

    # Initialize config
    config = Config()

    # Override with command-line arguments
    config.data.use_subset = args.subset
    config.train.epochs = args.epochs
    config.train.lr = args.lr
    config.data.batch_size = args.batch_size
    config.data.img_size = args.img_size
    config.model.img_size = args.img_size
    config.model.patch_size = args.patch_size
    config.model.embed_dim = args.embed_dim
    config.model.num_heads = args.num_heads
    config.model.transformer_units = args.transformer_units
    config.train.resume_from = args.resume
    config.train.early_stop_metric = args.early_stop_metric

    # Handle target labels and label map
    if args.target_labels is not None:
        config.data.target_labels = tuple(args.target_labels)
        # Re-generate label_map and class_names
        if args.label_map is not None:
            config.data.label_map = json.loads(args.label_map)
        else:
            # Auto-generate default map
            if "No Finding" in config.data.target_labels:
                disease = [l for l in config.data.target_labels if l != "No Finding"][0]
                config.data.label_map = {"No Finding": 0, disease: 1}
            else:
                config.data.label_map = {label: idx for idx, label in enumerate(sorted(config.data.target_labels))}
        # Re-generate class_names
        sorted_items = sorted(config.data.label_map.items(), key=lambda x: x[1])
        config.data.class_names = tuple(name for name, _ in sorted_items)
        # Update model num_classes
        config.model.num_classes = len(config.data.class_names)

    print("="*60)
    print("Configuration:")
    print(f"  Image size: {config.data.img_size}x{config.data.img_size}")
    print(f"  Patch size: {config.model.patch_size}")
    print(f"  Embed dim: {config.model.embed_dim}")
    print(f"  Attention heads: {config.model.num_heads}")
    print(f"  Transformer layers: {config.model.transformer_units}")
    print(f"  Batch size: {config.data.batch_size}")
    print(f"  Learning rate: {config.train.lr}")
    print(f"  Total epochs: {config.train.epochs}")
    print(f"  Use subset: {config.data.use_subset}")
    print(f"  Resume: {config.train.resume_from if config.train.resume_from else 'None'}")
    print(f"  Target labels: {config.data.target_labels}")
    print(f"  Label map: {config.data.label_map}")
    print(f"  Class names: {config.data.class_names}")
    print(f"  Early stop metric: {config.train.early_stop_metric}")
    print("="*60)

    # Download dataset
    download_dataset(config)

    # Load data
    print("\nLoading dataset...")
    df_train, df_val, df_test = load_and_split_data(config)
    train_loader, val_loader, test_loader = get_dataloaders(df_train, df_val, df_test, config)

    # Build model
    print("\nBuilding model...")
    model, device = build_model(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Train
    print("\nStarting training...")
    trainer = Trainer(model, device, config)
    history = trainer.fit(train_loader, val_loader)

    # Test evaluation
    print("\nTest set evaluation...")
    metrics, y_true, y_pred = evaluate_model(model, test_loader, device, 
                                                class_names=config.data.class_names)

    # Visualization
    print("\nGenerating visualizations...")
    plot_training_history(history)
    plot_confusion_matrix(y_true, y_pred, class_names=config.data.class_names)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
