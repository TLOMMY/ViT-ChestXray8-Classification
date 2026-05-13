"""
Model evaluation and result visualization
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import torch

from .utils import compute_metrics


@torch.no_grad()
def evaluate_model(model, test_loader, device, class_names=('No Finding', 'Infiltration')):
    """Evaluate model on test set"""
    model.eval()
    all_preds, all_labels = [], []

    for images, labels in test_loader:
        images = images.to(device)
        outputs = model(images)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())

    metrics = compute_metrics(all_labels, all_preds, pos_label=1)

    print("\n" + "="*50)
    print("Test Set Evaluation Results")
    print("="*50)
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"F1-Score: {metrics['f1']:.2f}%")
    print(f"Recall:   {metrics['recall']:.2f}%")
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, 
                                target_names=class_names))

    return metrics, all_labels, all_preds


def plot_training_history(history, save_path="outputs/training_history.png"):
    """Plot training history curves"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Training Loss
    axes[0].plot(epochs, history['train_loss'], 'r-o', label='Train Loss')
    axes[0].set_title('Training Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].grid(True)

    # 2. Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-o', label='Train Acc')

    val_epochs = [e for e, acc in zip(epochs, history['val_acc']) if acc is not None]
    val_accs = [acc for acc in history['val_acc'] if acc is not None]
    if val_accs:
        axes[1].plot(val_epochs, val_accs, 'g-s', label='Val Acc')

    axes[1].set_title('Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_ylim(0, 100)
    axes[1].legend()
    axes[1].grid(True)

    # 3. F1 & Recall
    val_f1s = [f for f in history['val_f1'] if f is not None]
    val_recalls = [r for r in history['val_recall'] if r is not None]

    if val_f1s:
        axes[2].plot(val_epochs, val_f1s, 'purple', marker='o', label='Val F1')
    if val_recalls:
        axes[2].plot(val_epochs, val_recalls, 'orange', marker='s', label='Val Recall')

    axes[2].set_title('Validation F1 & Recall')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Score (%)')
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Training curves saved: {save_path}")
    plt.show()


def plot_confusion_matrix(y_true, y_pred, class_names=('No Finding', 'Infiltration'), 
                          save_path="outputs/confusion_matrix.png"):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Confusion matrix saved: {save_path}")
    plt.show()
