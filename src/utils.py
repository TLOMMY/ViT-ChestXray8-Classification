"""
Utility functions: image preprocessing, path lookup, metric computation, checkpoint I/O
"""

import os
import re
import numpy as np
from PIL import Image
import torch
from sklearn.metrics import f1_score, recall_score, accuracy_score


def preprocess_image(image_path, target_size=(64, 64)):
    """Image preprocessing: grayscale -> resize -> normalize -> add channel dim"""
    img = Image.open(image_path).convert('L')
    img = img.resize(target_size, Image.LANCZOS)
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W)
    return img_array


def find_full_image_path(image_filename, base_path):
    """Search for image in NIH dataset subdirectories (images_001 ~ images_012)"""
    potential_subdirs = [f'images_{i:03d}' for i in range(1, 13)]

    for sub_dir in potential_subdirs:
        candidate_path = os.path.join(base_path, sub_dir, 'images', image_filename)
        if os.path.exists(candidate_path):
            return candidate_path

    # Fallback: direct path
    candidate_path_direct = os.path.join(base_path, 'images', image_filename)
    if os.path.exists(candidate_path_direct):
        return candidate_path_direct
    return None


def compute_metrics(y_true, y_pred, pos_label=1):
    """Compute classification metrics"""
    return {
        'accuracy': accuracy_score(y_true, y_pred) * 100,
        'f1': f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100,
        'recall': recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0) * 100,
    }


def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        **metrics
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None, device='cuda'):
    """Load training checkpoint"""
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint


def parse_training_log(log_text):
    """Parse metrics from training log text (legacy compatibility)"""
    pattern = re.compile(r'==> Epoch (\d+) Summary: Total Loss = ([\d.]+), Accuracy = ([\d.]+)%')

    losses, accs = [], []
    for line in log_text.split('\n'):
        match = pattern.search(line)
        if match:
            losses.append(float(match.group(2)))
            accs.append(float(match.group(3)))

    return losses, accs
