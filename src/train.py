"""
Training pipeline: training loop, validation, early stopping, checkpoint management
"""

import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm

from .utils import compute_metrics, save_checkpoint, load_checkpoint


class Trainer:
    """Trainer class"""

    def __init__(self, model, device, config):
        self.model = model
        self.device = device
        self.config = config
        self.train_cfg = config.train

        # Optimizer & Loss
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=self.train_cfg.lr,
            weight_decay=self.train_cfg.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()

        # Metrics tracking
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_acc': [], 'val_f1': [], 'val_recall': []
        }
        self.best_metric = 0.0
        self.patience_counter = 0
        self.start_epoch = 0

        # Resume training
        if self.train_cfg.resume_from and os.path.exists(self.train_cfg.resume_from):
            self.start_epoch, _ = load_checkpoint(
                self.train_cfg.resume_from, model, self.optimizer, device
            )
            print(f"Resumed training from epoch {self.start_epoch}")

    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(train_loader, desc="Training")
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*correct/total:.2f}%'
            })

        epoch_loss = total_loss / len(train_loader)
        epoch_acc = 100.0 * correct / total
        return epoch_loss, epoch_acc

    @torch.no_grad()
    def validate(self, val_loader):
        """Validation"""
        self.model.eval()
        all_preds, all_labels = [], []

        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device)
            outputs = self.model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        metrics = compute_metrics(all_labels, all_preds, pos_label=1)
        return metrics

    def fit(self, train_loader, val_loader):
        """Full training loop"""
        print(f"Starting training | Device: {self.device} | Total epochs: {self.train_cfg.epochs}")

        for epoch in range(self.start_epoch, self.train_cfg.epochs):
            print(f"\n{'='*50}")
            print(f"Epoch {epoch+1}/{self.train_cfg.epochs}")

            # Training
            t_start = time.time()
            train_loss, train_acc = self.train_epoch(train_loader)
            t_elapsed = time.time() - t_start

            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)

            print(f"Train | Loss: {train_loss:.4f} | Acc: {train_acc:.2f}% | Time: {t_elapsed:.1f}s")

            # Save checkpoint
            if (epoch + 1) % self.train_cfg.save_every == 0:
                ckpt_path = os.path.join(
                    self.train_cfg.checkpoint_dir, 
                    f"checkpoint_epoch_{epoch+1}.pth"
                )
                save_checkpoint(self.model, self.optimizer, epoch, self.history, ckpt_path)

            # Validation
            if (epoch + 1) % self.train_cfg.val_every == 0:
                val_metrics = self.validate(val_loader)

                self.history['val_acc'].append(val_metrics['accuracy'])
                self.history['val_f1'].append(val_metrics['f1'])
                self.history['val_recall'].append(val_metrics['recall'])

                print(f"Val   | Acc: {val_metrics['accuracy']:.2f}% | "
                      f"F1: {val_metrics['f1']:.2f}% | Recall: {val_metrics['recall']:.2f}%")

                # Early stopping check
                current_metric = val_metrics[self.train_cfg.early_stop_metric]
                if current_metric > self.best_metric:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    # Save best model
                    best_path = os.path.join(self.train_cfg.checkpoint_dir, "best_model.pth")
                    save_checkpoint(self.model, self.optimizer, epoch, self.history, best_path)
                    print(f">>> Best model saved ({self.train_cfg.early_stop_metric}: {current_metric:.2f}%)")
                else:
                    self.patience_counter += self.train_cfg.val_every
                    if self.patience_counter >= self.train_cfg.early_stop_patience:
                        print(f"Early stopping triggered! No improvement for {self.train_cfg.early_stop_patience} epochs")
                        break
            else:
                # Fill None for non-validation epochs to keep list lengths consistent
                self.history['val_acc'].append(None)
                self.history['val_f1'].append(None)
                self.history['val_recall'].append(None)

        return self.history
