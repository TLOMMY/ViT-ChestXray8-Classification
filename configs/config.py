"""
ViT NIH Chest X-ray Binary Classification Pipeline Configuration
All tunable parameters are centralized here for easy modification.
"""

import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass
class DataConfig:
    """Data-related configuration"""
    # Dataset path (auto-set after kagglehub download)
    dataset_name: str = "nih-chest-xrays/data"
    csv_filename: str = "Data_Entry_2017.csv"

    # Target labels (binary classification)
    # To change classification task, modify target_labels and label_map below.
    # Example for Pneumonia vs No Finding:
    #   target_labels: tuple = ("Pneumonia", "No Finding")
    #   label_map: Dict[str, int] = {"No Finding": 0, "Pneumonia": 1}
    target_labels: Tuple[str, ...] = ("Infiltration", "No Finding")

    # Label encoding map: text label -> integer class index
    # The key order determines class names in evaluation reports.
    # 'No Finding' should typically be class 0 (negative), disease class 1 (positive).
    label_map: Dict[str, int] = field(default_factory=lambda: {"No Finding": 0, "Infiltration": 1})

    # Class names for visualization and reports (order must match label_map values)
    class_names: Tuple[str, ...] = field(default_factory=lambda: ("No Finding", "Infiltration"))

    # Data split ratios
    test_size: float = 0.2
    val_size: float = 0.1  # Validation ratio of remaining train data
    random_state: int = 42

    # Image preprocessing
    img_size: int = 64          # Input image size (H/W)
    num_channels: int = 1       # Grayscale channel count

    # DataLoader
    batch_size: int = 64
    num_workers: int = 4

    # Subset for quick debugging
    use_subset: bool = False
    subset_train_size: int = 1000
    subset_val_size: int = 200


@dataclass  
class ModelConfig:
    """ViT model architecture configuration"""
    # === Core tunable parameters ===
    img_size: int = 64          # Must match DataConfig.img_size
    patch_size: int = 16        # Patch size (must divide img_size)
    num_channels: int = 1       # Input channel count

    # Transformer architecture
    embed_dim: int = 32         # Patch embedding dimension
    num_heads: int = 4          # Attention head count
    mlp_dim: int = 64           # MLP hidden dimension
    transformer_units: int = 1  # Transformer Encoder layer count

    # Classification head
    num_classes: int = 2        # Binary classification
    dropout: float = 0.1        # Dropout rate

    @property
    def num_patches(self):
        return (self.img_size // self.patch_size) ** 2


@dataclass
class TrainConfig:
    """Training configuration"""
    # Hardware
    device: str = "cuda"        # cuda or cpu

    # Optimizer
    lr: float = 0.01
    weight_decay: float = 1e-4

    # Training schedule
    epochs: int = 150
    save_every: int = 1         # Save checkpoint every N epochs
    val_every: int = 10         # Validate every N epochs

    # Paths
    checkpoint_dir: str = "checkpoints"
    resume_from: str = ""       # Path to resume training from checkpoint

    # Early stopping
    early_stop_patience: int = 20
    early_stop_metric: str = "f1"  # Metric to monitor: 'f1', 'accuracy', or 'recall'


@dataclass
class Config:
    """Master configuration"""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        # Sync image dimensions
        self.model.img_size = self.data.img_size
        self.model.num_channels = self.data.num_channels

        # Create checkpoint directory
        os.makedirs(self.train.checkpoint_dir, exist_ok=True)


# Default config instance
config = Config()
