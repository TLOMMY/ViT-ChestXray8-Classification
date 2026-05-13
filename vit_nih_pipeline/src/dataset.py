"""
NIH Chest X-ray dataset loading and preprocessing
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader

from .utils import preprocess_image


def get_patient_stratification_key(patient_df_group, all_disease_labels):
    """Generate patient-level stratification key (prevents data leakage when splitting by Patient ID)"""
    all_labels = []
    for labels_str in patient_df_group['Finding Labels'].tolist():
        all_labels.extend(labels_str.split('|'))

    unique_diseases = set()
    has_no_finding = False

    for label in all_labels:
        if label == 'No Finding':
            has_no_finding = True
        elif label in all_disease_labels:
            unique_diseases.add(label)

    if not unique_diseases and has_no_finding:
        return "No_Finding_Only"
    elif unique_diseases:
        num = len(unique_diseases)
        if num == 1:
            return "Diseases_1"
        elif num == 2:
            return "Diseases_2"
        else:
            return "Diseases_3Plus"
    return "No_Labels_Found"


def load_and_split_data(config):
    """
    Load dataset and split by Patient ID with stratification.
    Returns: df_train, df_val, df_test (filtered for binary labels)
    """
    data_cfg = config.data

    # Read CSV
    csv_path = os.path.join(data_cfg.dataset_name, data_cfg.csv_filename)
    df = pd.read_csv(csv_path)

    # Get all disease labels (for stratification)
    all_labels = []
    for labels_str in df['Finding Labels']:
        all_labels.extend(labels_str.split('|'))
    label_counts = pd.Series(all_labels).value_counts()
    all_disease_labels = sorted([l for l in label_counts.index if l != 'No Finding'])

    # Patient stratification keys
    patient_stratify_keys = df.groupby('Patient ID').apply(
        lambda x: get_patient_stratification_key(x, all_disease_labels)
    )

    patient_ids = patient_stratify_keys.index.to_numpy()
    stratify_y = patient_stratify_keys.to_numpy()

    # Split: train+val / test
    patient_train_val, patient_test, _, _ = train_test_split(
        patient_ids, stratify_y, test_size=data_cfg.test_size, 
        random_state=data_cfg.random_state, stratify=stratify_y
    )

    # Split: train / val
    val_ratio = data_cfg.val_size / (1 - data_cfg.test_size)
    stratify_train_val = patient_stratify_keys.loc[patient_train_val].to_numpy()

    patient_train, patient_val, _, _ = train_test_split(
        patient_train_val, stratify_train_val, test_size=val_ratio,
        random_state=data_cfg.random_state, stratify=stratify_train_val
    )

    # Build DataFrames
    df_train = df[df['Patient ID'].isin(patient_train)].copy()
    df_val = df[df['Patient ID'].isin(patient_val)].copy()
    df_test = df[df['Patient ID'].isin(patient_test)].copy()

    # Filter binary labels using target_labels from config
    target_labels = list(data_cfg.target_labels)
    df_train = df_train[df_train['Finding Labels'].isin(target_labels)].copy()
    df_val = df_val[df_val['Finding Labels'].isin(target_labels)].copy()
    df_test = df_test[df_test['Finding Labels'].isin(target_labels)].copy()

    print(f"Data split complete:")
    print(f"  Train: {len(df_train)} images ({len(patient_train)} patients)")
    print(f"  Val:   {len(df_val)} images ({len(patient_val)} patients)")
    print(f"  Test:  {len(df_test)} images ({len(patient_test)} patients)")

    return df_train, df_val, df_test


class NIHChestXrayDataset(Dataset):
    """NIH Chest X-ray PyTorch Dataset"""

    def __init__(self, dataframe, image_dir, img_size=64, label_map=None):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = image_dir
        self.img_size = img_size
        self.label_map = label_map if label_map is not None else {"No Finding": 0, "Infiltration": 1}

        # Pre-scan all image paths for speed
        self.all_image_paths = {}
        for sub_dir in [f'images_{i:03d}' for i in range(1, 13)]:
            full_sub_dir = os.path.join(image_dir, sub_dir, 'images')
            if os.path.exists(full_sub_dir):
                for f in os.listdir(full_sub_dir):
                    self.all_image_paths[f] = os.path.join(full_sub_dir, f)

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_name = self.dataframe.iloc[idx]['Image Index']
        img_path = self.all_image_paths.get(img_name)

        if img_path is None:
            raise FileNotFoundError(f"Image {img_name} not found")

        image = preprocess_image(img_path, target_size=(self.img_size, self.img_size))
        image_tensor = torch.from_numpy(image).float()

        label_text = self.dataframe.iloc[idx]['Finding Labels']
        # Use label_map from config for encoding
        label = self.label_map.get(label_text, 0)

        return image_tensor, label


def get_dataloaders(df_train, df_val, df_test, config):
    """Create DataLoaders"""
    data_cfg = config.data
    image_dir = config.data.dataset_name
    label_map = data_cfg.label_map

    # Subset for quick debugging
    if data_cfg.use_subset:
        df_train = df_train.sample(n=min(data_cfg.subset_train_size, len(df_train)), 
                                   random_state=data_cfg.random_state)
        df_val = df_val.sample(n=min(data_cfg.subset_val_size, len(df_val)),
                               random_state=data_cfg.random_state)
        print(f"Using subset: Train={len(df_train)}, Val={len(df_val)}")

    train_ds = NIHChestXrayDataset(df_train, image_dir, data_cfg.img_size, label_map)
    val_ds = NIHChestXrayDataset(df_val, image_dir, data_cfg.img_size, label_map)
    test_ds = NIHChestXrayDataset(df_test, image_dir, data_cfg.img_size, label_map)

    train_loader = DataLoader(
        train_ds, batch_size=data_cfg.batch_size, 
        shuffle=True, num_workers=data_cfg.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=data_cfg.batch_size,
        shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=data_cfg.batch_size,
        shuffle=False, num_workers=data_cfg.num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader
