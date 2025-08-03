# src/data_loader.py

import glob
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleSegDataset(Dataset):
    """
    A minimal Dataset for segmentation:
    - Finds all .jpg/.jpeg files in images_dir
    - Finds matching files in masks_dir (same filenames)
    - Returns (img, mask) pairs as FloatTensors in [0,1]
    """
    def __init__(self, images_dir, masks_dir):
        img_patterns = ["*.jpg", "*.jpeg"]
        msk_patterns = ["*.jpg", "*.jpeg", "*.png"]

        self.img_paths = []
        for pat in img_patterns:
            self.img_paths += glob.glob(os.path.join(images_dir, pat))
        self.img_paths = sorted(self.img_paths)

        self.mask_paths = []
        for pat in msk_patterns:
            self.mask_paths += glob.glob(os.path.join(masks_dir, pat))
        self.mask_paths = sorted(self.mask_paths)

        assert len(self.img_paths) == len(self.mask_paths), (
            f"Found {len(self.img_paths)} images but {len(self.mask_paths)} masks"
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx]).convert("RGB")
        msk = Image.open(self.mask_paths[idx]).convert("L")

        img_arr = np.array(img)
        msk_arr = np.array(msk)

        img_tensor = torch.from_numpy(img_arr).permute(2, 0, 1).float() / 255.0
        msk_tensor = torch.from_numpy(msk_arr).unsqueeze(0).float() / 255.0

        return img_tensor, msk_tensor

def get_simple_loaders(
    images_dir="data/raw/images",
    masks_dir="data/raw/masks",
    batch_size=8,
    num_workers=2,
    shuffle=True
):
    """
    Returns a DataLoader for (img, mask) pairs from your JPEG dataset.
    """
    dataset = SimpleSegDataset(images_dir, masks_dir)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader