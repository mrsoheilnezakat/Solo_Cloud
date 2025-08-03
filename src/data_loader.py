# src/data_loader.py

import glob
import os

from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class SimpleSegDataset(Dataset):
    """
    A minimal Dataset for segmentation with:
    - JPEG/.png support
    - On-the-fly resizing
    - Pixel normalization
    """
    def __init__(
        self,
        images_dir,
        masks_dir,
        img_size=(256, 256),        # resize target (H, W)
        normalize=True             # whether to apply mean/std normalization
    ):
        self.images_dir = images_dir
        self.masks_dir  = masks_dir
        self.img_size   = img_size
        self.normalize  = normalize

        # supported extensions
        img_pats = ["*.jpg", "*.jpeg"]
        msk_pats = ["*.jpg", "*.jpeg", "*.png"]

        # gather & sort
        self.img_paths  = sorted(
            sum((glob.glob(os.path.join(images_dir, p)) for p in img_pats), [])
        )
        self.mask_paths = sorted(
            sum((glob.glob(os.path.join(masks_dir, p)) for p in msk_pats), [])
        )

        assert len(self.img_paths) == len(self.mask_paths), (
            f"Found {len(self.img_paths)} images but {len(self.mask_paths)} masks"
        )

        # precompute normalization tensors if needed
        if self.normalize:
            # ImageNet stats
            self.mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
            self.std  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # 1) Load and resize
        img = Image.open(self.img_paths[idx]).convert("RGB")
        msk = Image.open(self.mask_paths[idx]).convert("L")
        img = img.resize(self.img_size, Image.BILINEAR)
        msk = msk.resize(self.img_size, Image.NEAREST)

        # 2) To NumPy
        img_arr = np.array(img)   # (H, W, 3)
        msk_arr = np.array(msk)   # (H, W)

        # 3) To PyTorch tensors
        img_tensor = torch.from_numpy(img_arr)              \
                         .permute(2, 0, 1).float() / 255.0  # (3, H, W)
        msk_tensor = torch.from_numpy(msk_arr)              \
                         .unsqueeze(0).float() / 255.0     # (1, H, W)
        msk_tensor = (msk_tensor > 0.5).float()             # ensure binary mask

        # 4) Normalize if asked
        if self.normalize:
            img_tensor = (img_tensor - self.mean) / self.std

        return img_tensor, msk_tensor

def get_simple_loaders(
    images_dir="data/raw/images",
    masks_dir="data/raw/masks",
    batch_size=8,
    img_size=(256,256),
    normalize=True,
    num_workers=2,
    shuffle=True
):
    """
    Returns a DataLoader that:
      - Resizes inputs to `img_size`
      - Optionally normalizes images
      - Batches and shuffles
    """
    dataset = SimpleSegDataset(
        images_dir, masks_dir,
        img_size=img_size,
        normalize=normalize
    )
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    return loader