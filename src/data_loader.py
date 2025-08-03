from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from .config import Config
import os
from PIL import Image

class CloudDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_paths = list((img_dir).glob("*.png"))
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img = Image.open(self.img_paths[idx])
        return self.transform(img)

def get_dataloaders():
    train_ds = CloudDataset(Config.PROC_DIR / "train")
    val_ds   = CloudDataset(Config.PROC_DIR / "val")
    return (
        DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True),
        DataLoader(val_ds,   batch_size=Config.BATCH_SIZE),
    )