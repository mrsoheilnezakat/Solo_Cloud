import torch
from .config import Config
import os

def save_checkpoint(model, optimizer, epoch, out_dir=None):
    odir = out_dir or Config.OUT_DIR
    os.makedirs(odir, exist_ok=True)
    path = os.path.join(odir, f"ckpt_epoch{epoch}.pt")
    torch.save({
        "epoch":    epoch,
        "model":    model.state_dict(),
        "optimizer":optimizer.state_dict()
    }, path)