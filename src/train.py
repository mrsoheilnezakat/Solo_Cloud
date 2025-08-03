import torch
import torch.nn as nn
from .config import Config
from .data_loader import get_dataloaders
from .model import SimpleCNN
from .utils import save_checkpoint

def train():
    device = Config.DEVICE
    model  = SimpleCNN().to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=Config.LR)
    loss_fn= nn.BCEWithLogitsLoss()

    train_dl, val_dl = get_dataloaders()
    for epoch in range(Config.EPOCHS):
        model.train()
        for imgs in train_dl:
            imgs = imgs.to(device)
            preds= model(imgs)
            loss = loss_fn(preds, imgs)  # example
            opt.zero_grad(); loss.backward(); opt.step()

        # optionally validate & save
        val_loss = evaluate(model, val_dl, loss_fn, device)
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}")
        save_checkpoint(model, opt, epoch, Config.OUT_DIR)

if __name__ == "__main__":
    train()