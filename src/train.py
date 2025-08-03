import torch
import torch.nn as nn
import torch.optim as optim
from src.data_loader import get_simple_loaders
from src.model import SimpleSegModel


# Hyperparameters
LR         = 1e-3
EPOCHS     = 50
BATCH_SIZE = 4
IMG_SIZE   = (256, 256)

def train():
    device = torch.device("cuda:0")        # explicitly pick GPU 0
    assert torch.cuda.is_available(), "CUDA not available – check your PyTorch + cudatoolkit install!"
    print("Forcing run on GPU:", torch.cuda.get_device_name(0))


    # 1) DataLoader
    loader = get_simple_loaders(
        images_dir="data/raw/images",
        masks_dir="data/raw/masks",
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        normalize=True,
        shuffle=True
    )

    # 2) Model, loss, optimizer
    model     = SimpleSegModel().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    # 3) Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        for imgs, masks in loader:
            imgs  = imgs.to(device)
            masks = masks.to(device)

            # forward + loss
            outputs = model(imgs)
            loss    = criterion(outputs, masks)

            # backward + step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {avg_loss:.4f}")

    # 4) Save final model weights
    torch.save(model.state_dict(), "cloud_seg_model.pth")
    print("Saved model to cloud_seg_model.pth")

if __name__ == "__main__":
    train()
