import torch

def evaluate(model, loader, loss_fn, device):
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for imgs in loader:
            imgs  = imgs.to(device)
            preds = model(imgs)
            total += loss_fn(preds, imgs).item()
            count += 1
    return total / count