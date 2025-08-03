import torch
import torch.nn as nn

class SimpleSegModel(nn.Module):
    """
    A tiny encoder–decoder for cloud segmentation.
    Input: 3×H×W → Output: 1×H×W logits
    """
    def __init__(self, in_ch=3, base_ch=32):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, base_ch,   kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # downscale H,W → H/2,W/2
        )
        # Decoder
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, 1,    kernel_size=1)  # output 1 channel (logits)
        )

    def forward(self, x):
        x = self.enc(x)
        x = self.dec(x)
        return x