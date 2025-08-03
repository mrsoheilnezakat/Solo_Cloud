import os
from pathlib import Path

class Config:
    # paths
    ROOT    = Path(__file__).parent.parent
    RAW_DIR = ROOT / "data" / "raw"
    PROC_DIR= ROOT / "data" / "processed"
    OUT_DIR = ROOT / "outputs"

    # training
    BATCH_SIZE  = 32
    LR          = 1e-3
    EPOCHS      = 20
    DEVICE      = "cuda" if __import__("torch").cuda.is_available() else "cpu"