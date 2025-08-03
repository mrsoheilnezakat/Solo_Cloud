# test.py

from src.data_loader import get_simple_loaders

def main():
    # Create a loader with a small batch
    loader = get_simple_loaders(
        images_dir="data/raw/images",
        masks_dir="data/raw/masks",
        batch_size=4,
        shuffle=False
    )

    # Pull one batch
    imgs, masks = next(iter(loader))

    # Print out the tensor shapes
    print("Images:", imgs.shape)  # expect: [4, 3, H, W]
    print("Masks: ", masks.shape) # expect: [4, 1, H, W]

if __name__ == "__main__":
    main()
