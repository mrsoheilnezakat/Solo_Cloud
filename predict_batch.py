import os
import sys
import torch
from PIL import Image
import torchvision.transforms as transforms

# Add src to Python path
sys.path.append("src")
from model import SimpleSegModel  # assuming model is defined in src/model.py

# Paths
test_dir = "test_images"
pred_dir = "pred_images"
model_path = "saved_models/cloud_seg_model.pth"

os.makedirs(pred_dir, exist_ok=True)

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),  # this will result in shape [1, H, W] for grayscale
])

# Load model
model = SimpleSegModel()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # change to 'cuda' if needed
model.eval()

# Predict
with torch.no_grad():
    for filename in os.listdir(test_dir):
        if filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")):
            img_path = os.path.join(test_dir, filename)
            img = Image.open(img_path).convert("L")  # still grayscale (1 channel)
            img = Image.merge("RGB", (img, img, img))  # replicate to 3 channels

            input_tensor = transform(img).unsqueeze(0)
            output = model(input_tensor)
            predicted_mask = torch.sigmoid(output).squeeze().numpy()

            save_path = os.path.join(pred_dir, filename)
            Image.fromarray((predicted_mask * 255).astype("uint8")).save(save_path)

print("✅ All masks saved to pred_images/")
