import sys
import os
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from vae.model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

ABLATION = 2
MODEL = f"vae/model_{ABLATION}.pth"
SAVE_DIR = f"vae/ablations/ablation{ABLATION}"
IMG_NAME = f"grid_{ABLATION}_1.png"

model = VAE(input_dim=64 * 64 * 3, hidden_dim=256, latent_dim=128).to(device)
model.load_state_dict(torch.load(MODEL, map_location=device))
model.eval()

# os.makedirs("vae/generated_images/checkpoint1_4", exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)

def save_image(img, name):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)

def generate_image(label, seed):
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(1, model.latent_dim, generator=g, device=device) * 1.2
        print(z)
        labels = torch.full((1, 1), float(label), device=device)

        image = model.decode(z, labels)
        return image

def generate_grid(c=3):
    fig, axes = plt.subplots(2, c, figsize=(3 * c, 6))
    
    for col in range(c):
        # Pick a random seed for this column to have the same "person" in both rows
        seed = torch.randint(0, 10000, (1,)).item()
        
        # 1st Row: No glasses (label = 0)
        img_0_tensor = generate_image(label=0, seed=seed)
        img_0_np = img_0_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_0_np = (img_0_np * 255).clip(0, 255).astype("uint8")
        
        axes[0, col].imshow(img_0_np)
        axes[0, col].set_xticks([])
        axes[0, col].set_yticks([])
        if col == 0:
            axes[0, col].set_ylabel("No Glasses", size='large')
        
        # 2nd Row: With glasses (label = 1)
        img_1_tensor = generate_image(label=1, seed=seed)
        img_1_np = img_1_tensor[0].permute(1, 2, 0).cpu().numpy()
        img_1_np = (img_1_np * 255).clip(0, 255).astype("uint8")
        
        axes[1, col].imshow(img_1_np)
        axes[1, col].set_xticks([])
        axes[1, col].set_yticks([])
        if col == 0:
            axes[1, col].set_ylabel("Glasses", size='large')

    plt.tight_layout()
    path = os.path.join(SAVE_DIR, IMG_NAME)
    plt.savefig(path)
    print(f"Saved grid: {path}")
    plt.show()
    plt.close()

generate_grid(c=3)