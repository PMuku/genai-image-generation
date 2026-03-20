import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from vae.model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(input_dim=128 * 128 * 3, hidden_dim=256, latent_dim=64).to(device)
model.load_state_dict(torch.load("vae/model_1.pth", map_location=device))
model.eval()

os.makedirs("vae/generated_images/checkpoint3", exist_ok=True)

def save_image(img, name):
    img = img[0].permute(1, 2, 0).cpu().numpy()
    img = (img * 255).clip(0, 255).astype("uint8")
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(name, img)

def generate_image(label, seed):
    with torch.no_grad():
        g = torch.Generator(device=device).manual_seed(seed)
        z = torch.randn(1, model.latent_dim, generator=g, device=device) * 2.5
        print(z)
        labels = torch.full((1, 1), float(label), device=device)

        image = model.decode(z, labels)
        save_image(image, f"vae/generated_images/checkpoint3/sample_{label}_{seed}.png")

# Generate 5 images with glasses and 5 without, using different seeds for variety
for i in range(10):
    seed = torch.randint(0, 10000, (1,)).item()
    generate_image(label=1, seed=seed)  # with glasses
    seed = torch.randint(0, 10000, (1,)).item()
    generate_image(label=0, seed=seed)  # without glasses
