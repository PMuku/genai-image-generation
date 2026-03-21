import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import cv2
import numpy as np
from vae.model import VAE

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(input_dim=64 * 64 * 3, hidden_dim=192, latent_dim=64).to(device)
model.load_state_dict(torch.load("vae/model_2.pth", map_location=device))
model.eval()

os.makedirs("vae/generated_images/checkpoint8", exist_ok=True)

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
        save_image(image, f"vae/generated_images/checkpoint8/sample_{label}_{seed}.png")

for i in range(8):
    seed = torch.randint(0, 10000, (1,)).item()
    generate_image(label=1, seed=seed)  # glasses
    generate_image(label=0, seed=seed)  # no glasses

# for EPOCH in [3, 5, 7]:

#     model = VAE(input_dim=128 * 128 * 3, hidden_dim=256, latent_dim=32).to(device)
#     model.load_state_dict(torch.load(f"vae/model_epoch_{EPOCH}.pth", map_location=device))
#     model.eval()

#     torch.manual_seed(42)


#     num_samples = 6
#     fixed_z = torch.randn(num_samples, model.latent_dim).to(device)

#     # labels: first half glasses, second half no glasses
#     labels = torch.cat([
#         torch.ones(num_samples // 2, 1),
#         torch.zeros(num_samples // 2, 1)
#     ]).to(device)

#     with torch.no_grad():
#         images = model.decode(fixed_z, labels)

#     save_dir = f"vae/generated_images/epoch_{EPOCH}"
#     os.makedirs(save_dir, exist_ok=True)

#     for i in range(num_samples):
#         label = int(labels[i].item())
#         save_image(images[i].unsqueeze(0), f"{save_dir}/sample_{i}_label_{label}.png")

# for i in range(num_samples):
#     save_image(images[i:i+1], f"vae/generated_images/checkpoint3/sample_{i}.png")

# z = fixed_z[:5]

# with torch.no_grad():
#     images_glasses = model.decode(z, torch.ones(5,1).to(device))
#     images_no = model.decode(z, torch.zeros(5,1).to(device))

# save_image(images_glasses, "vae/generated_images/checkpoint3/glasses.png")
# save_image(images_no, "vae/generated_images/checkpoint3/no_glasses.png")