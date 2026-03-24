import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from shared.dataset import GlassesDataset
from diffusion.model import UNet, noise_schedule, forward_diffusion, generate
from diffusion.evaluate import compute_fid

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

CSV_PATH = "data/processed/train.csv"
IMG_DIR = "data/raw/faces-spring-2020/faces-spring-2020"

parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=64)
args = parser.parse_args()

# hyperparameters
"""
T: no of diffusion steps
EPOCHS: no of epochs for training
BATCH_SIZE: batch size
LR = learning rate
"""
T = 500
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = 2e-4

beta, alpha, alpha_bar = noise_schedule(T)
beta = beta.to(device)
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

dataset = GlassesDataset(CSV_PATH, IMG_DIR, img_size=64)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

model = UNet(img_ch=3, base_ch=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)


def train_model(epochs):
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0

        for images, _ in loader:
            images = images.to(device)

            timesteps = torch.randint(0, T, (images.size(0),), device=device)
            noised_images, noise = forward_diffusion(images, timesteps, alpha_bar)

            optimizer.zero_grad()
            predicted_noise = model(noised_images, timesteps)
            loss = torch.nn.functional.mse_loss(predicted_noise, noise)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_loss = running_loss / max(len(loader), 1)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")

    return epoch_loss


epoch_loss = train_model(EPOCHS)
print(f"Training loss after {EPOCHS} epochs: {epoch_loss:.4f}")
torch.save(model.state_dict(), "diffusion/model.pth")

model.eval()

# Generate samples for visualization
samples = generate(model, shape=(4, 3, 64, 64), timesteps=T, beta=beta, alpha=alpha, alpha_bar=alpha_bar, device=device)
samples = samples.clamp(0, 1)
save_image(samples, "diffusion/generated_faces.png", nrow=2)
print("Saved generated samples to diffusion/generated_faces.png")

# Compute FID score
NUM_FID_SAMPLES = 64
print(f"\nGenerating {NUM_FID_SAMPLES} samples for FID evaluation...")
fid_samples = generate(model, shape=(NUM_FID_SAMPLES, 3, 64, 64), timesteps=T, beta=beta, alpha=alpha, alpha_bar=alpha_bar, device=device)
fid_samples = fid_samples.clamp(0, 1)

# Collect real images for comparison
real_images = torch.stack([dataset[i][0] for i in range(min(NUM_FID_SAMPLES, len(dataset)))])

fid_device = torch.device("cpu")  # InceptionV3 can be heavy on MPS, use CPU for stability
fid_score = compute_fid(real_images, fid_samples, fid_device)
print(f"FID Score: {fid_score:.2f} (lower is better)")
