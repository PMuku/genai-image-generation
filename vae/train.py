import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from shared.dataset import GlassesDataset
from vae.model import VAE


CSV_PATH = "data/processed/train.csv"
IMG_DIR = "data/raw/faces-spring-2020/faces-spring-2020"

dataset = GlassesDataset(CSV_PATH, IMG_DIR, img_size=64)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VAE(input_dim=64 * 64 * 3, hidden_dim=512, latent_dim=64).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)


def train_model(epochs=10):

	for epoch in range(epochs):
		model.train()
		running_loss = 0.0

		for images, labels in loader:
			images = images.to(device)
			labels = labels.float().unsqueeze(1).to(device)

			optimizer.zero_grad()
			recon_images, mu, logvar = model(images, labels)
			loss = model.loss_function(recon_images, images, mu, logvar)
			loss.backward()
			optimizer.step()

			running_loss += loss.item()
		epoch_loss = running_loss / max(len(loader), 1)
		print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}")
		
	return epoch_loss


epoch_loss = train_model(epochs=10)
print(f"Training loss after 10 epochs: {epoch_loss:.4f}")
torch.save(model.state_dict(), "vae/model.pth")
