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

model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)