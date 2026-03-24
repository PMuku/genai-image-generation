import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torchvision.utils import save_image

from diffusion.model import UNet, noise_schedule, generate

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, default="diffusion/model.pth", help="Path to model checkpoint")
parser.add_argument("--num_images", type=int, default=4, help="Number of images to generate")
parser.add_argument("--output", type=str, default="diffusion/generated_faces.png", help="Output image path")
parser.add_argument("--nrow", type=int, default=2, help="Number of images per row in the grid")
args = parser.parse_args()

T = 500
beta, alpha, alpha_bar = noise_schedule(T)
beta = beta.to(device)
alpha = alpha.to(device)
alpha_bar = alpha_bar.to(device)

model = UNet(img_ch=3, base_ch=64).to(device)
model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
model.eval()


with torch.no_grad():
    samples = generate(model, shape=(args.num_images, 3, 64, 64), timesteps=T, beta=beta, alpha=alpha, alpha_bar=alpha_bar, device=device)
    samples = samples.clamp(0, 1)

save_image(samples, args.output, nrow=args.nrow)