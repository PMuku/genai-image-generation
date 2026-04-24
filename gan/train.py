import os
import shutil
import pandas as pd
import random
import torchvision.transforms.functional as TF
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import torchvision.utils as vutils
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn as nn
from model import Generator, EnergyModel, EMA, z_dim

# --- Local Data Paths ---
src_img_path = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "faces-spring-2020", "faces-spring-2020")

dst_base = "./glassesdata/"
img_path = os.path.join(dst_base, "images")
train_folder = os.path.join(dst_base, "train")
test_folder = os.path.join(dst_base, "test")

os.makedirs(img_path, exist_ok=True)
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

for img_name in os.listdir(src_img_path):
    if img_name.startswith("face-") and img_name.endswith(".png"):
        new_name = img_name.replace("face-", "")
        src = os.path.join(src_img_path, img_name)
        dst = os.path.join(img_path, new_name)
        if not os.path.exists(dst):
            shutil.copy(src, dst)

images = sorted(
    [f for f in os.listdir(img_path) if f.endswith(".png")],
    key=lambda x: int(os.path.splitext(x)[0])
)

for img_name in images[:4500]:
    src = os.path.join(img_path, img_name)
    dst = os.path.join(train_folder, img_name)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

for img_name in images[4500:5000]:
    src = os.path.join(img_path, img_name)
    dst = os.path.join(test_folder, img_name)
    if not os.path.exists(dst):
        shutil.copy(src, dst)

print("dataset copied and split into ./glassesdata/train and ./glassesdata/test")

csv_path = os.path.join(os.path.dirname(__file__), "..", "data", "processed", "train.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)

    glasses_folder = os.path.join(train_folder, "glasses")
    no_glasses_folder = os.path.join(train_folder, "no_glasses")
    os.makedirs(glasses_folder, exist_ok=True)
    os.makedirs(no_glasses_folder, exist_ok=True)

    for idx, row in df.iterrows():
        img_name = f"{int(row['id'])}.png"
        src = os.path.join(train_folder, img_name)

        if not os.path.exists(src):
            print(f"Image {img_name} not found, skipping...")
            continue

        dst = glasses_folder if row['glasses'] == 1 else no_glasses_folder
        shutil.move(src, os.path.join(dst, img_name))

    print("train folder segregated into glasses and no_glasses")

for root, dirs, files in os.walk(dst_base):
    level = root.replace(dst_base, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files[:3]:
        print(f"{subindent}{f}")


# --- Data Loading ---
transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

train_data = datasets.ImageFolder(train_folder, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=0, pin_memory=False)

print("Classes:", train_data.classes)

def adaptive_augment(image, p=0.2):
    if random.random() < p: image = TF.hflip(image)
    if random.random() < p: image = TF.adjust_brightness(image, 0.8 + 0.4*random.random())
    if random.random() < p: image = TF.adjust_contrast(image, 0.8 + 0.4*random.random())
    return image

if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
print(f"Using device: {device}")

use_amp = device.type == 'cuda'

# --- Initialization ---
G = Generator(z_dim=z_dim).to(device)
E = EnergyModel().to(device)

if use_amp:
    scaler = torch.amp.GradScaler('cuda')

opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_E = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))

lambda_ent = 0.1  # entropy regularisation weight (Eq. 15)

epochs = 50
fixed_noise = torch.randn(6, z_dim, 1, 1, device=device)

save_path = "./GAN_Checkpoints"
os.makedirs(save_path, exist_ok=True)

ema = EMA(G, decay=0.999)

G_losses = []
E_losses = []
E_real_vals = []
E_fake_vals = []

# --- Training Loop ---
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

    for i, (imgs, _) in loop:
        imgs = imgs.to(device)
        bs = imgs.size(0)

        aug_imgs = adaptive_augment(imgs)

        # --- Energy model step (Eq. 7) ---
        # positive phase: push energy down on real data
        # negative phase: push energy up on generated data
        with torch.no_grad():
            fake_imgs = G(torch.randn(bs, z_dim, 1, 1, device=device))

        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                real_energy = E(aug_imgs)
                fake_energy = E(fake_imgs)
                e_loss = real_energy.mean() - fake_energy.mean()
            opt_E.zero_grad()
            scaler.scale(e_loss).backward()
            scaler.step(opt_E)
            scaler.update()
        else:
            real_energy = E(aug_imgs)
            fake_energy = E(fake_imgs)
            e_loss = real_energy.mean() - fake_energy.mean()
            opt_E.zero_grad()
            e_loss.backward()
            opt_E.step()

        E_losses.append(e_loss.item())
        E_real_vals.append(real_energy.mean().item())
        E_fake_vals.append(fake_energy.mean().item())

        # --- Generator step (Eq. 13-14) ---
        # minimise energy of generated samples + entropy regularisation via BN scales
        if use_amp:
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                gen_imgs = G(torch.randn(bs, z_dim, 1, 1, device=device))
                gen_energy = E(gen_imgs)
                entropy_reg = sum(
                    torch.log(m.weight.abs() + 1e-8).sum()
                    for m in G.modules() if isinstance(m, nn.BatchNorm2d)
                )
                g_loss = gen_energy.mean() - lambda_ent * entropy_reg
            opt_G.zero_grad()
            scaler.scale(g_loss).backward()
            scaler.step(opt_G)
            scaler.update()
        else:
            gen_imgs = G(torch.randn(bs, z_dim, 1, 1, device=device))
            gen_energy = E(gen_imgs)
            entropy_reg = sum(
                torch.log(m.weight.abs() + 1e-8).sum()
                for m in G.modules() if isinstance(m, nn.BatchNorm2d)
            )
            g_loss = gen_energy.mean() - lambda_ent * entropy_reg
            opt_G.zero_grad()
            g_loss.backward()
            opt_G.step()

        G_losses.append(g_loss.item())
        ema.update(G)

    torch.save({
        "epoch": epoch+1,
        "generator_state_dict": G.state_dict(),
        "energy_model_state_dict": E.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_E_state_dict": opt_E.state_dict(),
    }, f"{save_path}/GAN_epoch_{epoch+1}.pth")

    if (epoch+1) % 2 == 0:
        G_ema = Generator(z_dim=z_dim).to(device)
        G_ema.load_state_dict(ema.shadow, strict=False)
        G_ema.eval()

        with torch.no_grad():
            fake_samples = G_ema(fixed_noise).detach().cpu()

        grid = vutils.make_grid(fake_samples, nrow=3, padding=2, normalize=True)
        plt.figure(figsize=(6,4))
        plt.axis("off")
        plt.title(f"Generated Images - Epoch {epoch+1}")
        plt.imshow(np.transpose(grid, (1,2,0)))
        plt.savefig(f"{save_path}/samples_epoch_{epoch+1}.png", bbox_inches='tight')
        plt.close()

torch.save({
    "epoch": epochs,
    "generator_state_dict": G.state_dict(),
    "energy_model_state_dict": E.state_dict(),
    "opt_G_state_dict": opt_G.state_dict(),
    "opt_E_state_dict": opt_E.state_dict(),
    "ema_shadow": ema.shadow,
}, f"{save_path}/GAN_final.pth")

torch.save(
    ema.shadow,
    f"{save_path}/G_ema_final.pth"
)

print(f"Saved final checkpoint → {save_path}/GAN_final.pth")
print(f"Saved EMA generator   → {save_path}/G_ema_final.pth")

# --- Loss Plots ---
plt.figure(figsize=(8,4))
plt.plot(G_losses, label="G Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Generator Loss Curve")
plt.legend()
plt.savefig(f"{save_path}/g_loss.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,4))
plt.plot(E_losses, label="Energy Model Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Energy Model Loss (E_real - E_fake)")
plt.legend()
plt.savefig(f"{save_path}/e_loss.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,4))
plt.plot(G_losses, label="Generator Loss")
plt.plot(E_losses, label="Energy Model Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("EBM-GAN Loss (G vs E)")
plt.legend()
plt.savefig(f"{save_path}/gan_loss.png", bbox_inches='tight')
plt.close()

plt.figure(figsize=(8,4))
plt.plot(E_real_vals, label="Mean Real Energy")
plt.plot(E_fake_vals, label="Mean Fake Energy")
plt.xlabel("Iterations")
plt.ylabel("Energy")
plt.title("Energy Values Over Time (real should be lower)")
plt.legend()
plt.savefig(f"{save_path}/energy_vals.png", bbox_inches='tight')
plt.close()

print("All plots saved to", save_path)
