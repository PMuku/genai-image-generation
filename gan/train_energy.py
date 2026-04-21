import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from shared.dataset import GlassesDataset
from gan.model import Generator, Discriminator, EMA, z_dim
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
import random as rd
import torchvision.transforms.functional as TF
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision.utils as vutils
import numpy as np

# # --- Download & Extract Data ---
# path = kagglehub.dataset_download("jeffheaton/glasses-or-no-glasses")
# print("Path:", path)

# src_img_path = "/kaggle/input/datasets/jeffheaton/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020"

# dst_base = "/kaggle/working/glassesdata/"
# img_path = os.path.join(dst_base, "images")
# train_folder = os.path.join(dst_base, "train")
# test_folder = os.path.join(dst_base, "test")

# os.makedirs(img_path, exist_ok=True)
# os.makedirs(train_folder, exist_ok=True)
# os.makedirs(test_folder, exist_ok=True)

# for img_name in os.listdir(src_img_path):
#     if img_name.startswith("face-") and img_name.endswith(".png"):
#         new_name = img_name.replace("face-", "")
#         src = os.path.join(src_img_path, img_name)
#         dst = os.path.join(img_path, new_name)
#         if not os.path.exists(dst):  # avoid recopying if already done
#             shutil.copy(src, dst)

# images = sorted(
#     [f for f in os.listdir(img_path) if f.endswith(".png")],
#     key=lambda x: int(os.path.splitext(x)[0])
# )

# for img_name in images[:4500]:
#     src = os.path.join(img_path, img_name)
#     dst = os.path.join(train_folder, img_name)
#     if not os.path.exists(dst):
#         shutil.copy(src, dst)

# for img_name in images[4500:5000]:
#     src = os.path.join(img_path, img_name)
#     dst = os.path.join(test_folder, img_name)
#     if not os.path.exists(dst):
#         shutil.copy(src, dst)

# print("dataset copied and split into /kaggle/working/glassesdata/train and /kaggle/working/glassesdata/test")

# csv_path = "/kaggle/input/datasets/anvik029/correct-train/train_1.csv"
# if os.path.exists(csv_path):
#     df = pd.read_csv(csv_path)

#     glasses_folder = os.path.join(train_folder, "glasses")
#     no_glasses_folder = os.path.join(train_folder, "no_glasses")
#     os.makedirs(glasses_folder, exist_ok=True)
#     os.makedirs(no_glasses_folder, exist_ok=True)

#     for idx, row in df.iterrows():
#         img_name = f"{int(row['id'])}.png"  
#         src = os.path.join(train_folder, img_name)

#         if not os.path.exists(src):
#             print(f"Image {img_name} not found, skipping...")
#             continue

#         dst = glasses_folder if row['glasses'] == 1 else no_glasses_folder
#         shutil.move(src, os.path.join(dst, img_name))

#     print("train folder segregated into glasses and no_glasses")

# base_path = "/kaggle/working/glassesdata/"
# for root, dirs, files in os.walk(base_path):
#     level = root.replace(base_path, '').count(os.sep)
#     indent = ' ' * 2 * level
#     print(f"{indent}{os.path.basename(root)}/")
#     subindent = ' ' * 2 * (level + 1)
#     for f in files[:3]:  # show only 3 files for preview
#         print(f"{subindent}{f}")


CSV_PATH = "data/processed/train.csv"
IMG_DIR = "data/raw/faces-spring-2020/faces-spring-2020"

# --- Data Loading ---
transformations = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
])

def adaptive_augment(image, p=0.2):
    # augmentations with probability p
    if rd.random() < p: image = TF.hflip(image)
    if rd.random() < p: image = TF.adjust_brightness(image, 0.8 + 0.4*rd.random())
    if rd.random() < p: image = TF.adjust_contrast(image, 0.8 + 0.4*rd.random())
    return image

def main():
    dataset = GlassesDataset(CSV_PATH, IMG_DIR, transform=transformations)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    G = Generator(z_dim=z_dim).to(device)
    E = Discriminator().to(device)

    if torch.cuda.device_count() > 1:
        G = nn.DataParallel(G)
        E = nn.DataParallel(E)
    
    scaler = torch.amp.GradScaler(device)

    opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
    opt_E = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))

    ema = EMA(G.module if isinstance(G, nn.DataParallel) else G, decay=0.999)
    
    fixed_noise = torch.randn(6, z_dim, 1, 1, device=device)

    def smooth_labels(labels, smoothing=0.1):
        return labels * (1.0 - smoothing) + 0.5 * smoothing

    def train_model(epochs=50):
        G_losses, E_losses = [], []
        # --- Training Loop ---
        for epoch in range(epochs):
            loop = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch+1}/{epochs}")

            for i, (imgs, _) in loop:
                imgs = imgs.to(device)
                bs = imgs.size(0)
                
                # energy discriminator
                aug_imgs = adaptive_augment(imgs) 
                
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    z = torch.randn(bs, z_dim, 1, 1, device=device)
                    fake_imgs = G(z)
                    
                    energy_real = E(aug_imgs)
                    energy_fake = E(fake_imgs.detach())

                    e_loss = energy_real.mean() - energy_fake.mean()
                    E_losses.append(e_loss.item())

                opt_E.zero_grad()
                scaler.scale(e_loss).backward()
                scaler.step(opt_E)
                scaler.update()

                # generator
                with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
                    z_g = torch.randn(bs, z_dim, 1, 1, device=device)   
                    gen_imgs = G(z_g)                                  
                    
                    # freeze E for generator update
                    E.eval()
                    energy_fake = E(gen_imgs)           
                    E.train()

                    g_loss = energy_fake.mean()
                    G_losses.append(g_loss.item())
                
                opt_G.zero_grad()
                scaler.scale(g_loss).backward()
                scaler.step(opt_G)
                scaler.update()
                
                ema.update(G.module if isinstance(G, nn.DataParallel) else G)
        
        return G_losses, E_losses

    epochs = 50

    save_path = "/kaggle/working/GAN_Checkpoints"
    os.makedirs(save_path, exist_ok=True)
    G_losses, E_losses = train_model(epochs)
    torch.save({
        "epoch": epochs,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": E.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_E_state_dict": opt_E.state_dict(),
        "scaler": scaler.state_dict(),
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
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(E_losses, label="E Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Energy Discriminator Loss Curve")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    plt.plot(G_losses, label="Generator Loss")
    plt.plot(E_losses, label="Energy Discriminator Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("GAN Loss (G vs E)")
    plt.legend()
    plt.show()

    plt.figure(figsize=(8,4))
    #plt.plot(D_real_acc, label="Real Accuracy")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Energy Discriminator Accuracy Over Time")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()