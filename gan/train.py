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
import kagglehub

from model import Generator, Discriminator, EMA, z_dim

# --- Download & Extract Data ---
path = kagglehub.dataset_download("jeffheaton/glasses-or-no-glasses")
print("Path:", path)

src_img_path = "/kaggle/input/datasets/jeffheaton/glasses-or-no-glasses/faces-spring-2020/faces-spring-2020"

dst_base = "/kaggle/working/glassesdata/"
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
        if not os.path.exists(dst):  # avoid recopying if already done
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

print("dataset copied and split into /kaggle/working/glassesdata/train and /kaggle/working/glassesdata/test")

csv_path = "/kaggle/input/datasets/anvik029/correct-train/train_1.csv"
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

base_path = "/kaggle/working/glassesdata/"
for root, dirs, files in os.walk(base_path):
    level = root.replace(base_path, '').count(os.sep)
    indent = ' ' * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    subindent = ' ' * 2 * (level + 1)
    for f in files[:3]:  # show only 3 files for preview
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
train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4,pin_memory=True)

print("Classes:", train_data.classes)

def adaptive_augment(image, p=0.2):
    # augmentations with probability p
    if random.random() < p: image = TF.hflip(image)
    if random.random() < p: image = TF.adjust_brightness(image, 0.8 + 0.4*random.random())
    if random.random() < p: image = TF.adjust_contrast(image, 0.8 + 0.4*random.random())
    return image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Initialization ---
G = Generator(z_dim=z_dim).to(device)
D = Discriminator().to(device)

if torch.cuda.device_count() > 1:
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)

scaler = torch.amp.GradScaler(device)
criterion = nn.BCEWithLogitsLoss()
opt_G = optim.Adam(G.parameters(), lr=0.0002, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=0.0002, betas=(0.5, 0.999))

epochs = 50
fixed_noise = torch.randn(6, z_dim, 1, 1, device=device)

save_path = "/kaggle/working/GAN_Checkpoints"
os.makedirs(save_path, exist_ok=True)

def smooth_labels(labels, smoothing=0.1):
    return labels * (1.0 - smoothing) + 0.5 * smoothing

ema = EMA(G.module if isinstance(G, nn.DataParallel) else G, decay=0.999)

G_losses = []
D_losses = []
D_real_acc = []
D_fake_acc = []

# --- Training Loop ---
for epoch in range(epochs):
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Epoch {epoch+1}/{epochs}")

    for i, (imgs, _) in loop:
        imgs = imgs.to(device)
        bs = imgs.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)
        real_labels = smooth_labels(real_labels, smoothing=0.05)
        fake_labels = smooth_labels(fake_labels, smoothing=0.05)
        
        # discriminator
        aug_imgs = adaptive_augment(imgs) 
        
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            real_out = D(aug_imgs)
            d_loss_real = criterion(real_out, real_labels)
            
            z = torch.randn(bs, z_dim, 1, 1, device=device)
            fake_imgs = G(z)
            fake_out = D(fake_imgs.detach())
            d_loss_fake = criterion(fake_out, fake_labels)

            d_loss = d_loss_real + d_loss_fake
            D_losses.append(d_loss.item())
            
            with torch.no_grad():
                real_pred = torch.sigmoid(real_out)
                fake_pred = torch.sigmoid(fake_out)
            
                real_acc = (real_pred > 0.5).float().mean().item()
                fake_acc = (fake_pred < 0.5).float().mean().item()
            
            D_real_acc.append(real_acc)
            D_fake_acc.append(fake_acc)

        opt_D.zero_grad()
        scaler.scale(d_loss).backward()
        scaler.step(opt_D)
        scaler.update()

        with torch.amp.autocast(device_type=device.type, dtype=torch.float16):
            z_g = torch.randn(bs, z_dim, 1, 1, device=device)   
            gen_imgs = G(z_g)                                  
            fake_out_for_g = D(gen_imgs)           
            g_loss = criterion(fake_out_for_g, real_labels)
            G_losses.append(g_loss.item())
        
        opt_G.zero_grad()
        scaler.scale(g_loss).backward()
        scaler.step(opt_G)
        scaler.update()
        
        ema.update(G.module if isinstance(G, nn.DataParallel) else G)

    torch.save({
        "epoch": epoch+1,
        "generator_state_dict": G.state_dict(),
        "discriminator_state_dict": D.state_dict(),
        "opt_G_state_dict": opt_G.state_dict(),
        "opt_D_state_dict": opt_D.state_dict(),
        "scaler": scaler.state_dict()
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
        plt.show()

torch.save({
    "epoch": epochs,
    "generator_state_dict": G.state_dict(),
    "discriminator_state_dict": D.state_dict(),
    "opt_G_state_dict": opt_G.state_dict(),
    "opt_D_state_dict": opt_D.state_dict(),
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
plt.plot(D_losses, label="D Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Discriminator Loss Curve")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(G_losses, label="Generator Loss")
plt.plot(D_losses, label="Discriminator Loss")
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("GAN Loss (G vs D)")
plt.legend()
plt.show()

plt.figure(figsize=(8,4))
plt.plot(D_real_acc, label="Real Accuracy")
plt.xlabel("Iterations")
plt.ylabel("Accuracy")
plt.title("Discriminator Accuracy Over Time")
plt.legend()
plt.show()
