import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy import linalg
from scipy.stats import entropy

from model import Generator, EMA, z_dim

device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_path = "checkpoints"

import os
ckpt_path = f"/kaggle/working/GAN_Checkpoints/GAN_final.pth"

if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location=device)

    G_ema = Generator(z_dim=z_dim).to(device)
    G_ema.load_state_dict(ckpt["ema_shadow"], strict=False)
    G_ema.eval()

    with torch.no_grad():
        sample_noise = torch.randn(6, z_dim, 1, 1, device=device)
        generated    = G_ema(sample_noise).detach().cpu()

    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    for idx, ax in enumerate(axes.flat):
        img = generated[idx]
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
        ax.axis("off")
        ax.set_title(f"Sample {idx+1}", fontsize=10)

    plt.suptitle("Generated Images", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.show()

    # --- Evaluation ---
    N_SAMPLES   = 4000   
    BATCH_SIZE  = 64
    NUM_WORKERS = 4

    G_eval = G_ema

    #generate fake images
    print(f"Generating {N_SAMPLES} fake images...")
    all_fake = []
    with torch.no_grad():
        while sum(t.shape[0] for t in all_fake) < N_SAMPLES:
            z = torch.randn(BATCH_SIZE, z_dim, 1, 1, device=device)
            imgs = G_eval(z).cpu()
            all_fake.append(imgs)
    fake_tensor = torch.cat(all_fake, dim=0)[:N_SAMPLES] 
    print(f"Fake images shape: {fake_tensor.shape}")

    #generate real images
    print(f"Loading {N_SAMPLES} real images...")
    real_transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])

    trainfolder="/kaggle/working/glassesdata/train"
    if os.path.exists(trainfolder):
        real_dataset = datasets.ImageFolder(trainfolder, transform=real_transform)
        real_loader  = DataLoader(real_dataset, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=NUM_WORKERS)
        all_real = []
        with torch.no_grad():
            for imgs, _ in real_loader:
                all_real.append(imgs)
                if sum(t.shape[0] for t in all_real) >= N_SAMPLES:
                    break
        real_tensor = torch.cat(all_real, dim=0)[:N_SAMPLES]
        print(f"Real images shape: {real_tensor.shape}")

        print("Loading Inception-v3...")
        weights = Inception_V3_Weights.DEFAULT

        inception_feat = inception_v3(weights=weights, transform_input=False)
        inception_feat.fc = nn.Identity()        
        inception_feat.aux_logits = False
        inception_feat = inception_feat.to(device).eval()

        inception_cls = inception_v3(weights=weights, transform_input=False)
        inception_cls.aux_logits = False
        inception_cls = inception_cls.to(device).eval()
        print("Inception-v3 loaded")

        @torch.no_grad()
        def get_features(img_tensor, model, batch_size=32):
            """Extract 2048-d pool3 features for FID."""
            feats = []
            for i in range(0, img_tensor.shape[0], batch_size):
                batch = img_tensor[i : i + batch_size].to(device)
                # Inception-v3 expects 299×299 input
                batch = nn.functional.interpolate(
                    batch, size=(299, 299), mode="bilinear", align_corners=False
                )
                feats.append(model(batch).cpu().numpy())
            return np.concatenate(feats, axis=0)

        print("Extracting Inception features (real)...")
        real_feats = get_features(real_tensor, inception_feat)
        print("Extracting Inception features (fake)...")
        fake_feats = get_features(fake_tensor, inception_feat)

        #fid compute
        def compute_fid(real_f, fake_f):
            mu_r, sigma_r = real_f.mean(0), np.cov(real_f, rowvar=False)
            mu_f, sigma_f = fake_f.mean(0), np.cov(fake_f, rowvar=False)
            diff = mu_r - mu_f
            covmean, _ = linalg.sqrtm(sigma_r @ sigma_f, disp=False)
            if np.iscomplexobj(covmean):
                covmean = covmean.real
            return float(diff @ diff + np.trace(sigma_r + sigma_f - 2 * covmean))

        fid = compute_fid(real_feats, fake_feats)

        #is compute
        @torch.no_grad()
        def compute_is(img_tensor, model, batch_size=32, splits=10):
            preds = []
            for i in range(0, img_tensor.shape[0], batch_size):
                batch = img_tensor[i : i + batch_size].to(device)
                batch = nn.functional.interpolate(
                    batch, size=(299, 299), mode="bilinear", align_corners=False
                )
                logits = model(batch)                          
                preds.append(torch.softmax(logits, dim=1).cpu().numpy())
            preds = np.concatenate(preds, axis=0)               

            chunk = preds.shape[0] // splits
            scores = []
            for k in range(splits):
                part = preds[k * chunk : (k + 1) * chunk]
                py   = part.mean(axis=0)                     
                scores.append(np.exp(np.mean([entropy(p, py) for p in part])))
            return float(np.mean(scores)), float(np.std(scores))

        is_mean, is_std = compute_is(fake_tensor, inception_cls)

        print("\n" + "="*45)
        print(f"  Image Generation Metrics")
        print("="*45)
        print(f"  FID  = {fid:.2f}   (lower is better)")
        print(f"  IS   = {is_mean:.3f} ± {is_std:.3f}  (higher is better)")
        print("="*45)
        print(f"\n  Evaluated on {N_SAMPLES} real + {N_SAMPLES} fake images")
    else:
        print("Could not find train folder for computing FID on real images.")
else:
    print(f"Could not find checkpoint at {ckpt_path}")
