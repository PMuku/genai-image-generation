import torch
import numpy as np
from scipy.linalg import sqrtm
from torchvision.models import inception_v3, Inception_V3_Weights
import torch.nn.functional as F


def get_inception_features(images, model, device):
    # Inception expects 299x299 RGB images normalized to [-1, 1]
    images = F.interpolate(images, size=(299, 299), mode="bilinear", align_corners=False)
    images = 2.0 * images - 1.0  # [0,1] -> [-1,1]
    images = images.to(device)

    with torch.no_grad():
        features = model(images)

    return features.cpu().numpy()


def build_inception_feature_extractor(device):
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT)
    # Remove the final fc layer to get 2048-d features
    model.fc = torch.nn.Identity()
    model.eval()
    model.to(device)
    return model


def compute_fid(real_images, generated_images, device):
    model = build_inception_feature_extractor(device)

    # Extract features
    real_feats = get_inception_features(real_images, model, device)
    gen_feats = get_inception_features(generated_images, model, device)

    # Compute mean and covariance
    mu_real, sigma_real = real_feats.mean(axis=0), np.cov(real_feats, rowvar=False)
    mu_gen, sigma_gen = gen_feats.mean(axis=0), np.cov(gen_feats, rowvar=False)

    # FID = ||mu_real - mu_gen||^2 + Tr(sigma_real + sigma_gen - 2*sqrt(sigma_real @ sigma_gen))
    diff = mu_real - mu_gen
    covmean, _ = sqrtm(sigma_real @ sigma_gen, disp=False)

    # Handle numerical instability: remove imaginary components
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff @ diff + np.trace(sigma_real + sigma_gen - 2.0 * covmean)
    return float(fid)
