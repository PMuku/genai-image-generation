import torch
import torch.nn as nn
import torch.nn.functional as F

z_dim = 128  # latent vector size
class Generator(nn.Module):
    def __init__(self, z_dim=128, img_channels=3, feature_g=32):  
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, feature_g*8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_g*8),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*8, feature_g*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_g*4),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*4, feature_g*2, 4, 2, 1, bias=False), 
            nn.BatchNorm2d(feature_g*2),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g*2, feature_g, 4, 2, 1, bias=False),  
            nn.BatchNorm2d(feature_g),
            nn.ReLU(True),

            nn.ConvTranspose2d(feature_g, img_channels, 4, 2, 1, bias=False),  
            nn.Tanh()
        )

    def forward(self, z):
        return self.net(z)
    
    # def forward(self, z, return_activations=False):
    #     activations = []
    #     x = z
    #     for layer in self.net:
    #         x = layer(x)
    #         if isinstance(layer, nn.BatchNorm2d):
    #             activations.append(x) 
    #     if return_activations:
    #         return x, activations
    #     return x

class EnergyModel(nn.Module):
    def __init__(self, img_channels=3, feature_d=32, n_experts=256):
        super().__init__()
        # feature extractor f_φ — same conv stack as the old Discriminator
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(img_channels, feature_d, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d, feature_d*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*2, feature_d*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(feature_d*4, feature_d*8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_d*8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # n_experts linear projections of the feature space (W_i, b_i in Eq. 11)
        # input: feature_d*8 maps × 4×4 spatial = 4096 for feature_d=32
        # self.experts = nn.Linear(feature_d * 8 * 4 * 4, n_experts)

    def forward(self, x):
        # feature extractor with flattening
        # feat = self.feature_extractor(x).view(x.size(0), -1)

        # # product-of-experts term: -Σ log(1 + exp(W_i^T f(x) + b_i))  [Eq. 11]
        # poe = -F.softplus(self.experts(feat)).sum(dim=1, keepdim=True)
        
        # quadratic term: ½ x^T x  (σ²=1 for inputs normalised to [-1, 1])  [Eq. 11]
        # quad = 0.001 * x.view(x.size(0), -1).pow(2).sum(dim=1, keepdim=True)

        # feature extractor without flattening
        feat = self.feature_extractor(x)

        # product-of-experts term: -Σ log(1 + exp(W_i^T f(x) + b_i))  [Eq. 11]
        poe = -F.softplus(feat).sum(dim=[1,2,3], keepdim=True)

        # quadratic term: ½ x^T x  (σ²=1 for inputs normalised to [-1, 1])  [Eq. 11]
        quad = 0.001 * x.pow(2).sum(dim=[1,2,3], keepdim=True)

        return quad + poe  # scalar energy per image; lower = more real

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v * (1.0 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)
