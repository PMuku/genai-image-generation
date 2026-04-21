import torch
import torch.nn as nn

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

class Discriminator(nn.Module):
    def __init__(self, img_channels=3, feature_d=32):
        super().__init__()
        self.net = nn.Sequential(
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

            nn.Conv2d(feature_d*8, 1, 4, 1, 0, bias=False)             
        )

    def forward(self, img):
        return self.net(img).view(-1, 1) 

class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}

    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k] = self.shadow[k] * self.decay + v * (1.0 - self.decay)

    def apply(self, model):
        model.load_state_dict(self.shadow)
