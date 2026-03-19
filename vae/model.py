import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = None
        self.mu = None
        self.logvar = None # log of variance for wider range of values
        self.decoder = None

    def reparameterize(self, mu, logvar):
        pass

    def forward(self, x):
        pass

    def loss_function(self, recon_x, x, mu, logvar):
        pass
