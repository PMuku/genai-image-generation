import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        encoder_channels = [
            max(hidden_dim // 8, 32),
            max(hidden_dim // 4, 32),
            max(hidden_dim // 2, 32),
            hidden_dim,
        ]
        self.img_dim = int((input_dim // 3) ** 0.5) # assuming input_dim is C*H*W and C=3
        self.encoder_output_channels = encoder_channels[-1]
        self.encoder_output_spatial = self.img_dim // (2 ** len(encoder_channels)) # each conv layer halves spatial dimensions
        # flattening
        self.encoder_output_dim = self.encoder_output_channels * self.encoder_output_spatial * self.encoder_output_spatial

        self.encoder = nn.Sequential(
            nn.Conv2d(4, encoder_channels[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(encoder_channels[1], encoder_channels[2], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(encoder_channels[2], encoder_channels[3], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
        )

        self.fc_mu = nn.Linear(self.encoder_output_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_output_dim, latent_dim)

        # label embedding for conditioning
        self.label_embed = nn.Embedding(2, latent_dim)
        # flattened input to decoder
        self.fc_decode = nn.Linear(latent_dim, self.encoder_output_dim)

        self.decoder_layers = [nn.ConvTranspose2d(encoder_channels[3], encoder_channels[2], 4, 2, 1),
                            nn.ConvTranspose2d(encoder_channels[2], encoder_channels[1], 4, 2, 1),
                            nn.ConvTranspose2d(encoder_channels[1], encoder_channels[0], 4, 2, 1),
                            nn.ConvTranspose2d(encoder_channels[0], 3, 4, 2, 1)]
        self.act = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout2d(0.2)

    # B -> B, 1 standard torch scalar dimension
    def _prepare_condition(self, labels, batch_size, device):
        if labels is None:
            return torch.zeros(batch_size, 1, device=device)

        if labels.dim() == 1:
            labels = labels.unsqueeze(1)

        return labels.float().to(device)

    # expand label to match spatial dims of image features for concat
    # B, 1 -> B, 1, 1, 1 -> B, 1, 64, 64
    def _expand_condition(self, labels, height, width):
        return labels.unsqueeze(-1).unsqueeze(-1).expand(-1, 1, height, width)

    def encode(self, x, labels):
        labels = self._prepare_condition(labels, x.size(0), x.device)
        labels_map = self._expand_condition(labels, x.size(2), x.size(3))

        encoded = self.encoder(torch.cat([x, labels_map], dim=1))
        encoded = torch.flatten(encoded, start_dim=1)
        
        mu = self.fc_mu(encoded)
        logvar = self.fc_logvar(encoded)
        return mu, logvar

    def decode(self, z, labels):
        labels = self._prepare_condition(labels, z.size(0), z.device)
        decoded = self.fc_decode(torch.cat([z, labels], dim=1))
        # unflatten and reshape
        decoded = decoded.view(z.size(0), self.encoder_output_channels, self.encoder_output_spatial, self.encoder_output_spatial)
        decoded = self.decoder(decoded)
        return decoded

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar) # standard deviation
        eps = torch.randn_like(std) # random noise
        return mu + eps * std # reparameterization trick

    def forward(self, x, labels=None):
        mu, logvar = self.encode(x, labels)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, labels)
        return recon_x, mu, logvar

    def loss_function(self, recon_x, x, mu, logvar, beta = 1.0):
        recon_loss = 0.8 * F.l1_loss(recon_x, x) + 0.2 * F.mse_loss(recon_x, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        # print(f"Reconstruction Loss: {recon_loss.item():.4f}, KL Divergence: {kl_loss.item():.4f}")
        return recon_loss + beta * kl_loss
