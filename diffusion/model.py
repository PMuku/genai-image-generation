import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def noise_schedule(T):
    beta = torch.linspace(1e-4, 0.02, T)
    alpha = 1.0 - beta
    alpha_bar = torch.cumprod(alpha, dim=0)
    return beta, alpha, alpha_bar


def forward_diffusion(x, t, alpha_bar, noise=None):
    if noise is None:
        noise = torch.randn_like(x)

    alpha_bar_at_t = alpha_bar[t].view(-1, 1, 1, 1)
    x_t = torch.sqrt(alpha_bar_at_t) * x + torch.sqrt(1.0 - alpha_bar_at_t) * noise
    return x_t, noise


@torch.no_grad()
def generate(model, shape, timesteps, beta, alpha, alpha_bar, device):
    sample = torch.randn(shape, device=device)

    for current_step in reversed(range(timesteps)):
        t_tensor = torch.full((shape[0],), current_step, device=device, dtype=torch.long)
        predicted_noise = model(sample, t_tensor)

        alpha_at_t = alpha[current_step]
        alpha_bar_at_t = alpha_bar[current_step]
        beta_at_t = beta[current_step]

        sample = (1.0 / torch.sqrt(alpha_at_t)) * (
            sample - (beta_at_t / torch.sqrt(1.0 - alpha_bar_at_t)) * predicted_noise
        )

        if current_step > 0:
            random_noise = torch.randn_like(sample)
            sample = sample + torch.sqrt(beta_at_t) * random_noise

    return sample


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Block(nn.Module):
    def __init__(self, img_ch, output_channels, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(img_ch, output_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(output_channels)
        self.batch_norm2 = nn.BatchNorm2d(output_channels)
        self.time_proj = nn.Linear(time_emb_dim, output_channels)

    def forward(self, feature_map, time_emb):
        hidden = F.relu(self.batch_norm1(self.conv1(feature_map)))
        hidden = hidden + self.time_proj(time_emb)[:, :, None, None]
        hidden = F.relu(self.batch_norm2(self.conv2(hidden)))
        return hidden


class UNet(nn.Module):
    """architecture consists of 3 encoder blocks sizing down to a bottleneck and sizing back up 3 decoder blocks."""
    def __init__(self, img_ch=3, base_ch=64, time_emb_dim=256):
        super().__init__()

        # time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # encoder
        self.encoder_block1 = Block(img_ch, base_ch, time_emb_dim)
        self.encoder_block2 = Block(base_ch, base_ch * 2, time_emb_dim)
        self.encoder_block3 = Block(base_ch * 2, base_ch * 4, time_emb_dim)
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = Block(base_ch * 4, base_ch * 4, time_emb_dim)

        # decoder (input channels doubled for skip connections)
        self.upsample3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=2, stride=2)
        self.decoder_block3 = Block(base_ch * 8, base_ch * 2, time_emb_dim)
        self.upsample2 = nn.ConvTranspose2d(base_ch * 2, base_ch * 2, kernel_size=2, stride=2)
        self.decoder_block2 = Block(base_ch * 4, base_ch, time_emb_dim)
        self.upsample1 = nn.ConvTranspose2d(base_ch, base_ch, kernel_size=2, stride=2)
        self.decoder_block1 = Block(base_ch * 2, base_ch, time_emb_dim)

        self.output_conv = nn.Conv2d(base_ch, img_ch, kernel_size=1)

    def forward(self, image, t):
        time_emb = self.time_mlp(self.time_embedding(t))

        e1 = self.encoder_block1(image, time_emb)
        e2 = self.encoder_block2(self.pool(e1), time_emb)
        e3 = self.encoder_block3(self.pool(e2), time_emb)

        bottleneck = self.bottleneck(self.pool(e3), time_emb)

        d3 = self.upsample3(bottleneck)
        d3 = self.decoder_block3(torch.cat([d3, e3], dim=1), time_emb)
        d2 = self.upsample2(d3)
        d2 = self.decoder_block2(torch.cat([d2, e2], dim=1), time_emb)
        d1 = self.upsample1(d2)
        d1 = self.decoder_block1(torch.cat([d1, e1], dim=1), time_emb)

        return self.output_conv(d1)
