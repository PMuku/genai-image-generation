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
        assert dim % 2 == 0, f"time_emb_dim must be even, got {dim}"
        self.dim = dim

    def forward(self, t):
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None].float() * emb[None, :]
        return torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(num_groups, out_ch)
        self.norm2 = nn.GroupNorm(num_groups, out_ch)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.residual = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, feature_map, time_emb):
        hidden = F.relu(self.norm1(self.conv1(feature_map)))
        hidden = hidden + self.time_proj(time_emb)[:, :, None, None]
        hidden = F.relu(self.norm2(self.conv2(hidden)))
        return hidden + self.residual(feature_map)


class UNet(nn.Module):
    def __init__(self, img_ch=3, base_ch=64, time_emb_dim=256, n_layers=3):
        super().__init__()

        # time embedding
        self.time_embedding = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # encoder
        # channel sizes: base_ch, base_ch*2, base_ch*4, ... base_ch * 2^(n_layers-1)
        enc_channels = [base_ch * (2 ** i) for i in range(n_layers)]
        self.encoders = nn.ModuleList()
        in_ch = img_ch
        for out_ch in enc_channels:
            self.encoders.append(Block(in_ch, out_ch, time_emb_dim))
            in_ch = out_ch
        self.pool = nn.MaxPool2d(2)

        # bottleneck
        self.bottleneck = Block(enc_channels[-1], enc_channels[-1], time_emb_dim)

        # decoder (input channels doubled for skip connections)
        self.upsamplers = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for i in reversed(range(n_layers)):
            up_in = enc_channels[i]
            self.upsamplers.append(nn.ConvTranspose2d(up_in, up_in, kernel_size=2, stride=2))
            dec_out = enc_channels[i - 1] if i > 0 else base_ch
            self.decoders.append(Block(up_in * 2, dec_out, time_emb_dim))

        self.output_conv = nn.Conv2d(base_ch, img_ch, kernel_size=1)

    def forward(self, image, t):
        time_emb = self.time_mlp(self.time_embedding(t))

        # encoder
        skips = []
        x = image
        for encoder in self.encoders:
            x = encoder(x, time_emb)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x, time_emb)

        # decoder
        for upsample, decoder, skip in zip(self.upsamplers, self.decoders, reversed(skips)):
            x = upsample(x)
            x = decoder(torch.cat([x, skip], dim=1), time_emb)

        return self.output_conv(x)
