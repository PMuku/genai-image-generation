# EBM-GAN Implementation Review

Based on: *Deep Directed Generative Models with Energy-Based Probability Estimation* — Kim & Bengio, 2016 (arXiv:1606.03439)

---

## Core Idea

The standard GAN `Discriminator` is a binary classifier trained with BCE to output a probability of real/fake. This paper replaces it with an **Energy Model** `E_Θ(x)` that outputs a scalar energy value where **low energy = real, high energy = fake**. The generator is trained to push its outputs toward low-energy regions rather than to fool a classifier.

A key advantage over standard GAN: the discriminator converges to `D(x) = 0.5` everywhere as training succeeds — it becomes uninformative. The energy function does not degenerate this way; as G improves, E_Θ's gradient becomes constant rather than its output, so it retains signal throughout training (Sec. 3.3).

---

## Changes to `model.py`

### Added import
```python
import torch.nn.functional as F
```

### Replaced `Discriminator` with `EnergyModel`

The old `Discriminator` ended with `Conv2d(feature_d*8, 1, 4, 1, 0)` to produce a single logit. The `EnergyModel` keeps the identical 4-layer convolutional feature extractor `f_φ`, then computes a scalar energy via the **product-of-experts** formula from Eq. 11:

```
E_Θ(x) = (1/σ²) xᵀx  −  Σᵢ log(1 + exp(Wᵢᵀ f_φ(x) + bᵢ))
          └─ quadratic ─┘  └──────── product of experts ────────┘
```

- **Quadratic term** `½ xᵀx`: always positive; grows with pixel magnitude; pushes energy up
- **Product of experts** `−Σ softplus(Wᵢᵀ f_φ(x) + bᵢ)`: experts fire on real image features, pulling energy down; generated images produce weaker expert responses so the quadratic term dominates → higher energy

The `Conv2d(256, 1, 4, 1, 0)` final layer is replaced by `nn.Linear(feature_d*8 * 4*4, n_experts)` (4096 → 256 for the default `feature_d=32`).

```python
class EnergyModel(nn.Module):
    def __init__(self, img_channels=3, feature_d=32, n_experts=256):
        super().__init__()
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
        self.experts = nn.Linear(feature_d * 8 * 4 * 4, n_experts)

    def forward(self, x):
        feat = self.feature_extractor(x).view(x.size(0), -1)
        poe  = -F.softplus(self.experts(feat)).sum(dim=1, keepdim=True)
        quad = 0.5 * x.view(x.size(0), -1).pow(2).sum(dim=1, keepdim=True)
        return quad + poe  # scalar energy; lower = more real
```

---

## Changes to `train.py`

### Import
```python
# before
from model import Generator, Discriminator, EMA, z_dim
# after
from model import Generator, EnergyModel, EMA, z_dim
```

### Initialisation

Removed: `D = Discriminator()`, `criterion = nn.BCEWithLogitsLoss()`, `opt_D`, `smooth_labels()`, `real_labels`, `fake_labels`, `D_losses`, `D_real_acc`, `D_fake_acc`

Added:
```python
E = EnergyModel().to(device)
opt_E = optim.Adam(E.parameters(), lr=0.0002, betas=(0.5, 0.999))
lambda_ent = 0.1          # entropy regularisation weight (Eq. 15)
E_losses   = []
E_real_vals = []
E_fake_vals = []
```

### Energy model step (replaces discriminator step)

**Old** — BCE contrastive classification:
```python
d_loss = criterion(D(real), real_labels) + criterion(D(fake), fake_labels)
```

**New** — contrastive divergence from Eq. 7:
```python
# positive phase: lower energy on real data
# negative phase: raise energy on generated data
with torch.no_grad():
    fake_imgs = G(torch.randn(bs, z_dim, 1, 1, device=device))

real_energy = E(aug_imgs)
fake_energy = E(fake_imgs)
e_loss = real_energy.mean() - fake_energy.mean()

opt_E.zero_grad()
e_loss.backward()
opt_E.step()
```

G is frozen (`torch.no_grad()`) during this step so gradients only update E.

### Generator step (replaces BCE fooling trick)

**Old** — fool discriminator into predicting real:
```python
g_loss = criterion(D(fake), real_labels)
```

**New** — minimise KL(P_φ || P_Θ) from Eq. 13–14, with entropy regularisation from Eq. 15:

```python
gen_imgs = G(torch.randn(bs, z_dim, 1, 1, device=device))
gen_energy = E(gen_imgs)   # gradient flows back through E into G

# Eq. 15: H(P_φ) ≈ Σ log(γ_aᵢ) over all BN scale params in G
entropy_reg = sum(
    torch.log(m.weight.abs() + 1e-8).sum()
    for m in G.modules() if isinstance(m, nn.BatchNorm2d)
)
g_loss = gen_energy.mean() - lambda_ent * entropy_reg

opt_G.zero_grad()
g_loss.backward()
opt_G.step()
```

Without `entropy_reg`, the generator collapses to the single deepest energy minimum (mode collapse). Maximising the log of BN scale parameters encourages diverse activations throughout G, approximating entropy maximisation.

### Checkpoint keys
```python
# before
"discriminator_state_dict": D.state_dict()
"opt_D_state_dict": opt_D.state_dict()

# after
"energy_model_state_dict": E.state_dict()
"opt_E_state_dict": opt_E.state_dict()
```

### Loss plots
| Old plot | New plot |
|---|---|
| `D Loss` (BCE) | `Energy Model Loss` (E_real − E_fake) |
| `GAN Loss (G vs D)` | `EBM-GAN Loss (G vs E)` |
| `Discriminator Accuracy` | `Energy Values Over Time` (real vs fake mean energy — real should stay lower) |

---

## What stays the same

| Component | Status |
|---|---|
| Generator architecture | Unchanged |
| EMA on Generator | Unchanged |
| Optimizer (Adam, lr=0.0002, betas=(0.5, 0.999)) | Unchanged |
| DataLoader, transforms, augmentation | Unchanged |
| MPS/CUDA/CPU device detection | Unchanged |
| Sample image saving every 2 epochs | Unchanged |

---

## Data pipeline changes (earlier session)

### Corrected labels (`data/processed/`)
- `corrections/flipped_1-1500.txt`, `flipped_1501-3000.txt`, `flipped_3001-4500.txt`: 484 total IDs with flipped `glasses` labels in the raw CSV — applied by inverting those rows in `data/raw/train.csv` → `data/processed/train.csv`
- `corrections/test_labels.txt`: 500 ground-truth labels for test images (IDs 4501–5000) — attached to `data/raw/test.csv` → `data/processed/test.csv`
- `train.py` reads labels from `data/processed/train.csv` (not raw)

### M4/MPS compatibility (earlier session)
- Device: MPS → CUDA → CPU priority order; `use_amp` only true on CUDA
- `GradScaler` and `autocast` gated behind `use_amp`
- `DataParallel` and `.module` unwrapping removed (MPS is single-device)
- `num_workers=0`, `pin_memory=False` for MPS DataLoader compatibility
- All `plt.show()` replaced with `plt.savefig()` + `plt.close()`
- All Kaggle paths replaced with local paths via `os.path.dirname(__file__)`
