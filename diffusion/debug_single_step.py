import torch
from torch.utils.data import DataLoader
import os, sys

# ---------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from scripts.mask_generator import MaskGenerator
from diffusion_loss import DiffusionLossPerImage
from unet_diffusion import UNetDiffusion, NoiseScheduler

# ---------------------------------------------------------------------
# Device setup
# ---------------------------------------------------------------------
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {device}")

# ---------------------------------------------------------------------
# Configuration and dataset
# ---------------------------------------------------------------------
config = Config()
noise_cfg = NoiseConfig()

dataset = CelebADataset(
    root_dir=config.data.data_path,
    split='train',
    image_size=config.data.image_size,
    download=False,
)
loader = DataLoader(dataset, batch_size=1, shuffle=True)

train_mask_gen = MaskGenerator.for_train(config.mask)
loss_fn = DiffusionLossPerImage()

# ---------------------------------------------------------------------
# Noise scheduler and model
# ---------------------------------------------------------------------
noise_sched = NoiseScheduler(
    num_timesteps=noise_cfg.num_timesteps,
    beta_start=noise_cfg.beta_start,
    beta_end=noise_cfg.beta_end,
    schedule_type=noise_cfg.schedule_type,
).to(device)

model = UNetDiffusion(
    input_channels=config.model.input_channels,
    hidden_dims=config.model.hidden_dims,
    use_attention=config.model.use_attention,
    use_skip_connections=config.model.use_skip_connections,
    pretrained_encoder=config.model.pretrained_encoder,
    encoder_checkpoint=config.model.encoder_checkpoint,
    freeze_encoder_stages=config.model.freeze_encoder_stages,
    input_size=config.data.image_size,
).to(device)

# ---------------------------------------------------------------------
# Single-batch sanity test
# ---------------------------------------------------------------------
batch = next(iter(loader))
x = batch["image"].to(device)
mask = train_mask_gen.generate((x.size(0), 1, x.size(2), x.size(3))).to(device)
t = torch.randint(0, noise_sched.num_timesteps, (x.size(0),), device=device)

# Forward diffusion + prediction
x_noisy, noise = noise_sched.add_noise(x, t, mask)
pred = model(x_noisy, t, mask)
loss = loss_fn(pred, noise, mask)

# ---------------------------------------------------------------------
# Print checks
# ---------------------------------------------------------------------
print("\n=== Forward Pass Sanity Check ===")
print(f"Predicted noise shape : {tuple(pred.shape)}")
print(f"True noise shape       : {tuple(noise.shape)}")
print(f"Mask mean (0–1)        : {mask.mean().item():.3f}")
print(f"Loss value             : {loss.item():.6f}")

# ---------------------------------------------------------------------
# Gradient update test
# ---------------------------------------------------------------------
print("\n=== Gradient Update Test ===")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
optimizer.zero_grad()
loss.backward()

grad_norm = 0.0
for p in model.parameters():
    if p.grad is not None:
        grad_norm += p.grad.data.norm(2).item()

optimizer.step()
print(f"Gradient norm (L2): {grad_norm:.4f}")
print("✅ One optimization step completed successfully.")

print("\nIf loss is finite and gradient norm is > 0, your pipeline is fully consistent.")
