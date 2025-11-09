# debug_noise_scheduler.py
import os
import torch
import torchvision
import torchvision.transforms as T
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from noise_scheduler_config import NoiseConfig
from unet_diffusion import NoiseScheduler
from scripts.mask_generator import MaskGenerator  # adjust path if needed

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMG_SIZE = 256
DEBUG_DIR = "debug_noise_scheduler"
os.makedirs(DEBUG_DIR, exist_ok=True)

# ---------------------------------------------------------------------
# LOAD TEST IMAGE
# ---------------------------------------------------------------------
transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),  # [-1,1] normalization
])

# Load one image (e.g., from CelebA or any folder)
dataset = torchvision.datasets.CelebA(
    root="./assets/datasets",
    split="test",
    transform=transform,
    download=False
)
image, _ = dataset[0]  # pick one
image = image.unsqueeze(0).to(DEVICE)  # [1,3,H,W]
print(f"[DEBUG] Loaded image shape: {image.shape}, range: {image.min():.3f}→{image.max():.3f}")

# ---------------------------------------------------------------------
# CREATE MASK
# ---------------------------------------------------------------------
mask_gen = MaskGenerator(mask_type='random', mask_ratio=0.4, min_size=32, max_size=128, deterministic=True)
mask = mask_gen.generate((1, 1, IMG_SIZE, IMG_SIZE)).to(DEVICE)
print(f"[DEBUG] Mask mean={mask.mean():.4f}, min={mask.min().item()}, max={mask.max().item()}")

# ---------------------------------------------------------------------
# INIT NOISE SCHEDULER
# ---------------------------------------------------------------------
scheduler = NoiseScheduler(num_timesteps=100, schedule_type='cosine').to(DEVICE)

# pick a timestep
t = torch.tensor([80], device=DEVICE, dtype=torch.long)
x_t, noise = scheduler.add_noise(image, t, mask)

# ---------------------------------------------------------------------
# DEBUG CHECKS
# ---------------------------------------------------------------------
with torch.no_grad():
    diff_unmasked = ((x_t - image) * (1 - mask)).abs().mean().item()
    diff_masked   = ((x_t - image) * mask).abs().mean().item()
    print(f"[DEBUG] Difference (unmasked): {diff_unmasked:.6f}  (should be ≈0)")
    print(f"[DEBUG] Difference (masked):   {diff_masked:.6f}  (should be >0)")

# ---------------------------------------------------------------------
# SAVE VISUALS
# ---------------------------------------------------------------------
def save_img(tensor, name):
    vutils.save_image(tensor, os.path.join(DEBUG_DIR, f"{name}.png"),
                      normalize=True, value_range=(-1,1))

save_img(image, "01_clean_image")
save_img(mask, "02_mask")
save_img(image * (1 - mask), "03_masked_image")
save_img(x_t, "04_noisy_image")
save_img((x_t - image) * mask, "05_diff_masked_only")

print(f"✅ Saved debug images to: {DEBUG_DIR}/")

# Optional: matplotlib grid
def show_side_by_side(*imgs, titles):
    fig, axs = plt.subplots(1, len(imgs), figsize=(15,5))
    for ax, img, title in zip(axs, imgs, titles):
        img = img.detach().cpu()
        img = (img + 1) / 2  # back to [0,1]
        ax.imshow(img.permute(1,2,0).clamp(0,1))
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()

show_side_by_side(
    image[0], (mask.repeat(1,3,1,1))[0], x_t[0],
    titles=["Original", "Mask", "After add_noise()"]
)
