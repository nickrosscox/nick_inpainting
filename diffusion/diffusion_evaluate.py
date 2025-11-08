# diffusion_evaluate.py
import torch
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import numpy as np
import torchvision.utils as vutils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from scripts.mask_generator import MaskGenerator
from evaluation.metrics import InpaintingMetrics


@torch.no_grad()
def sample_ddpm(model, scheduler, images, mask, num_timesteps=None):
    """
    Perform DDPM reverse sampling for inpainting.
    Model predicts noise ε_θ(x_t, t).
    
    Args:
        model: trained UNet predicting noise
        scheduler: NoiseScheduler instance
        images: clean original images [B, 3, H, W]
        mask: binary mask [B, 1, H, W] (1 = region to inpaint)
        num_timesteps: optional override for number of reverse steps
    Returns:
        x_0_pred: final denoised reconstruction [B, 3, H, W]
    """
    device = images.device
    B = images.size(0)
    T = scheduler.num_timesteps if num_timesteps is None else num_timesteps

    # start from pure noise in masked region, keep original in known region
    t_full = torch.full((B,), T - 1, device=device, dtype=torch.long)
    x_t, _ = scheduler.add_noise(images, t_full, mask)

    for step in reversed(range(T)):
        t = torch.full((B,), step, device=device, dtype=torch.long)

        # predict noise ε
        eps_pred = model(x_t, t, mask)

        # gather scalar coefficients
        beta_t       = scheduler.betas[t]              # [B]
        alpha_t      = scheduler.alphas[t]             # [B]
        alpha_bar_t  = scheduler.alpha_bars[t]         # [B]
        alpha_bar_prev = scheduler.alpha_bars_prev[t]  # [B]

        # posterior variance  (Eq. 7)
        posterior_var = ((1 - alpha_bar_prev) / (1 - alpha_bar_t)) * beta_t  # [B]

        # mean using ε-parameterization  (Eq. 12)
        mean = (1.0 / torch.sqrt(alpha_t))[:, None, None, None] * (
            x_t - (beta_t / torch.sqrt(1 - alpha_bar_t))[:, None, None, None] * eps_pred
        )

        # add noise except at t = 0
        if step > 0:
            noise = torch.randn_like(x_t)
            x_t = mean + torch.sqrt(posterior_var)[:, None, None, None] * noise
        else:
            x_t = mean

        # Re-inject known (unmasked) pixels every step  (RePaint trick)
        x_t = mask * x_t + (1 - mask) * images

    # clamp final output to valid range
    x_0_pred = x_t.clamp(-1, 1)
    return x_0_pred



def run_evaluation(model, test_loader, noise_scheduler, mask_generator, device, save_dir='results/diffusion'):
    """
    Run comprehensive evaluation on test set.
    """
    model.eval()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Initialize metrics calculator
    metrics_calc = InpaintingMetrics(device=device)
    print("✅ Metrics initialized (PSNR, LPIPS)")
    
    # Collect metrics
    all_psnr = []
    all_lpips = []
    all_mse = []
    all_mae = []
    
    print("\nStarting evaluation...")
    print(f"Computing: PSNR, LPIPS, MSE, MAE")
    
    for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
        images = batch['image'].to(device)
        filenames = batch['filename']
        
        B, C, H, W = images.shape
        
        # Generate deterministic masks
        masks = mask_generator.generate_for_filenames(
            filenames=filenames,
            shape=(1, H, W)
        ).to(device)
        
        # Add full noise to masked regions (start from complete noise)
        t = torch.full((B,), noise_scheduler.num_timesteps - 1, device=device)
        # noisy_images, _ = noise_scheduler.add_noise(images, t, masks)
        
        # Denoise using DDPM sampling
        inpainted = sample_ddpm(model, noise_scheduler, images, masks)
        
        # Compute all metrics on full images
        psnr_val = metrics_calc.psnr(inpainted, images)
        lpips_val = metrics_calc.lpips_distance(inpainted, images)
        
        # Compute MSE and MAE
        mse_val = torch.mean((inpainted - images) ** 2).item()
        mae_val = torch.mean(torch.abs(inpainted - images)).item()
        
        all_psnr.append(psnr_val)
        all_lpips.append(lpips_val)
        all_mse.append(mse_val)
        all_mae.append(mae_val)
        
        # Save first batch for visualization
        if batch_idx == 0:
            # Create masked input: show original with black mask region
            masked_input = images * (1 - masks)  # Zero out masked region
            
            comparison = torch.cat([
                images[:8],           # Row 1: Original
                masked_input[:8],     # Row 2: Input with black mask
                inpainted[:8]         # Row 3: Inpainted result
            ], dim=0)
            
            vutils.save_image(
                comparison,
                os.path.join(save_dir, 'samples.png'),
                nrow=8,
                normalize=True,
                value_range=(-1, 1)
            )
            print(f"\n✅ Saved sample images to {save_dir}/samples.png")
    
    # Compute summary statistics
    results = {
        'psnr_mean': np.mean(all_psnr),
        'psnr_std': np.std(all_psnr),
        'lpips_mean': np.mean(all_lpips),
        'lpips_std': np.std(all_lpips),
        'mse_mean': np.mean(all_mse),
        'mse_std': np.std(all_mse),
        'mae_mean': np.mean(all_mae),
        'mae_std': np.std(all_mae)
    }
    
    # Print results
    print("\n" + "="*60)
    print("DIFFUSION MODEL EVALUATION RESULTS")
    print("="*60)
    print(f"PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB")
    print(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}")
    print(f"MSE:   {results['mse_mean']:.6f} ± {results['mse_std']:.6f}")
    print(f"MAE:   {results['mae_mean']:.6f} ± {results['mae_std']:.6f}")
    print("="*60)
    
    # Save results
    results_file = os.path.join(save_dir, 'metrics.txt')
    with open(results_file, 'w') as f:
        f.write("="*60 + "\n")
        f.write("DIFFUSION MODEL EVALUATION RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"PSNR:  {results['psnr_mean']:.2f} ± {results['psnr_std']:.2f} dB\n")
        f.write(f"LPIPS: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f}\n")
        f.write(f"MSE:   {results['mse_mean']:.6f} ± {results['mse_std']:.6f}\n")
        f.write(f"MAE:   {results['mae_mean']:.6f} ± {results['mae_std']:.6f}\n")
        f.write("="*60 + "\n")
    
    print(f"\n✅ Results saved to {results_file}")
    
    return results


def evaluate():
    """Evaluate trained diffusion model on test set."""
    
    # Load config
    config = Config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create test dataset
    test_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
        image_size=config.data.image_size,
        download=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Create model
    model = UNetDiffusion(
        input_channels=config.model.input_channels,
        hidden_dims=config.model.hidden_dims,
        use_attention=config.model.use_attention,
        use_skip_connections=config.model.use_skip_connections,
        pretrained_encoder=config.model.pretrained_encoder,
        encoder_checkpoint=config.model.encoder_checkpoint,
        freeze_encoder_stages=config.model.freeze_encoder_stages,
        input_size=config.data.image_size
    ).to(device)
    
    # Load checkpoint
    checkpoint_path = os.path.join(config.logging.checkpoint_dir, 'diffusion_best_model.pt')
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    model.eval()
    
    # Create noise scheduler
    noise_config = NoiseConfig()
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_config.num_timesteps,
        beta_start=noise_config.beta_start,
        beta_end=noise_config.beta_end,
        schedule_type=noise_config.schedule_type
    ).to(device)
    
    # Create deterministic mask generator
    mask_generator = MaskGenerator.for_eval(config.mask)
    
    # Run evaluation
    results = run_evaluation(
        model=model,
        test_loader=test_loader,
        noise_scheduler=noise_scheduler,
        mask_generator=mask_generator,
        device=device,
        save_dir='results/diffusion'
    )
    
    return results


if __name__ == '__main__':
    evaluate()