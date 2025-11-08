import torch
import numpy as np
from typing import Tuple
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torchvision.models import inception_v3
import lpips


class InpaintingMetrics:
    """Metrics for evaluating inpainting quality."""
    
    def __init__(self, device: str = 'cuda'):
        self.device = device
        
        # LPIPS for perceptual distance
        self.lpips = lpips.LPIPS(net='alex').to(device)
        
        # Inception model for FID
        self.inception = inception_v3(pretrained=True, transform_input=False).to(device)
        self.inception.eval()
    
    def psnr(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Peak Signal-to-Noise Ratio."""
        mse = torch.mean((pred - target) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(2.0 / torch.sqrt(mse)).item()
    
    def ssim(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Structural Similarity Index."""
        from skimage.metrics import structural_similarity
        
        pred_np = pred.cpu().numpy().transpose(0, 2, 3, 1)
        target_np = target.cpu().numpy().transpose(0, 2, 3, 1)
        
        ssim_values = []
        for p, t in zip(pred_np, target_np):
            ssim_val = structural_similarity(p, t, multichannel=True, data_range=2.0)
            ssim_values.append(ssim_val)
        
        return np.mean(ssim_values)
    
    def lpips_distance(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate LPIPS perceptual distance."""
        with torch.no_grad():
            distance = self.lpips(pred, target)
        return distance.mean().item()
    
    def calculate_fid(self, real_features: np.ndarray, fake_features: np.ndarray) -> float:
        """Calculate Fréchet Inception Distance."""
        mu_real = np.mean(real_features, axis=0)
        mu_fake = np.mean(fake_features, axis=0)
        
        sigma_real = np.cov(real_features, rowvar=False)
        sigma_fake = np.cov(fake_features, rowvar=False)
        
        diff = mu_real - mu_fake
        
        covmean, _ = linalg.sqrtm(sigma_real @ sigma_fake, disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff @ diff + np.trace(sigma_real + sigma_fake - 2 * covmean)
        return fid
    
    def extract_inception_features(self, images: torch.Tensor) -> np.ndarray:
        """Extract features from Inception model for FID calculation."""
        with torch.no_grad():
            # Resize to inception input size
            images = adaptive_avg_pool2d(images, (299, 299))
            
            # Get features
            features = self.inception(images)
            
        return features.cpu().numpy()