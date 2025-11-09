import torch
from torch.utils.data import DataLoader
import sys
import os

# Ensure the project root (the directory containing diffusion/, data/, etc.) is in sys.path
project_root = os.path.dirname(os.path.abspath(__file__))  # e.g. .../in_paint_structure/diffusion
project_root = os.path.dirname(project_root)               # go one level up → .../in_paint_structure
if project_root not in sys.path:
    sys.path.insert(0, project_root)


from config import Config
from noise_scheduler_config import NoiseConfig
from data.celeba_dataset import CelebADataset
from unet_diffusion import UNetDiffusion, NoiseScheduler
from diffusion_loss import DiffusionLossPerImage, DiffusionLoss
from diffusion_trainer import DiffusionTrainer
from scripts.mask_generator import MaskGenerator


def main():
    config = Config()
    noise_scheduler = NoiseConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create mask generator
    train_mask_generator = MaskGenerator.for_train(config.mask)
    val_mask_generator = MaskGenerator.for_eval(config.mask)
    

    model = UNetDiffusion(
        input_channels=config.model.input_channels,
        hidden_dims=config.model.hidden_dims,
        use_attention=config.model.use_attention,
        use_skip_connections=config.model.use_skip_connections,
        pretrained_encoder=config.model.pretrained_encoder,
        encoder_checkpoint=config.model.encoder_checkpoint,
        freeze_encoder_stages=config.model.freeze_encoder_stages
    )

    train_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='train',
        image_size=config.data.image_size,
        download=True
    )
    val_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='val',
        image_size=config.data.image_size,
        download=False
    )

    #config.training.batch_size = 128
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        pin_memory=True
    )

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create noise scheduler
    noise_scheduler = NoiseScheduler(
        num_timesteps=noise_scheduler.num_timesteps,
        beta_start=noise_scheduler.beta_start,
        beta_end=noise_scheduler.beta_end,
        schedule_type=noise_scheduler.schedule_type
    )

    # Create loss function
    loss_fn = DiffusionLoss()

    # Create trainer
    trainer = DiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        noise_scheduler=noise_scheduler,
        config=config,
        device=device,
        train_mask_generator=train_mask_generator,  # Random for training
        val_mask_generator=val_mask_generator, # Deterministic for Validation
        patience=10,
        use_ema=False
    )

    # Start training
    print("Starting training...")
    trainer.train()
    print("Training completed!")

if __name__ == '__main__':
    main()