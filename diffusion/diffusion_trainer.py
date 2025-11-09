import os
import torch
import copy

class DiffusionTrainer:
    """
    Trainer for diffusion inpainting model with optional Exponential Moving Average (EMA) stabilization.
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, noise_scheduler,
                 config, device, train_mask_generator, val_mask_generator, patience,
                 use_ema=True):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.noise_scheduler = noise_scheduler.to(device)
        self.config = config
        self.device = device
        self.train_mask_generator = train_mask_generator
        self.val_mask_generator = val_mask_generator
        self.patience = patience
        self.use_ema = use_ema  # 👈 flag controls EMA usage

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Cosine LR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.training.epochs,
            eta_min=1e-6
        )

        # Training parameters
        self.num_epochs = config.training.epochs
        self.num_epochs = 1
        self.num_timesteps = noise_scheduler.num_timesteps

        # --- EMA setup (optional) ---
        if self.use_ema:
            self.ema_decay = 0.9999
            self.ema_model = copy.deepcopy(self.model).to(self.device)
            self.ema_model.eval()
            print("🟢 EMA is ENABLED — using exponential moving average with decay =", self.ema_decay)
        else:
            self.ema_model = None
            print("🔴 EMA is DISABLED — training and validation will use raw model weights only.")

        # Internal counters
        self.global_step = 0

    # -------------------------------------------------------------------------
    # EMA helper
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def update_ema(self):
        """Update exponential moving average of model parameters."""
        if not self.use_ema:
            return
        for ema_param, param in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1 - self.ema_decay)

    # -------------------------------------------------------------------------
    # Training epoch
    # -------------------------------------------------------------------------
    def train_epoch(self, epoch):
        """Train the model for one epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(self.train_loader):
            images = batch['image'].to(self.device)
            B, C, H, W = images.shape

            # Generate random masks
            masks = self.train_mask_generator.generate((B, 1, H, W)).to(self.device)

            batch_size = images.size(0)
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

            # Add noise (only masked region)
            noisy_images, noise = self.noise_scheduler.add_noise(images, t, masks)

            # ====== DEBUG BLOCK (only for first batch) ======
            if epoch == 1 and batch_idx == 0:
                import torchvision.utils as vutils
                import matplotlib.pyplot as plt
                import os

                debug_dir = os.path.join(self.config.logging.checkpoint_dir, "debug_inputs")
                os.makedirs(debug_dir, exist_ok=True)

                print(f"\n[DEBUG] Epoch {epoch}, Batch {batch_idx}")
                print(f"Images shape:       {images.shape}")
                print(f"Noisy images shape: {noisy_images.shape}")
                print(f"Mask shape:         {masks.shape}")
                print(f"t range: [{t.min().item()}, {t.max().item()}]")
                print(f"Mask mean value (should be ~0.1–0.4): {masks.mean().item():.4f}")

                # Verify channel composition
                combined_input = torch.cat([noisy_images, masks.repeat(1,3,1,1)], dim=0)
                vutils.save_image(combined_input, os.path.join(debug_dir, "noisy_plus_mask.png"),
                                nrow=8, normalize=True, value_range=(-1,1))
                print(f"[DEBUG] Saved noisy+mask composite grid to {debug_dir}/noisy_plus_mask.png")

                # Visualize each channel separately
                for c in range(3):
                    vutils.save_image(noisy_images[:, c:c+1],
                                    os.path.join(debug_dir, f"noisy_channel_{c}.png"),
                                    nrow=8, normalize=True, value_range=(-1,1))
                vutils.save_image(masks, os.path.join(debug_dir, "mask_channel.png"),
                                nrow=8, normalize=True, value_range=(0,1))
                print(f"[DEBUG] Saved RGB + mask channels separately for manual inspection")

                # Quick sanity figure
                img_show = (noisy_images[0].detach().cpu().permute(1,2,0) + 1)/2
                mask_show = masks[0].detach().cpu().permute(1,2,0)
                plt.figure(figsize=(10,4))
                plt.subplot(1,3,1); plt.imshow(((images[0].cpu().permute(1,2,0)+1)/2).clamp(0,1)); plt.title("Original")
                plt.subplot(1,3,2); plt.imshow(mask_show, cmap="gray"); plt.title("Mask")
                plt.subplot(1,3,3); plt.imshow(img_show.clamp(0,1)); plt.title("Noisy Image Input")
                plt.tight_layout()
                plt.savefig(os.path.join(debug_dir, "debug_triplet.png"))
                plt.close()
            # ====== END DEBUG BLOCK ======


            # Forward pass
            predicted_noise = self.model(noisy_images, t, masks)
            loss = self.loss_fn(predicted_noise, noise, masks)

            # Backpropagation
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Update EMA (only if enabled)
            self.update_ema()

            total_loss += loss.item()
            self.global_step += 1

            if batch_idx % 100 == 0:
                print(f"Epoch [{epoch}/{self.num_epochs}] "
                      f"Batch [{batch_idx}/{len(self.train_loader)}] "
                      f"Loss: {loss.item():.4f}")

        avg_loss = total_loss / len(self.train_loader)
        return avg_loss

    # -------------------------------------------------------------------------
    # Validation
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def validate(self, epoch):
        """Validate using EMA weights if enabled, otherwise raw model."""
        model_to_eval = self.ema_model if self.use_ema else self.model
        model_to_eval.eval()

        total_loss = 0.0
        for batch in self.val_loader:
            images = batch['image'].to(self.device)
            filenames = batch['filename']

            B, C, H, W = images.shape
            masks = self.val_mask_generator.generate_for_filenames(
                filenames=filenames, shape=(1, H, W)
            ).to(self.device)

            batch_size = images.size(0)
            t = torch.randint(0, self.num_timesteps, (batch_size,), device=self.device)

            noisy_images, noise = self.noise_scheduler.add_noise(images, t, masks)
            predicted_noise = model_to_eval(noisy_images, t, masks)
            loss = self.loss_fn(predicted_noise, noise, masks)
            total_loss += loss.item()

        avg_loss = total_loss / len(self.val_loader)
        return avg_loss

    # -------------------------------------------------------------------------
    # Training loop
    # -------------------------------------------------------------------------
    def train(self):
        """Full training loop with early stopping and checkpointing."""
        best_val_loss = float('inf')
        epochs_no_improve = 0

        print("\n🚀 Starting training...")
        # print(f"EMA Status: {'ENABLED' if self.use_ema else 'DISABLED'}")
        print("-" * 60)

        for epoch in range(1, self.num_epochs + 1):
            print(f"\nEpoch {epoch}/{self.num_epochs}")
            print("-" * 50)

            train_loss = self.train_epoch(epoch)
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = self.validate(epoch)
            print(f"Validation Loss: {val_loss:.4f}")

            self.scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
                self.save_checkpoint(epoch, "diffusion_best_model.pt")
                print(f"✅ New best model saved (val_loss={val_loss:.4f})")
            else:
                epochs_no_improve += 1

            # if epochs_no_improve >= self.patience:
            #     print(f"\n⏹️ Early stopping after {epoch} epochs (no improvement for {self.patience})")
            #     break

        self.save_checkpoint(epoch, "diffusion_final_model.pt")
        print("💾 Final model saved after all epochs.")
        print(f"\n🏁 Training complete — Best validation loss: {best_val_loss:.4f}")

    # -------------------------------------------------------------------------
    # Checkpointing
    # -------------------------------------------------------------------------
    def save_checkpoint(self, epoch, filename):
        """Save model and EMA state (if available)."""
        checkpoint_dir = self.config.logging.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }

        if self.use_ema and self.ema_model is not None:
            checkpoint["ema_state_dict"] = self.ema_model.state_dict()

        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)
        print(f"💾 Checkpoint saved: {path}")
