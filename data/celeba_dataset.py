import os
import json
import random
from glob import glob
from typing import Dict, Any, Optional
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class CelebADataset(Dataset):
    """
    Leak-proof, deterministic CelebA dataset wrapper.
    - Expects images under: <data_path>/celeba/img_align_celeba/
    - Saves celeba_split.json in the same directory as this file
    - Guarantees identical train/val/test splits across runs & scripts
    """

    def __init__(
        self,
        root_dir: str,                     # e.g., DataConfig.data_path → "./assets/datasets"
        split: str = "train",
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,            # kept for API parity
        verify_integrity: bool = True,     # unused, for API parity
    ):
        self.root_dir = root_dir
        self.split = split
        self.image_size = image_size

        # ------------------------------------------------------------------
        # 1. Locate the true CelebA folder and split file path
        # ------------------------------------------------------------------
        celeba_root = os.path.join(root_dir, "celeba")
        image_dir = os.path.join(celeba_root, "img_align_celeba")

        # Save split file in the same directory as this Python file
        script_dir = os.path.dirname(os.path.abspath(__file__))
        split_file = os.path.join(script_dir, "celeba_split.json")

        # Ensure images exist
        if not os.path.exists(image_dir):
            raise FileNotFoundError(
                f"[ERROR] Expected images under: {image_dir}\n"
                f"Please verify your DataConfig.data_path points to './assets/datasets'."
            )

        # ------------------------------------------------------------------
        # 2. Create or load a deterministic split file
        # ------------------------------------------------------------------
        if not os.path.exists(split_file):
            print(f"[INFO] No split file found at {split_file}. Creating one now (deterministic).")

            all_imgs = sorted(glob(os.path.join(image_dir, "*.jpg")))
            if not all_imgs:
                raise FileNotFoundError(f"No .jpg files found under {image_dir}")

            random.seed(42)  # fixed for determinism
            random.shuffle(all_imgs)

            n = len(all_imgs)
            train_end = int(0.8 * n)
            val_end = int(0.9 * n)
            split_dict = {
                "train": all_imgs[:train_end],
                "val":   all_imgs[train_end:val_end],
                "test":  all_imgs[val_end:],
            }

            with open(split_file, "w") as f:
                json.dump(split_dict, f, indent=2)
            print(f"[INFO] Saved persistent split file ({n} images) → {split_file}")
        else:
            with open(split_file) as f:
                split_dict = json.load(f)

        if split not in split_dict:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(split_dict.keys())}.")

        self.file_list = split_dict[split]

        # ------------------------------------------------------------------
        # 3. Define default transforms
        # ------------------------------------------------------------------
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        self.transform = transform

        # ------------------------------------------------------------------
        # 4. Cache filenames
        # ------------------------------------------------------------------
        self.filename = [os.path.basename(f) for f in self.file_list]

        print(f"[INFO] Loaded CelebA split '{split}' with {len(self.file_list)} images.")
        print(f"[INFO] Split file path: {split_file}")

    # ----------------------------------------------------------------------
    # 5. Sample retrieval
    # ----------------------------------------------------------------------
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path = self.file_list[idx]
        image = Image.open(path).convert("RGB")
        image = self.transform(image)

        return {
            "image": image,
            "filename": self.filename[idx],
            "idx": idx,
        }

    def __len__(self) -> int:
        return len(self.file_list)
