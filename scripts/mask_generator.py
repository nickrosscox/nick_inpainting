import os
import hashlib
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import torch


class MaskGenerator:
    """
    Mask generator that produces masks only (boolean/{0,1} float tensors).

    Features:
    - Supports 'random' (bbox), 'center', and 'irregular' masks
    - Batch and single-image generation
    - Class-level factory methods for config-driven creation
    - Deterministic mask generation for validation/test via per-filename hashing
    - Optional on-disk caching for deterministic masks
    """

    def __init__(
        self,
        mask_type: str = 'random',
        mask_ratio: float = 0.4,
        min_size: int = 32,
        max_size: int = 128,
        seed: Optional[int] = None,
        cache_dir: Optional[str] = None,
        deterministic: bool = False,
    ):
        self.mask_type = mask_type
        self.mask_ratio = float(mask_ratio)
        self.min_size = int(min_size)
        self.max_size = int(max_size)
        self.cache_dir = cache_dir
        self.deterministic = deterministic

        # Base RNG for dynamic generation; for deterministic per-filename we use derived seeds
        self._base_seed = seed
        self._rng = np.random.RandomState(seed) if seed is not None else np.random

        if self.cache_dir is not None:
            os.makedirs(self.cache_dir, exist_ok=True)

    # -------------------------
    # Class-level factories
    # -------------------------
    @classmethod
    def from_config(cls, config, deterministic: bool = False) -> "MaskGenerator":
        """Create MaskGenerator from MaskConfig object."""
        return cls(
            mask_type=config.type,
            mask_ratio=config.mask_ratio,
            min_size=config.min_size,
            max_size=config.max_size,
            seed=config.seed,
            cache_dir=config.cache_dir,
            deterministic=deterministic,
        )

    @classmethod
    def for_train(cls, config: Dict[str, Any]) -> "MaskGenerator":
        """
        Factory for dynamic masks during training.
        """
        return cls.from_config(config, deterministic=False)

    @classmethod
    def for_eval(cls, config, cache_dir: Optional[str] = None) -> "MaskGenerator":
        """
        Factory for deterministic masks during validation/test.
        """
        # Create a copy if we need to override cache_dir
        if cache_dir is not None:
            from copy import deepcopy
            config = deepcopy(config)
            config.cache_dir = cache_dir
        return cls.from_config(config, deterministic=True)

    # -------------------------
    # Public API
    # -------------------------
    def set_seed(self, seed: int):
        self._base_seed = seed
        self._rng = np.random.RandomState(seed)

    def generate(self, shape: Tuple[int, ...]) -> torch.Tensor:
        """
        Generate mask(s) with the configured strategy.

        Args:
            shape:
              - (B, 1, H, W) for batch
              - (1, H, W) for single image

        Returns:
            Mask tensor of shape (B, 1, H, W) or (1, H, W) with dtype float32 in {0,1}.
        """
        if len(shape) == 4:
            B, C, H, W = shape
            if C != 1:
                raise ValueError(f"Expected channel dimension C=1 for mask generation, got C={C}")
            masks = [self._generate_single((1, H, W), rng=self._rng) for _ in range(B)]
            return torch.stack(masks, dim=0)
        elif len(shape) == 3:
            C, H, W = shape
            if C != 1:
                raise ValueError(f"Expected channel dimension C=1 for mask generation, got C={C}")
            return self._generate_single((1, H, W), rng=self._rng).unsqueeze(0)  # -> (1,1,H,W)
        else:
            raise ValueError(f"Unsupported shape for mask generation: {shape}")

    def generate_for_filenames(
        self,
        filenames: List[str],
        shape: Tuple[int, int, int],
        cache_dir: Optional[str] = None
    ) -> torch.Tensor:
        """
        Deterministically generate masks for a batch of filenames.

        Args:
            filenames: list of image filenames (used to derive per-image seeds)
            shape: (1, H, W)
            cache_dir: override cache directory; if None, uses self.cache_dir

        Returns:
            Tensor of shape (B, 1, H, W)
        """
        if len(shape) != 3 or shape[0] != 1:
            raise ValueError("shape must be (1, H, W) for per-filename deterministic generation")

        H, W = shape[1], shape[2]
        use_cache = cache_dir is not None or self.cache_dir is not None
        cache_root = cache_dir or self.cache_dir

        masks: List[torch.Tensor] = []
        for fname in filenames:
            cached = None
            cache_path = None

            if use_cache:
                cache_path = self._cache_path_for_filename(fname, H, W, cache_root)
                if os.path.isfile(cache_path):
                    try:
                        cached = torch.load(cache_path, map_location='cpu')
                        # Basic sanity check on shape
                        if not torch.is_tensor(cached) or cached.shape != (1, 1, H, W):
                            cached = None
                    except Exception:
                        cached = None

            if cached is not None:
                masks.append(cached)
                continue

            # Derive a deterministic seed from filename and base seed
            seed = self._seed_from_filename(fname)
            rng = np.random.RandomState(seed)

            mask = self._generate_single((1, H, W), rng=rng).unsqueeze(0)  # (1,1,H,W)
            if use_cache and cache_path is not None:
                try:
                    torch.save(mask, cache_path)
                except Exception:
                    pass  # caching is best-effort

            masks.append(mask)

        return torch.cat(masks, dim=0)  # (B,1,H,W)

    # -------------------------
    # Internal helpers
    # -------------------------
    def _generate_single(self, shape: Tuple[int, int, int], rng: np.random.RandomState) -> torch.Tensor:
        """
        Generate a single mask of shape (1, H, W) using the provided RNG.
        """
        _, H, W = shape
        if self.mask_type == 'random':
            mask = self._random_bbox_mask(H, W, rng)
        elif self.mask_type == 'center':
            mask = self._center_mask(H, W)
        elif self.mask_type == 'irregular':
            mask = self._irregular_mask(H, W, rng)
        else:
            raise ValueError(f"Unknown mask type: {self.mask_type}")

        return torch.from_numpy(mask.astype(np.float32)).view(1, H, W)

    def _random_bbox_mask(self, H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.float32)
        # Determine box size bounds
        min_h = max(1, self.min_size)
        min_w = max(1, self.min_size)
        max_h = max(min(H, self.max_size), min_h + 1)
        max_w = max(min(W, self.max_size), min_w + 1)

        box_h = int(rng.randint(min_h, max_h))
        box_w = int(rng.randint(min_w, max_w))
        if box_h >= H:
            box_h = H - 1
        if box_w >= W:
            box_w = W - 1

        y = int(rng.randint(0, max(1, H - box_h)))
        x = int(rng.randint(0, max(1, W - box_w)))

        mask[y:y + box_h, x:x + box_w] = 1.0
        return mask

    def _center_mask(self, H: int, W: int) -> np.ndarray:
        mask = np.zeros((H, W), dtype=np.float32)
        size = int(min(H, W) * float(self.mask_ratio))
        size = max(1, min(size, min(H, W) - 1))
        y = (H - size) // 2
        x = (W - size) // 2
        mask[y:y + size, x:x + size] = 1.0
        return mask

    def _irregular_mask(self, H: int, W: int, rng: np.random.RandomState) -> np.ndarray:
        """
        Simple free-form irregular mask using random walks with thickness.
        Avoids OpenCV dependency to keep things lightweight.
        """
        mask = np.zeros((H, W), dtype=np.float32)
        num_strokes = int(rng.randint(1, 5))
        for _ in range(num_strokes):
            y = int(rng.randint(0, H))
            x = int(rng.randint(0, W))
            num_points = int(rng.randint(5, 15))

            for _ in range(num_points):
                angle = float(rng.uniform(0, 2 * np.pi))
                length = int(rng.randint(10, 30))
                y_end = int(np.clip(y + length * np.sin(angle), 0, H - 1))
                x_end = int(np.clip(x + length * np.cos(angle), 0, W - 1))

                thickness = int(rng.randint(5, 15))
                y0 = max(0, min(y, y_end) - thickness)
                y1 = min(H, max(y, y_end) + thickness)
                x0 = max(0, min(x, x_end) - thickness)
                x1 = min(W, max(x, x_end) + thickness)
                mask[y0:y1, x0:x1] = 1.0

                y, x = y_end, x_end

        return mask

    def _seed_from_filename(self, filename: str) -> int:
        """
        Derive a deterministic integer seed from filename and base seed.
        """
        h = hashlib.sha1(filename.encode('utf-8')).hexdigest()
        fname_seed = int(h[:8], 16)  # 32-bit component
        base = self._base_seed if self._base_seed is not None else 0x31415926
        combined = (fname_seed ^ base ^ 0x9E3779B9) & 0x7FFFFFFF
        return combined

    def _cache_path_for_filename(self, filename: str, H: int, W: int, cache_root: str) -> str:
        """
        Produce a cache file path for a given filename and geometry.
        """
        # Normalize filename to a safe token
        token = hashlib.sha1(filename.encode('utf-8')).hexdigest()
        # Partition masks by mask_type/geometry to avoid collisions across settings
        subdir = os.path.join(cache_root, f"{self.mask_type}_{H}x{W}")
        os.makedirs(subdir, exist_ok=True)
        return os.path.join(subdir, f"{token}.pt")