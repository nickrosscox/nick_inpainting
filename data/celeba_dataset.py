import os
from typing import Dict, Any, Optional
import torch
from torchvision import transforms
from torchvision.datasets import CelebA as TorchvisionCelebA


class CelebADataset(TorchvisionCelebA):
    """
    Wrapper around torchvision.datasets.CelebA that:
    - Returns dict format compatible with existing pipeline
    - Maps 'val' split to 'valid' for torchvision
    - Provides default transforms matching original behavior
    - Enables subclassing for specialized datasets
    """
    
    SPLIT_MAP = {
        'train': 'train',
        'val': 'valid',      # Map val → valid
        'test': 'test',
        'all': 'all'
    }
    
    def __init__(
        self,
        root_dir: str,
        split: str = 'train',
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
        download: bool = False,
        verify_integrity: bool = True,
    ):
        """
        Args:
            root_dir: Path to CelebA dataset directory
            split: One of ['train', 'val', 'test', 'all']
            image_size: Target image size for resizing
            transform: Optional torchvision transform pipeline
            download: If True, download dataset if missing
            verify_integrity: Verify dataset integrity (unused, kept for API compatibility)
        """
        # Validate and map split
        if split not in self.SPLIT_MAP:
            raise ValueError(f"Invalid split '{split}'. Must be one of {list(self.SPLIT_MAP.keys())}.")
        
        torchvision_split = self.SPLIT_MAP[split]
        
        # Store for later use
        self.split = split
        self.image_size = image_size
        
        # Default transforms if none provided
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ])
        
        # Initialize parent with torchvision split name
        super().__init__(
            root=root_dir,
            split=torchvision_split,
            target_type='attr',  # Required param, we don't use attributes
            transform=transform,
            download=download
        )
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Override to return dict format instead of tuple.
        
        Returns:
            Dict with keys:
                - 'image': Transformed image tensor
                - 'filename': Original filename (e.g., '000001.jpg')
                - 'idx': Index in dataset
        """
        # Call parent to get (image, attr)
        image, _ = super().__getitem__(idx)
        
        # Get filename from internal filename list
        filename = self.filename[idx]
        
        return {
            'image': image,
            'filename': filename,
            'idx': idx,
        }