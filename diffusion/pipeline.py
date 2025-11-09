import os
import sys

project_root = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(project_root)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from data.celeba_dataset import CelebADataset
from config import Config

config = Config()

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

test_dataset = CelebADataset(
        root_dir=config.data.data_path,
        split='test',
        image_size=config.data.image_size,
        download=False
)

train_files = set(train_dataset.file_list)
val_files   = set(val_dataset.file_list)
test_files  = set(test_dataset.file_list)

print(len(train_files & val_files),
      len(train_files & test_files),
      len(val_files & test_files))