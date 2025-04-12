import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torchvision.datasets import ImageFolder
from torchvision import transforms
from timm.data import create_transform
from timm.data.mixup import Mixup

# Define paths
PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(PROJECT_PATH, "data/tiny-imagenet-200/")
TRAIN_PATH = os.path.join(DATA_PATH, "train")
VAL_PATH = os.path.join(DATA_PATH, "val")
VAL_IMG_PATH = os.path.join(VAL_PATH, "images")
VAL_ANNO_PATH = os.path.join(VAL_PATH, "val_annotations.txt")

# Constants
NUM_CLASSES = 200  # Tiny ImageNet has 200 classes

# Data augmentation configuration
AUG_CONFIG = {
    "use_autoaugment": True,
    "random_erase_prob": 0.25,
    "mixup_alpha": 0.8,
    "cutmix_alpha": 1.0,
    "mixup_prob": 1.0,
    "switch_prob": 0.5,
    "label_smoothing": 0.1,
}

def image_loader(image_path):
    """Load an image as a PIL Image, not torch tensor"""
    return Image.open(image_path).convert('RGB')

def build_train_transforms(img_size=224):
    """Create advanced transforms for training data"""
    transform = create_transform(
        input_size=img_size,
        is_training=True,
        auto_augment='rand-m9-mstd0.5-inc1',
        re_prob=AUG_CONFIG["random_erase_prob"],
        re_mode='pixel',
        re_count=1,
        interpolation='bicubic',
    )
    return transform

def build_val_transforms(img_size=224):
    """Create transforms for validation data"""
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

def build_mixup_fn():
    """Create mixup/cutmix augmentation function"""
    mixup_fn = None
    if AUG_CONFIG["mixup_alpha"] > 0 or AUG_CONFIG["cutmix_alpha"] > 0:
        mixup_fn = Mixup(
            mixup_alpha=AUG_CONFIG["mixup_alpha"],
            cutmix_alpha=AUG_CONFIG["cutmix_alpha"],
            cutmix_minmax=None,
            prob=AUG_CONFIG["mixup_prob"],
            switch_prob=AUG_CONFIG["switch_prob"],
            mode='batch',
            label_smoothing=AUG_CONFIG["label_smoothing"],
            num_classes=NUM_CLASSES
        )
    return mixup_fn

class TinyImageNetValidation(Dataset):
    """Custom dataset for Tiny ImageNet validation set"""
    def __init__(self, transform=None):
        self.root = VAL_IMG_PATH
        self.transform = transform
        self.imgs = sorted(os.listdir(VAL_IMG_PATH))
        
        # Load class IDs and build mapping
        self.class_to_idx = {}
        train_dataset = ImageFolder(root=TRAIN_PATH)
        self.class_to_idx = train_dataset.class_to_idx
        
        # Load image to class mappings from annotations
        self.img_to_class = {}
        with open(VAL_ANNO_PATH) as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2:
                    self.img_to_class[parts[0]] = parts[1]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img_name = self.imgs[idx]
        img_path = os.path.join(self.root, img_name)
        
        # Load image
        image = image_loader(img_path)
        
        # Get class ID
        class_id = self.img_to_class.get(img_name, None)
        if class_id is None:
            raise ValueError(f"Class ID not found for image {img_name}")
        
        # Get class index
        class_idx = self.class_to_idx.get(class_id, 0)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, class_idx

def get_loaders(batch_size=128, num_workers=4, img_size=224, use_mixup=True):
    """
    Create data loaders for Tiny ImageNet
    
    Args:
        batch_size: Batch size for training and validation
        num_workers: Number of workers for data loading
        img_size: Image size after resizing
        use_mixup: Whether to use mixup/cutmix augmentation
        
    Returns:
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        mixup_fn: Mixup function or None
    """
    print("Creating data loaders...")
    
    # Create transforms
    train_transform = build_train_transforms(img_size)
    val_transform = build_val_transforms(img_size)
    
    # Create datasets
    train_dataset = ImageFolder(
        root=TRAIN_PATH,
        loader=image_loader,
        transform=train_transform
    )
    
    val_dataset = TinyImageNetValidation(transform=val_transform)
    
    # Create data loaders
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create mixup function if requested
    mixup_fn = build_mixup_fn() if use_mixup else None
    
    print(f"Created data loaders: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    return train_loader, val_loader, mixup_fn

if __name__ == "__main__":
    # For testing
    train_loader, val_loader, mixup_fn = get_loaders(batch_size=4)
    
    # Test training data loader
    for images, labels in train_loader:
        print(f"Training batch - Images: {images.shape}, Labels: {labels.shape}")
        # Apply mixup if available
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)
            print(f"After mixup - Images: {images.shape}, Labels: {labels.shape}")
        break
    
    # Test validation data loader
    for images, labels in val_loader:
        print(f"Validation batch - Images: {images.shape}, Labels: {labels.shape}")
        break 