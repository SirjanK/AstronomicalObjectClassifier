"""
Data loader for astronomical object classification dataset.
Supports training/validation/test splits with automatic label mapping.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import csv

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class AstronomicalObjectDataset(Dataset):
    """
    PyTorch dataset for astronomical object classification.
    
    Args:
        data_path: Root directory containing training/validation/test subdirectories
        split: One of 'training', 'validation', or 'test'
        label_mapping: Dictionary mapping class names to integer labels
        preprocessor: Image preprocessing function (transforms.Compose)
    """
    
    def __init__(
        self,
        data_path: Path,
        split: str,
        label_mapping: Dict[str, int],
        preprocessor: transforms.Compose
    ):
        self.data_path = Path(data_path)
        self.split = split
        self.label_mapping = label_mapping
        self.preprocessor = preprocessor
        
        # Load all image file paths and labels
        self.samples = self._load_samples()
    
    def _load_samples(self) -> List[Tuple[Path, int]]:
        """Load all image samples from the specified split directory."""
        samples = []
        split_dir = self.data_path / self.split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory {split_dir} does not exist")
        
        # Iterate through class directories
        for class_dir in sorted(split_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            class_name = class_dir.name
            if class_name not in self.label_mapping:
                print(f"Warning: Class '{class_name}' not in label mapping, skipping")
                continue
            
            label = self.label_mapping[class_name]
            
            # Find all image files in this class directory
            for ext in ['*.png', '*.jpg']:
                for img_path in class_dir.glob(ext):
                    samples.append((img_path, label))
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Load and preprocess image, return features and label."""
        img_path, label = self.samples[idx]
        
        # Load image
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a black image as fallback
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        
        # Apply preprocessing (feature extraction)
        features = self.preprocessor(img)
        
        return features, label


class AstronomicalObjectDataLoader:
    """
    Data loader for astronomical object classification.
    Provides training/validation/test dataloaders with automatic label mapping.
    """
    
    def __init__(
        self,
        data_path: Path,
        preprocessor: transforms.Compose,
        batch_size: int = 32,
        num_workers: int = 4,
        shuffle_training: bool = True
    ):
        """
        Initialize the data loader.
        
        Args:
            data_path: Root directory containing training/validation/test subdirectories
            preprocessor: transforms.Compose object for preprocessing images (required)
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            shuffle_training: Whether to shuffle training data
        """
        if preprocessor is None:
            raise ValueError("preprocessor must be specified, cannot be None")
        
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_training = shuffle_training
        self.preprocessor = preprocessor
        
        # Load or create label mapping
        self.label_mapping = self._load_or_create_label_mapping()
        
        # Create datasets and dataloaders
        self._create_dataloaders()
    
    def _load_or_create_label_mapping(self) -> Dict[str, int]:
        """
        Load existing label mapping from assets/label_mapping.csv,
        or create it from training directory structure.
        """
        assets_dir = Path('assets')
        mapping_file = assets_dir / 'label_mapping.csv'
        
        # Try to load existing mapping
        if mapping_file.exists():
            label_mapping = {}
            with open(mapping_file, 'r') as f:
                reader = csv.reader(f)
                next(reader)  # Skip header
                for row in reader:
                    if len(row) >= 2:
                        class_name, label = row[0], int(row[1])
                        label_mapping[class_name] = label
            print(f"Loaded label mapping from {mapping_file}")
            return label_mapping
        
        # Create new mapping from training directory
        training_dir = self.data_path / 'training'
        if not training_dir.exists():
            raise ValueError(f"Training directory {training_dir} does not exist")
        
        # Get all class subdirectories and sort them
        classes = sorted([d.name for d in training_dir.iterdir() if d.is_dir()])
        label_mapping = {cls: idx for idx, cls in enumerate(classes)}
        
        # Create assets directory and save mapping
        assets_dir.mkdir(exist_ok=True)
        with open(mapping_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['class_name', 'label'])
            for class_name, label in label_mapping.items():
                writer.writerow([class_name, label])
        
        print(f"Created label mapping and saved to {mapping_file}")
        print(f"Classes: {list(label_mapping.keys())}")
        
        return label_mapping
    
    def _create_dataloaders(self):
        """Create PyTorch dataloaders for training, validation, and test splits."""
        splits = ['training', 'validation', 'test']
        self.dataloaders = {}
        self.datasets = {}
        
        for split in splits:
            try:
                dataset = AstronomicalObjectDataset(
                    data_path=self.data_path,
                    split=split,
                    label_mapping=self.label_mapping,
                    preprocessor=self.preprocessor
                )
                self.datasets[split] = dataset
                
                shuffle = (split == 'training' and self.shuffle_training)
                dataloader = DataLoader(
                    dataset,
                    batch_size=self.batch_size,
                    shuffle=shuffle,
                    num_workers=self.num_workers,
                    pin_memory=torch.cuda.is_available(),
                    persistent_workers=(self.num_workers > 0)
                )
                self.dataloaders[split] = dataloader
                
                print(f"Created {split} dataloader with {len(dataset)} samples")
            except ValueError as e:
                print(f"Warning: Could not create {split} dataloader: {e}")
                self.datasets[split] = None
                self.dataloaders[split] = None
    
    def get_dataloader(self, split: str) -> Optional[DataLoader]:
        """Get dataloader for specified split."""
        return self.dataloaders.get(split)
    
    def get_dataset(self, split: str) -> Optional[AstronomicalObjectDataset]:
        """Get dataset for specified split."""
        return self.datasets.get(split)
    
    def get_label_mapping(self) -> Dict[str, int]:
        """Get the label mapping dictionary."""
        return self.label_mapping
    
    def get_num_classes(self) -> int:
        """Get the number of classes."""
        return len(self.label_mapping)
