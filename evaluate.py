"""
Evaluation script for astronomical object classification.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict
import json
import csv

from data_loader import AstronomicalObjectDataLoader
from models.create_model import (
    create_and_load_model,
    load_model_config,
    get_model_preprocessor
)
from metrics import compute_comprehensive_metrics


def evaluate(data_path: str, model_names: List[str], device: torch.device = None) -> pd.DataFrame:
    """
    Evaluate multiple models on the test set and return metrics.
    
    Args:
        data_path: Path to data directory
        model_names: List of model names (subdirectories under assets/)
        device: Device to run evaluation on (defaults to cuda if available)
    
    Returns:
        DataFrame with columns:
        - model_name
        - top1_accuracy
        - top3_accuracy
        - loss
        - precision_<class_name> for each class
        - recall_<class_name> for each class
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Log device information
    print("=" * 80)
    print("EVALUATION SETUP")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ Device: {device}")
    else:
        print("✗ GPU not available, using CPU")
        print(f"✓ Device: {device}")
    print("=" * 80)
    
    # Load label mapping
    label_mapping_path = Path('assets') / 'label_mapping.csv'
    with open(label_mapping_path, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # Skip header
        label_to_name = {}
        for row in reader:
            if len(row) >= 2:
                class_name, label = row[0], int(row[1])
                label_to_name[label] = class_name
    
    num_classes = len(label_to_name)
    class_names = [label_to_name[i] for i in range(num_classes)]
    
    # We'll get the preprocessor from the first model's config
    # All models should use the same preprocessor in this setup
    # Load the first model config to determine preprocessor
    assets_dir = Path('assets')
    first_model_dir = assets_dir / model_names[0]
    first_config_path = first_model_dir / 'model_config.json'
    
    if not first_config_path.exists():
        raise ValueError(f"Model config not found for {model_names[0]}")
    
    config = load_model_config(first_config_path)
    model_type = config['model_type']
    preprocessor = get_model_preprocessor(model_type)
    
    # Create dataloader for test set
    data_path = Path(data_path)
    
    loader = AstronomicalObjectDataLoader(
        data_path=data_path,
        preprocessor=preprocessor,
        batch_size=1,  # Single batch to load all samples
        num_workers=0  # No parallel processing for evaluation
    )
    
    test_loader = loader.get_dataloader('test')
    
    # Collect all test data
    all_features = []
    all_labels = []
    
    print("Loading test data...")
    for features, labels in test_loader:
        all_features.append(features)
        all_labels.append(labels)
    
    all_features = torch.cat(all_features, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    print(f"Loaded {len(all_features)} test samples")
    
    # Evaluate each model
    results = []
    
    for model_name in model_names:
        print(f"\nEvaluating model: {model_name}")
        
        # Load model
        assets_dir = Path('assets') / model_name
        config_path = assets_dir / 'model_config.json'
        checkpoint_path = assets_dir / 'best_model.pth'
        
        if not config_path.exists() or not checkpoint_path.exists():
            print(f"Warning: Model {model_name} not found, skipping")
            continue
        
        model = create_and_load_model(config_path, checkpoint_path, device)
        
        # Run inference
        model.eval()
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            # Get predictions
            all_features_cuda = all_features.to(device)
            outputs = model(all_features_cuda)
            loss = criterion(outputs, all_labels.to(device))
            
            # Compute comprehensive metrics using shared function
            all_labels_cuda = all_labels.to(device)
            metrics = compute_comprehensive_metrics(outputs, all_labels_cuda, num_classes)
            
            # Store results
            result = {
                'model_name': model_name,
                'top1_accuracy': metrics['top1'],
                'top3_accuracy': metrics['top3'],
                'loss': loss.item()
            }
            
            # Extract precision and recall per class
            precision_per_class = [metrics[f'precision_{i}'] for i in range(num_classes)]
            recall_per_class = [metrics[f'recall_{i}'] for i in range(num_classes)]
            
            # Add per-class precision and recall
            for class_name, prec, rec in zip(class_names, precision_per_class, recall_per_class):
                result[f'precision_{class_name}'] = prec
                result[f'recall_{class_name}'] = rec
            
            results.append(result)
            
            print(f"  Top-1 Accuracy: {result['top1_accuracy']:.2f}%")
            print(f"  Top-3 Accuracy: {result['top3_accuracy']:.2f}%")
            print(f"  Loss: {loss.item():.4f}")
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    return df
