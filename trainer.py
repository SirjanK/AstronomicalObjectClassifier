"""
Training script for astronomical object classification.

Usage:
    # Train logistic regression model (uses semantic features automatically)
    python trainer.py --name my_model_v1 --model logistic_regression --batch_size 32 --num_epochs 50
    
    # With custom hyperparameters
    python trainer.py --name my_model_v2 --batch_size 64 --learning_rate 0.001 --weight_decay 0.0001 --dropout 0.2 --num_epochs 100
    
    # View TensorBoard logs
    tensorboard --logdir logs
"""

import argparse
import os
from pathlib import Path
import time
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from data_loader import AstronomicalObjectDataLoader
from models.create_model import (
    save_model_config, 
    create_model_from_args_and_sample,
    get_model_preprocessor
)
from metrics import compute_top_k_accuracy
import json


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train astronomical object classifier')
    
    # Data arguments
    parser.add_argument('--data_path', type=str, default='data', 
                       help='Path to data directory')
    parser.add_argument('--assets_path', type=str, default='assets',
                       help='Path to assets directory (for label mapping)')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of worker processes for data loading')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001,
                       help='L2 regularization weight')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout probability')
    parser.add_argument('--num_epochs', type=int, default=100,
                       help='Number of training epochs')
    
    # Model arguments
    parser.add_argument('--model', type=str, default='logistic_regression',
                       choices=['logistic_regression'],
                       help='Model architecture to use')
    
    # Logging arguments
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory for TensorBoard logs and checkpoints')
    parser.add_argument('--log_freq', type=int, default=10,
                       help='Logging frequency (in batches)')
    parser.add_argument('--val_freq', type=int, default=5,
                       help='Validation frequency (in epochs)')
    
    # Smoothing arguments
    parser.add_argument('--smooth_window', type=int, default=100,
                       help='Window size for loss smoothing (in batches)')
    
    # Model name (required)
    parser.add_argument('--name', type=str, required=True,
                       help='Name for this model run (used for saving checkpoints and config)')
    
    return parser.parse_args()


def validate(model: nn.Module, dataloader, criterion: nn.Module, device: torch.device) -> Dict[str, float]:
    """Validate model and return metrics."""
    model.eval()
    total_loss = 0.0
    correct_top1 = 0
    correct_top3 = 0
    total_samples = 0
    
    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Compute metrics
            total_loss += loss.item()
            total_samples += labels.size(0)
            
            # Compute accuracies using shared function
            # Note: We need to accumulate across batches, so compute for this batch
            batch_size = labels.size(0)
            
            # Top-1 accuracy
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t()
            batch_correct_top1 = pred.eq(labels.view(1, -1).expand_as(pred)).reshape(-1).float().sum().item()
            correct_top1 += batch_correct_top1
            
            # Top-3 accuracy
            _, pred_top3 = outputs.topk(3, 1, True, True)
            pred_top3 = pred_top3.t()
            batch_correct_top3 = pred_top3.eq(labels.view(1, -1).expand_as(pred_top3)).any(dim=0).float().sum().item()
            correct_top3 += batch_correct_top3
    
    return {
        'loss': total_loss / len(dataloader),
        'acc_top1': 100.0 * correct_top1 / total_samples,
        'acc_top3': 100.0 * correct_top3 / total_samples
    }


def train():
    """Main training loop."""
    args = parse_args()
    
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("TRAINING SETUP")
    print("=" * 80)
    if torch.cuda.is_available():
        print(f"✓ GPU Available: {torch.cuda.get_device_name(0)}")
        print(f"✓ CUDA Version: {torch.version.cuda}")
        print(f"✓ Device: {device}")
    else:
        print("✗ GPU not available, using CPU")
        print(f"✓ Device: {device}")
    print("=" * 80)
    
    # Set up logging
    log_dir = Path(args.log_dir)
    log_dir.mkdir(exist_ok=True, parents=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    run_dir = log_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    writer = SummaryWriter(log_dir=str(run_dir))
    
    # Create data loader with model-specific preprocessor
    print("Loading data...")
    data_path = Path(args.data_path)
    preprocessor = get_model_preprocessor(args.model)
    
    loader = AstronomicalObjectDataLoader(
        data_path=data_path,
        preprocessor=preprocessor,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle_training=True
    )
    
    num_classes = loader.get_num_classes()
    print(f"Number of classes: {num_classes}")
    print(f"Label mapping: {loader.get_label_mapping()}")
    
    # Set up assets directory for saving model
    assets_dir = Path(args.assets_path)
    model_assets_dir = assets_dir / args.name
    model_assets_dir.mkdir(parents=True, exist_ok=True)
    
    # Get sample to determine input dimension and create model
    train_loader = loader.get_dataloader('training')
    features_sample, _ = next(iter(train_loader))
    
    # Create model and get its config (all handled in one function)
    model, model_config = create_model_from_args_and_sample(args, features_sample, num_classes)
    model = model.to(device)
    print(f"Model: {model}")
    
    # Save model configuration
    model_config_path = model_assets_dir / 'model_config.json'
    save_model_config(model_config, model_config_path)
    print(f"Saved model configuration to {model_config_path}")
    
    # Create optimizer and loss function
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    
    print("\nStarting training...")
    print("=" * 80)
    
    for epoch in range(args.num_epochs):
        model.train()
        running_loss = 0.0
        correct_top1 = 0
        correct_top3 = 0
        total_samples = 0
        
        epoch_start_time = time.time()
        
        for batch_idx, (features, labels) in enumerate(train_loader):
            features = features.to(device)
            labels = labels.to(device)
            
            # Reshape if needed (for image inputs)
            if len(features.shape) == 4:
                batch_size = features.shape[0]
                features = features.view(batch_size, -1)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Accumulate metrics
            running_loss += loss.item()
            total_samples += labels.size(0)
            
            # Compute accuracies
            _, pred = outputs.topk(1, 1, True, True)
            pred = pred.t()
            correct_top1 += pred.eq(labels.view(1, -1).expand_as(pred)).reshape(-1).float().sum().item()
            
            _, pred_top3 = outputs.topk(3, 1, True, True)
            pred_top3 = pred_top3.t()
            correct_top3 += pred_top3.eq(labels.view(1, -1).expand_as(pred_top3)).any(dim=0).float().sum().item()
            
            # Smoothed loss
            train_losses.append(loss.item())
            if len(train_losses) > args.smooth_window:
                train_losses.pop(0)
            
            smoothed_loss = np.mean(train_losses)
            
            # Logging
            if (batch_idx + 1) % args.log_freq == 0:
                writer.add_scalar('train/smoothed_loss', smoothed_loss, epoch * len(train_loader) + batch_idx)
                writer.add_scalar('train/instance_loss', loss.item(), epoch * len(train_loader) + batch_idx)
        
        # End of epoch metrics
        epoch_loss = running_loss / len(train_loader)
        epoch_acc_top1 = 100.0 * correct_top1 / total_samples
        epoch_acc_top3 = 100.0 * correct_top3 / total_samples
        
        # Validation
        val_loader = loader.get_dataloader('validation')
        if val_loader is not None and epoch % args.val_freq == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            
            print(f"Epoch {epoch+1}/{args.num_epochs} ({time.time() - epoch_start_time:.1f}s)")
            print(f"  Train Loss: {epoch_loss:.4f}, Train Acc (Top-1): {epoch_acc_top1:.2f}%, Train Acc (Top-3): {epoch_acc_top3:.2f}%")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, Val Acc (Top-1): {val_metrics['acc_top1']:.2f}%, Val Acc (Top-3): {val_metrics['acc_top3']:.2f}%")
            
            # Log to TensorBoard
            writer.add_scalar('loss/train', epoch_loss, epoch)
            writer.add_scalar('acc_top1/train', epoch_acc_top1, epoch)
            writer.add_scalar('acc_top3/train', epoch_acc_top3, epoch)
            writer.add_scalar('loss/val', val_metrics['loss'], epoch)
            writer.add_scalar('acc_top1/val', val_metrics['acc_top1'], epoch)
            writer.add_scalar('acc_top3/val', val_metrics['acc_top3'], epoch)
            
            # Save best model to assets directory
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_metrics['loss'],
                    'val_acc_top1': val_metrics['acc_top1'],
                    'val_acc_top3': val_metrics['acc_top3'],
                    'args': vars(args),  # Convert namespace to dict
                }
                checkpoint_path = model_assets_dir / 'best_model.pth'
                torch.save(checkpoint, checkpoint_path)
                print(f"  ✓ Saved best model to {checkpoint_path} (val loss: {best_val_loss:.4f})")
        else:
            print(f"Epoch {epoch+1}/{args.num_epochs} ({time.time() - epoch_start_time:.1f}s)")
            print(f"  Train Loss: {epoch_loss:.4f}, Train Acc (Top-1): {epoch_acc_top1:.2f}%, Train Acc (Top-3): {epoch_acc_top3:.2f}%")
            
            writer.add_scalar('train/epoch_loss', epoch_loss, epoch)
            writer.add_scalar('train/epoch_acc_top1', epoch_acc_top1, epoch)
            writer.add_scalar('train/epoch_acc_top3', epoch_acc_top3, epoch)
        
        print("-" * 80)
    
    print("\nTraining complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Logs saved to: {run_dir}")
    
    writer.close()


if __name__ == "__main__":
    train()
