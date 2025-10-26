"""
Metrics computation for model evaluation.
"""

import torch
from typing import Dict, Tuple
import numpy as np


def compute_top_k_accuracy(outputs: torch.Tensor, targets: torch.Tensor, top_k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        outputs: Model output logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        top_k: Top-k accuracy to compute
    
    Returns:
        Accuracy as a float percentage
    """
    with torch.no_grad():
        batch_size = targets.size(0)
        _, pred = outputs.topk(top_k, 1, True, True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        correct_k = correct[:top_k].reshape(-1).float().sum(0, keepdim=True)
        acc = correct_k.mul_(100.0 / batch_size).item()
        return acc


def compute_per_class_metrics(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Tuple[Dict[int, float], Dict[int, float]]:
    """
    Compute per-class precision and recall.
    
    Args:
        outputs: Model output logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes
    
    Returns:
        Tuple of (precision_dict, recall_dict) where each dict maps class_idx to metric value
    """
    with torch.no_grad():
        _, pred = outputs.topk(1, dim=1)
        pred = pred.squeeze()
        
        precision_dict = {}
        recall_dict = {}
        
        for class_idx in range(num_classes):
            # True positives: predicted this class and was correct
            tp = ((pred == class_idx) & (targets == class_idx)).sum().item()
            # False positives: predicted this class but was wrong
            fp = ((pred == class_idx) & (targets != class_idx)).sum().item()
            # False negatives: should have predicted this class but didn't
            fn = ((pred != class_idx) & (targets == class_idx)).sum().item()
            
            # Precision = TP / (TP + FP)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            # Recall = TP / (TP + FN)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            
            precision_dict[class_idx] = precision
            recall_dict[class_idx] = recall
        
        return precision_dict, recall_dict


def compute_accuracy(outputs: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
    """
    Compute comprehensive accuracy metrics.
    
    Args:
        outputs: Model output logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
    
    Returns:
        Dictionary with 'top1' and 'top3' accuracy percentages
    """
    return {
        'top1': compute_top_k_accuracy(outputs, targets, top_k=1),
        'top3': compute_top_k_accuracy(outputs, targets, top_k=3)
    }


def compute_comprehensive_metrics(outputs: torch.Tensor, targets: torch.Tensor, num_classes: int) -> Dict[str, float]:
    """
    Compute all metrics including per-class precision and recall.
    
    Args:
        outputs: Model output logits [batch_size, num_classes]
        targets: Ground truth labels [batch_size]
        num_classes: Number of classes
    
    Returns:
        Dictionary with all metrics:
        - 'top1': Top-1 accuracy
        - 'top3': Top-3 accuracy
        - 'precision_<class_idx>': Precision for each class
        - 'recall_<class_idx>': Recall for each class
    """
    metrics = compute_accuracy(outputs, targets)
    precision_dict, recall_dict = compute_per_class_metrics(outputs, targets, num_classes)
    
    for class_idx in range(num_classes):
        metrics[f'precision_{class_idx}'] = precision_dict[class_idx]
        metrics[f'recall_{class_idx}'] = recall_dict[class_idx]
    
    return metrics

