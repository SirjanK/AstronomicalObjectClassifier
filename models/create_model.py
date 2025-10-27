"""
Model factory functions for creating models from configuration.
"""

import torch
import torch.nn as nn
import json
from pathlib import Path
from typing import Dict, Any
from enum import Enum

from models.logistic_regression import LogisticRegressionModel
from preprocessors import (
    get_salient_features_v2_preprocessor, 
    get_resize_preprocessor
)


class ModelType(Enum):
    """Enum for model types."""
    LOGISTIC_REGRESSION_V2 = "logistic_regression_v2"


class PreprocessorType(Enum):
    """Enum for preprocessor types."""
    SALIENT_FEATURES_V2 = "salient_features_v2"
    RESIZE = "resize"


def create_model_from_config(config: Dict[str, Any]) -> nn.Module:
    """
    Create a model from a configuration dictionary.
    
    Args:
        config: Configuration dictionary containing:
            - model_type: Type of model ('logistic_regression', etc.)
            - input_dim: Input dimension
            - num_classes: Number of classes
            - dropout: Dropout probability
    
    Returns:
        Model instance
    """
    model_type = config['model_type']
    input_dim = config['input_dim']
    num_classes = config['num_classes']
    dropout = config.get('dropout', 0.0)
    
    if model_type == 'logistic_regression_v2':
        return LogisticRegressionModel(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout=dropout
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def save_model_config(config: Dict[str, Any], save_path: Path):
    """
    Save model configuration to a JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save JSON file
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_model_config(load_path: Path) -> Dict[str, Any]:
    """
    Load model configuration from a JSON file.
    
    Args:
        load_path: Path to load JSON file from
    
    Returns:
        Configuration dictionary
    """
    with open(load_path, 'r') as f:
        return json.load(f)


def get_input_dim_from_sample(feature_sample: torch.Tensor) -> int:
    """
    Determine input dimension from a feature sample.
    
    Args:
        feature_sample: Sample feature tensor from dataloader
    
    Returns:
        Input dimension for model
    """
    if len(feature_sample.shape) == 4:  # Image format [B, C, H, W]
        input_dim = feature_sample.shape[1] * feature_sample.shape[2] * feature_sample.shape[3]
    elif len(feature_sample.shape) == 2:  # Feature vector [B, D]
        input_dim = feature_sample.shape[1]
    else:
        raise ValueError(f"Unexpected input shape: {feature_sample.shape}")
    return input_dim


def get_preprocessor_for_model(model_type: str) -> PreprocessorType:
    """
    Get the preprocessor type for a given model type.
    
    Args:
        model_type: Model type string
    
    Returns:
        PreprocessorType enum
    """
    if model_type == 'logistic_regression_v2':
        return PreprocessorType.SALIENT_FEATURES_V2
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def create_model_from_args_and_sample(args, feature_sample: torch.Tensor, num_classes: int):
    """
    Create a model based on arguments and infer input dimension from feature sample.
    
    This is useful during training when you want to create a model with arguments
    but let it infer the input dimension automatically.
    
    Args:
        args: Parsed command line arguments
        feature_sample: Sample feature tensor to infer input dimension
        num_classes: Number of classes
    
    Returns:
        Tuple of (model, config_dict)
    """
    input_dim = get_input_dim_from_sample(feature_sample)
    
    # Map model type to preprocessor type
    preprocessor_type = get_preprocessor_for_model(args.model)
    
    config = {
        'model_type': args.model,
        'preprocessor_type': preprocessor_type.value,
        'input_dim': int(input_dim),
        'num_classes': int(num_classes),
        'dropout': args.dropout
    }
    
    model = create_model_from_config(config)
    
    return model, config


def get_model_preprocessor(model_type: str):
    """
    Get the preprocessor associated with a model type.
    
    Args:
        model_type: Model type string
    
    Returns:
        Preprocessor transforms.Compose
    """
    preprocessor_type = get_preprocessor_for_model(model_type)
    
    if preprocessor_type == PreprocessorType.SALIENT_FEATURES_V2:
        return get_salient_features_v2_preprocessor(
            image_resize_size=(224, 224),
            grid_size=(32, 32)
        )
    elif preprocessor_type == PreprocessorType.RESIZE:
        return get_resize_preprocessor(output_size=(224, 224))
    else:
        raise ValueError(f"Unknown preprocessor type: {preprocessor_type}")


def create_and_load_model(config_path: Path, checkpoint_path: Path, device: torch.device) -> nn.Module:
    """
    Create a model from configuration and load its checkpoint.
    
    Args:
        config_path: Path to model configuration JSON
        checkpoint_path: Path to model checkpoint
        device: Device to load model on
    
    Returns:
        Loaded model
    """
    config = load_model_config(config_path)
    model = create_model_from_config(config)
    model = model.to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model

