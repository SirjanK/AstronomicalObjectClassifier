"""
Preprocessing functions for astronomical object classification.
"""

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from typing import Tuple
from functools import partial
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from skimage.color import rgb2hsv


def get_resize_preprocessor(output_size: Tuple[int, int] = (224, 224)) -> transforms.Compose:
    """
    Simple resize preprocessor that resizes images and scales to [0, 1].
    
    Args:
        output_size: Target size (height, width)
    
    Returns:
        transforms.Compose: Preprocessor pipeline
    """
    return transforms.Compose([
        transforms.Resize(output_size),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1] automatically
    ])


def get_salient_features_preprocessor(image_resize_size: Tuple[int, int] = (224, 224), grid_size: Tuple[int, int] = (8, 8)) -> transforms.Compose:
    """
    Extract salient features from images for simple baseline models.
    Features include:
    - Average channel values (R, G, B)
    - Brightness (average across all channels)
    - Grid-based features: mean and max values for each grid_size[0]xgrid_size[1] grid in the image_resize_size image
    - Texture features: standard deviation per channel
    - Edge detection features: approximate using gradient magnitude
    
    Args:
        image_resize_size: Size to resize the image to
        grid_size: Number of grids (height, width) - will create grid_size[0]*grid_size[1] grids
    
    Returns:
        transforms.Compose: Custom preprocessor that returns a feature vector
    """
    # Use functools.partial to create a pickle-able function
    extract_features_fn = partial(extract_salient_features, grid_size=grid_size)
    
    return transforms.Compose([
        transforms.Resize(image_resize_size),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1] automatically
        transforms.Lambda(extract_features_fn)
    ])


def get_salient_features_v2_preprocessor(image_resize_size: Tuple[int, int] = (224, 224), grid_size: Tuple[int, int] = (8, 8)) -> transforms.Compose:
    """
    Extract salient features from images for simple baseline models.
    Features include:
      - Global stats (mean, std, min, max, brightness)
      - Grid-based stats
      - Edge/gradient features (magnitude & angle)
      - LBP histogram
      - HOG features
      - HSV stats
    
    Args:
        image_resize_size: Size to resize the image to
        grid_size: Number of grids (height, width) - will create grid_size[0]*grid_size[1] grids
    
    Returns:
        transforms.Compose: Custom preprocessor that returns a feature vector
    """
    # Use functools.partial to create a pickle-able function
    extract_features_fn = partial(extract_salient_features_v2, grid_size=grid_size)
    
    return transforms.Compose([
        transforms.Resize(image_resize_size),
        transforms.ToTensor(),  # Converts to [C, H, W] and scales to [0, 1] automatically
        transforms.Lambda(extract_features_fn)
    ])


def extract_salient_features(img_input: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Extract salient features from an image.
    
    Args:
        img_input: torch.Tensor of shape [C, H, W] and values in [0, 1]
        grid_size: Size of each grid (grid_h, grid_w) in pixels
    
    Returns:
        torch.Tensor: Feature vector of shape [n_features] (float32)
    """
    # Convert to numpy and transpose from [C, H, W] to [H, W, C] for easier processing
    img_array = img_input.numpy().transpose(1, 2, 0)  # [H, W, C]
    h, w, c = img_array.shape
    
    features = []
    
    # 1. Average channel values (global)
    for channel in range(c):
        features.append(np.mean(img_array[:, :, channel]))
    
    # 2. Standard deviation per channel (texture measure)
    for channel in range(c):
        features.append(np.std(img_array[:, :, channel]))
    
    # 3. Brightness (average across all channels)
    brightness = np.mean(img_array)
    features.append(brightness)
    
    # 4. Max and min pixel values (across all channels)
    features.append(np.max(img_array))
    features.append(np.min(img_array))
    
    # 5. Grid-based features (mean and max for each grid)
    # Calculate number of grids that will fit in the image
    n_grids_h = h // grid_size[0]  # Number of grids vertically
    n_grids_w = w // grid_size[1]  # Number of grids horizontally
    
    # For each grid
    for i in range(n_grids_h):
        for j in range(n_grids_w):
            # Extract the grid region
            h_start = i * grid_size[0]
            h_end = (i + 1) * grid_size[0]
            w_start = j * grid_size[1]
            w_end = (j + 1) * grid_size[1]
            
            grid = img_array[h_start:h_end, w_start:w_end, :]
            
            # Calculate mean and max for each channel in this grid
            for channel in range(c):
                features.append(np.mean(grid[:,:,channel]))
                features.append(np.max(grid[:,:,channel]))
                features.append(np.min(grid[:,:,channel]))
                features.append(np.std(grid[:,:,channel]))
    
    # 6. Edge detection features (simple gradient approximation)
    # Convert to grayscale
    gray = np.mean(img_array, axis=2)  # [H, W]
    
    # Compute gradients
    grad_x = np.diff(gray, axis=1)  # [H, W-1]
    grad_y = np.diff(gray, axis=0)  # [H-1, W]
    
    # Pad to original size
    grad_x = np.pad(grad_x, ((0, 0), (0, 1)), mode='edge')
    grad_y = np.pad(grad_y, ((0, 1), (0, 0)), mode='edge')
    
    # Compute gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Statistics of gradient magnitude
    features.append(np.mean(gradient_magnitude))
    features.append(np.max(gradient_magnitude))
    features.append(np.std(gradient_magnitude))
    
    # Convert to tensor
    feature_vector = torch.tensor(features, dtype=torch.float32)
    
    return feature_vector


def extract_salient_features_v2(img_input: torch.Tensor, grid_size: Tuple[int, int]) -> torch.Tensor:
    """
    Extract salient features from an image, including:
      - Global stats (mean, std, min, max, brightness)
      - Grid-based stats
      - Edge gradients (magnitude & angle)
      - LBP histogram
      - HOG features
      - HSV stats
    
    Args:
        img_input: torch.Tensor of shape [C, H, W] and values in [0, 1]
        grid_size: Size of each grid (grid_h, grid_w) in pixels
    
    Returns:
        torch.Tensor: Feature vector of shape [n_features] (float32)
    """
    # Convert to numpy and transpose from [C,H,W] to [H,W,C]
    img_array = img_input.numpy().transpose(1, 2, 0)
    h, w, c = img_array.shape
    
    features = []

    # --------------------------
    # 1. Global channel statistics
    for channel in range(c):
        features.append(np.mean(img_array[:,:,channel]))
        features.append(np.std(img_array[:,:,channel]))
    
    brightness = np.mean(img_array)
    features.append(brightness)
    
    features.append(np.max(img_array))
    features.append(np.min(img_array))
    
    # --------------------------
    # 2. Grid-based features
    n_grids_h = h // grid_size[0]
    n_grids_w = w // grid_size[1]
    
    for i in range(n_grids_h):
        for j in range(n_grids_w):
            h_start = i * grid_size[0]
            h_end = (i + 1) * grid_size[0]
            w_start = j * grid_size[1]
            w_end = (j + 1) * grid_size[1]
            
            grid = img_array[h_start:h_end, w_start:w_end, :]
            for channel in range(c):
                features.append(np.mean(grid[:,:,channel]))
                features.append(np.max(grid[:,:,channel]))
                features.append(np.min(grid[:,:,channel]))
                features.append(np.std(grid[:,:,channel]))
    
    # --------------------------
    # 3. Edge/gradient features
    gray = np.mean(img_array, axis=2)
    grad_x = np.diff(gray, axis=1)
    grad_y = np.diff(gray, axis=0)
    grad_x = np.pad(grad_x, ((0,0),(0,1)), mode='edge')
    grad_y = np.pad(grad_y, ((0,1),(0,0)), mode='edge')
    
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    grad_angle = np.arctan2(grad_y, grad_x)
    
    # gradient magnitude statistics
    features.append(np.mean(grad_mag))
    features.append(np.max(grad_mag))
    features.append(np.std(grad_mag))
    
    # gradient angle statistics
    features.append(np.mean(grad_angle))
    features.append(np.std(grad_angle))
    features.append(np.max(grad_angle))
    
    # --------------------------
    # 4. LBP histogram (grayscale)
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), density=True)
    features.extend(lbp_hist.tolist())
    
    # --------------------------
    # 5. HOG features (grayscale, small cell for speed)
    hog_features = hog(gray, pixels_per_cell=(16,16), cells_per_block=(1,1), 
                       orientations=8, feature_vector=True)
    features.extend(hog_features.tolist())
    
    # --------------------------
    # 6. HSV statistics
    hsv = rgb2hsv(img_array)
    for i in range(3):  # H, S, V channels
        features.append(np.mean(hsv[:,:,i]))
        features.append(np.std(hsv[:,:,i]))
        features.append(np.max(hsv[:,:,i]))
        features.append(np.min(hsv[:,:,i]))
    
    # --------------------------
    feature_vector = torch.tensor(features, dtype=torch.float32)
    return feature_vector
