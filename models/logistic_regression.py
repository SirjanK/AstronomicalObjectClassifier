"""
Simple logistic regression model for astronomical object classification.
"""

import torch
import torch.nn as nn


class LogisticRegressionModel(nn.Module):
    """
    Logistic regression model.
    
    Args:
        input_dim: Size of input feature vector
        num_classes: Number of output classes
        dropout: Dropout probability (optional, applied before final layer)
    """
    
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.0):
        super().__init__()
        
        # Optional dropout layer
        if dropout > 0:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None
        
        # Single linear layer
        self.linear = nn.Linear(input_dim, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
        
        Returns:
            Logits tensor of shape [batch_size, num_classes]
        """
        if self.dropout is not None:
            x = self.dropout(x)
        
        return self.linear(x)
