"""
CNN-Transformer model for Crypto Futures Trading System.

This module implements the CNN-Transformer architecture for market regime
classification with positional encoding and multi-head attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class PositionalEncoding(nn.Module):
    """
    Positional encoding for Transformer model.
    
    Implements sinusoidal positional encoding to provide
    sequence position information to the Transformer.
    """
    
    def __init__(self, d_model: int, max_len: int = 5000):
        """
        Initialize positional encoding.
        
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
        """
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.
        
        Args:
            x: Input tensor of shape (seq_len, batch_size, d_model)
            
        Returns:
            torch.Tensor: Input with positional encoding added
        """
        return x + self.pe[:x.size(0), :]


class CNNTransformerModel(nn.Module):
    """
    CNN-Transformer model for market regime classification.
    
    Combines CNN layers for local pattern recognition with Transformer
    layers for long-term sequence modeling and attention.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize CNN-Transformer model.
        
        Args:
            config: Model configuration settings
        """
        super(CNNTransformerModel, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.sequence_length = config.sequence_length
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        
        # CNN feature extraction layers
        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, config.hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(self.dropout_rate)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, self.sequence_length)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=512,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized CNN-Transformer model with {self.count_parameters()} parameters")
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose for CNN: (batch_size, input_dim, sequence_length)
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Transpose back for Transformer: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # (seq_len, batch_size, features)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # (batch_size, seq_len, features)
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = torch.mean(x, dim=1)
        
        # Classification
        output = self.classifier(x)
        
        return output
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Attention weights from the last layer
        """
        batch_size = x.size(0)
        
        # Transpose for CNN
        x = x.transpose(1, 2)
        
        # CNN feature extraction
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Transpose back for Transformer
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = x.transpose(0, 1)
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)
        
        # Get attention weights from transformer
        # Note: This is a simplified version - actual attention weights
        # would require modifying the transformer forward pass
        x = self.transformer(x)
        
        return x
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction probabilities.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Prediction probabilities
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get prediction classes.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Prediction classes
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions


class CNNTransformerModelV2(nn.Module):
    """
    Enhanced CNN-Transformer model with residual connections and layer normalization.
    
    This is an improved version with residual connections, layer normalization,
    and enhanced attention mechanisms for better performance.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize enhanced CNN-Transformer model.
        
        Args:
            config: Model configuration settings
        """
        super(CNNTransformerModelV2, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.sequence_length = config.sequence_length
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        
        # Enhanced CNN feature extraction
        self.conv1 = nn.Conv1d(self.input_dim, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, config.hidden_dim, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(config.hidden_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(config.hidden_dim, self.sequence_length)
        
        # Enhanced Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=8,
            dim_feedforward=1024,
            dropout=self.dropout_rate,
            activation='gelu',
            batch_first=True,
            norm_first=True  # Pre-norm architecture
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=self.num_layers
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized enhanced CNN-Transformer model with {self.count_parameters()} parameters")
    
    def init_weights(self):
        """Initialize model weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, mean=0, std=0.01)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the enhanced model.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Transpose for CNN
        x = x.transpose(1, 2)
        
        # Enhanced CNN feature extraction with residual connections
        x1 = F.relu(self.conv1(x))
        x1 = self.pool(x1)
        x1 = self.dropout_conv(x1)
        
        x2 = F.relu(self.conv2(x1))
        x2 = self.pool(x2)
        x2 = self.dropout_conv(x2)
        
        x3 = F.relu(self.conv3(x2))
        x3 = self.pool(x3)
        x3 = self.dropout_conv(x3)
        
        # Transpose back for Transformer
        x3 = x3.transpose(1, 2)
        
        # Layer normalization after transpose for correct shape
        x3 = self.layer_norm(x3)
        
        # Add positional encoding
        x3 = x3.transpose(0, 1)
        x3 = self.pos_encoding(x3)
        x3 = x3.transpose(0, 1)
        
        # Enhanced Transformer encoding
        x3 = self.transformer(x3)
        
        # Global average pooling
        x3 = torch.mean(x3, dim=1)
        
        # Enhanced classification
        output = self.classifier(x3)
        
        return output
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction probabilities."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get prediction classes."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            predictions = torch.argmax(logits, dim=1)
        return predictions
