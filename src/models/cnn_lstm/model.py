"""
CNN-LSTM model for Crypto Futures Trading System.

This module implements the CNN-LSTM architecture for market regime
classification with attention mechanism and dropout regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import ModelConfig

logger = logging.getLogger(__name__)


class CNNLSTMModel(nn.Module):
    """
    CNN-LSTM model for market regime classification.
    
    Combines CNN layers for local pattern recognition with LSTM layers
    for sequence modeling and attention mechanism for focus.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize CNN-LSTM model.
        
        Args:
            config: Model configuration settings
        """
        super(CNNLSTMModel, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.sequence_length = config.sequence_length
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        
        # CNN layers for local pattern recognition
        self.conv1 = nn.Conv1d(self.input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(self.dropout_rate)
        
        # LSTM layers for sequence modeling
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, self.num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized CNN-LSTM model with {self.count_parameters()} parameters")
    
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
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Transpose back for LSTM: (batch_size, seq_len, features)
        x = x.transpose(1, 2)
        
        # LSTM sequence modeling
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Classification
        output = self.classifier(pooled)
        
        return output
    
    def count_parameters(self) -> int:
        """Count the number of trainable parameters in the model."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get attention weights for visualization.
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_dim)
            
        Returns:
            torch.Tensor: Attention weights
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
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)
        x = self.dropout_conv(x)
        
        # Transpose back for LSTM
        x = x.transpose(1, 2)
        
        # LSTM sequence modeling
        lstm_out, _ = self.lstm(x)
        
        # Attention mechanism
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        return attn_weights
    
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


class CNNLSTMModelV2(nn.Module):
    """
    Enhanced CNN-LSTM model with residual connections and layer normalization.
    
    This is an improved version with residual connections and layer normalization
    for better training stability and performance.
    """
    
    def __init__(self, config: ModelConfig):
        """
        Initialize enhanced CNN-LSTM model.
        
        Args:
            config: Model configuration settings
        """
        super(CNNLSTMModelV2, self).__init__()
        
        self.config = config
        self.input_dim = config.input_dim
        self.sequence_length = config.sequence_length
        self.hidden_dim = config.hidden_dim
        self.num_layers = config.num_layers
        self.num_classes = config.num_classes
        self.dropout_rate = config.dropout_rate
        
        # Enhanced CNN layers with residual connections
        self.conv1 = nn.Conv1d(self.input_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(128, 128, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool1d(2)
        self.dropout_conv = nn.Dropout(self.dropout_rate)
        self.layer_norm = nn.LayerNorm(128)
        
        # Enhanced LSTM layers
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=self.hidden_dim,
            num_layers=self.num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=self.dropout_rate if self.num_layers > 1 else 0
        )
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=self.hidden_dim * 2,
            num_heads=8,
            dropout=0.1,
            batch_first=True
        )
        
        # Enhanced classifier with residual connections
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, self.num_classes)
        )
        
        # Initialize weights
        self.init_weights()
        
        logger.info(f"Initialized enhanced CNN-LSTM model with {self.count_parameters()} parameters")
    
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
        
        # Residual connection
        x4 = F.relu(self.conv4(x3))
        x4 = x4 + x3  # Residual connection
        # Apply layer norm after transpose for correct shape
        x4 = self.dropout_conv(x4)
        
        # Transpose back for LSTM
        x4 = x4.transpose(1, 2)
        
        # Apply layer norm after transpose for correct shape
        x4 = self.layer_norm(x4)
        
        # LSTM sequence modeling
        lstm_out, (hidden, cell) = self.lstm(x4)
        
        # Multi-head attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )
        
        # Global average pooling
        pooled = torch.mean(attn_out, dim=1)
        
        # Enhanced classification
        output = self.classifier(pooled)
        
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
