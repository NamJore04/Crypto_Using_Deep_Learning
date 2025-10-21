"""
Ensemble model for Crypto Futures Trading System.

This module implements ensemble methods combining CNN-LSTM and CNN-Transformer
models for improved market regime classification performance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import ModelConfig
from ..cnn_lstm.model import CNNLSTMModel, CNNLSTMModelV2
from ..transformer.model import CNNTransformerModel, CNNTransformerModelV2

logger = logging.getLogger(__name__)


class ModelEnsemble(nn.Module):
    """
    Ensemble model combining multiple CNN-LSTM and CNN-Transformer models.
    
    Uses weighted averaging or voting to combine predictions from multiple
    models for improved performance and robustness.
    """
    
    def __init__(self, models: List[nn.Module], weights: Optional[List[float]] = None,
                 ensemble_method: str = 'weighted_average'):
        """
        Initialize ensemble model.
        
        Args:
            models: List of models to ensemble
            weights: Weights for each model (equal weights if None)
            ensemble_method: Method for combining predictions ('weighted_average', 'voting')
        """
        super(ModelEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        
        if weights is None:
            self.weights = [1.0 / len(models)] * len(models)
        else:
            self.weights = weights
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        logger.info(f"Initialized ensemble with {len(models)} models using {ensemble_method}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Ensemble predictions
        """
        predictions = []
        
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        if self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            ensemble_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += weight * pred
            return ensemble_pred
        
        elif self.ensemble_method == 'voting':
            # Voting-based ensemble
            # Convert logits to probabilities
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            
            # Weighted voting
            ensemble_prob = torch.zeros_like(probs[0])
            for prob, weight in zip(probs, self.weights):
                ensemble_prob += weight * prob
            
            # Convert back to logits
            ensemble_pred = torch.log(ensemble_prob + 1e-8)
            return ensemble_pred
        
        else:
            raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")
    
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
    
    def get_individual_predictions(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get predictions from individual models.
        
        Args:
            x: Input tensor
            
        Returns:
            List[torch.Tensor]: Individual model predictions
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
        
        return predictions
    
    def predict_weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """Get weighted average predictions."""
        self.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            # Weighted average
            ensemble_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += weight * pred
            
            return ensemble_pred
    
    def predict_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Get voting-based predictions."""
        self.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            # Convert to probabilities and vote
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            
            # Weighted voting
            ensemble_prob = torch.zeros_like(probs[0])
            for prob, weight in zip(probs, self.weights):
                ensemble_prob += weight * prob
            
            # Convert back to logits
            ensemble_pred = torch.log(ensemble_prob + 1e-8)
            return ensemble_pred


class AdaptiveEnsemble(nn.Module):
    """
    Adaptive ensemble that learns to weight models based on input characteristics.
    
    Uses a meta-learner to dynamically adjust model weights based on
    input features and historical performance.
    """
    
    def __init__(self, models: List[nn.Module], meta_hidden_dim: int = 64):
        """
        Initialize adaptive ensemble.
        
        Args:
            models: List of models to ensemble
            meta_hidden_dim: Hidden dimension for meta-learner
        """
        super(AdaptiveEnsemble, self).__init__()
        
        self.models = nn.ModuleList(models)
        self.num_models = len(models)
        
        # Meta-learner for adaptive weighting
        self.meta_learner = nn.Sequential(
            nn.Linear(models[0].input_dim, meta_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(meta_hidden_dim, meta_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(meta_hidden_dim // 2, self.num_models),
            nn.Softmax(dim=1)
        )
        
        logger.info(f"Initialized adaptive ensemble with {len(models)} models")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through adaptive ensemble.
        
        Args:
            x: Input tensor
            
        Returns:
            torch.Tensor: Ensemble predictions
        """
        # Get individual predictions
        predictions = []
        for model in self.models:
            pred = model(x)
            predictions.append(pred)
        
        # Get adaptive weights from meta-learner
        # Use mean of input features for meta-learner
        input_features = torch.mean(x, dim=1)  # (batch_size, input_dim)
        weights = self.meta_learner(input_features)  # (batch_size, num_models)
        
        # Weighted combination
        ensemble_pred = torch.zeros_like(predictions[0])
        for i, pred in enumerate(predictions):
            ensemble_pred += weights[:, i:i+1] * pred
        
        return ensemble_pred
    
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
    
    def predict_weighted_average(self, x: torch.Tensor) -> torch.Tensor:
        """Get weighted average predictions."""
        self.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            # Weighted average
            ensemble_pred = torch.zeros_like(predictions[0])
            for pred, weight in zip(predictions, self.weights):
                ensemble_pred += weight * pred
            
            return ensemble_pred
    
    def predict_voting(self, x: torch.Tensor) -> torch.Tensor:
        """Get voting-based predictions."""
        self.eval()
        with torch.no_grad():
            predictions = []
            for model in self.models:
                pred = model(x)
                predictions.append(pred)
            
            # Convert to probabilities and vote
            probs = [F.softmax(pred, dim=1) for pred in predictions]
            
            # Weighted voting
            ensemble_prob = torch.zeros_like(probs[0])
            for prob, weight in zip(probs, self.weights):
                ensemble_prob += weight * prob
            
            # Convert back to logits
            ensemble_pred = torch.log(ensemble_prob + 1e-8)
            return ensemble_pred


def create_ensemble_from_configs(configs: List[ModelConfig], 
                                model_types: List[str]) -> ModelEnsemble:
    """
    Create ensemble from multiple model configurations.
    
    Args:
        configs: List of model configurations
        model_types: List of model types ('cnn_lstm', 'cnn_lstm_v2', 'transformer', 'transformer_v2')
        
    Returns:
        ModelEnsemble: Ensemble model
    """
    models = []
    
    for config, model_type in zip(configs, model_types):
        if model_type == 'cnn_lstm':
            model = CNNLSTMModel(config)
        elif model_type == 'cnn_lstm_v2':
            model = CNNLSTMModelV2(config)
        elif model_type == 'transformer':
            model = CNNTransformerModel(config)
        elif model_type == 'transformer_v2':
            model = CNNTransformerModelV2(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    return ModelEnsemble(models)


def create_adaptive_ensemble_from_configs(configs: List[ModelConfig],
                                       model_types: List[str]) -> AdaptiveEnsemble:
    """
    Create adaptive ensemble from multiple model configurations.
    
    Args:
        configs: List of model configurations
        model_types: List of model types
        
    Returns:
        AdaptiveEnsemble: Adaptive ensemble model
    """
    models = []
    
    for config, model_type in zip(configs, model_types):
        if model_type == 'cnn_lstm':
            model = CNNLSTMModel(config)
        elif model_type == 'cnn_lstm_v2':
            model = CNNLSTMModelV2(config)
        elif model_type == 'transformer':
            model = CNNTransformerModel(config)
        elif model_type == 'transformer_v2':
            model = CNNTransformerModelV2(config)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models.append(model)
    
    return AdaptiveEnsemble(models)
