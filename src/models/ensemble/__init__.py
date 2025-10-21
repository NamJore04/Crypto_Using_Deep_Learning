"""
Ensemble models package for Crypto Futures Trading System.

This package contains ensemble methods for combining multiple
CNN-LSTM and CNN-Transformer models.
"""

from .model_ensemble import (
    ModelEnsemble,
    AdaptiveEnsemble,
    create_ensemble_from_configs,
    create_adaptive_ensemble_from_configs
)

__all__ = [
    'ModelEnsemble',
    'AdaptiveEnsemble',
    'create_ensemble_from_configs',
    'create_adaptive_ensemble_from_configs'
]
