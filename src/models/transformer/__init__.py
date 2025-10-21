"""
CNN-Transformer model package for Crypto Futures Trading System.

This package contains the CNN-Transformer model implementation for
market regime classification with enhanced attention mechanisms.
"""

from .model import CNNTransformerModel, CNNTransformerModelV2, PositionalEncoding

__all__ = [
    'CNNTransformerModel',
    'CNNTransformerModelV2',
    'PositionalEncoding'
]
