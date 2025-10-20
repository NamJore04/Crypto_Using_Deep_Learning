"""
Configuration settings for Crypto Futures Trading System.

This module contains all configuration settings for the trading system,
including database, API, trading, and model configurations.
"""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional


@dataclass
class DatabaseConfig:
    """Database configuration settings."""
    host: str = os.getenv('MONGO_HOST', 'localhost')
    port: int = int(os.getenv('MONGO_PORT', '27017'))
    database: str = os.getenv('MONGO_DB', 'crypto_trading')
    username: str = os.getenv('MONGO_USER', '')
    password: str = os.getenv('MONGO_PASS', '')
    
    @property
    def connection_string(self) -> str:
        """Get MongoDB connection string."""
        if self.username and self.password:
            return f"mongodb://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"
        return f"mongodb://{self.host}:{self.port}/{self.database}"


@dataclass
class TradingConfig:
    """Trading configuration settings."""
    initial_capital: float = 10000.0
    commission: float = 0.001
    max_position_size: float = 0.1
    stop_loss_atr: float = 2.0
    take_profit_atr: float = 3.0
    max_drawdown: float = 0.2
    risk_per_trade: float = 0.02


@dataclass
class APIConfig:
    """API configuration settings."""
    binance_api_key: str = os.getenv('BINANCE_API_KEY', '')
    binance_secret: str = os.getenv('BINANCE_SECRET', '')
    sandbox_mode: bool = True
    rate_limit: int = 1200
    timeout: int = 30


@dataclass
class DataConfig:
    """Data configuration settings."""
    feature_columns: list = None
    sequence_length: int = 60
    lookback_period: int = 20
    normalization_method: str = 'z_score'
    
    def __post_init__(self):
        if self.feature_columns is None:
            self.feature_columns = [
                'open', 'high', 'low', 'close', 'volume',
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'atr'
            ]


@dataclass
class BreakoutConfig:
    """Breakout detection configuration settings."""
    squeeze_threshold: float = 0.3
    donchian_period: int = 20
    future_return_threshold: float = 0.004
    lookback_hours: int = 4
    min_volume_ratio: float = 1.5
    future_horizon: int = 8


@dataclass
class ModelConfig:
    """Model configuration settings."""
    input_dim: int = 20
    sequence_length: int = 60
    hidden_dim: int = 64
    num_layers: int = 2
    num_classes: int = 4
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    early_stopping_patience: int = 20
    dropout_rate: float = 0.3


class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.trading = TradingConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.data = DataConfig()
        self.breakout = BreakoutConfig()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            'database': self.database.__dict__,
            'trading': self.trading.__dict__,
            'model': self.model.__dict__,
            'api': self.api.__dict__,
            'data': self.data.__dict__,
            'breakout': self.breakout.__dict__
        }
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        # Check required API credentials
        if not self.api.binance_api_key and not self.api.sandbox_mode:
            raise ValueError("Binance API key is required for live trading")
        
        # Check database connection
        if not self.database.host:
            raise ValueError("Database host is required")
        
        # Check trading parameters
        if self.trading.max_position_size > 1.0:
            raise ValueError("Maximum position size cannot exceed 100%")
        
        if self.trading.max_drawdown > 1.0:
            raise ValueError("Maximum drawdown cannot exceed 100%")
        
        return True


# Global configuration instance
config = Config()