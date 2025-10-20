"""
Trading system package for Crypto Futures Trading System.

This package provides comprehensive trading functionality including
backtesting, risk management, and signal generation.
"""

from .backtest.backtest_engine import BacktestEngine, BacktestResults, Trade
from .risk_management.risk_manager import RiskManager, RiskMetrics, RiskAlert, RiskLevel
from .signals.signal_generator import SignalGenerator, Signal, SignalType, MarketRegime, SignalConfig

__all__ = [
    # Backtest engine
    'BacktestEngine',
    'BacktestResults', 
    'Trade',
    
    # Risk management
    'RiskManager',
    'RiskMetrics',
    'RiskAlert',
    'RiskLevel',
    
    # Signal generation
    'SignalGenerator',
    'Signal',
    'SignalType',
    'MarketRegime',
    'SignalConfig'
]

__version__ = "1.0.0"
__author__ = "Crypto Trading Team"
__description__ = "Trading system for crypto futures with AI-powered signals"
