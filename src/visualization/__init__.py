"""
Visualization package for Crypto Futures Trading System.

This package provides comprehensive visualization functionality including
interactive dashboards, chart generation, and performance analysis.
"""

from .dashboard import TradingDashboard
from .chart_utils import ChartGenerator, PerformanceAnalyzer

__all__ = [
    'TradingDashboard',
    'ChartGenerator', 
    'PerformanceAnalyzer'
]

__version__ = "1.0.0"
__author__ = "Crypto Trading Team"
__description__ = "Visualization tools for crypto trading system"
