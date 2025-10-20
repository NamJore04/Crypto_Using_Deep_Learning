"""
Unit tests for trading system components.

Tests backtest engine, risk management, and signal generation.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from trading.backtest.backtest_engine import BacktestEngine, BacktestResults, Trade
from trading.risk_management.risk_manager import RiskManager, RiskLevel
from trading.signals.signal_generator import SignalGenerator, Signal, SignalType
from config.settings import TradingConfig


class TestBacktestEngine:
    """Test cases for BacktestEngine."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.trading_config = TradingConfig(
            initial_capital=10000.0,
            commission=0.001,
            max_position_size=0.1,
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
            max_drawdown=0.2,
            risk_per_trade=0.02
        )
        
        self.backtest_engine = BacktestEngine(self.trading_config)
    
    def test_initialization(self):
        """Test backtest engine initialization."""
        assert self.backtest_engine is not None
        assert self.backtest_engine.initial_capital == 10000.0
        assert self.backtest_engine.commission == 0.001
        assert self.backtest_engine.position == 0
        assert self.backtest_engine.capital == 10000.0
    
    def test_reset(self):
        """Test backtest engine reset."""
        # Execute a trade to change state
        self.backtest_engine.execute_trade('LONG', 50000.0, datetime.now(), 1000.0)
        
        # Reset
        self.backtest_engine.reset()
        
        assert self.backtest_engine.capital == 10000.0
        assert self.backtest_engine.position == 0
        assert self.backtest_engine.position_size == 0
        assert len(self.backtest_engine.trades) == 0
    
    def test_execute_long_trade(self):
        """Test long trade execution."""
        timestamp = datetime.now()
        price = 50000.0
        atr = 1000.0
        
        self.backtest_engine.execute_trade('LONG', price, timestamp, atr)
        
        assert self.backtest_engine.position == 1
        assert self.backtest_engine.position_size > 0
        assert self.backtest_engine.entry_price == price
        assert len(self.backtest_engine.trades) == 1
        
        # Check trade record
        trade = self.backtest_engine.trades[0]
        assert trade.action == 'OPEN_LONG'
        assert trade.price == price
        assert trade.timestamp == timestamp
    
    def test_execute_short_trade(self):
        """Test short trade execution."""
        timestamp = datetime.now()
        price = 50000.0
        atr = 1000.0
        
        self.backtest_engine.execute_trade('SHORT', price, timestamp, atr)
        
        assert self.backtest_engine.position == -1
        assert self.backtest_engine.position_size > 0
        assert self.backtest_engine.entry_price == price
        assert len(self.backtest_engine.trades) == 1
        
        # Check trade record
        trade = self.backtest_engine.trades[0]
        assert trade.action == 'OPEN_SHORT'
        assert trade.price == price
        assert trade.timestamp == timestamp
    
    def test_execute_close_trade(self):
        """Test close trade execution."""
        # First open a position
        self.backtest_engine.execute_trade('LONG', 50000.0, datetime.now(), 1000.0)
        initial_capital = self.backtest_engine.capital
        
        # Close the position
        close_price = 51000.0
        close_timestamp = datetime.now()
        self.backtest_engine.execute_trade('CLOSE', close_price, close_timestamp)
        
        assert self.backtest_engine.position == 0
        assert self.backtest_engine.position_size == 0
        assert len(self.backtest_engine.trades) == 2
        
        # Check that capital changed (P&L)
        assert self.backtest_engine.capital != initial_capital
    
    def test_calculate_metrics(self):
        """Test metrics calculation."""
        # Execute some trades
        self.backtest_engine.execute_trade('LONG', 50000.0, datetime.now(), 1000.0)
        self.backtest_engine.execute_trade('CLOSE', 51000.0, datetime.now())
        
        metrics = self.backtest_engine.calculate_metrics()
        
        assert isinstance(metrics, BacktestResults)
        assert metrics.total_return is not None
        assert metrics.sharpe_ratio is not None
        assert metrics.max_drawdown is not None
        assert metrics.win_rate is not None
        assert metrics.total_trades is not None
        assert metrics.final_capital is not None
    
    def test_position_sizing(self):
        """Test position sizing logic."""
        # Test with ATR-based sizing
        atr = 1000.0
        price = 50000.0
        
        self.backtest_engine.execute_trade('LONG', price, datetime.now(), atr)
        
        # Position size should be calculated based on risk
        expected_size = (self.backtest_engine.capital * self.trading_config.risk_per_trade) / (atr * 2)
        assert abs(self.backtest_engine.position_size - expected_size) < 0.01
    
    def test_commission_calculation(self):
        """Test commission calculation."""
        # Execute a trade
        self.backtest_engine.execute_trade('LONG', 50000.0, datetime.now(), 1000.0)
        
        # Check that commission is recorded
        trade = self.backtest_engine.trades[0]
        expected_commission = 50000.0 * self.backtest_engine.position_size * self.trading_config.commission
        assert abs(trade.commission - expected_commission) < 0.01


class TestRiskManager:
    """Test cases for RiskManager."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.trading_config = TradingConfig(
            initial_capital=10000.0,
            commission=0.001,
            max_position_size=0.1,
            stop_loss_atr=2.0,
            take_profit_atr=3.0,
            max_drawdown=0.2,
            risk_per_trade=0.02
        )
        
        self.risk_manager = RiskManager(self.trading_config)
    
    def test_initialization(self):
        """Test risk manager initialization."""
        assert self.risk_manager is not None
        assert self.risk_manager.max_position_size == 0.1
        assert self.risk_manager.stop_loss_atr == 2.0
        assert self.risk_manager.take_profit_atr == 3.0
    
    def test_calculate_position_size(self):
        """Test position size calculation."""
        capital = 10000.0
        price = 50000.0
        atr = 1000.0
        risk_per_trade = 0.02
        
        position_size = self.risk_manager.calculate_position_size(
            capital, price, atr, risk_per_trade
        )
        
        # Expected size based on risk
        expected_size = (capital * risk_per_trade) / (atr * self.risk_manager.stop_loss_atr)
        max_size = capital * self.risk_manager.max_position_size / price
        
        assert position_size == min(expected_size, max_size)
        assert position_size > 0
    
    def test_check_stop_loss_long(self):
        """Test stop loss check for long position."""
        entry_price = 50000.0
        atr = 1000.0
        
        # Price below stop loss
        current_price = entry_price - (atr * self.risk_manager.stop_loss_atr) - 100
        assert self.risk_manager.check_stop_loss(entry_price, current_price, atr, 'LONG') == True
        
        # Price above stop loss
        current_price = entry_price - (atr * self.risk_manager.stop_loss_atr) + 100
        assert self.risk_manager.check_stop_loss(entry_price, current_price, atr, 'LONG') == False
    
    def test_check_stop_loss_short(self):
        """Test stop loss check for short position."""
        entry_price = 50000.0
        atr = 1000.0
        
        # Price above stop loss
        current_price = entry_price + (atr * self.risk_manager.stop_loss_atr) + 100
        assert self.risk_manager.check_stop_loss(entry_price, current_price, atr, 'SHORT') == True
        
        # Price below stop loss
        current_price = entry_price + (atr * self.risk_manager.stop_loss_atr) - 100
        assert self.risk_manager.check_stop_loss(entry_price, current_price, atr, 'SHORT') == False
    
    def test_check_take_profit_long(self):
        """Test take profit check for long position."""
        entry_price = 50000.0
        atr = 1000.0
        
        # Price above take profit
        current_price = entry_price + (atr * self.risk_manager.take_profit_atr) + 100
        assert self.risk_manager.check_take_profit(entry_price, current_price, atr, 'LONG') == True
        
        # Price below take profit
        current_price = entry_price + (atr * self.risk_manager.take_profit_atr) - 100
        assert self.risk_manager.check_take_profit(entry_price, current_price, atr, 'LONG') == False
    
    def test_check_take_profit_short(self):
        """Test take profit check for short position."""
        entry_price = 50000.0
        atr = 1000.0
        
        # Price below take profit
        current_price = entry_price - (atr * self.risk_manager.take_profit_atr) - 100
        assert self.risk_manager.check_take_profit(entry_price, current_price, atr, 'SHORT') == True
        
        # Price above take profit
        current_price = entry_price - (atr * self.risk_manager.take_profit_atr) + 100
        assert self.risk_manager.check_take_profit(entry_price, current_price, atr, 'SHORT') == False
    
    def test_assess_risk(self):
        """Test risk assessment."""
        # Test different risk levels
        low_risk = self.risk_manager.assess_risk(0.01, 0.1, 0.05)
        medium_risk = self.risk_manager.assess_risk(0.05, 0.3, 0.15)
        high_risk = self.risk_manager.assess_risk(0.1, 0.5, 0.25)
        
        assert low_risk == RiskLevel.LOW
        assert medium_risk == RiskLevel.MEDIUM
        assert high_risk == RiskLevel.HIGH
    
    def test_portfolio_risk_controls(self):
        """Test portfolio-level risk controls."""
        # Test maximum drawdown check
        current_capital = 8000.0  # 20% drawdown
        max_drawdown = 0.2
        
        assert self.risk_manager.check_max_drawdown(current_capital, 10000.0, max_drawdown) == True
        
        # Test position size limits
        capital = 10000.0
        price = 50000.0
        atr = 1000.0
        
        position_size = self.risk_manager.calculate_position_size(capital, price, atr)
        max_allowed_size = capital * self.risk_manager.max_position_size / price
        
        assert position_size <= max_allowed_size


class TestSignalGenerator:
    """Test cases for SignalGenerator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.signal_config = {
            'breakout_threshold': 0.6,
            'trend_threshold': 0.5,
            'confidence_threshold': 0.4
        }
        
        self.trading_config = TradingConfig()
        
        self.signal_generator = SignalGenerator(
            signal_config=self.signal_config,
            trading_config=self.trading_config
        )
    
    def test_initialization(self):
        """Test signal generator initialization."""
        assert self.signal_generator is not None
        assert self.signal_generator.breakout_threshold == 0.6
        assert self.signal_generator.trend_threshold == 0.5
        assert self.signal_generator.confidence_threshold == 0.4
    
    def test_generate_signal_breakout_long(self):
        """Test signal generation for breakout long."""
        predictions = np.array([0.1, 0.1, 0.1, 0.7])  # High breakout probability
        price = 50000.0
        timestamp = datetime.now()
        atr = 1000.0
        
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            price=price,
            timestamp=timestamp,
            atr=atr
        )
        
        assert signal.signal_type == SignalType.LONG
        assert signal.confidence > 0.5
        assert signal.price == price
        assert signal.timestamp == timestamp
    
    def test_generate_signal_breakout_short(self):
        """Test signal generation for breakout short."""
        predictions = np.array([0.1, 0.1, 0.7, 0.1])  # High downtrend probability
        price = 50000.0
        timestamp = datetime.now()
        atr = 1000.0
        
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            price=price,
            timestamp=timestamp,
            atr=atr
        )
        
        assert signal.signal_type == SignalType.SHORT
        assert signal.confidence > 0.5
        assert signal.price == price
        assert signal.timestamp == timestamp
    
    def test_generate_signal_hold(self):
        """Test signal generation for hold."""
        predictions = np.array([0.4, 0.3, 0.2, 0.1])  # Low confidence
        price = 50000.0
        timestamp = datetime.now()
        atr = 1000.0
        
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            price=price,
            timestamp=timestamp,
            atr=atr
        )
        
        assert signal.signal_type == SignalType.HOLD
        assert signal.confidence < 0.5
        assert signal.price == price
        assert signal.timestamp == timestamp
    
    def test_generate_signal_close(self):
        """Test signal generation for close."""
        predictions = np.array([0.8, 0.1, 0.05, 0.05])  # High sideway probability
        price = 50000.0
        timestamp = datetime.now()
        atr = 1000.0
        
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            price=price,
            timestamp=timestamp,
            atr=atr
        )
        
        assert signal.signal_type == SignalType.CLOSE
        assert signal.confidence > 0.5
        assert signal.price == price
        assert signal.timestamp == timestamp
    
    def test_signal_filtering(self):
        """Test signal filtering logic."""
        # Test signal filtering
        signals = [
            Signal(datetime.now(), SignalType.LONG, 0.8, 0, 50000.0, {}),
            Signal(datetime.now(), SignalType.SHORT, 0.3, 1, 50000.0, {}),
            Signal(datetime.now(), SignalType.HOLD, 0.6, 2, 50000.0, {})
        ]
        
        filtered_signals = self.signal_generator.filter_signals(signals)
        
        # Should filter out low confidence signals
        assert len(filtered_signals) <= len(signals)
        for signal in filtered_signals:
            assert signal.confidence >= self.signal_generator.confidence_threshold
    
    def test_signal_confidence_scoring(self):
        """Test signal confidence scoring."""
        # Test different prediction scenarios
        high_confidence = np.array([0.1, 0.1, 0.1, 0.7])
        low_confidence = np.array([0.25, 0.25, 0.25, 0.25])
        
        signal1 = self.signal_generator.generate_signal(
            predictions=high_confidence,
            price=50000.0,
            timestamp=datetime.now(),
            atr=1000.0
        )
        
        signal2 = self.signal_generator.generate_signal(
            predictions=low_confidence,
            price=50000.0,
            timestamp=datetime.now(),
            atr=1000.0
        )
        
        assert signal1.confidence > signal2.confidence


class TestTradingSystemIntegration:
    """Integration tests for trading system."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.trading_config = TradingConfig()
        self.backtest_engine = BacktestEngine(self.trading_config)
        self.risk_manager = RiskManager(self.trading_config)
        self.signal_generator = SignalGenerator(
            signal_config={'breakout_threshold': 0.6, 'trend_threshold': 0.5, 'confidence_threshold': 0.4},
            trading_config=self.trading_config
        )
    
    def test_complete_trading_cycle(self):
        """Test complete trading cycle."""
        # Generate signals
        predictions = np.array([0.1, 0.1, 0.1, 0.7])  # Breakout signal
        signal = self.signal_generator.generate_signal(
            predictions=predictions,
            price=50000.0,
            timestamp=datetime.now(),
            atr=1000.0
        )
        
        # Execute trade
        self.backtest_engine.execute_trade(
            signal.signal_type.value,
            signal.price,
            signal.timestamp,
            atr=1000.0
        )
        
        # Check position
        assert self.backtest_engine.position != 0
        assert len(self.backtest_engine.trades) == 1
        
        # Close position
        self.backtest_engine.execute_trade('CLOSE', 51000.0, datetime.now())
        
        # Check final state
        assert self.backtest_engine.position == 0
        assert len(self.backtest_engine.trades) == 2
        
        # Calculate metrics
        metrics = self.backtest_engine.calculate_metrics()
        assert metrics.total_trades == 1
        assert metrics.final_capital != self.backtest_engine.initial_capital
    
    def test_risk_management_integration(self):
        """Test risk management integration."""
        # Test position sizing with risk management
        capital = 10000.0
        price = 50000.0
        atr = 1000.0
        
        position_size = self.risk_manager.calculate_position_size(capital, price, atr)
        
        # Execute trade with calculated position size
        self.backtest_engine.execute_trade('LONG', price, datetime.now(), atr)
        
        # Check that position size is within risk limits
        max_allowed_size = capital * self.risk_manager.max_position_size / price
        assert self.backtest_engine.position_size <= max_allowed_size
    
    def test_signal_to_trade_pipeline(self):
        """Test signal to trade pipeline."""
        # Create multiple signals
        signals = []
        for i in range(5):
            predictions = np.random.rand(4)
            predictions = predictions / predictions.sum()  # Normalize
            
            signal = self.signal_generator.generate_signal(
                predictions=predictions,
                price=50000.0 + i * 100,
                timestamp=datetime.now() + timedelta(minutes=i),
                atr=1000.0
            )
            signals.append(signal)
        
        # Execute trades based on signals
        for signal in signals:
            self.backtest_engine.execute_trade(
                signal.signal_type.value,
                signal.price,
                signal.timestamp,
                atr=1000.0
            )
        
        # Check results
        assert len(self.backtest_engine.trades) == len(signals)
        
        # Calculate final metrics
        metrics = self.backtest_engine.calculate_metrics()
        assert metrics.total_trades == len(signals)
        assert metrics.final_capital is not None


if __name__ == "__main__":
    pytest.main([__file__])
