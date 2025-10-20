"""
Backtest engine for Crypto Futures Trading System.

This module provides comprehensive backtesting functionality with realistic
assumptions, transaction costs, and performance metrics calculation.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import dataclass

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.settings import TradingConfig

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Trade record data class."""
    timestamp: datetime
    action: str  # 'OPEN_LONG', 'OPEN_SHORT', 'CLOSE'
    price: float
    size: float
    capital: float
    pnl: Optional[float] = None
    commission: Optional[float] = None
    atr: Optional[float] = None


@dataclass
class BacktestResults:
    """Backtest results data class."""
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    final_capital: float
    equity_curve: pd.DataFrame
    trades: List[Trade]
    performance_metrics: Dict[str, Any]


class BacktestEngine:
    """
    Comprehensive backtest engine for crypto futures trading.
    
    Handles trade execution, position management, risk controls,
    and performance calculation with realistic assumptions.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize backtest engine.
        
        Args:
            config: Trading configuration settings
        """
        self.config = config
        self.initial_capital = config.initial_capital
        self.commission = config.commission
        self.max_position_size = config.max_position_size
        self.max_drawdown = config.max_drawdown
        
        # State variables
        self.capital = self.initial_capital
        self.position = 0  # 0: no position, 1: long, -1: short
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_timestamp = None
        
        # Records
        self.trades: List[Trade] = []
        self.equity_curve: List[Dict[str, Any]] = []
        self.daily_returns: List[float] = []
        
        # Performance tracking
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        
        logger.info(f"Initialized backtest engine with capital: {self.initial_capital}")
    
    def reset(self):
        """Reset backtest state to initial conditions."""
        self.capital = self.initial_capital
        self.position = 0
        self.position_size = 0.0
        self.entry_price = 0.0
        self.entry_timestamp = None
        
        self.trades.clear()
        self.equity_curve.clear()
        self.daily_returns.clear()
        
        self.peak_capital = self.initial_capital
        self.current_drawdown = 0.0
        self.max_drawdown_reached = 0.0
        
        logger.info("Backtest engine reset to initial state")
    
    def execute_trade(self, signal: str, price: float, timestamp: datetime, 
                    atr: Optional[float] = None, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Execute a trade based on signal.
        
        Args:
            signal: Trading signal ('LONG', 'SHORT', 'HOLD', 'CLOSE')
            price: Current price
            timestamp: Trade timestamp
            atr: Average True Range for position sizing
            metadata: Additional trade metadata
            
        Returns:
            bool: True if trade was executed, False otherwise
        """
        try:
            # Check for maximum drawdown limit
            if self.current_drawdown >= self.max_drawdown:
                logger.warning(f"Maximum drawdown limit reached: {self.current_drawdown:.2%}")
                return False
            
            # Execute trade based on signal
            if signal == 'LONG' and self.position <= 0:
                return self._open_long(price, timestamp, atr, metadata)
            elif signal == 'SHORT' and self.position >= 0:
                return self._open_short(price, timestamp, atr, metadata)
            elif signal == 'CLOSE' and self.position != 0:
                return self._close_position(price, timestamp, metadata)
            elif signal == 'HOLD':
                # Update equity curve without trading
                self._update_equity_curve(price, timestamp)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def _open_long(self, price: float, timestamp: datetime, atr: Optional[float] = None,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Open long position."""
        try:
            # Close existing short position first
            if self.position < 0:
                self._close_position(price, timestamp, metadata)
            
            # Calculate position size
            position_size = self._calculate_position_size(price, atr)
            if position_size <= 0:
                return False
            
            # Check if we have enough capital
            required_capital = position_size * price
            if required_capital > self.capital:
                logger.warning(f"Insufficient capital: {required_capital:.2f} > {self.capital:.2f}")
                return False
            
            # Execute long position
            self.position = 1
            self.position_size = position_size
            self.entry_price = price
            self.entry_timestamp = timestamp
            
            # Calculate commission
            commission_cost = required_capital * self.commission
            
            # Update capital
            self.capital -= commission_cost
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                action='OPEN_LONG',
                price=price,
                size=position_size,
                capital=self.capital,
                commission=commission_cost,
                atr=atr
            )
            self.trades.append(trade)
            
            # Update equity curve
            self._update_equity_curve(price, timestamp)
            
            logger.info(f"Opened LONG position: {position_size:.4f} @ {price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening long position: {e}")
            return False
    
    def _open_short(self, price: float, timestamp: datetime, atr: Optional[float] = None,
                   metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Open short position."""
        try:
            # Close existing long position first
            if self.position > 0:
                self._close_position(price, timestamp, metadata)
            
            # Calculate position size
            position_size = self._calculate_position_size(price, atr)
            if position_size <= 0:
                return False
            
            # Check if we have enough capital
            required_capital = position_size * price
            if required_capital > self.capital:
                logger.warning(f"Insufficient capital: {required_capital:.2f} > {self.capital:.2f}")
                return False
            
            # Execute short position
            self.position = -1
            self.position_size = position_size
            self.entry_price = price
            self.entry_timestamp = timestamp
            
            # Calculate commission
            commission_cost = required_capital * self.commission
            
            # Update capital
            self.capital -= commission_cost
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                action='OPEN_SHORT',
                price=price,
                size=position_size,
                capital=self.capital,
                commission=commission_cost,
                atr=atr
            )
            self.trades.append(trade)
            
            # Update equity curve
            self._update_equity_curve(price, timestamp)
            
            logger.info(f"Opened SHORT position: {position_size:.4f} @ {price:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error opening short position: {e}")
            return False
    
    def _close_position(self, price: float, timestamp: datetime, 
                      metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Close current position."""
        try:
            if self.position == 0:
                return False
            
            # Calculate P&L
            if self.position > 0:  # Long position
                pnl = (price - self.entry_price) * self.position_size
            else:  # Short position
                pnl = (self.entry_price - price) * self.position_size
            
            # Calculate commission
            trade_value = self.entry_price * self.position_size
            commission_cost = trade_value * self.commission
            
            # Apply commission to P&L
            pnl -= commission_cost
            
            # Update capital
            self.capital += pnl
            
            # Record trade
            trade = Trade(
                timestamp=timestamp,
                action='CLOSE',
                price=price,
                size=self.position_size,
                capital=self.capital,
                pnl=pnl,
                commission=commission_cost
            )
            self.trades.append(trade)
            
            # Reset position
            self.position = 0
            self.position_size = 0.0
            self.entry_price = 0.0
            self.entry_timestamp = None
            
            # Update equity curve
            self._update_equity_curve(price, timestamp)
            
            logger.info(f"Closed position: P&L = {pnl:.2f}, Capital = {self.capital:.2f}")
            return True
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False
    
    def _calculate_position_size(self, price: float, atr: Optional[float] = None) -> float:
        """
        Calculate position size based on risk management rules.
        
        Args:
            price: Current price
            atr: Average True Range for volatility-based sizing
            
        Returns:
            float: Position size
        """
        try:
            # Maximum position size based on capital
            max_size_by_capital = self.capital * self.max_position_size / price
            
            if atr is not None and atr > 0:
                # ATR-based position sizing
                risk_per_trade = self.config.risk_per_trade
                risk_amount = self.capital * risk_per_trade
                stop_distance = atr * self.config.stop_loss_atr
                atr_based_size = risk_amount / stop_distance
                
                # Use the smaller of the two
                position_size = min(max_size_by_capital, atr_based_size)
            else:
                # Fixed percentage of capital
                position_size = max_size_by_capital
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def _update_equity_curve(self, price: float, timestamp: datetime):
        """Update equity curve with current portfolio value."""
        try:
            # Calculate current portfolio value
            if self.position > 0:  # Long position
                portfolio_value = self.capital + (price - self.entry_price) * self.position_size
            elif self.position < 0:  # Short position
                portfolio_value = self.capital + (self.entry_price - price) * self.position_size
            else:  # No position
                portfolio_value = self.capital
            
            # Update peak capital and drawdown
            if portfolio_value > self.peak_capital:
                self.peak_capital = portfolio_value
                self.current_drawdown = 0.0
            else:
                self.current_drawdown = (self.peak_capital - portfolio_value) / self.peak_capital
                self.max_drawdown_reached = max(self.max_drawdown_reached, self.current_drawdown)
            
            # Record equity curve point
            equity_point = {
                'timestamp': timestamp,
                'capital': self.capital,
                'portfolio_value': portfolio_value,
                'position': self.position,
                'position_size': self.position_size,
                'entry_price': self.entry_price,
                'current_price': price,
                'drawdown': self.current_drawdown
            }
            self.equity_curve.append(equity_point)
            
        except Exception as e:
            logger.error(f"Error updating equity curve: {e}")
    
    def calculate_metrics(self) -> BacktestResults:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            BacktestResults: Complete backtest results
        """
        try:
            if not self.trades:
                logger.warning("No trades executed, returning empty results")
                return self._empty_results()
            
            # Convert equity curve to DataFrame
            equity_df = pd.DataFrame(self.equity_curve)
            if equity_df.empty:
                return self._empty_results()
            
            # Calculate returns
            equity_df['returns'] = equity_df['portfolio_value'].pct_change()
            equity_df['cumulative_returns'] = (1 + equity_df['returns']).cumprod() - 1
            
            # Basic metrics
            total_return = (equity_df['portfolio_value'].iloc[-1] / self.initial_capital) - 1
            final_capital = equity_df['portfolio_value'].iloc[-1]
            
            # Risk metrics
            returns = equity_df['returns'].dropna()
            if len(returns) > 1:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
            else:
                sharpe_ratio = 0
            
            max_drawdown = self.max_drawdown_reached
            
            # Trade metrics
            closed_trades = [t for t in self.trades if t.action == 'CLOSE' and t.pnl is not None]
            total_trades = len(closed_trades)
            
            if closed_trades:
                winning_trades = [t for t in closed_trades if t.pnl > 0]
                win_rate = len(winning_trades) / total_trades
            else:
                win_rate = 0
            
            # Additional performance metrics
            performance_metrics = self._calculate_additional_metrics(equity_df, closed_trades)
            
            results = BacktestResults(
                total_return=total_return,
                sharpe_ratio=sharpe_ratio,
                max_drawdown=max_drawdown,
                win_rate=win_rate,
                total_trades=total_trades,
                final_capital=final_capital,
                equity_curve=equity_df,
                trades=self.trades,
                performance_metrics=performance_metrics
            )
            
            logger.info(f"Backtest completed: Return={total_return:.2%}, Sharpe={sharpe_ratio:.2f}, "
                       f"MaxDD={max_drawdown:.2%}, WinRate={win_rate:.2%}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            return self._empty_results()
    
    def _calculate_additional_metrics(self, equity_df: pd.DataFrame, 
                                    closed_trades: List[Trade]) -> Dict[str, Any]:
        """Calculate additional performance metrics."""
        try:
            metrics = {}
            
            # Volatility
            returns = equity_df['returns'].dropna()
            if len(returns) > 1:
                metrics['volatility'] = returns.std() * np.sqrt(252)
                metrics['skewness'] = returns.skew()
                metrics['kurtosis'] = returns.kurtosis()
            else:
                metrics['volatility'] = 0
                metrics['skewness'] = 0
                metrics['kurtosis'] = 0
            
            # Trade statistics
            if closed_trades:
                pnls = [t.pnl for t in closed_trades]
                metrics['avg_trade_pnl'] = np.mean(pnls)
                metrics['best_trade'] = max(pnls)
                metrics['worst_trade'] = min(pnls)
                metrics['profit_factor'] = sum([p for p in pnls if p > 0]) / abs(sum([p for p in pnls if p < 0])) if any(p < 0 for p in pnls) else float('inf')
            else:
                metrics['avg_trade_pnl'] = 0
                metrics['best_trade'] = 0
                metrics['worst_trade'] = 0
                metrics['profit_factor'] = 0
            
            # Drawdown analysis
            equity_df['peak'] = equity_df['portfolio_value'].expanding().max()
            equity_df['drawdown'] = (equity_df['portfolio_value'] - equity_df['peak']) / equity_df['peak']
            metrics['avg_drawdown'] = equity_df['drawdown'].mean()
            metrics['max_drawdown_duration'] = self._calculate_max_drawdown_duration(equity_df)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating additional metrics: {e}")
            return {}
    
    def _calculate_max_drawdown_duration(self, equity_df: pd.DataFrame) -> int:
        """Calculate maximum drawdown duration in periods."""
        try:
            in_drawdown = equity_df['drawdown'] < 0
            drawdown_periods = []
            current_period = 0
            
            for is_dd in in_drawdown:
                if is_dd:
                    current_period += 1
                else:
                    if current_period > 0:
                        drawdown_periods.append(current_period)
                    current_period = 0
            
            if current_period > 0:
                drawdown_periods.append(current_period)
            
            return max(drawdown_periods) if drawdown_periods else 0
            
        except Exception as e:
            logger.error(f"Error calculating max drawdown duration: {e}")
            return 0
    
    def _empty_results(self) -> BacktestResults:
        """Return empty results when no trades were executed."""
        return BacktestResults(
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            win_rate=0.0,
            total_trades=0,
            final_capital=self.initial_capital,
            equity_curve=pd.DataFrame(),
            trades=[],
            performance_metrics={}
        )
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current backtest status."""
        return {
            'capital': self.capital,
            'position': self.position,
            'position_size': self.position_size,
            'entry_price': self.entry_price,
            'current_drawdown': self.current_drawdown,
            'total_trades': len(self.trades),
            'equity_points': len(self.equity_curve)
        }
    
    def export_trades(self, filepath: str) -> bool:
        """
        Export trades to CSV file.
        
        Args:
            filepath: Path to save trades CSV
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.trades:
                logger.warning("No trades to export")
                return False
            
            trades_data = []
            for trade in self.trades:
                trades_data.append({
                    'timestamp': trade.timestamp,
                    'action': trade.action,
                    'price': trade.price,
                    'size': trade.size,
                    'capital': trade.capital,
                    'pnl': trade.pnl,
                    'commission': trade.commission,
                    'atr': trade.atr
                })
            
            trades_df = pd.DataFrame(trades_data)
            trades_df.to_csv(filepath, index=False)
            
            logger.info(f"Exported {len(self.trades)} trades to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting trades: {e}")
            return False
