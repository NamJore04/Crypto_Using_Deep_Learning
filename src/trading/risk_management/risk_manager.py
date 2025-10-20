"""
Risk management system for Crypto Futures Trading System.

This module provides comprehensive risk management functionality including
position sizing, stop loss, take profit, and portfolio-level risk controls.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List
import logging
from dataclasses import dataclass
from enum import Enum

from ...config.settings import TradingConfig

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    """Risk level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskMetrics:
    """Risk metrics data class."""
    portfolio_value: float
    position_size: float
    position_value: float
    risk_per_trade: float
    max_position_size: float
    current_drawdown: float
    max_drawdown: float
    volatility: float
    var_95: float  # Value at Risk 95%
    expected_shortfall: float


@dataclass
class RiskAlert:
    """Risk alert data class."""
    timestamp: datetime
    level: RiskLevel
    message: str
    metric: str
    value: float
    threshold: float
    action_required: bool


class RiskManager:
    """
    Comprehensive risk management system.
    
    Handles position sizing, stop loss, take profit, portfolio-level
    risk controls, and risk monitoring with real-time alerts.
    """
    
    def __init__(self, config: TradingConfig):
        """
        Initialize risk manager.
        
        Args:
            config: Trading configuration settings
        """
        self.config = config
        self.max_position_size = config.max_position_size
        self.stop_loss_atr = config.stop_loss_atr
        self.take_profit_atr = config.take_profit_atr
        self.max_drawdown = config.max_drawdown
        self.risk_per_trade = config.risk_per_trade
        
        # Risk monitoring
        self.risk_alerts: List[RiskAlert] = []
        self.portfolio_history: List[Dict[str, Any]] = []
        self.current_risk_level = RiskLevel.LOW
        
        # Risk limits
        self.max_daily_loss = 0.05  # 5% max daily loss
        self.max_weekly_loss = 0.15  # 15% max weekly loss
        self.max_consecutive_losses = 5
        self.consecutive_losses = 0
        
        # Volatility tracking
        self.volatility_window = 20
        self.volatility_history: List[float] = []
        
        logger.info("Risk manager initialized with comprehensive risk controls")
    
    def calculate_position_size(self, capital: float, price: float, atr: Optional[float] = None,
                              volatility: Optional[float] = None, risk_level: Optional[RiskLevel] = None) -> float:
        """
        Calculate optimal position size based on risk management rules.
        
        Args:
            capital: Available capital
            price: Current price
            atr: Average True Range for volatility-based sizing
            volatility: Historical volatility
            risk_level: Current risk level
            
        Returns:
            float: Recommended position size
        """
        try:
            # Base position size from capital
            max_size_by_capital = capital * self.max_position_size / price
            
            # ATR-based position sizing
            if atr is not None and atr > 0:
                risk_amount = capital * self.risk_per_trade
                stop_distance = atr * self.stop_loss_atr
                atr_based_size = risk_amount / stop_distance
            else:
                atr_based_size = max_size_by_capital
            
            # Volatility-based adjustment
            if volatility is not None and volatility > 0:
                # Reduce position size for high volatility
                volatility_adjustment = min(1.0, 0.3 / volatility)  # Cap at 30% of normal volatility
                atr_based_size *= volatility_adjustment
            
            # Risk level adjustment
            if risk_level is not None:
                if risk_level == RiskLevel.HIGH:
                    atr_based_size *= 0.5  # Reduce by 50%
                elif risk_level == RiskLevel.CRITICAL:
                    atr_based_size *= 0.25  # Reduce by 75%
            
            # Use the smaller of the calculated sizes
            position_size = min(max_size_by_capital, atr_based_size)
            
            # Ensure minimum viable position
            min_position = 0.001  # 0.1% of capital
            position_size = max(position_size, min_position)
            
            logger.debug(f"Position size calculated: {position_size:.4f} (Capital: {capital:.2f}, "
                        f"Price: {price:.2f}, ATR: {atr}, Vol: {volatility})")
            
            return max(0, position_size)
            
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
            return 0.0
    
    def check_stop_loss(self, entry_price: float, current_price: float, atr: float,
                       position_type: str) -> Tuple[bool, float]:
        """
        Check if stop loss is triggered.
        
        Args:
            entry_price: Entry price of position
            current_price: Current market price
            atr: Average True Range
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple[bool, float]: (is_triggered, stop_price)
        """
        try:
            if atr <= 0:
                return False, 0.0
            
            if position_type == 'LONG':
                stop_price = entry_price - (atr * self.stop_loss_atr)
                is_triggered = current_price <= stop_price
            elif position_type == 'SHORT':
                stop_price = entry_price + (atr * self.stop_loss_atr)
                is_triggered = current_price >= stop_price
            else:
                return False, 0.0
            
            if is_triggered:
                logger.info(f"Stop loss triggered for {position_type}: {current_price:.2f} vs {stop_price:.2f}")
            
            return is_triggered, stop_price
            
        except Exception as e:
            logger.error(f"Error checking stop loss: {e}")
            return False, 0.0
    
    def check_take_profit(self, entry_price: float, current_price: float, atr: float,
                         position_type: str) -> Tuple[bool, float]:
        """
        Check if take profit is triggered.
        
        Args:
            entry_price: Entry price of position
            current_price: Current market price
            atr: Average True Range
            position_type: 'LONG' or 'SHORT'
            
        Returns:
            Tuple[bool, float]: (is_triggered, take_profit_price)
        """
        try:
            if atr <= 0:
                return False, 0.0
            
            if position_type == 'LONG':
                tp_price = entry_price + (atr * self.take_profit_atr)
                is_triggered = current_price >= tp_price
            elif position_type == 'SHORT':
                tp_price = entry_price - (atr * self.take_profit_atr)
                is_triggered = current_price <= tp_price
            else:
                return False, 0.0
            
            if is_triggered:
                logger.info(f"Take profit triggered for {position_type}: {current_price:.2f} vs {tp_price:.2f}")
            
            return is_triggered, tp_price
            
        except Exception as e:
            logger.error(f"Error checking take profit: {e}")
            return False, 0.0
    
    def check_portfolio_risk(self, portfolio_value: float, position_value: float,
                           daily_pnl: float, weekly_pnl: float) -> List[RiskAlert]:
        """
        Check portfolio-level risk limits.
        
        Args:
            portfolio_value: Current portfolio value
            position_value: Current position value
            daily_pnl: Daily P&L
            weekly_pnl: Weekly P&L
            
        Returns:
            List[RiskAlert]: List of risk alerts
        """
        alerts = []
        timestamp = datetime.now()
        
        try:
            # Check daily loss limit
            daily_loss_pct = abs(daily_pnl) / portfolio_value if portfolio_value > 0 else 0
            if daily_loss_pct > self.max_daily_loss:
                alert = RiskAlert(
                    timestamp=timestamp,
                    level=RiskLevel.CRITICAL,
                    message=f"Daily loss limit exceeded: {daily_loss_pct:.2%} > {self.max_daily_loss:.2%}",
                    metric="daily_loss",
                    value=daily_loss_pct,
                    threshold=self.max_daily_loss,
                    action_required=True
                )
                alerts.append(alert)
            
            # Check weekly loss limit
            weekly_loss_pct = abs(weekly_pnl) / portfolio_value if portfolio_value > 0 else 0
            if weekly_loss_pct > self.max_weekly_loss:
                alert = RiskAlert(
                    timestamp=timestamp,
                    level=RiskLevel.CRITICAL,
                    message=f"Weekly loss limit exceeded: {weekly_loss_pct:.2%} > {self.max_weekly_loss:.2%}",
                    metric="weekly_loss",
                    value=weekly_loss_pct,
                    threshold=self.max_weekly_loss,
                    action_required=True
                )
                alerts.append(alert)
            
            # Check position size limit
            position_pct = position_value / portfolio_value if portfolio_value > 0 else 0
            if position_pct > self.max_position_size:
                alert = RiskAlert(
                    timestamp=timestamp,
                    level=RiskLevel.HIGH,
                    message=f"Position size exceeded: {position_pct:.2%} > {self.max_position_size:.2%}",
                    metric="position_size",
                    value=position_pct,
                    threshold=self.max_position_size,
                    action_required=True
                )
                alerts.append(alert)
            
            # Check consecutive losses
            if self.consecutive_losses >= self.max_consecutive_losses:
                alert = RiskAlert(
                    timestamp=timestamp,
                    level=RiskLevel.HIGH,
                    message=f"Too many consecutive losses: {self.consecutive_losses} >= {self.max_consecutive_losses}",
                    metric="consecutive_losses",
                    value=self.consecutive_losses,
                    threshold=self.max_consecutive_losses,
                    action_required=True
                )
                alerts.append(alert)
            
            # Update risk level based on alerts
            if alerts:
                critical_alerts = [a for a in alerts if a.level == RiskLevel.CRITICAL]
                high_alerts = [a for a in alerts if a.level == RiskLevel.HIGH]
                
                if critical_alerts:
                    self.current_risk_level = RiskLevel.CRITICAL
                elif high_alerts:
                    self.current_risk_level = RiskLevel.HIGH
                else:
                    self.current_risk_level = RiskLevel.MEDIUM
            else:
                self.current_risk_level = RiskLevel.LOW
            
            # Store alerts
            self.risk_alerts.extend(alerts)
            
            return alerts
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk: {e}")
            return []
    
    def update_trade_result(self, pnl: float, is_winning: bool):
        """
        Update risk manager with trade result.
        
        Args:
            pnl: Trade P&L
            is_winning: Whether trade was profitable
        """
        try:
            if is_winning:
                self.consecutive_losses = 0
            else:
                self.consecutive_losses += 1
            
            logger.debug(f"Trade result updated: PnL={pnl:.2f}, Winning={is_winning}, "
                        f"Consecutive losses={self.consecutive_losses}")
            
        except Exception as e:
            logger.error(f"Error updating trade result: {e}")
    
    def calculate_risk_metrics(self, portfolio_value: float, position_value: float,
                              returns_history: List[float]) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics.
        
        Args:
            portfolio_value: Current portfolio value
            position_value: Current position value
            returns_history: Historical returns
            
        Returns:
            RiskMetrics: Comprehensive risk metrics
        """
        try:
            # Basic metrics
            position_size = position_value / portfolio_value if portfolio_value > 0 else 0
            risk_per_trade = self.risk_per_trade
            max_position_size = self.max_position_size
            
            # Calculate drawdown
            if len(self.portfolio_history) > 1:
                portfolio_values = [h['portfolio_value'] for h in self.portfolio_history]
                peak = max(portfolio_values)
                current_drawdown = (peak - portfolio_value) / peak if peak > 0 else 0
                max_drawdown = max([(peak - val) / peak for val in portfolio_values]) if peak > 0 else 0
            else:
                current_drawdown = 0
                max_drawdown = 0
            
            # Calculate volatility and VaR
            if len(returns_history) > 1:
                returns_array = np.array(returns_history)
                volatility = np.std(returns_array) * np.sqrt(252)  # Annualized
                var_95 = np.percentile(returns_array, 5)  # 5th percentile
                expected_shortfall = np.mean(returns_array[returns_array <= var_95])
            else:
                volatility = 0
                var_95 = 0
                expected_shortfall = 0
            
            metrics = RiskMetrics(
                portfolio_value=portfolio_value,
                position_size=position_size,
                position_value=position_value,
                risk_per_trade=risk_per_trade,
                max_position_size=max_position_size,
                current_drawdown=current_drawdown,
                max_drawdown=max_drawdown,
                volatility=volatility,
                var_95=var_95,
                expected_shortfall=expected_shortfall
            )
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating risk metrics: {e}")
            return RiskMetrics(
                portfolio_value=portfolio_value,
                position_size=0,
                position_value=position_value,
                risk_per_trade=self.risk_per_trade,
                max_position_size=self.max_position_size,
                current_drawdown=0,
                max_drawdown=0,
                volatility=0,
                var_95=0,
                expected_shortfall=0
            )
    
    def update_portfolio_history(self, portfolio_value: float, position_value: float,
                               timestamp: datetime, additional_data: Optional[Dict[str, Any]] = None):
        """
        Update portfolio history for risk monitoring.
        
        Args:
            portfolio_value: Current portfolio value
            position_value: Current position value
            timestamp: Timestamp of update
            additional_data: Additional data to store
        """
        try:
            history_entry = {
                'timestamp': timestamp,
                'portfolio_value': portfolio_value,
                'position_value': position_value,
                'risk_level': self.current_risk_level.value
            }
            
            if additional_data:
                history_entry.update(additional_data)
            
            self.portfolio_history.append(history_entry)
            
            # Keep only recent history (last 1000 entries)
            if len(self.portfolio_history) > 1000:
                self.portfolio_history = self.portfolio_history[-1000:]
            
        except Exception as e:
            logger.error(f"Error updating portfolio history: {e}")
    
    def get_risk_summary(self) -> Dict[str, Any]:
        """Get current risk summary."""
        try:
            return {
                'current_risk_level': self.current_risk_level.value,
                'consecutive_losses': self.consecutive_losses,
                'max_consecutive_losses': self.max_consecutive_losses,
                'max_daily_loss': self.max_daily_loss,
                'max_weekly_loss': self.max_weekly_loss,
                'max_position_size': self.max_position_size,
                'risk_per_trade': self.risk_per_trade,
                'total_alerts': len(self.risk_alerts),
                'recent_alerts': len([a for a in self.risk_alerts if a.timestamp > datetime.now() - timedelta(hours=24)])
            }
            
        except Exception as e:
            logger.error(f"Error getting risk summary: {e}")
            return {}
    
    def reset_risk_state(self):
        """Reset risk manager state."""
        try:
            self.risk_alerts.clear()
            self.portfolio_history.clear()
            self.current_risk_level = RiskLevel.LOW
            self.consecutive_losses = 0
            self.volatility_history.clear()
            
            logger.info("Risk manager state reset")
            
        except Exception as e:
            logger.error(f"Error resetting risk state: {e}")
    
    def export_risk_report(self, filepath: str) -> bool:
        """
        Export risk report to CSV.
        
        Args:
            filepath: Path to save risk report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.portfolio_history:
                logger.warning("No portfolio history to export")
                return False
            
            # Create risk report DataFrame
            risk_df = pd.DataFrame(self.portfolio_history)
            
            # Add alerts if any
            if self.risk_alerts:
                alerts_data = []
                for alert in self.risk_alerts:
                    alerts_data.append({
                        'timestamp': alert.timestamp,
                        'level': alert.level.value,
                        'message': alert.message,
                        'metric': alert.metric,
                        'value': alert.value,
                        'threshold': alert.threshold,
                        'action_required': alert.action_required
                    })
                
                alerts_df = pd.DataFrame(alerts_data)
                risk_df.to_csv(filepath, index=False)
                
                # Save alerts separately
                alerts_filepath = filepath.replace('.csv', '_alerts.csv')
                alerts_df.to_csv(alerts_filepath, index=False)
            else:
                risk_df.to_csv(filepath, index=False)
            
            logger.info(f"Risk report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting risk report: {e}")
            return False
