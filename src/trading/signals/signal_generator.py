"""
Signal generation system for Crypto Futures Trading System.

This module provides signal generation from model predictions, including
signal conversion, filtering, and risk-based signal adjustment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple, List, Union
import logging
from dataclasses import dataclass
from enum import Enum

from ...config.settings import TradingConfig, ModelConfig

logger = logging.getLogger(__name__)


class SignalType(Enum):
    """Trading signal types."""
    LONG = "LONG"
    SHORT = "SHORT"
    HOLD = "HOLD"
    CLOSE = "CLOSE"


class MarketRegime(Enum):
    """Market regime types."""
    SIDEWAY = 0
    UPTREND = 1
    DOWNTREND = 2
    BREAKOUT = 3


@dataclass
class Signal:
    """Trading signal data class."""
    timestamp: datetime
    signal_type: SignalType
    confidence: float
    market_regime: MarketRegime
    price: float
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class SignalConfig:
    """Signal generation configuration."""
    breakout_threshold: float = 0.6
    trend_threshold: float = 0.5
    confidence_threshold: float = 0.4
    min_hold_period: int = 3  # Minimum periods to hold position
    max_hold_period: int = 20  # Maximum periods to hold position
    signal_hysteresis: float = 0.1  # Hysteresis to prevent signal flickering


class SignalGenerator:
    """
    Signal generation system for trading decisions.
    
    Converts model predictions to trading signals with risk management,
    signal filtering, and position management logic.
    """
    
    def __init__(self, signal_config: SignalConfig, trading_config: TradingConfig):
        """
        Initialize signal generator.
        
        Args:
            signal_config: Signal generation configuration
            trading_config: Trading configuration
        """
        self.signal_config = signal_config
        self.trading_config = trading_config
        
        # Signal state tracking
        self.current_signal = SignalType.HOLD
        self.signal_history: List[Signal] = []
        self.position_hold_period = 0
        self.last_signal_time = None
        
        # Signal filtering
        self.signal_buffer: List[Tuple[datetime, SignalType, float]] = []
        self.buffer_size = 3
        
        logger.info("Signal generator initialized with risk-based filtering")
    
    def generate_signal(self, predictions: np.ndarray, price: float, timestamp: datetime,
                       atr: Optional[float] = None, additional_features: Optional[Dict[str, Any]] = None) -> Signal:
        """
        Generate trading signal from model predictions.
        
        Args:
            predictions: Model predictions (probabilities for 4 classes)
            price: Current price
            timestamp: Current timestamp
            atr: Average True Range
            additional_features: Additional market features
            
        Returns:
            Signal: Generated trading signal
        """
        try:
            # Ensure predictions is 1D array
            if predictions.ndim > 1:
                predictions = predictions.flatten()
            
            # Get probabilities for each market regime
            sideway_prob = predictions[MarketRegime.SIDEWAY.value]
            uptrend_prob = predictions[MarketRegime.UPTREND.value]
            downtrend_prob = predictions[MarketRegime.DOWNTREND.value]
            breakout_prob = predictions[MarketRegime.BREAKOUT.value]
            
            # Determine market regime
            regime_probs = {
                MarketRegime.SIDEWAY: sideway_prob,
                MarketRegime.UPTREND: uptrend_prob,
                MarketRegime.DOWNTREND: downtrend_prob,
                MarketRegime.BREAKOUT: breakout_prob
            }
            
            predicted_regime = max(regime_probs, key=regime_probs.get)
            confidence = regime_probs[predicted_regime]
            
            # Generate signal based on regime and confidence
            signal_type = self._determine_signal_type(predicted_regime, confidence, 
                                                    uptrend_prob, downtrend_prob, breakout_prob)
            
            # Apply signal filtering and risk management
            filtered_signal = self._apply_signal_filtering(signal_type, confidence, timestamp)
            
            # Create signal object
            signal = Signal(
                timestamp=timestamp,
                signal_type=filtered_signal,
                confidence=confidence,
                market_regime=predicted_regime,
                price=price,
                metadata={
                    'atr': atr,
                    'regime_probs': regime_probs,
                    'additional_features': additional_features
                }
            )
            
            # Update signal history
            self.signal_history.append(signal)
            self._update_signal_state(signal)
            
            logger.debug(f"Signal generated: {signal.signal_type.value} with confidence {confidence:.3f}")
            
            return signal
            
        except Exception as e:
            logger.error(f"Error generating signal: {e}")
            return self._create_hold_signal(timestamp, price)
    
    def _determine_signal_type(self, predicted_regime: MarketRegime, confidence: float,
                              uptrend_prob: float, downtrend_prob: float, 
                              breakout_prob: float) -> SignalType:
        """
        Determine signal type based on predicted regime and probabilities.
        
        Args:
            predicted_regime: Predicted market regime
            confidence: Prediction confidence
            uptrend_prob: Uptrend probability
            downtrend_prob: Downtrend probability
            breakout_prob: Breakout probability
            
        Returns:
            SignalType: Determined signal type
        """
        try:
            # Check confidence threshold
            if confidence < self.signal_config.confidence_threshold:
                return SignalType.HOLD
            
            # Breakout signals (highest priority)
            if breakout_prob >= self.signal_config.breakout_threshold:
                if uptrend_prob > downtrend_prob:
                    return SignalType.LONG
                else:
                    return SignalType.SHORT
            
            # Trend signals
            elif predicted_regime == MarketRegime.UPTREND:
                if uptrend_prob >= self.signal_config.trend_threshold:
                    return SignalType.LONG
                else:
                    return SignalType.HOLD
            
            elif predicted_regime == MarketRegime.DOWNTREND:
                if downtrend_prob >= self.signal_config.trend_threshold:
                    return SignalType.SHORT
                else:
                    return SignalType.HOLD
            
            # Sideway market - hold or close
            elif predicted_regime == MarketRegime.SIDEWAY:
                if self.current_signal in [SignalType.LONG, SignalType.SHORT]:
                    return SignalType.CLOSE
                else:
                    return SignalType.HOLD
            
            return SignalType.HOLD
            
        except Exception as e:
            logger.error(f"Error determining signal type: {e}")
            return SignalType.HOLD
    
    def _apply_signal_filtering(self, signal_type: SignalType, confidence: float, 
                              timestamp: datetime) -> SignalType:
        """
        Apply signal filtering to prevent noise and improve signal quality.
        
        Args:
            signal_type: Raw signal type
            confidence: Signal confidence
            timestamp: Signal timestamp
            
        Returns:
            SignalType: Filtered signal type
        """
        try:
            # Add to signal buffer
            self.signal_buffer.append((timestamp, signal_type, confidence))
            
            # Keep only recent signals
            if len(self.signal_buffer) > self.buffer_size:
                self.signal_buffer = self.signal_buffer[-self.buffer_size:]
            
            # Apply hysteresis to prevent signal flickering
            if self.current_signal != SignalType.HOLD:
                if signal_type == self.current_signal:
                    return signal_type
                elif signal_type == SignalType.HOLD:
                    # Only change to HOLD if confidence is significantly lower
                    if confidence < (1.0 - self.signal_config.signal_hysteresis):
                        return SignalType.HOLD
                    else:
                        return self.current_signal
                else:
                    # Only change signal if new signal is significantly stronger
                    if confidence > (1.0 + self.signal_config.signal_hysteresis):
                        return signal_type
                    else:
                        return self.current_signal
            
            # Check for signal consistency in buffer
            if len(self.signal_buffer) >= self.buffer_size:
                recent_signals = [s[1] for s in self.signal_buffer[-3:]]
                if len(set(recent_signals)) == 1 and recent_signals[0] == signal_type:
                    return signal_type
                else:
                    return SignalType.HOLD
            
            return signal_type
            
        except Exception as e:
            logger.error(f"Error applying signal filtering: {e}")
            return SignalType.HOLD
    
    def _update_signal_state(self, signal: Signal):
        """Update internal signal state."""
        try:
            self.current_signal = signal.signal_type
            self.last_signal_time = signal.timestamp
            
            # Update position hold period
            if signal.signal_type in [SignalType.LONG, SignalType.SHORT]:
                self.position_hold_period += 1
            else:
                self.position_hold_period = 0
            
        except Exception as e:
            logger.error(f"Error updating signal state: {e}")
    
    def _create_hold_signal(self, timestamp: datetime, price: float) -> Signal:
        """Create a default HOLD signal."""
        return Signal(
            timestamp=timestamp,
            signal_type=SignalType.HOLD,
            confidence=0.0,
            market_regime=MarketRegime.SIDEWAY,
            price=price,
            metadata={'error': 'Signal generation failed'}
        )
    
    def should_close_position(self, current_signal: SignalType, position_hold_period: int) -> bool:
        """
        Determine if position should be closed based on hold period and signal.
        
        Args:
            current_signal: Current signal type
            position_hold_period: Number of periods position has been held
            
        Returns:
            bool: True if position should be closed
        """
        try:
            # Close if signal is CLOSE
            if current_signal == SignalType.CLOSE:
                return True
            
            # Close if held too long
            if position_hold_period >= self.signal_config.max_hold_period:
                return True
            
            # Close if signal changed to opposite direction
            if (self.current_signal == SignalType.LONG and current_signal == SignalType.SHORT) or \
               (self.current_signal == SignalType.SHORT and current_signal == SignalType.LONG):
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking position close: {e}")
            return False
    
    def get_signal_statistics(self) -> Dict[str, Any]:
        """Get signal generation statistics."""
        try:
            if not self.signal_history:
                return {}
            
            # Count signals by type
            signal_counts = {}
            for signal in self.signal_history:
                signal_type = signal.signal_type.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in self.signal_history])
            
            # Calculate signal frequency
            if len(self.signal_history) > 1:
                time_span = (self.signal_history[-1].timestamp - self.signal_history[0].timestamp).total_seconds()
                signal_frequency = len(self.signal_history) / (time_span / 3600)  # signals per hour
            else:
                signal_frequency = 0
            
            return {
                'total_signals': len(self.signal_history),
                'signal_counts': signal_counts,
                'average_confidence': avg_confidence,
                'signal_frequency': signal_frequency,
                'current_signal': self.current_signal.value,
                'position_hold_period': self.position_hold_period
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal statistics: {e}")
            return {}
    
    def reset_signal_state(self):
        """Reset signal generator state."""
        try:
            self.current_signal = SignalType.HOLD
            self.signal_history.clear()
            self.position_hold_period = 0
            self.last_signal_time = None
            self.signal_buffer.clear()
            
            logger.info("Signal generator state reset")
            
        except Exception as e:
            logger.error(f"Error resetting signal state: {e}")
    
    def export_signal_history(self, filepath: str) -> bool:
        """
        Export signal history to CSV.
        
        Args:
            filepath: Path to save signal history
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.signal_history:
                logger.warning("No signal history to export")
                return False
            
            # Convert signals to DataFrame
            signals_data = []
            for signal in self.signal_history:
                signal_data = {
                    'timestamp': signal.timestamp,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'market_regime': signal.market_regime.value,
                    'price': signal.price
                }
                
                # Add metadata if available
                if signal.metadata:
                    for key, value in signal.metadata.items():
                        if key != 'regime_probs':  # Skip complex objects
                            signal_data[f'metadata_{key}'] = value
                
                signals_data.append(signal_data)
            
            signals_df = pd.DataFrame(signals_data)
            signals_df.to_csv(filepath, index=False)
            
            logger.info(f"Exported {len(self.signal_history)} signals to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting signal history: {e}")
            return False
    
    def get_recent_signals(self, hours: int = 24) -> List[Signal]:
        """
        Get recent signals within specified time window.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            List[Signal]: Recent signals
        """
        try:
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_signals = [s for s in self.signal_history if s.timestamp >= cutoff_time]
            return recent_signals
            
        except Exception as e:
            logger.error(f"Error getting recent signals: {e}")
            return []
    
    def get_signal_trend(self, window: int = 10) -> Dict[str, Any]:
        """
        Analyze signal trend over recent window.
        
        Args:
            window: Number of recent signals to analyze
            
        Returns:
            Dict[str, Any]: Signal trend analysis
        """
        try:
            if len(self.signal_history) < window:
                return {}
            
            recent_signals = self.signal_history[-window:]
            
            # Count signal types
            signal_counts = {}
            for signal in recent_signals:
                signal_type = signal.signal_type.value
                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
            
            # Calculate trend direction
            long_signals = signal_counts.get('LONG', 0)
            short_signals = signal_counts.get('SHORT', 0)
            
            if long_signals > short_signals:
                trend_direction = 'BULLISH'
            elif short_signals > long_signals:
                trend_direction = 'BEARISH'
            else:
                trend_direction = 'NEUTRAL'
            
            # Calculate average confidence
            avg_confidence = np.mean([s.confidence for s in recent_signals])
            
            return {
                'trend_direction': trend_direction,
                'signal_counts': signal_counts,
                'average_confidence': avg_confidence,
                'long_short_ratio': long_signals / max(short_signals, 1),
                'total_signals': len(recent_signals)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signal trend: {e}")
            return {}
