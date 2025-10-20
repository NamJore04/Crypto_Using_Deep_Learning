"""
Data processor for Crypto Futures Trading System.

This module handles data preprocessing, technical indicator calculations,
and feature engineering for the trading system.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import talib
import logging
from datetime import datetime, timedelta

from ...config.settings import BreakoutConfig

logger = logging.getLogger(__name__)


class DataProcessor:
    """
    Data processor for crypto trading data.
    
    Handles data preprocessing, technical indicator calculations,
    and feature engineering for the trading system.
    """
    
    def __init__(self, breakout_config: BreakoutConfig):
        """
        Initialize data processor.
        
        Args:
            breakout_config: Breakout detection configuration
        """
        self.breakout_config = breakout_config
        self.indicators = {}
    
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with technical indicators
        """
        try:
            # Create a copy to avoid modifying original data
            data = df.copy()
            
            # Price data
            high = data['high'].values
            low = data['low'].values
            close = data['close'].values
            volume = data['volume'].values
            
            # Momentum indicators
            data['rsi'] = talib.RSI(close, timeperiod=14)
            data['macd'], data['macd_signal'], data['macd_hist'] = talib.MACD(close)
            data['stoch_k'], data['stoch_d'] = talib.STOCH(high, low, close)
            data['williams_r'] = talib.WILLR(high, low, close)
            
            # Trend indicators
            data['ema_12'] = talib.EMA(close, timeperiod=12)
            data['ema_26'] = talib.EMA(close, timeperiod=26)
            data['ema_50'] = talib.EMA(close, timeperiod=50)
            data['ema_200'] = talib.EMA(close, timeperiod=200)
            data['sma_20'] = talib.SMA(close, timeperiod=20)
            data['sma_50'] = talib.SMA(close, timeperiod=50)
            
            # Volatility indicators
            data['atr'] = talib.ATR(high, low, close, timeperiod=14)
            data['bb_upper'], data['bb_middle'], data['bb_lower'] = talib.BBANDS(close)
            data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
            data['bb_pctb'] = (close - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
            
            # Volume indicators
            data['obv'] = talib.OBV(close, volume)
            data['ad'] = talib.AD(high, low, close, volume)
            data['cmf'] = self._calculate_cmf(high, low, close, volume, 20)
            
            # Price patterns
            data['doji'] = talib.CDLDOJI(high, low, close, data['open'])
            data['hammer'] = talib.CDLHAMMER(high, low, close, data['open'])
            data['shooting_star'] = talib.CDLSHOOTINGSTAR(high, low, close, data['open'])
            
            # Custom indicators
            data = self._calculate_custom_indicators(data)
            
            logger.info(f"Calculated technical indicators for {len(data)} records")
            return data
            
        except Exception as e:
            logger.error(f"Failed to calculate technical indicators: {e}")
            return df
    
    def _calculate_cmf(self, high: np.ndarray, low: np.ndarray, 
                      close: np.ndarray, volume: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate Chaikin Money Flow (CMF).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            period: Period for calculation
            
        Returns:
            np.ndarray: CMF values
        """
        try:
            # Calculate money flow multiplier
            mf_multiplier = ((close - low) - (high - close)) / (high - low)
            mf_multiplier = np.where(high == low, 0, mf_multiplier)
            
            # Calculate money flow volume
            mf_volume = mf_multiplier * volume
            
            # Calculate CMF
            cmf = pd.Series(mf_volume).rolling(period).sum() / pd.Series(volume).rolling(period).sum()
            return cmf.values
            
        except Exception as e:
            logger.error(f"Failed to calculate CMF: {e}")
            return np.zeros(len(close))
    
    def _calculate_custom_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate custom indicators for breakout detection.
        
        Args:
            df: DataFrame with OHLCV data and basic indicators
            
        Returns:
            pd.DataFrame: DataFrame with custom indicators
        """
        try:
            # Keltner Channel
            df['kc_middle'] = df['close'].ewm(span=20).mean()
            df['kc_range'] = df['high'].rolling(20).max() - df['low'].rolling(20).min()
            df['kc_upper'] = df['kc_middle'] + 1.5 * df['kc_range']
            df['kc_lower'] = df['kc_middle'] - 1.5 * df['kc_range']
            df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
            
            # Donchian Channel
            df['donchian_high'] = df['high'].rolling(20).max()
            df['donchian_low'] = df['low'].rolling(20).min()
            df['donchian_width'] = (df['donchian_high'] - df['donchian_low']) / df['close']
            
            # Squeeze detection
            df['squeeze'] = (df['bb_width'] < df['kc_width']).astype(int)
            
            # Price position indicators
            df['price_position'] = (df['close'] - df['donchian_low']) / (df['donchian_high'] - df['donchian_low'])
            df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
            
            # Volatility indicators
            df['volatility'] = df['close'].pct_change().rolling(20).std()
            df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50).mean()
            
            # Volume indicators
            df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
            
            # Price momentum
            df['momentum_1'] = df['close'].pct_change(1)
            df['momentum_3'] = df['close'].pct_change(3)
            df['momentum_6'] = df['close'].pct_change(6)
            df['momentum_12'] = df['close'].pct_change(12)
            
            # Trend strength
            df['trend_strength'] = abs(df['close'].pct_change().rolling(20).mean())
            df['trend_direction'] = np.where(df['close'].pct_change().rolling(20).mean() > 0, 1, -1)
            
            logger.info("Calculated custom indicators")
            return df
            
        except Exception as e:
            logger.error(f"Failed to calculate custom indicators: {e}")
            return df
    
    def detect_breakouts(self, df: pd.DataFrame) -> pd.Series:
        """
        Detect breakout events using quantitative criteria.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            pd.Series: Breakout labels (0: no breakout, 1: breakout)
        """
        try:
            labels = np.zeros(len(df))
            
            for t in range(self.breakout_config.donchian_period, 
                          len(df) - self.breakout_config.future_horizon):
                
                # 1. Check squeeze condition
                squeeze_window = df['squeeze'].iloc[t-self.breakout_config.squeeze_window:t]
                squeeze_ratio = squeeze_window.mean()
                
                # 2. Check breakout condition
                current_price = df['close'].iloc[t]
                donchian_high = df['donchian_high'].iloc[t]
                donchian_low = df['donchian_low'].iloc[t]
                
                # 3. Check future return
                future_price = df['close'].iloc[t + self.breakout_config.future_horizon]
                future_return = abs(future_price / current_price - 1)
                
                # Breakout criteria
                if (squeeze_ratio >= self.breakout_config.squeeze_threshold and
                    (current_price > donchian_high or current_price < donchian_low) and
                    future_return >= self.breakout_config.return_threshold):
                    labels[t] = 1
            
            logger.info(f"Detected {np.sum(labels)} breakout events")
            return pd.Series(labels, index=df.index)
            
        except Exception as e:
            logger.error(f"Failed to detect breakouts: {e}")
            return pd.Series(np.zeros(len(df)), index=df.index)
    
    def label_market_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Label market regimes based on price action and indicators.
        
        Args:
            df: DataFrame with OHLCV data and indicators
            
        Returns:
            pd.Series: Market regime labels (0: SIDEWAY, 1: UPTREND, 2: DOWNTREND, 3: BREAKOUT)
        """
        try:
            labels = np.zeros(len(df))
            
            # Calculate returns and volatility
            returns = df['close'].pct_change()
            volatility = returns.rolling(20).std()
            trend_strength = abs(returns.rolling(20).mean())
            
            # SIDEWAY (0): Low volatility, no clear trend
            sideway_mask = (
                (volatility < volatility.quantile(0.3)) & 
                (trend_strength < trend_strength.quantile(0.3))
            )
            labels[sideway_mask] = 0
            
            # UPTREND (1): Positive trend with momentum
            uptrend_mask = (
                (returns.rolling(20).mean() > 0) & 
                (df['rsi'] > 50) &
                (df['close'] > df['ema_50'])
            )
            labels[uptrend_mask] = 1
            
            # DOWNTREND (2): Negative trend with momentum
            downtrend_mask = (
                (returns.rolling(20).mean() < 0) & 
                (df['rsi'] < 50) &
                (df['close'] < df['ema_50'])
            )
            labels[downtrend_mask] = 2
            
            # BREAKOUT (3): Use breakout detection
            breakout_labels = self.detect_breakouts(df)
            labels[breakout_labels == 1] = 3
            
            # Ensure no overlap - breakout takes precedence
            breakout_mask = breakout_labels == 1
            labels[breakout_mask] = 3
            
            logger.info(f"Labeled market regimes: {np.bincount(labels.astype(int))}")
            # Return a DataFrame with a 'market_regime' column as expected by unit tests
            result_df = df.copy()
            result_df['market_regime'] = labels.astype(int)
            return result_df
            
        except Exception as e:
            logger.error(f"Failed to label market regimes: {e}")
            result_df = df.copy()
            result_df['market_regime'] = 0
            return result_df
    
    def normalize_features(self, df: pd.DataFrame, method: str = 'zscore') -> pd.DataFrame:
        """
        Normalize features for machine learning.
        
        Args:
            df: DataFrame with features
            method: Normalization method ('zscore', 'minmax', 'robust')
            
        Returns:
            pd.DataFrame: Normalized features
        """
        try:
            # Select numeric columns only
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            normalized_df = df.copy()
            
            for col in numeric_cols:
                if method == 'zscore':
                    # Z-score normalization
                    normalized_df[col] = (df[col] - df[col].mean()) / df[col].std()
                elif method == 'minmax':
                    # Min-max normalization
                    normalized_df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                elif method == 'robust':
                    # Robust normalization (median and IQR)
                    median = df[col].median()
                    iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
                    normalized_df[col] = (df[col] - median) / iqr
            
            logger.info(f"Normalized features using {method} method")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Failed to normalize features: {e}")
            return df
    
    def create_sequences(self, df: pd.DataFrame, sequence_length: int, 
                        target_col: str = 'label') -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for time series machine learning.
        
        Args:
            df: DataFrame with features and targets
            sequence_length: Length of sequences
            target_col: Name of target column
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Features and targets arrays
        """
        try:
            # Select feature columns (exclude target and timestamp)
            feature_cols = [col for col in df.columns if col not in [target_col, 'timestamp']]
            features = df[feature_cols].values
            targets = df[target_col].values
            
            # Create sequences
            X, y = [], []
            for i in range(sequence_length, len(features)):
                X.append(features[i-sequence_length:i])
                y.append(targets[i])
            
            X = np.array(X)
            y = np.array(y)
            
            logger.info(f"Created {len(X)} sequences of length {sequence_length}")
            return X, y
            
        except Exception as e:
            logger.error(f"Failed to create sequences: {e}")
            return np.array([]), np.array([])
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Validate data quality and report issues.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Dict[str, Any]: Validation results
        """
        try:
            issues = []
            
            # Check for missing values
            missing_values = df.isnull().sum()
            if missing_values.any():
                issues.append(f"Missing values found: {missing_values[missing_values > 0].to_dict()}")
            
            # Check for infinite values
            infinite_values = np.isinf(df.select_dtypes(include=[np.number])).sum()
            if infinite_values.any():
                issues.append(f"Infinite values found: {infinite_values[infinite_values > 0].to_dict()}")
            
            # Check for negative prices
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in df.columns and (df[col] <= 0).any():
                    issues.append(f"Non-positive prices found in {col}")
            
            # Check for negative volume
            if 'volume' in df.columns and (df['volume'] < 0).any():
                issues.append("Negative volume found")
            
            # Check OHLC relationships
            if all(col in df.columns for col in price_cols):
                if not (df['high'] >= df['low']).all():
                    issues.append("High < Low relationship violated")
                if not (df['high'] >= df['open']).all():
                    issues.append("High < Open relationship violated")
                if not (df['high'] >= df['close']).all():
                    issues.append("High < Close relationship violated")
            
            # Check for extreme outliers
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if df[col].abs().max() > 1e6:
                    issues.append(f"Extreme outliers found in {col}")
            
            validation_result = {
                'valid': len(issues) == 0,
                'issues': issues,
                'data_shape': df.shape,
                'missing_values': missing_values.to_dict() if 'missing_values' in locals() else {},
                'infinite_values': infinite_values.to_dict() if 'infinite_values' in locals() else {}
            }
            
            logger.info(f"Data validation completed: {'PASS' if validation_result['valid'] else 'FAIL'}")
            return validation_result
            
        except Exception as e:
            logger.error(f"Failed to validate data: {e}")
            return {'valid': False, 'issues': [f"Validation error: {e}"], 'data_shape': (0, 0)}
    
    def calculate_breakout_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate breakout-specific features.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: DataFrame with breakout features
        """
        try:
            features_df = df.copy()
            
            # First calculate technical indicators if not present
            if 'bb_width' not in features_df.columns:
                features_df = self.calculate_technical_indicators(features_df)
            
            # Donchian Channel features
            features_df['donchian_high'] = features_df['high'].rolling(self.breakout_config.donchian_period).max()
            features_df['donchian_low'] = features_df['low'].rolling(self.breakout_config.donchian_period).min()
            features_df['donchian_width'] = (features_df['donchian_high'] - features_df['donchian_low']) / features_df['close']
            
            # Squeeze features (Bollinger vs Keltner)
            kc_middle = features_df['close'].ewm(span=self.breakout_config.donchian_period).mean()
            kc_range = features_df['high'].rolling(self.breakout_config.donchian_period).max() - features_df['low'].rolling(self.breakout_config.donchian_period).min()
            kc_upper = kc_middle + 1.5 * kc_range
            kc_lower = kc_middle - 1.5 * kc_range
            kc_width = (kc_upper - kc_lower) / kc_middle
            
            # Squeeze detection
            features_df['squeeze'] = (features_df['bb_width'] < kc_width).astype(int)
            
            # Volume features
            features_df['volume_ratio'] = features_df['volume'] / features_df['volume'].rolling(20).mean()
            features_df['volume_spike'] = (features_df['volume_ratio'] > self.breakout_config.min_volume_ratio).astype(int)
            
            logger.info("Breakout features calculated successfully")
            return features_df
            
        except Exception as e:
            logger.error(f"Failed to calculate breakout features: {e}")
            return df
    
    def normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data using specified method.
        
        Args:
            df: DataFrame to normalize
            
        Returns:
            pd.DataFrame: Normalized DataFrame
        """
        try:
            normalized_df = df.copy()
            
            # Select numeric columns for normalization
            numeric_cols = normalized_df.select_dtypes(include=[np.number]).columns
            
            # Skip only timestamp and target columns from normalization
            skip_cols = ['timestamp', 'market_regime']
            normalize_cols = [col for col in numeric_cols if col not in skip_cols]
            
            normalization_method = 'z_score'  # Default normalization method
            if normalization_method == 'z_score':
                # Z-score normalization
                for col in normalize_cols:
                    mean = normalized_df[col].mean()
                    std = normalized_df[col].std()
                    if std > 0:
                        normalized_df[col] = (normalized_df[col] - mean) / std
                        
            elif self.breakout_config.normalization_method == 'min_max':
                # Min-max normalization
                for col in normalize_cols:
                    min_val = normalized_df[col].min()
                    max_val = normalized_df[col].max()
                    if max_val > min_val:
                        normalized_df[col] = (normalized_df[col] - min_val) / (max_val - min_val)
            
            logger.info(f"Data normalization completed using {normalization_method}")
            return normalized_df
            
        except Exception as e:
            logger.error(f"Failed to normalize data: {e}")
            return df

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data by handling missing values and outliers.
        
        Args:
            df: DataFrame to clean
            
        Returns:
            pd.DataFrame: Cleaned DataFrame
        """
        try:
            cleaned_df = df.copy()
            
            # Forward fill missing values
            cleaned_df = cleaned_df.fillna(method='ffill')
            
            # Remove rows with all NaN values
            cleaned_df = cleaned_df.dropna(how='all')
            
            # Handle infinite values
            cleaned_df = cleaned_df.replace([np.inf, -np.inf], np.nan)
            cleaned_df = cleaned_df.fillna(method='ffill')
            
            # Remove extreme outliers (beyond 5 standard deviations)
            numeric_cols = cleaned_df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col in ['open', 'high', 'low', 'close', 'volume']:
                    continue  # Skip price and volume columns
                
                mean = cleaned_df[col].mean()
                std = cleaned_df[col].std()
                if std > 0:
                    cleaned_df = cleaned_df[abs(cleaned_df[col] - mean) <= 5 * std]
            
            logger.info(f"Data cleaning completed: {len(df)} -> {len(cleaned_df)} records")
            return cleaned_df
            
        except Exception as e:
            logger.error(f"Failed to clean data: {e}")
            return df
