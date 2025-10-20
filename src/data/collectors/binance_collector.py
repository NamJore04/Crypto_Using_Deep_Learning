"""
Binance data collector for Crypto Futures Trading System.

This module handles data collection from Binance Futures API
using CCXT library with proper rate limiting and error handling.
"""

import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
import time
import logging
from tqdm import tqdm

from ...config.settings import APIConfig, DataConfig

logger = logging.getLogger(__name__)


class BinanceDataCollector:
    """
    Binance data collector for crypto futures data.
    
    Handles data collection from Binance Futures API with
    rate limiting, error handling, and data validation.
    """
    
    def __init__(self, api_config: APIConfig, data_config: DataConfig):
        """
        Initialize Binance data collector.
        
        Args:
            api_config: API configuration settings
            data_config: Data configuration settings
        """
        self.api_config = api_config
        self.data_config = data_config
        self.exchange = None
        self._initialize_exchange()
    
    def _initialize_exchange(self):
        """Initialize CCXT exchange instance."""
        try:
            self.exchange = ccxt.binance({
                'apiKey': self.api_config.binance_api_key,
                'secret': self.api_config.binance_secret,
                'sandbox': self.api_config.sandbox_mode,
                'rateLimit': self.api_config.rate_limit,
                'timeout': self.api_config.timeout * 1000,
                'enableRateLimit': True,
                'options': {
                    'defaultType': 'future'  # Use futures market
                }
            })
            
            logger.info(f"Initialized Binance exchange (sandbox: {self.api_config.sandbox_mode})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Binance exchange: {e}")
            raise
    
    def test_connection(self) -> bool:
        """
        Test connection to Binance API.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test with a simple API call
            markets = self.exchange.load_markets()
            logger.info(f"Connection test successful. Found {len(markets)} markets")
            return True
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available trading symbols.
        
        Returns:
            List[str]: List of available symbols
        """
        try:
            markets = self.exchange.load_markets()
            symbols = [symbol for symbol, market in markets.items() 
                      if market['type'] == 'future' and market['active']]
            
            logger.info(f"Found {len(symbols)} available symbols")
            return symbols
            
        except Exception as e:
            logger.error(f"Failed to get available symbols: {e}")
            return []
    
    def collect_historical_data(self, symbol: str, timeframe: str, 
                              days: Optional[int] = None) -> pd.DataFrame:
        """
        Collect historical OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe (e.g., '30m', '1h', '4h')
            days: Number of days to collect (default from config)
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Always reinitialize exchange to respect current patched/mocked context
            # This avoids stale exchange instances across tests
            self._initialize_exchange()

            if days is None:
                days = self.data_config.lookback_days
            
            # Calculate start and end times
            end_time = datetime.now()
            start_time = end_time - timedelta(days=days)
            
            logger.info(f"Collecting {symbol} {timeframe} data from {start_time} to {end_time}")
            
            # Convert to milliseconds
            since = int(start_time.timestamp() * 1000)
            until = int(end_time.timestamp() * 1000)
            
            all_data = []
            current_since = since
            
            # Collect data in chunks to handle rate limits
            with tqdm(desc=f"Collecting {symbol} {timeframe}") as pbar:
                while current_since < until:
                    try:
                        # Fetch OHLCV data
                        ohlcv = self.exchange.fetch_ohlcv(
                            symbol=symbol,
                            timeframe=timeframe,
                            since=current_since,
                            limit=1000  # Maximum per request
                        )
                        
                        if not ohlcv:
                            logger.warning(f"No data received for {symbol} {timeframe}")
                            break
                        
                        all_data.extend(ohlcv)
                        
                        # Update progress
                        pbar.update(len(ohlcv))
                        
                        # Update since timestamp
                        current_since = ohlcv[-1][0] + 1
                        
                        # Rate limiting
                        time.sleep(0.1)
                        
                    except ccxt.RateLimitExceeded:
                        logger.warning("Rate limit exceeded, waiting...")
                        time.sleep(1)
                        continue
                    except ccxt.NetworkError as e:
                        logger.warning(f"Network error: {e}, retrying...")
                        time.sleep(1)
                        continue
                    except Exception as e:
                        logger.error(f"Error collecting data: {e}")
                        break
            
            if not all_data:
                logger.warning(f"No data collected for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame (keep 'timestamp' as a column in ms to align with unit tests)
            df = pd.DataFrame(all_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Remove duplicates by timestamp and sort by timestamp
            df = df.drop_duplicates(subset=['timestamp'], keep='last')
            df = df.sort_values('timestamp')
            
            # Validate data
            df = self._validate_ohlcv_data(df)
            
            logger.info(f"Collected {len(df)} records for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect historical data: {e}")
            return pd.DataFrame()
    
    def collect_realtime_data(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Collect real-time OHLCV data from Binance.
        
        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            
        Returns:
            pd.DataFrame: Real-time OHLCV data
        """
        try:
            # Always reinitialize exchange to respect current patched/mocked context
            self._initialize_exchange()

            # Get latest data
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=1)
            
            if not ohlcv:
                logger.warning(f"No real-time data for {symbol} {timeframe}")
                return pd.DataFrame()
            
            # Convert to DataFrame (keep timestamp column in ms for consistency)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Validate data
            df = self._validate_ohlcv_data(df)
            
            logger.info(f"Collected real-time data for {symbol} {timeframe}")
            return df
            
        except Exception as e:
            logger.error(f"Failed to collect real-time data: {e}")
            return pd.DataFrame()
    
    def collect_multiple_symbols(self, symbols: List[str], timeframe: str,
                                days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple symbols.
        
        Args:
            symbols: List of trading symbols
            timeframe: Timeframe
            days: Number of days to collect
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with symbol as key and DataFrame as value
        """
        results = {}
        
        for symbol in symbols:
            logger.info(f"Collecting data for {symbol}")
            data = self.collect_historical_data(symbol, timeframe, days)
            if not data.empty:
                results[symbol] = data
            else:
                logger.warning(f"No data collected for {symbol}")
        
        return results
    
    def collect_multiple_timeframes(self, symbol: str, timeframes: List[str],
                                  days: Optional[int] = None) -> Dict[str, pd.DataFrame]:
        """
        Collect data for multiple timeframes.
        
        Args:
            symbol: Trading symbol
            timeframes: List of timeframes
            days: Number of days to collect
            
        Returns:
            Dict[str, pd.DataFrame]: Dictionary with timeframe as key and DataFrame as value
        """
        results = {}
        
        for timeframe in timeframes:
            logger.info(f"Collecting {symbol} {timeframe} data")
            data = self.collect_historical_data(symbol, timeframe, days)
            if not data.empty:
                results[timeframe] = data
            else:
                logger.warning(f"No data collected for {symbol} {timeframe}")
        
        return results
    
    def _validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            pd.DataFrame: Cleaned and validated data
        """
        try:
            original_length = len(df)
            
            # Remove rows with missing values
            df = df.dropna()
            
            # Remove rows with zero or negative prices
            df = df[(df['open'] > 0) & (df['high'] > 0) & (df['low'] > 0) & (df['close'] > 0)]
            
            # Remove rows with negative volume
            df = df[df['volume'] >= 0]
            
            # Validate OHLC relationships
            df = df[df['high'] >= df['low']]
            df = df[df['high'] >= df['open']]
            df = df[df['high'] >= df['close']]
            df = df[df['low'] <= df['open']]
            df = df[df['low'] <= df['close']]
            
            # Remove extreme outliers (prices > 10x median)
            for col in ['open', 'high', 'low', 'close']:
                median_price = df[col].median()
                df = df[df[col] <= median_price * 10]
                df = df[df[col] >= median_price / 10]
            
            cleaned_length = len(df)
            if cleaned_length < original_length:
                logger.warning(f"Cleaned data: {original_length} -> {cleaned_length} records")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to validate OHLCV data: {e}")
            return df
    
    def get_market_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get market information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Market information
        """
        try:
            markets = self.exchange.load_markets()
            market = markets.get(symbol)
            
            if not market:
                logger.warning(f"Market not found for {symbol}")
                return {}
            
            return {
                'symbol': symbol,
                'base': market['base'],
                'quote': market['quote'],
                'active': market['active'],
                'type': market['type'],
                'spot': market['spot'],
                'future': market['future'],
                'precision': market['precision'],
                'limits': market['limits']
            }
            
        except Exception as e:
            logger.error(f"Failed to get market info for {symbol}: {e}")
            return {}
    
    def get_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get current ticker information for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dict[str, Any]: Ticker information
        """
        try:
            ticker = self.exchange.fetch_ticker(symbol)
            return ticker
            
        except Exception as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            return {}
    
    def get_orderbook(self, symbol: str, limit: int = 20) -> Dict[str, Any]:
        """
        Get order book for a symbol.
        
        Args:
            symbol: Trading symbol
            limit: Number of orders to retrieve
            
        Returns:
            Dict[str, Any]: Order book information
        """
        try:
            orderbook = self.exchange.fetch_order_book(symbol, limit)
            return orderbook
            
        except Exception as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            return {}
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate collected data."""
        # Timestamp may be provided as a column or as the DataFrame index.
        # Only enforce OHLCV columns.
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        
        if data.empty:
            return False
        
        if not all(col in data.columns for col in required_columns):
            return False
        
        if data.isnull().any().any():
            return False
        
        if (data[['open', 'high', 'low', 'close', 'volume']] <= 0).any().any():
            return False
        
        # If timestamp is present, ensure it is parseable; otherwise accept index
        if 'timestamp' in data.columns:
            try:
                pd.to_datetime(data['timestamp'])
            except Exception:
                return False
        
        return True
    
    def close(self):
        """Close the exchange connection."""
        if self.exchange:
            self.exchange.close()
            logger.info("Closed Binance exchange connection")
