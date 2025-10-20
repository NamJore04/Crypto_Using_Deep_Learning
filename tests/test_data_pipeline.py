"""
Unit tests for data pipeline components.

Tests data collection, processing, and storage functionality.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.collectors.binance_collector import BinanceDataCollector
from src.data.processors.data_processor import DataProcessor
from src.data.storage.mongodb_storage import MongoDBStorage
from src.config.settings import DatabaseConfig, APIConfig, DataConfig


class TestBinanceDataCollector:
    """Test cases for BinanceDataCollector."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create mock configurations
        self.api_config = APIConfig(
            binance_api_key='test_key',
            binance_secret='test_secret',
            sandbox_mode=True,
            rate_limit=1200
        )
        self.data_config = DataConfig()
        
        # Mock the exchange initialization to avoid real API calls
        with patch('ccxt.binance'):
            self.collector = BinanceDataCollector(self.api_config, self.data_config)
    
    def test_initialization(self):
        """Test collector initialization."""
        assert self.collector is not None
        assert hasattr(self.collector, 'exchange')
    
    @patch('ccxt.binance')
    def test_collect_historical_data(self, mock_binance):
        """Test historical data collection."""
        # Mock exchange response
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],
            [1640998800000, 47500, 48500, 47000, 48000, 1200]
        ]
        mock_binance.return_value = mock_exchange
        
        data = self.collector.collect_historical_data("BTC/USDT", "30m", 1)
        
        assert data is not None
        assert len(data) == 2
        assert list(data.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    def test_validate_data(self):
        """Test data validation."""
        # Valid data
        valid_data = pd.DataFrame({
            'timestamp': [1640995200000, 1640998800000],
            'open': [47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000],
            'volume': [1000, 1200]
        })
        
        assert self.collector.validate_data(valid_data) == True
        
        # Invalid data (negative prices)
        invalid_data = pd.DataFrame({
            'timestamp': [1640995200000, 1640998800000],
            'open': [-47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000],
            'volume': [1000, 1200]
        })
        
        assert self.collector.validate_data(invalid_data) == False


class TestDataProcessor:
    """Test cases for DataProcessor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        from src.config.settings import BreakoutConfig
        breakout_config = BreakoutConfig()
        self.processor = DataProcessor(breakout_config)
        
        # Create sample data
        self.sample_data = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 10000
        })
        self.sample_data.index = pd.date_range('2023-01-01', periods=100, freq='30T')
    
    def test_initialization(self):
        """Test processor initialization."""
        assert self.processor is not None
        assert hasattr(self.processor, 'indicators')
    
    def test_calculate_technical_indicators(self):
        """Test technical indicator calculation."""
        processed_data = self.processor.calculate_technical_indicators(self.sample_data)
        
        # Check that indicators are added
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                              'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv']
        
        for indicator in expected_indicators:
            assert indicator in processed_data.columns
    
    def test_calculate_breakout_features(self):
        """Test breakout feature calculation."""
        processed_data = self.processor.calculate_breakout_features(self.sample_data)
        
        # Check that breakout features are added
        expected_features = ['squeeze', 'donchian_high', 'donchian_low', 'donchian_width']
        
        for feature in expected_features:
            assert feature in processed_data.columns
    
    def test_label_market_regimes(self):
        """Test market regime labeling."""
        # Add required indicators first
        data_with_indicators = self.processor.calculate_technical_indicators(self.sample_data)
        labeled_data = self.processor.label_market_regimes(data_with_indicators)
        
        assert 'market_regime' in labeled_data.columns
        assert labeled_data['market_regime'].isin([0, 1, 2, 3]).all()
    
    def test_create_sequences(self):
        """Test sequence creation for model training."""
        # Add required indicators and labels
        data_with_indicators = self.processor.calculate_technical_indicators(self.sample_data)
        labeled_data = self.processor.label_market_regimes(data_with_indicators)
        
        X, y = self.processor.create_sequences(labeled_data, sequence_length=10, target_col='market_regime')
        
        assert X is not None
        assert y is not None
        assert len(X) == len(y)
        assert X.shape[1] == 10  # sequence length
        assert X.shape[2] > 0  # number of features
    
    def test_normalize_data(self):
        """Test data normalization."""
        normalized_data = self.processor.normalize_data(self.sample_data)
        
        # Check that data is normalized (mean close to 0, std close to 1)
        numeric_cols = normalized_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col != 'market_regime':  # Skip target column
                assert abs(normalized_data[col].mean()) < 0.1
                assert abs(normalized_data[col].std() - 1.0) < 0.1


class TestMongoDBStorage:
    """Test cases for MongoDBStorage."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = DatabaseConfig(
            host='localhost',
            port=27017,
            database='test_crypto_trading'
        )
        self.storage = MongoDBStorage(self.config)
    
    @patch('pymongo.MongoClient')
    def test_initialization(self, mock_client):
        """Test storage initialization."""
        mock_client.return_value = Mock()
        storage = MongoDBStorage(self.config)
        assert storage is not None
    
    @patch('pymongo.MongoClient')
    def test_store_ohlcv(self, mock_client):
        """Test OHLCV data storage."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        mock_client.return_value.__getitem__.return_value = mock_db
        
        storage = MongoDBStorage(self.config)
        # Mock the collections directly
        storage._collections = {'ohlcv_data': mock_collection}
        
        # Mock insert_many result
        mock_result = Mock()
        mock_result.inserted_ids = ['id1', 'id2']
        mock_collection.insert_many.return_value = mock_result
        
        test_data = pd.DataFrame({
            'timestamp': [1640995200000, 1640998800000],
            'open': [47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000],
            'volume': [1000, 1200]
        })
        
        result = storage.store_ohlcv("BTC/USDT", "30m", test_data)
        assert result == True
        mock_collection.insert_many.assert_called_once()
    
    @patch('pymongo.MongoClient')
    def test_get_ohlcv(self, mock_client):
        """Test OHLCV data retrieval."""
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        
        # Mock cursor
        mock_cursor = Mock()
        mock_cursor.sort.return_value = mock_cursor
        mock_cursor.limit.return_value = [
            {
                '_id': 'mock_id',
                'symbol': 'BTC/USDT',
                'timeframe': '30m',
                'timestamp': 1640995200000,
                'open': 47000,
                'high': 48000,
                'low': 46000,
                'close': 47500,
                'volume': 1000,
                'created_at': '2023-01-01T00:00:00Z'
            }
        ]
        mock_collection.find.return_value = mock_cursor
        
        mock_client.return_value.__getitem__.return_value = mock_db
        
        storage = MongoDBStorage(self.config)
        # Mock the collections directly
        storage._collections = {'ohlcv_data': mock_collection}
        data = storage.get_ohlcv("BTC/USDT", "30m", limit=1)
        
        assert data is not None
        assert len(data) == 1
        assert 'close' in data.columns


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        from src.config.settings import BreakoutConfig, APIConfig, DataConfig, DatabaseConfig
        
        self.api_config = APIConfig(
            binance_api_key='test_key',
            binance_secret='test_secret',
            sandbox_mode=True,
            rate_limit=1200
        )
        self.data_config = DataConfig()
        self.breakout_config = BreakoutConfig()
        
        self.collector = BinanceDataCollector(self.api_config, self.data_config)
        self.processor = DataProcessor(self.breakout_config)
        self.config = DatabaseConfig(
            host='localhost',
            port=27017,
            database='test_crypto_trading'
        )
    
    @patch('ccxt.binance')
    @patch('pymongo.MongoClient')
    def test_complete_data_pipeline(self, mock_client, mock_binance):
        """Test complete data pipeline integration."""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],
            [1640998800000, 47500, 48500, 47000, 48000, 1200]
        ]
        mock_binance.return_value = mock_exchange
        
        # Mock database
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        mock_client.return_value.__getitem__.return_value = mock_db
        
        # Test complete pipeline
        collector = BinanceDataCollector(self.api_config, self.data_config)
        processor = DataProcessor(self.breakout_config)
        storage = MongoDBStorage(self.config)
        
        # Collect data
        data = collector.collect_historical_data("BTC/USDT", "30m", 1)
        assert data is not None
        
        # Process data
        processed_data = processor.calculate_technical_indicators(data)
        assert processed_data is not None
        
        # Store data
        result = storage.store_ohlcv("BTC/USDT", "30m", processed_data)
        assert result == True
        
        # Verify pipeline success
        assert len(processed_data) > 0
        assert 'rsi' in processed_data.columns


if __name__ == "__main__":
    pytest.main([__file__])
