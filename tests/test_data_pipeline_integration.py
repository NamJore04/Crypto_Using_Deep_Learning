"""
Integration tests for data pipeline components.

Tests the complete data flow from collection to storage.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data.collectors.binance_collector import BinanceDataCollector
from src.data.processors.data_processor import DataProcessor
from src.data.storage.mongodb_storage import MongoDBStorage
from src.config.settings import DatabaseConfig, APIConfig, DataConfig, BreakoutConfig


class TestDataPipelineIntegration:
    """Integration tests for complete data pipeline."""
    
    def setup_method(self):
        """Setup test fixtures."""
        # Create configurations
        self.api_config = APIConfig(
            binance_api_key='test_key',
            binance_secret='test_secret',
            sandbox_mode=True,
            rate_limit=1200
        )
        self.data_config = DataConfig()
        self.breakout_config = BreakoutConfig()
        self.db_config = DatabaseConfig(
            host='localhost',
            port=27017,
            database='test_crypto_trading'
        )
    
    @patch('ccxt.binance')
    def test_data_collection_integration(self, mock_binance):
        """Test data collection with proper mocking."""
        # Mock exchange response
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],
            [1640998800000, 47500, 48500, 47000, 48000, 1200],
            [1641002400000, 48000, 49000, 47500, 48500, 1500]
        ]
        mock_binance.return_value = mock_exchange
        
        # Create collector
        collector = BinanceDataCollector(self.api_config, self.data_config)
        
        # Test data collection
        data = collector.collect_historical_data("BTC/USDT", "30m", 3)
        
        # Verify results
        assert data is not None
        assert len(data) == 3
        assert 'open' in data.columns
        assert 'high' in data.columns
        assert 'low' in data.columns
        assert 'close' in data.columns
        assert 'volume' in data.columns
        
        # Verify data types
        assert data['open'].dtype in [np.float64, np.int64]
        assert data['high'].dtype in [np.float64, np.int64]
        assert data['low'].dtype in [np.float64, np.int64]
        assert data['close'].dtype in [np.float64, np.int64]
        assert data['volume'].dtype in [np.float64, np.int64]
    
    def test_data_processing_integration(self):
        """Test data processing with technical indicators."""
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [47000, 47500, 48000, 48500, 49000],
            'high': [48000, 48500, 49000, 49500, 50000],
            'low': [46000, 47000, 47500, 48000, 48500],
            'close': [47500, 48000, 48500, 49000, 49500],
            'volume': [1000, 1200, 1500, 1800, 2000]
        })
        sample_data.index = pd.date_range('2023-01-01', periods=5, freq='30T')
        
        # Create processor
        processor = DataProcessor(self.breakout_config)
        
        # Test technical indicators
        processed_data = processor.calculate_technical_indicators(sample_data)
        
        # Verify indicators are added
        expected_indicators = ['rsi', 'macd', 'macd_signal', 'macd_hist', 
                              'bb_upper', 'bb_middle', 'bb_lower', 'atr', 'obv']
        
        for indicator in expected_indicators:
            assert indicator in processed_data.columns
        
        # Test breakout features
        breakout_data = processor.calculate_breakout_features(processed_data)
        
        # Verify breakout features
        expected_features = ['squeeze', 'donchian_high', 'donchian_low', 'donchian_width']
        
        for feature in expected_features:
            assert feature in breakout_data.columns
    
    @patch('pymongo.MongoClient')
    def test_data_storage_integration(self, mock_client):
        """Test data storage with proper mocking."""
        # Mock database
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_client.return_value.admin.command.return_value = {'ok': 1}
        
        # Mock insert_many to return inserted_ids
        mock_insert_result = Mock()
        mock_insert_result.inserted_ids = ['id1', 'id2', 'id3']
        mock_collection.insert_many.return_value = mock_insert_result
        
        # Create storage
        storage = MongoDBStorage(self.db_config)
        
        # Test data storage
        sample_data = pd.DataFrame({
            'open': [47000, 47500, 48000],
            'high': [48000, 48500, 49000],
            'low': [46000, 47000, 47500],
            'close': [47500, 48000, 48500],
            'volume': [1000, 1200, 1500]
        })
        
        result = storage.store_ohlcv("BTC/USDT", "30m", sample_data)
        
        # Verify storage
        assert result == True
        mock_collection.insert_many.assert_called_once()
    
    @patch('ccxt.binance')
    @patch('pymongo.MongoClient')
    def test_complete_pipeline_integration(self, mock_client, mock_binance):
        """Test complete data pipeline from collection to storage."""
        # Mock exchange
        mock_exchange = Mock()
        mock_exchange.fetch_ohlcv.return_value = [
            [1640995200000, 47000, 48000, 46000, 47500, 1000],
            [1640998800000, 47500, 48500, 47000, 48000, 1200],
            [1641002400000, 48000, 49000, 47500, 48500, 1500]
        ]
        mock_binance.return_value = mock_exchange
        
        # Mock database
        mock_db = Mock()
        mock_collection = Mock()
        mock_db.__getitem__ = Mock(return_value=mock_collection)
        mock_client.return_value.__getitem__.return_value = mock_db
        mock_client.return_value.admin.command.return_value = {'ok': 1}
        
        # Mock insert_many
        mock_insert_result = Mock()
        mock_insert_result.inserted_ids = ['id1', 'id2', 'id3']
        mock_collection.insert_many.return_value = mock_insert_result
        
        # Create components
        collector = BinanceDataCollector(self.api_config, self.data_config)
        processor = DataProcessor(self.breakout_config)
        storage = MongoDBStorage(self.db_config)
        
        # Step 1: Collect data
        raw_data = collector.collect_historical_data("BTC/USDT", "30m", 3)
        assert raw_data is not None
        assert len(raw_data) == 3
        
        # Step 2: Process data
        processed_data = processor.calculate_technical_indicators(raw_data)
        assert processed_data is not None
        assert 'rsi' in processed_data.columns
        
        # Step 3: Add breakout features
        breakout_data = processor.calculate_breakout_features(processed_data)
        assert breakout_data is not None
        assert 'squeeze' in breakout_data.columns
        
        # Step 4: Store data
        result = storage.store_ohlcv("BTC/USDT", "30m", breakout_data)
        assert result == True
        
        # Verify complete pipeline
        assert len(breakout_data) == 3
        assert 'close' in breakout_data.columns
        assert 'rsi' in breakout_data.columns
        assert 'squeeze' in breakout_data.columns
    
    def test_data_validation_integration(self):
        """Test data validation across pipeline."""
        # Create sample data
        sample_data = pd.DataFrame({
            'open': [47000, 47500, 48000],
            'high': [48000, 48500, 49000],
            'low': [46000, 47000, 47500],
            'close': [47500, 48000, 48500],
            'volume': [1000, 1200, 1500]
        })
        
        # Test data validation
        collector = BinanceDataCollector(self.api_config, self.data_config)
        
        # Valid data should pass
        assert collector.validate_data(sample_data) == True
        
        # Invalid data should fail
        invalid_data = pd.DataFrame({
            'open': [47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000]
            # Missing volume column
        })
        
        assert collector.validate_data(invalid_data) == False
    
    def test_error_handling_integration(self):
        """Test error handling across pipeline."""
        # Test with empty data
        empty_data = pd.DataFrame()
        
        processor = DataProcessor(self.breakout_config)
        
        # Should handle empty data gracefully
        try:
            result = processor.calculate_technical_indicators(empty_data)
            # Should return empty DataFrame or handle gracefully
            assert result is not None
        except Exception as e:
            # Should log error and handle gracefully
            assert "error" in str(e).lower() or "empty" in str(e).lower()
    
    def test_data_consistency_integration(self):
        """Test data consistency across pipeline."""
        # Create consistent sample data
        sample_data = pd.DataFrame({
            'open': [47000, 47500, 48000, 48500, 49000],
            'high': [48000, 48500, 49000, 49500, 50000],
            'low': [46000, 47000, 47500, 48000, 48500],
            'close': [47500, 48000, 48500, 49000, 49500],
            'volume': [1000, 1200, 1500, 1800, 2000]
        })
        sample_data.index = pd.date_range('2023-01-01', periods=5, freq='30T')
        
        processor = DataProcessor(self.breakout_config)
        
        # Process data
        processed_data = processor.calculate_technical_indicators(sample_data)
        
        # Verify data consistency
        assert len(processed_data) == len(sample_data)
        assert processed_data.index.equals(sample_data.index)
        
        # Verify original data is preserved
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in processed_data.columns
            assert processed_data[col].equals(sample_data[col])


if __name__ == "__main__":
    pytest.main([__file__])
