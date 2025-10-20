"""
Integration tests for complete system.

Tests end-to-end system integration and performance.
"""

import pytest
import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path
import json
import os

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from main import TradingSystem
from system_integration import SystemMonitor, SystemIntegrator, SystemStatus, SystemHealth
from config.settings import TradingConfig, ModelConfig, DatabaseConfig


class TestSystemIntegration:
    """Test cases for system integration."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'trading': {
                'initial_capital': 10000.0,
                'commission': 0.001,
                'max_position_size': 0.1,
                'stop_loss_atr': 2.0,
                'take_profit_atr': 3.0,
                'max_drawdown': 0.2,
                'risk_per_trade': 0.02
            },
            'model': {
                'input_dim': 20,
                'sequence_length': 60,
                'hidden_dim': 64,
                'num_layers': 2,
                'num_classes': 4,
                'learning_rate': 0.001,
                'batch_size': 32,
                'epochs': 10,
                'dropout_rate': 0.3
            },
            'database': {
                'host': 'localhost',
                'port': 27017,
                'database': 'test_crypto_trading'
            },
            'signal': {
                'breakout_threshold': 0.6,
                'trend_threshold': 0.5,
                'confidence_threshold': 0.4
            },
            'model_type': 'cnn_lstm'
        }
        
        self.trading_system = TradingSystem(self.config)
    
    def test_system_initialization(self):
        """Test system initialization."""
        assert self.trading_system is not None
        assert self.trading_system.config == self.config
        assert not self.trading_system.is_initialized
    
    @patch('src.data.collectors.binance_collector.BinanceDataCollector')
    @patch('src.data.processors.data_processor.DataProcessor')
    @patch('src.data.storage.mongodb_storage.MongoDBStorage')
    @patch('src.trading.backtest.backtest_engine.BacktestEngine')
    @patch('src.trading.risk_management.risk_manager.RiskManager')
    @patch('src.trading.signals.signal_generator.SignalGenerator')
    @patch('src.visualization.dashboard.TradingDashboard')
    def test_initialize_system(self, mock_dashboard, mock_signal_gen, mock_risk, 
                              mock_backtest, mock_storage, mock_processor, mock_collector):
        """Test system initialization."""
        # Mock all components
        mock_collector.return_value = Mock()
        mock_processor.return_value = Mock()
        mock_storage.return_value = Mock()
        mock_backtest.return_value = Mock()
        mock_risk.return_value = Mock()
        mock_signal_gen.return_value = Mock()
        mock_dashboard.return_value = Mock()
        
        # Initialize system
        result = self.trading_system.initialize_system()
        
        assert result == True
        assert self.trading_system.is_initialized == True
        assert self.trading_system.data_collector is not None
        assert self.trading_system.data_processor is not None
        assert self.trading_system.storage is not None
        assert self.trading_system.backtest_engine is not None
        assert self.trading_system.risk_manager is not None
        assert self.trading_system.signal_generator is not None
        assert self.trading_system.dashboard is not None
    
    @patch('src.data.collectors.binance_collector.BinanceDataCollector')
    @patch('src.data.processors.data_processor.DataProcessor')
    @patch('src.data.storage.mongodb_storage.MongoDBStorage')
    def test_collect_data(self, mock_storage, mock_processor, mock_collector):
        """Test data collection."""
        # Mock components
        mock_collector_instance = Mock()
        mock_collector_instance.collect_historical_data.return_value = pd.DataFrame({
            'open': [47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000],
            'volume': [1000, 1200]
        })
        mock_collector.return_value = mock_collector_instance
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_data.return_value = pd.DataFrame({
            'open': [47000, 47500],
            'high': [48000, 48500],
            'low': [46000, 47000],
            'close': [47500, 48000],
            'volume': [1000, 1200],
            'market_regime': [0, 1]
        })
        mock_processor.return_value = mock_processor_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.store_ohlcv.return_value = True
        mock_storage.return_value = mock_storage_instance
        
        # Initialize system
        self.trading_system.initialize_system()
        
        # Collect data
        result = self.trading_system.collect_data("BTC/USDT", "30m", 1)
        
        assert result == True
        assert self.trading_system.current_data is not None
        assert self.trading_system.current_data['symbol'] == "BTC/USDT"
        assert self.trading_system.current_data['timeframe'] == "30m"
    
    @patch('src.models.cnn_lstm.model.CNNLSTMModel')
    @patch('src.models.cnn_lstm.trainer.ModelTrainer')
    def test_train_model(self, mock_trainer, mock_model):
        """Test model training."""
        # Mock model and trainer
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = None
        mock_trainer.return_value = mock_trainer_instance
        
        # Initialize system
        self.trading_system.initialize_system()
        
        # Create sample data
        self.trading_system.current_data = {
            'data': pd.DataFrame({
                'open': np.random.randn(100) * 100 + 50000,
                'high': np.random.randn(100) * 100 + 50100,
                'low': np.random.randn(100) * 100 + 49900,
                'close': np.random.randn(100) * 100 + 50000,
                'volume': np.random.randn(100) * 1000 + 10000,
                'market_regime': np.random.randint(0, 4, 100)
            })
        }
        
        # Train model
        result = self.trading_system.train_model(model_type='cnn_lstm')
        
        assert result == True
        assert self.trading_system.model is not None
        assert self.trading_system.trainer is not None
    
    @patch('src.models.cnn_lstm.model.CNNLSTMModel')
    def test_load_model(self, mock_model):
        """Test model loading."""
        # Mock model
        mock_model_instance = Mock()
        mock_model.return_value = mock_model_instance
        
        # Create temporary model file
        os.makedirs('models', exist_ok=True)
        torch.save({'test': 'data'}, 'models/test_model.pth')
        
        # Load model
        model = self.trading_system.load_model('models/test_model.pth')
        
        assert model is not None
        assert model == mock_model_instance
        
        # Clean up
        os.remove('models/test_model.pth')
    
    def test_generate_signals(self):
        """Test signal generation."""
        # Mock model
        mock_model = Mock()
        mock_model.return_value = torch.tensor([[0.1, 0.1, 0.1, 0.7]])  # Breakout prediction
        self.trading_system.model = mock_model
        
        # Mock data processor
        mock_processor = Mock()
        mock_processor.create_sequences.return_value = (np.random.randn(10, 60, 20), None)
        self.trading_system.data_processor = mock_processor
        
        # Mock signal generator
        mock_signal_gen = Mock()
        mock_signal_gen.generate_signal.return_value = Mock()
        self.trading_system.signal_generator = mock_signal_gen
        
        # Initialize system
        self.trading_system.is_initialized = True
        
        # Create sample data
        self.trading_system.current_data = {
            'data': pd.DataFrame({
                'close': np.random.randn(100) * 100 + 50000,
                'atr': np.random.randn(100) * 100 + 1000
            })
        }
        
        # Generate signals
        signals = self.trading_system.generate_signals()
        
        assert signals is not None
        assert len(signals) > 0
    
    def test_run_backtest(self):
        """Test backtest execution."""
        # Mock backtest engine
        mock_backtest_engine = Mock()
        mock_backtest_engine.calculate_metrics.return_value = Mock()
        self.trading_system.backtest_engine = mock_backtest_engine
        
        # Mock signals
        mock_signals = [Mock() for _ in range(5)]
        for signal in mock_signals:
            signal.signal_type.value = 'LONG'
            signal.price = 50000.0
            signal.timestamp = datetime.now()
            signal.confidence = 0.8
        
        # Initialize system
        self.trading_system.is_initialized = True
        
        # Create sample data
        self.trading_system.current_data = {
            'data': pd.DataFrame({
                'close': np.random.randn(100) * 100 + 50000,
                'atr': np.random.randn(100) * 100 + 1000
            })
        }
        
        # Run backtest
        results = self.trading_system.run_backtest(signals=mock_signals)
        
        assert results is not None
        assert mock_backtest_engine.calculate_metrics.called
    
    @patch('src.data.collectors.binance_collector.BinanceDataCollector')
    @patch('src.data.processors.data_processor.DataProcessor')
    @patch('src.data.storage.mongodb_storage.MongoDBStorage')
    @patch('src.models.cnn_lstm.model.CNNLSTMModel')
    @patch('src.models.cnn_lstm.trainer.ModelTrainer')
    @patch('src.trading.backtest.backtest_engine.BacktestEngine')
    @patch('src.trading.risk_management.risk_manager.RiskManager')
    @patch('src.trading.signals.signal_generator.SignalGenerator')
    @patch('src.visualization.dashboard.TradingDashboard')
    def test_run_complete_pipeline(self, mock_dashboard, mock_signal_gen, mock_risk,
                                 mock_backtest, mock_trainer, mock_model, mock_storage,
                                 mock_processor, mock_collector):
        """Test complete pipeline execution."""
        # Mock all components
        mock_collector_instance = Mock()
        mock_collector_instance.collect_historical_data.return_value = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 10000
        })
        mock_collector.return_value = mock_collector_instance
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_data.return_value = pd.DataFrame({
            'open': np.random.randn(100) * 100 + 50000,
            'high': np.random.randn(100) * 100 + 50100,
            'low': np.random.randn(100) * 100 + 49900,
            'close': np.random.randn(100) * 100 + 50000,
            'volume': np.random.randn(100) * 1000 + 10000,
            'market_regime': np.random.randint(0, 4, 100)
        })
        mock_processor_instance.create_sequences.return_value = (np.random.randn(50, 60, 20), np.random.randint(0, 4, 50))
        mock_processor.return_value = mock_processor_instance
        
        mock_storage_instance = Mock()
        mock_storage_instance.store_ohlcv.return_value = True
        mock_storage.return_value = mock_storage_instance
        
        mock_model_instance = Mock()
        mock_model_instance.return_value = torch.tensor([[0.1, 0.1, 0.1, 0.7]])
        mock_model.return_value = mock_model_instance
        
        mock_trainer_instance = Mock()
        mock_trainer_instance.train.return_value = None
        mock_trainer.return_value = mock_trainer_instance
        
        mock_backtest_instance = Mock()
        mock_backtest_instance.calculate_metrics.return_value = Mock()
        mock_backtest.return_value = mock_backtest_instance
        
        mock_risk_instance = Mock()
        mock_risk.return_value = mock_risk_instance
        
        mock_signal_gen_instance = Mock()
        mock_signal_gen_instance.generate_signal.return_value = Mock()
        mock_signal_gen.return_value = mock_signal_gen_instance
        
        mock_dashboard_instance = Mock()
        mock_dashboard_instance.update_data.return_value = None
        mock_dashboard.return_value = mock_dashboard_instance
        
        # Run complete pipeline
        results = self.trading_system.run_complete_pipeline(
            symbol="BTC/USDT",
            timeframe="30m",
            days=1,
            model_type="cnn_lstm"
        )
        
        assert results['success'] == True
        assert 'backtest_results' in results
        assert 'performance_metrics' in results
        assert 'signals_generated' in results
    
    def test_get_system_status(self):
        """Test system status retrieval."""
        # Test uninitialized system
        status = self.trading_system.get_system_status()
        
        assert status['initialized'] == False
        assert status['model_loaded'] == False
        assert status['data_available'] == False
        assert status['signals_generated'] == 0
        assert status['backtest_completed'] == False
        assert status['dashboard_available'] == False
        assert 'timestamp' in status


class TestSystemMonitor:
    """Test cases for SystemMonitor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'database': {
                'host': 'localhost',
                'port': 27017,
                'database': 'test_crypto_trading'
            },
            'trading': {
                'initial_capital': 10000.0,
                'commission': 0.001
            }
        }
        
        self.monitor = SystemMonitor(self.config)
    
    def test_initialization(self):
        """Test monitor initialization."""
        assert self.monitor is not None
        assert self.monitor.config == self.config
        assert not self.monitor.monitoring
        assert len(self.monitor.health_history) == 0
    
    @patch('src.data.storage.mongodb_storage.MongoDBStorage')
    def test_check_database_health(self, mock_storage):
        """Test database health check."""
        # Mock storage
        mock_storage_instance = Mock()
        mock_storage_instance.store_ohlcv.return_value = True
        mock_storage_instance.get_ohlcv.return_value = pd.DataFrame()
        mock_storage_instance.delete_ohlcv.return_value = True
        mock_storage.return_value = mock_storage_instance
        
        # Check database health
        health = self.monitor.check_database_health()
        
        assert health.name == "database"
        assert health.status in [SystemStatus.HEALTHY, SystemStatus.CRITICAL]
        assert health.last_check is not None
        assert health.response_time >= 0
    
    def test_check_model_health(self):
        """Test model health check."""
        # Test with non-existent model
        health = self.monitor.check_model_health()
        
        assert health.name == "model"
        assert health.status == SystemStatus.WARNING
        assert health.error_message is not None
        
        # Test with existing model
        os.makedirs('models', exist_ok=True)
        torch.save({'test': 'data'}, 'models/test_model.pth')
        
        health = self.monitor.check_model_health('models/test_model.pth')
        
        assert health.name == "model"
        assert health.status == SystemStatus.HEALTHY
        assert health.metrics is not None
        
        # Clean up
        os.remove('models/test_model.pth')
    
    @patch('src.trading.backtest.backtest_engine.BacktestEngine')
    def test_check_trading_system_health(self, mock_backtest):
        """Test trading system health check."""
        # Mock backtest engine
        mock_backtest_instance = Mock()
        mock_backtest_instance.execute_trade.return_value = None
        mock_backtest_instance.calculate_metrics.return_value = Mock()
        mock_backtest.return_value = mock_backtest_instance
        
        # Check trading system health
        health = self.monitor.check_trading_system_health()
        
        assert health.name == "trading_system"
        assert health.status in [SystemStatus.HEALTHY, SystemStatus.CRITICAL]
        assert health.last_check is not None
        assert health.response_time >= 0
    
    def test_check_system_resources(self):
        """Test system resources health check."""
        health = self.monitor.check_system_resources()
        
        assert health.name == "system_resources"
        assert health.status in [SystemStatus.HEALTHY, SystemStatus.WARNING, SystemStatus.CRITICAL]
        assert health.last_check is not None
        assert health.response_time >= 0
        assert health.metrics is not None
        assert 'cpu_percent' in health.metrics
        assert 'memory_percent' in health.metrics
    
    @patch('src.data.collectors.binance_collector.BinanceDataCollector')
    @patch('src.data.processors.data_processor.DataProcessor')
    def test_check_data_pipeline_health(self, mock_processor, mock_collector):
        """Test data pipeline health check."""
        # Mock components
        mock_collector_instance = Mock()
        mock_collector.return_value = mock_collector_instance
        
        mock_processor_instance = Mock()
        mock_processor_instance.process_data.return_value = pd.DataFrame({
            'test': [1, 2, 3]
        })
        mock_processor.return_value = mock_processor_instance
        
        # Check data pipeline health
        health = self.monitor.check_data_pipeline_health()
        
        assert health.name == "data_pipeline"
        assert health.status in [SystemStatus.HEALTHY, SystemStatus.CRITICAL]
        assert health.last_check is not None
        assert health.response_time >= 0
    
    def test_run_health_check(self):
        """Test comprehensive health check."""
        health = self.monitor.run_health_check()
        
        assert isinstance(health, SystemHealth)
        assert health.overall_status is not None
        assert health.timestamp is not None
        assert len(health.components) > 0
        assert health.system_metrics is not None
        assert health.uptime >= 0
        
        # Check that health is stored in history
        assert len(self.monitor.health_history) == 1
        assert self.monitor.health_history[0] == health
    
    def test_get_health_summary(self):
        """Test health summary retrieval."""
        # Run health check first
        self.monitor.run_health_check()
        
        summary = self.monitor.get_health_summary()
        
        assert 'current_status' in summary
        assert 'timestamp' in summary
        assert 'uptime' in summary
        assert 'components' in summary
        assert 'system_metrics' in summary
        assert 'history_length' in summary


class TestSystemIntegrator:
    """Test cases for SystemIntegrator."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.config = {
            'database': {
                'host': 'localhost',
                'port': 27017,
                'database': 'test_crypto_trading'
            },
            'trading': {
                'initial_capital': 10000.0,
                'commission': 0.001
            },
            'model': {
                'input_dim': 20,
                'sequence_length': 60,
                'num_classes': 4
            },
            'signal': {
                'breakout_threshold': 0.6,
                'trend_threshold': 0.5,
                'confidence_threshold': 0.4
            }
        }
        
        self.integrator = SystemIntegrator(self.config)
    
    def test_initialization(self):
        """Test integrator initialization."""
        assert self.integrator is not None
        assert self.integrator.config == self.config
        assert self.integrator.monitor is not None
        assert len(self.integrator.integration_tests) == 0
    
    def test_run_integration_tests(self):
        """Test integration tests execution."""
        results = self.integrator.run_integration_tests()
        
        assert 'timestamp' in results
        assert 'tests' in results
        assert 'overall_success' in results
        assert 'summary' in results
        assert len(results['tests']) > 0
        assert 'total_tests' in results['summary']
        assert 'passed_tests' in results['summary']
        assert 'success_rate' in results['summary']
    
    def test_run_performance_tests(self):
        """Test performance tests execution."""
        results = self.integrator.run_performance_tests()
        
        assert 'timestamp' in results
        assert 'tests' in results
        assert 'overall_performance' in results
        assert 'summary' in results
        assert len(results['tests']) > 0
        assert 'total_tests' in results['summary']
        assert 'passed_tests' in results['summary']
        assert 'success_rate' in results['summary']
    
    def test_generate_integration_report(self):
        """Test integration report generation."""
        report = self.integrator.generate_integration_report()
        
        assert 'timestamp' in report
        assert 'system_health' in report
        assert 'integration_tests' in report
        assert 'performance_tests' in report
        assert 'recommendations' in report
        
        # Check that report is saved
        assert os.path.exists('reports/integration_report.json')
        
        # Clean up
        if os.path.exists('reports/integration_report.json'):
            os.remove('reports/integration_report.json')
        if os.path.exists('reports'):
            os.rmdir('reports')


if __name__ == "__main__":
    pytest.main([__file__])
