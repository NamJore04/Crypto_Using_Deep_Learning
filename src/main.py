"""
Main system integration for Crypto Futures Trading System.

This module provides the main entry point and system integration
for the complete trading system pipeline.
"""

import torch
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

from config.settings import TradingConfig, ModelConfig, DatabaseConfig
from data.collectors.binance_collector import BinanceDataCollector
from data.processors.data_processor import DataProcessor
from data.storage.mongodb_storage import MongoDBStorage
from models.cnn_lstm.model import CNNLSTMModel, CNNLSTMModelV2
from models.transformer.model import CNNTransformerModel, CNNTransformerModelV2
from models.ensemble.model_ensemble import ModelEnsemble, AdaptiveEnsemble
from models.cnn_lstm.trainer import ModelTrainer, create_data_loaders
from trading.backtest.backtest_engine import BacktestEngine, BacktestResults
from trading.risk_management.risk_manager import RiskManager, RiskLevel
from trading.signals.signal_generator import SignalGenerator, Signal, SignalType
from visualization.dashboard import TradingDashboard
from visualization.chart_utils import ChartGenerator, PerformanceAnalyzer

logger = logging.getLogger(__name__)


class TradingSystem:
    """
    Main trading system integration class.
    
    Orchestrates the complete trading pipeline from data collection
    to signal generation and backtesting.
    """
    
    def __init__(self, config: Dict[str, Any], model_path: Optional[str] = None):
        """
        Initialize trading system.
        
        Args:
            config: System configuration dictionary
            model_path: Path to pre-trained model (optional)
        """
        self.config = config
        self.model_path = model_path
        
        # Initialize configurations
        self.trading_config = TradingConfig(**config.get('trading', {}))
        self.model_config = ModelConfig(**config.get('model', {}))
        self.database_config = DatabaseConfig(**config.get('database', {}))
        
        # Initialize components
        self.data_collector = None
        self.data_processor = None
        self.storage = None
        self.model = None
        self.trainer = None
        self.backtest_engine = None
        self.risk_manager = None
        self.signal_generator = None
        self.dashboard = None
        
        # System state
        self.is_initialized = False
        self.current_data = None
        self.signals = []
        self.backtest_results = None
        
        logger.info("Trading system initialized")
    
    def initialize_system(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            logger.info("Initializing trading system components...")
            
            # Initialize data components
            self.data_collector = BinanceDataCollector()
            self.data_processor = DataProcessor()
            self.storage = MongoDBStorage(self.database_config)
            
            # Initialize trading components
            self.backtest_engine = BacktestEngine(self.trading_config)
            self.risk_manager = RiskManager(self.trading_config)
            self.signal_generator = SignalGenerator(
                signal_config=self.config.get('signal', {}),
                trading_config=self.trading_config
            )
            
            # Initialize model if path provided
            if self.model_path and os.path.exists(self.model_path):
                self.model = self.load_model(self.model_path)
                logger.info(f"Model loaded from {self.model_path}")
            else:
                logger.warning("No model path provided or model not found")
            
            # Initialize dashboard
            self.dashboard = TradingDashboard(
                config=self.trading_config,
                title="Crypto Trading System Dashboard"
            )
            
            self.is_initialized = True
            logger.info("Trading system initialization completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing trading system: {e}")
            return False
    
    def load_model(self, model_path: str) -> torch.nn.Module:
        """
        Load pre-trained model.
        
        Args:
            model_path: Path to model file
            
        Returns:
            torch.nn.Module: Loaded model
        """
        try:
            # Determine model type from path or config
            model_type = self.config.get('model_type', 'cnn_lstm')
            
            if model_type == 'cnn_lstm':
                model = CNNLSTMModel(
                    input_dim=self.model_config.input_dim,
                    sequence_length=self.model_config.sequence_length,
                    hidden_dim=self.model_config.hidden_dim,
                    num_layers=self.model_config.num_layers,
                    num_classes=self.model_config.num_classes,
                    dropout_rate=self.model_config.dropout_rate
                )
            elif model_type == 'cnn_transformer':
                model = CNNTransformerModel(
                    input_dim=self.model_config.input_dim,
                    sequence_length=self.model_config.sequence_length,
                    d_model=self.model_config.hidden_dim,
                    num_classes=self.model_config.num_classes,
                    dropout_rate=self.model_config.dropout_rate
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Load model weights
            model.load_state_dict(torch.load(model_path, map_location='cpu'))
            model.eval()
            
            logger.info(f"Model loaded successfully: {model_type}")
            return model
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def collect_data(self, symbol: str = "BTC/USDT", timeframe: str = "30m", 
                    days: int = 365) -> bool:
        """
        Collect market data.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            days: Number of days to collect
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return False
            
            logger.info(f"Collecting data for {symbol} ({timeframe}) for {days} days")
            
            # Collect historical data
            data = self.data_collector.collect_historical_data(symbol, timeframe, days)
            
            if data is None or data.empty:
                logger.error("No data collected")
                return False
            
            # Process data
            processed_data = self.data_processor.process_data(data)
            
            if processed_data is None or processed_data.empty:
                logger.error("Data processing failed")
                return False
            
            # Store data
            self.storage.store_ohlcv(symbol, timeframe, processed_data)
            
            # Store current data for system use
            self.current_data = {
                'symbol': symbol,
                'timeframe': timeframe,
                'data': processed_data,
                'timestamp': datetime.now()
            }
            
            logger.info(f"Data collection completed: {len(processed_data)} records")
            return True
            
        except Exception as e:
            logger.error(f"Error collecting data: {e}")
            return False
    
    def train_model(self, data: Optional[pd.DataFrame] = None, 
                   model_type: str = "cnn_lstm") -> bool:
        """
        Train model on collected data.
        
        Args:
            data: Training data (uses current data if None)
            model_type: Type of model to train
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return False
            
            # Use current data if no data provided
            if data is None:
                if self.current_data is None:
                    logger.error("No data available for training")
                    return False
                data = self.current_data['data']
            
            logger.info(f"Training {model_type} model...")
            
            # Create sequences
            X, y = self.data_processor.create_sequences(
                data, 
                sequence_length=self.model_config.sequence_length,
                target_col='market_regime'
            )
            
            if X is None or y is None:
                logger.error("Failed to create training sequences")
                return False
            
            # Create data loaders
            train_loader, val_loader, test_loader = create_data_loaders(
                X, y, 
                sequence_length=self.model_config.sequence_length,
                batch_size=self.model_config.batch_size,
                test_size=0.15,
                val_size=0.15
            )
            
            # Initialize model
            if model_type == 'cnn_lstm':
                self.model = CNNLSTMModel(
                    input_dim=self.model_config.input_dim,
                    sequence_length=self.model_config.sequence_length,
                    hidden_dim=self.model_config.hidden_dim,
                    num_layers=self.model_config.num_layers,
                    num_classes=self.model_config.num_classes,
                    dropout_rate=self.model_config.dropout_rate
                )
            elif model_type == 'cnn_transformer':
                self.model = CNNTransformerModel(
                    input_dim=self.model_config.input_dim,
                    sequence_length=self.model_config.sequence_length,
                    d_model=self.model_config.hidden_dim,
                    num_classes=self.model_config.num_classes,
                    dropout_rate=self.model_config.dropout_rate
                )
            else:
                raise ValueError(f"Unknown model type: {model_type}")
            
            # Initialize trainer
            self.trainer = ModelTrainer(self.model, self.model_config)
            self.trainer.setup_training(
                learning_rate=self.model_config.learning_rate,
                weight_decay=self.model_config.weight_decay
            )
            
            # Train model
            model_save_path = f"models/{model_type}_best.pth"
            os.makedirs("models", exist_ok=True)
            
            self.trainer.train(train_loader, val_loader, model_save_path)
            
            # Evaluate model
            class_names = ['SIDEWAY', 'UPTREND', 'DOWNTREND', 'BREAKOUT']
            metrics = self.trainer.evaluate_model(test_loader, class_names)
            
            logger.info(f"Model training completed. Accuracy: {metrics['accuracy']:.4f}")
            return True
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return False
    
    def generate_signals(self, data: Optional[pd.DataFrame] = None) -> List[Signal]:
        """
        Generate trading signals from model predictions.
        
        Args:
            data: Market data (uses current data if None)
            
        Returns:
            List[Signal]: Generated trading signals
        """
        try:
            if not self.is_initialized or self.model is None:
                logger.error("System not initialized or model not loaded")
                return []
            
            # Use current data if no data provided
            if data is None:
                if self.current_data is None:
                    logger.error("No data available for signal generation")
                    return []
                data = self.current_data['data']
            
            logger.info("Generating trading signals...")
            
            # Create sequences for prediction
            X, _ = self.data_processor.create_sequences(
                data,
                sequence_length=self.model_config.sequence_length,
                target_col='market_regime'
            )
            
            if X is None:
                logger.error("Failed to create prediction sequences")
                return []
            
            # Generate predictions
            signals = []
            with torch.no_grad():
                for i in range(len(X)):
                    # Get sequence
                    sequence = torch.tensor(X[i], dtype=torch.float32).unsqueeze(0)
                    
                    # Get prediction
                    prediction = self.model(sequence)
                    probabilities = torch.softmax(prediction, dim=1).numpy()[0]
                    
                    # Get current price and timestamp
                    price = data.iloc[i + self.model_config.sequence_length]['close']
                    timestamp = data.index[i + self.model_config.sequence_length]
                    atr = data.iloc[i + self.model_config.sequence_length].get('atr', None)
                    
                    # Generate signal
                    signal = self.signal_generator.generate_signal(
                        predictions=probabilities,
                        price=price,
                        timestamp=timestamp,
                        atr=atr,
                        additional_features={'sequence_index': i}
                    )
                    
                    signals.append(signal)
            
            self.signals = signals
            logger.info(f"Generated {len(signals)} trading signals")
            return signals
            
        except Exception as e:
            logger.error(f"Error generating signals: {e}")
            return []
    
    def run_backtest(self, data: Optional[pd.DataFrame] = None, 
                    signals: Optional[List[Signal]] = None) -> BacktestResults:
        """
        Run complete backtest simulation.
        
        Args:
            data: Market data (uses current data if None)
            signals: Trading signals (generates if None)
            
        Returns:
            BacktestResults: Backtest results
        """
        try:
            if not self.is_initialized:
                logger.error("System not initialized")
                return None
            
            # Use current data if no data provided
            if data is None:
                if self.current_data is None:
                    logger.error("No data available for backtest")
                    return None
                data = self.current_data['data']
            
            # Generate signals if not provided
            if signals is None:
                signals = self.generate_signals(data)
                if not signals:
                    logger.error("Failed to generate signals for backtest")
                    return None
            
            logger.info(f"Running backtest with {len(signals)} signals...")
            
            # Reset backtest engine
            self.backtest_engine.reset()
            
            # Execute trades based on signals
            for i, signal in enumerate(signals):
                if i >= len(data):
                    break
                
                # Get market data for this signal
                market_data = data.iloc[i + self.model_config.sequence_length]
                price = market_data['close']
                timestamp = data.index[i + self.model_config.sequence_length]
                atr = market_data.get('atr', None)
                
                # Execute trade
                self.backtest_engine.execute_trade(
                    signal=signal.signal_type.value,
                    price=price,
                    timestamp=timestamp,
                    atr=atr,
                    metadata={'confidence': signal.confidence}
                )
            
            # Calculate results
            results = self.backtest_engine.calculate_metrics()
            self.backtest_results = results
            
            logger.info(f"Backtest completed. Total return: {results.total_return:.2%}")
            return results
            
        except Exception as e:
            logger.error(f"Error running backtest: {e}")
            return None
    
    def run_complete_pipeline(self, symbol: str = "BTC/USDT", 
                            timeframe: str = "30m", days: int = 365,
                            model_type: str = "cnn_lstm") -> Dict[str, Any]:
        """
        Run complete trading system pipeline.
        
        Args:
            symbol: Trading symbol
            timeframe: Data timeframe
            days: Number of days to collect
            model_type: Type of model to use
            
        Returns:
            Dict[str, Any]: Complete pipeline results
        """
        try:
            logger.info("Starting complete trading system pipeline...")
            
            # Initialize system
            if not self.initialize_system():
                return {'success': False, 'error': 'System initialization failed'}
            
            # Collect data
            if not self.collect_data(symbol, timeframe, days):
                return {'success': False, 'error': 'Data collection failed'}
            
            # Train model
            if not self.train_model(model_type=model_type):
                return {'success': False, 'error': 'Model training failed'}
            
            # Generate signals
            signals = self.generate_signals()
            if not signals:
                return {'success': False, 'error': 'Signal generation failed'}
            
            # Run backtest
            backtest_results = self.run_backtest(signals=signals)
            if backtest_results is None:
                return {'success': False, 'error': 'Backtest failed'}
            
            # Update dashboard
            self.dashboard.update_data(
                backtest_results=backtest_results,
                current_data=self.current_data,
                signals=signals
            )
            
            # Prepare results
            results = {
                'success': True,
                'symbol': symbol,
                'timeframe': timeframe,
                'days': days,
                'model_type': model_type,
                'data_points': len(self.current_data['data']),
                'signals_generated': len(signals),
                'backtest_results': backtest_results,
                'performance_metrics': {
                    'total_return': backtest_results.total_return,
                    'sharpe_ratio': backtest_results.sharpe_ratio,
                    'max_drawdown': backtest_results.max_drawdown,
                    'win_rate': backtest_results.win_rate,
                    'total_trades': backtest_results.total_trades
                }
            }
            
            logger.info("Complete trading system pipeline finished successfully")
            return results
            
        except Exception as e:
            logger.error(f"Error in complete pipeline: {e}")
            return {'success': False, 'error': str(e)}
    
    def start_dashboard(self, host: str = "127.0.0.1", port: int = 8050):
        """
        Start the trading dashboard.
        
        Args:
            host: Dashboard host
            port: Dashboard port
        """
        try:
            if not self.is_initialized or self.dashboard is None:
                logger.error("Dashboard not initialized")
                return
            
            logger.info(f"Starting dashboard on {host}:{port}")
            self.dashboard.run_server(host=host, port=port, debug=False)
            
        except Exception as e:
            logger.error(f"Error starting dashboard: {e}")
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get current system status.
        
        Returns:
            Dict[str, Any]: System status information
        """
        return {
            'initialized': self.is_initialized,
            'model_loaded': self.model is not None,
            'data_available': self.current_data is not None,
            'signals_generated': len(self.signals),
            'backtest_completed': self.backtest_results is not None,
            'dashboard_available': self.dashboard is not None,
            'timestamp': datetime.now().isoformat()
        }


def main():
    """Main entry point for the trading system."""
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # System configuration
    config = {
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
            'epochs': 100,
            'dropout_rate': 0.3
        },
        'database': {
            'host': 'localhost',
            'port': 27017,
            'database': 'crypto_trading'
        },
        'signal': {
            'breakout_threshold': 0.6,
            'trend_threshold': 0.5,
            'confidence_threshold': 0.4
        },
        'model_type': 'cnn_lstm'
    }
    
    # Initialize trading system
    trading_system = TradingSystem(config)
    
    # Run complete pipeline
    results = trading_system.run_complete_pipeline(
        symbol="BTC/USDT",
        timeframe="30m",
        days=365,
        model_type="cnn_lstm"
    )
    
    if results['success']:
        print("✅ Trading system pipeline completed successfully!")
        print(f"📊 Performance: {results['performance_metrics']}")
        
        # Start dashboard
        print("🚀 Starting dashboard...")
        trading_system.start_dashboard()
    else:
        print(f"❌ Pipeline failed: {results['error']}")


if __name__ == "__main__":
    main()
