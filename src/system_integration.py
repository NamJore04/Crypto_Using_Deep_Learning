"""
System integration utilities for Crypto Futures Trading System.

This module provides utilities for system integration, monitoring,
and health checks across all system components.
"""

import logging
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import os
from pathlib import Path

from .config.settings import TradingConfig, ModelConfig, DatabaseConfig
from .data.storage.mongodb_storage import MongoDBStorage
from .trading.backtest.backtest_engine import BacktestEngine
from .models.cnn_lstm.trainer import ModelTrainer

logger = logging.getLogger(__name__)


class SystemStatus(Enum):
    """System status enumeration."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    OFFLINE = "offline"


@dataclass
class ComponentStatus:
    """Component status information."""
    name: str
    status: SystemStatus
    last_check: datetime
    response_time: float
    error_message: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None


@dataclass
class SystemHealth:
    """System health information."""
    overall_status: SystemStatus
    timestamp: datetime
    components: List[ComponentStatus]
    system_metrics: Dict[str, Any]
    uptime: float


class SystemMonitor:
    """
    System monitoring and health check utilities.
    
    Provides comprehensive monitoring of all system components
    including data pipeline, models, trading system, and database.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize system monitor.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.monitoring = False
        self.monitor_thread = None
        self.health_history = []
        self.alert_thresholds = {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0,
            'error_rate': 0.05
        }
        
        # Initialize components for monitoring
        self.database_config = DatabaseConfig(**config.get('database', {}))
        self.trading_config = TradingConfig(**config.get('trading', {}))
        
        logger.info("System monitor initialized")
    
    def check_database_health(self) -> ComponentStatus:
        """
        Check database health and connectivity.
        
        Returns:
            ComponentStatus: Database health status
        """
        start_time = time.time()
        
        try:
            # Test database connection
            storage = MongoDBStorage(self.database_config)
            
            # Test basic operations
            test_data = {
                'timestamp': datetime.now(),
                'test': True,
                'component': 'system_monitor'
            }
            
            # Test write
            storage.store_ohlcv('TEST/USDT', '1m', pd.DataFrame([test_data]))
            
            # Test read
            data = storage.get_ohlcv('TEST/USDT', '1m', limit=1)
            
            # Test delete
            storage.delete_ohlcv('TEST/USDT', '1m')
            
            response_time = time.time() - start_time
            
            return ComponentStatus(
                name="database",
                status=SystemStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={
                    'connection_test': True,
                    'read_write_test': True,
                    'response_time': response_time
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Database health check failed: {e}")
            
            return ComponentStatus(
                name="database",
                status=SystemStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
    
    def check_model_health(self, model_path: Optional[str] = None) -> ComponentStatus:
        """
        Check model health and availability.
        
        Args:
            model_path: Path to model file
            
        Returns:
            ComponentStatus: Model health status
        """
        start_time = time.time()
        
        try:
            if model_path and os.path.exists(model_path):
                # Check model file
                file_size = os.path.getsize(model_path)
                file_age = time.time() - os.path.getmtime(model_path)
                
                # Check if model can be loaded
                import torch
                model_state = torch.load(model_path, map_location='cpu')
                
                response_time = time.time() - start_time
                
                return ComponentStatus(
                    name="model",
                    status=SystemStatus.HEALTHY,
                    last_check=datetime.now(),
                    response_time=response_time,
                    metrics={
                        'model_path': model_path,
                        'file_size_mb': file_size / (1024 * 1024),
                        'file_age_hours': file_age / 3600,
                        'model_loaded': True,
                        'state_dict_keys': len(model_state.keys())
                    }
                )
            else:
                response_time = time.time() - start_time
                
                return ComponentStatus(
                    name="model",
                    status=SystemStatus.WARNING,
                    last_check=datetime.now(),
                    response_time=response_time,
                    error_message="Model file not found or not specified",
                    metrics={'model_available': False}
                )
                
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Model health check failed: {e}")
            
            return ComponentStatus(
                name="model",
                status=SystemStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
    
    def check_trading_system_health(self) -> ComponentStatus:
        """
        Check trading system health.
        
        Returns:
            ComponentStatus: Trading system health status
        """
        start_time = time.time()
        
        try:
            # Test backtest engine
            backtest_engine = BacktestEngine(self.trading_config)
            
            # Test basic operations
            backtest_engine.reset()
            
            # Test trade execution
            test_trade = backtest_engine.execute_trade(
                signal='LONG',
                price=50000.0,
                timestamp=datetime.now(),
                atr=1000.0
            )
            
            # Test metrics calculation
            metrics = backtest_engine.calculate_metrics()
            
            response_time = time.time() - start_time
            
            return ComponentStatus(
                name="trading_system",
                status=SystemStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={
                    'backtest_engine': True,
                    'trade_execution': True,
                    'metrics_calculation': True,
                    'initial_capital': backtest_engine.initial_capital
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Trading system health check failed: {e}")
            
            return ComponentStatus(
                name="trading_system",
                status=SystemStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
    
    def check_system_resources(self) -> ComponentStatus:
        """
        Check system resource usage.
        
        Returns:
            ComponentStatus: System resources health status
        """
        start_time = time.time()
        
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            # Determine status based on thresholds
            status = SystemStatus.HEALTHY
            if (cpu_percent > self.alert_thresholds['cpu_usage'] or 
                memory.percent > self.alert_thresholds['memory_usage'] or
                disk.percent > self.alert_thresholds['disk_usage']):
                status = SystemStatus.WARNING
            
            if (cpu_percent > 95 or memory.percent > 95 or disk.percent > 95):
                status = SystemStatus.CRITICAL
            
            response_time = time.time() - start_time
            
            return ComponentStatus(
                name="system_resources",
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': memory.available / (1024**3),
                    'disk_percent': disk.percent,
                    'disk_free_gb': disk.free / (1024**3),
                    'load_average': os.getloadavg() if hasattr(os, 'getloadavg') else None
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"System resources health check failed: {e}")
            
            return ComponentStatus(
                name="system_resources",
                status=SystemStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
    
    def check_data_pipeline_health(self) -> ComponentStatus:
        """
        Check data pipeline health.
        
        Returns:
            ComponentStatus: Data pipeline health status
        """
        start_time = time.time()
        
        try:
            # Test data collector
            from .data.collectors.binance_collector import BinanceDataCollector
            collector = BinanceDataCollector()
            
            # Test data processor
            from .data.processors.data_processor import DataProcessor
            processor = DataProcessor()
            
            # Test with sample data
            import pandas as pd
            import numpy as np
            
            sample_data = pd.DataFrame({
                'open': np.random.randn(100) * 100 + 50000,
                'high': np.random.randn(100) * 100 + 50100,
                'low': np.random.randn(100) * 100 + 49900,
                'close': np.random.randn(100) * 100 + 50000,
                'volume': np.random.randn(100) * 1000 + 10000
            })
            
            # Test data processing
            processed_data = processor.process_data(sample_data)
            
            response_time = time.time() - start_time
            
            return ComponentStatus(
                name="data_pipeline",
                status=SystemStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=response_time,
                metrics={
                    'data_collector': True,
                    'data_processor': True,
                    'sample_processing': True,
                    'processed_features': len(processed_data.columns) if processed_data is not None else 0
                }
            )
            
        except Exception as e:
            response_time = time.time() - start_time
            logger.error(f"Data pipeline health check failed: {e}")
            
            return ComponentStatus(
                name="data_pipeline",
                status=SystemStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                error_message=str(e)
            )
    
    def run_health_check(self, model_path: Optional[str] = None) -> SystemHealth:
        """
        Run comprehensive system health check.
        
        Args:
            model_path: Path to model file
            
        Returns:
            SystemHealth: Complete system health status
        """
        start_time = time.time()
        
        logger.info("Running comprehensive system health check...")
        
        # Check all components
        components = [
            self.check_database_health(),
            self.check_model_health(model_path),
            self.check_trading_system_health(),
            self.check_system_resources(),
            self.check_data_pipeline_health()
        ]
        
        # Determine overall status
        statuses = [comp.status for comp in components]
        if SystemStatus.CRITICAL in statuses:
            overall_status = SystemStatus.CRITICAL
        elif SystemStatus.WARNING in statuses:
            overall_status = SystemStatus.WARNING
        else:
            overall_status = SystemStatus.HEALTHY
        
        # Calculate system metrics
        total_response_time = sum(comp.response_time for comp in components)
        avg_response_time = total_response_time / len(components)
        
        system_metrics = {
            'total_components': len(components),
            'healthy_components': len([c for c in components if c.status == SystemStatus.HEALTHY]),
            'warning_components': len([c for c in components if c.status == SystemStatus.WARNING]),
            'critical_components': len([c for c in components if c.status == SystemStatus.CRITICAL]),
            'average_response_time': avg_response_time,
            'total_response_time': total_response_time
        }
        
        # Create health object
        health = SystemHealth(
            overall_status=overall_status,
            timestamp=datetime.now(),
            components=components,
            system_metrics=system_metrics,
            uptime=time.time() - start_time
        )
        
        # Store in history
        self.health_history.append(health)
        
        # Keep only last 100 health checks
        if len(self.health_history) > 100:
            self.health_history = self.health_history[-100:]
        
        logger.info(f"Health check completed. Overall status: {overall_status.value}")
        return health
    
    def start_monitoring(self, interval: int = 60, model_path: Optional[str] = None):
        """
        Start continuous system monitoring.
        
        Args:
            interval: Monitoring interval in seconds
            model_path: Path to model file
        """
        if self.monitoring:
            logger.warning("Monitoring already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval, model_path),
            daemon=True
        )
        self.monitor_thread.start()
        
        logger.info(f"Started system monitoring with {interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
        
        logger.info("Stopped system monitoring")
    
    def _monitoring_loop(self, interval: int, model_path: Optional[str] = None):
        """Internal monitoring loop."""
        while self.monitoring:
            try:
                health = self.run_health_check(model_path)
                
                # Log status
                if health.overall_status == SystemStatus.CRITICAL:
                    logger.critical(f"System health critical: {health}")
                elif health.overall_status == SystemStatus.WARNING:
                    logger.warning(f"System health warning: {health}")
                else:
                    logger.info(f"System health: {health.overall_status.value}")
                
                # Save health data
                self._save_health_data(health)
                
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
            
            time.sleep(interval)
    
    def _save_health_data(self, health: SystemHealth):
        """Save health data to file."""
        try:
            health_data = {
                'timestamp': health.timestamp.isoformat(),
                'overall_status': health.overall_status.value,
                'uptime': health.uptime,
                'system_metrics': health.system_metrics,
                'components': [
                    {
                        'name': comp.name,
                        'status': comp.status.value,
                        'response_time': comp.response_time,
                        'error_message': comp.error_message,
                        'metrics': comp.metrics
                    }
                    for comp in health.components
                ]
            }
            
            # Save to file
            os.makedirs('logs', exist_ok=True)
            with open('logs/health_check.json', 'w') as f:
                json.dump(health_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving health data: {e}")
    
    def get_health_summary(self) -> Dict[str, Any]:
        """
        Get system health summary.
        
        Returns:
            Dict[str, Any]: Health summary
        """
        if not self.health_history:
            return {'error': 'No health data available'}
        
        latest_health = self.health_history[-1]
        
        return {
            'current_status': latest_health.overall_status.value,
            'timestamp': latest_health.timestamp.isoformat(),
            'uptime': latest_health.uptime,
            'components': {
                comp.name: {
                    'status': comp.status.value,
                    'response_time': comp.response_time,
                    'error': comp.error_message
                }
                for comp in latest_health.components
            },
            'system_metrics': latest_health.system_metrics,
            'history_length': len(self.health_history)
        }


class SystemIntegrator:
    """
    System integration coordinator.
    
    Coordinates integration between all system components
    and provides unified interface for system operations.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize system integrator.
        
        Args:
            config: System configuration
        """
        self.config = config
        self.monitor = SystemMonitor(config)
        self.integration_tests = []
        
        logger.info("System integrator initialized")
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """
        Run comprehensive integration tests.
        
        Returns:
            Dict[str, Any]: Integration test results
        """
        logger.info("Running system integration tests...")
        
        test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_success': True,
            'summary': {}
        }
        
        # Test 1: Data Pipeline Integration
        try:
            from .data.collectors.binance_collector import BinanceDataCollector
            from .data.processors.data_processor import DataProcessor
            from .data.storage.mongodb_storage import MongoDBStorage
            
            collector = BinanceDataCollector()
            processor = DataProcessor()
            storage = MongoDBStorage(self.monitor.database_config)
            
            test_results['tests'].append({
                'name': 'data_pipeline_integration',
                'status': 'passed',
                'message': 'Data pipeline components integrated successfully'
            })
            
        except Exception as e:
            test_results['tests'].append({
                'name': 'data_pipeline_integration',
                'status': 'failed',
                'message': f'Data pipeline integration failed: {e}'
            })
            test_results['overall_success'] = False
        
        # Test 2: Model Integration
        try:
            from .models.cnn_lstm.model import CNNLSTMModel
            from .models.transformer.model import CNNTransformerModel
            from .models.ensemble.model_ensemble import ModelEnsemble
            
            # Test model instantiation
            cnn_lstm = CNNLSTMModel(
                input_dim=self.config.get('model', {}).get('input_dim', 20),
                sequence_length=self.config.get('model', {}).get('sequence_length', 60),
                num_classes=4
            )
            
            test_results['tests'].append({
                'name': 'model_integration',
                'status': 'passed',
                'message': 'Model components integrated successfully'
            })
            
        except Exception as e:
            test_results['tests'].append({
                'name': 'model_integration',
                'status': 'failed',
                'message': f'Model integration failed: {e}'
            })
            test_results['overall_success'] = False
        
        # Test 3: Trading System Integration
        try:
            from .trading.backtest.backtest_engine import BacktestEngine
            from .trading.risk_management.risk_manager import RiskManager
            from .trading.signals.signal_generator import SignalGenerator
            
            backtest_engine = BacktestEngine(self.monitor.trading_config)
            risk_manager = RiskManager(self.monitor.trading_config)
            signal_generator = SignalGenerator(
                signal_config=self.config.get('signal', {}),
                trading_config=self.monitor.trading_config
            )
            
            test_results['tests'].append({
                'name': 'trading_system_integration',
                'status': 'passed',
                'message': 'Trading system components integrated successfully'
            })
            
        except Exception as e:
            test_results['tests'].append({
                'name': 'trading_system_integration',
                'status': 'failed',
                'message': f'Trading system integration failed: {e}'
            })
            test_results['overall_success'] = False
        
        # Test 4: Visualization Integration
        try:
            from .visualization.dashboard import TradingDashboard
            from .visualization.chart_utils import ChartGenerator, PerformanceAnalyzer
            
            dashboard = TradingDashboard(
                config=self.monitor.trading_config,
                title="Integration Test Dashboard"
            )
            
            test_results['tests'].append({
                'name': 'visualization_integration',
                'status': 'passed',
                'message': 'Visualization components integrated successfully'
            })
            
        except Exception as e:
            test_results['tests'].append({
                'name': 'visualization_integration',
                'status': 'failed',
                'message': f'Visualization integration failed: {e}'
            })
            test_results['overall_success'] = False
        
        # Test 5: End-to-End Integration
        try:
            # Test complete pipeline integration
            from .main import TradingSystem
            
            trading_system = TradingSystem(self.config)
            
            # Test system initialization
            if trading_system.initialize_system():
                test_results['tests'].append({
                    'name': 'end_to_end_integration',
                    'status': 'passed',
                    'message': 'End-to-end system integration successful'
                })
            else:
                test_results['tests'].append({
                    'name': 'end_to_end_integration',
                    'status': 'failed',
                    'message': 'End-to-end system integration failed'
                })
                test_results['overall_success'] = False
                
        except Exception as e:
            test_results['tests'].append({
                'name': 'end_to_end_integration',
                'status': 'failed',
                'message': f'End-to-end integration failed: {e}'
            })
            test_results['overall_success'] = False
        
        # Generate summary
        passed_tests = len([t for t in test_results['tests'] if t['status'] == 'passed'])
        total_tests = len(test_results['tests'])
        
        test_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        logger.info(f"Integration tests completed. Success rate: {test_results['summary']['success_rate']:.2%}")
        return test_results
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """
        Run system performance tests.
        
        Returns:
            Dict[str, Any]: Performance test results
        """
        logger.info("Running system performance tests...")
        
        performance_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_performance': 'good',
            'summary': {}
        }
        
        # Test 1: Data Processing Performance
        try:
            import time
            import pandas as pd
            import numpy as np
            
            from .data.processors.data_processor import DataProcessor
            
            processor = DataProcessor()
            
            # Create test data
            test_data = pd.DataFrame({
                'open': np.random.randn(1000) * 100 + 50000,
                'high': np.random.randn(1000) * 100 + 50100,
                'low': np.random.randn(1000) * 100 + 49900,
                'close': np.random.randn(1000) * 100 + 50000,
                'volume': np.random.randn(1000) * 1000 + 10000
            })
            
            # Measure processing time
            start_time = time.time()
            processed_data = processor.process_data(test_data)
            processing_time = time.time() - start_time
            
            performance_results['tests'].append({
                'name': 'data_processing_performance',
                'status': 'passed',
                'processing_time': processing_time,
                'data_points': len(test_data),
                'throughput': len(test_data) / processing_time,
                'message': f'Data processing: {processing_time:.3f}s for {len(test_data)} points'
            })
            
        except Exception as e:
            performance_results['tests'].append({
                'name': 'data_processing_performance',
                'status': 'failed',
                'message': f'Data processing performance test failed: {e}'
            })
        
        # Test 2: Model Inference Performance
        try:
            import torch
            
            from .models.cnn_lstm.model import CNNLSTMModel
            
            model = CNNLSTMModel(
                input_dim=20,
                sequence_length=60,
                num_classes=4
            )
            model.eval()
            
            # Create test input
            test_input = torch.randn(1, 20, 60)
            
            # Warm up
            with torch.no_grad():
                for _ in range(10):
                    _ = model(test_input)
            
            # Measure inference time
            start_time = time.time()
            with torch.no_grad():
                for _ in range(100):
                    _ = model(test_input)
            inference_time = (time.time() - start_time) / 100
            
            performance_results['tests'].append({
                'name': 'model_inference_performance',
                'status': 'passed',
                'inference_time': inference_time,
                'throughput': 1 / inference_time,
                'message': f'Model inference: {inference_time*1000:.2f}ms per prediction'
            })
            
        except Exception as e:
            performance_results['tests'].append({
                'name': 'model_inference_performance',
                'status': 'failed',
                'message': f'Model inference performance test failed: {e}'
            })
        
        # Test 3: Backtest Performance
        try:
            from .trading.backtest.backtest_engine import BacktestEngine
            
            backtest_engine = BacktestEngine(self.monitor.trading_config)
            
            # Simulate backtest with many trades
            start_time = time.time()
            
            for i in range(1000):
                signal = 'LONG' if i % 2 == 0 else 'SHORT'
                price = 50000 + i * 10
                timestamp = datetime.now() + timedelta(minutes=i)
                atr = 1000 + i * 5
                
                backtest_engine.execute_trade(signal, price, timestamp, atr)
            
            backtest_time = time.time() - start_time
            
            performance_results['tests'].append({
                'name': 'backtest_performance',
                'status': 'passed',
                'backtest_time': backtest_time,
                'trades_executed': 1000,
                'throughput': 1000 / backtest_time,
                'message': f'Backtest: {backtest_time:.3f}s for 1000 trades'
            })
            
        except Exception as e:
            performance_results['tests'].append({
                'name': 'backtest_performance',
                'status': 'failed',
                'message': f'Backtest performance test failed: {e}'
            })
        
        # Generate summary
        passed_tests = len([t for t in performance_results['tests'] if t['status'] == 'passed'])
        total_tests = len(performance_results['tests'])
        
        performance_results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': passed_tests / total_tests if total_tests > 0 else 0
        }
        
        logger.info(f"Performance tests completed. Success rate: {performance_results['summary']['success_rate']:.2%}")
        return performance_results
    
    def generate_integration_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive integration report.
        
        Returns:
            Dict[str, Any]: Integration report
        """
        logger.info("Generating integration report...")
        
        # Run health check
        health = self.monitor.run_health_check()
        
        # Run integration tests
        integration_tests = self.run_integration_tests()
        
        # Run performance tests
        performance_tests = self.run_performance_tests()
        
        # Generate report
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': {
                'overall_status': health.overall_status.value,
                'components': [
                    {
                        'name': comp.name,
                        'status': comp.status.value,
                        'response_time': comp.response_time
                    }
                    for comp in health.components
                ]
            },
            'integration_tests': integration_tests,
            'performance_tests': performance_tests,
            'recommendations': self._generate_recommendations(health, integration_tests, performance_tests)
        }
        
        # Save report
        os.makedirs('reports', exist_ok=True)
        with open('reports/integration_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info("Integration report generated and saved")
        return report
    
    def _generate_recommendations(self, health: SystemHealth, 
                                integration_tests: Dict[str, Any],
                                performance_tests: Dict[str, Any]) -> List[str]:
        """Generate system recommendations based on test results."""
        recommendations = []
        
        # Health-based recommendations
        if health.overall_status == SystemStatus.CRITICAL:
            recommendations.append("CRITICAL: System health is critical. Immediate attention required.")
        
        for comp in health.components:
            if comp.status == SystemStatus.CRITICAL:
                recommendations.append(f"CRITICAL: {comp.name} component is critical: {comp.error_message}")
            elif comp.status == SystemStatus.WARNING:
                recommendations.append(f"WARNING: {comp.name} component needs attention")
        
        # Integration test recommendations
        if not integration_tests['overall_success']:
            recommendations.append("Integration tests failed. Review component integration.")
        
        # Performance recommendations
        for test in performance_tests['tests']:
            if test['status'] == 'failed':
                recommendations.append(f"Performance issue: {test['name']} - {test['message']}")
        
        return recommendations
