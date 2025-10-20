"""
System utility functions for Crypto Futures Trading System.

This module provides utility functions for system monitoring,
performance analysis, and common operations.
"""

import os
import json
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class SystemMetrics:
    """System metrics collection and analysis."""
    
    def __init__(self):
        """Initialize system metrics collector."""
        self.metrics_history = []
        self.start_time = time.time()
    
    def collect_system_metrics(self) -> Dict[str, Any]:
        """
        Collect current system metrics.
        
        Returns:
            Dict[str, Any]: System metrics
        """
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_available = memory.available / (1024**3)  # GB
            memory_total = memory.total / (1024**3)  # GB
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = disk.percent
            disk_free = disk.free / (1024**3)  # GB
            disk_total = disk.total / (1024**3)  # GB
            
            # Network metrics
            network = psutil.net_io_counters()
            network_bytes_sent = network.bytes_sent
            network_bytes_recv = network.bytes_recv
            
            # Process metrics
            process = psutil.Process()
            process_memory = process.memory_info().rss / (1024**2)  # MB
            process_cpu = process.cpu_percent()
            
            # System uptime
            uptime = time.time() - self.start_time
            
            metrics = {
                'timestamp': datetime.now().isoformat(),
                'uptime_seconds': uptime,
                'cpu': {
                    'percent': cpu_percent,
                    'count': cpu_count
                },
                'memory': {
                    'percent': memory_percent,
                    'available_gb': memory_available,
                    'total_gb': memory_total
                },
                'disk': {
                    'percent': disk_percent,
                    'free_gb': disk_free,
                    'total_gb': disk_total
                },
                'network': {
                    'bytes_sent': network_bytes_sent,
                    'bytes_recv': network_bytes_recv
                },
                'process': {
                    'memory_mb': process_memory,
                    'cpu_percent': process_cpu
                }
            }
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Keep only last 1000 records
            if len(self.metrics_history) > 1000:
                self.metrics_history = self.metrics_history[-1000:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary statistics.
        
        Returns:
            Dict[str, Any]: Metrics summary
        """
        if not self.metrics_history:
            return {'error': 'No metrics data available'}
        
        # Calculate averages
        cpu_values = [m['cpu']['percent'] for m in self.metrics_history]
        memory_values = [m['memory']['percent'] for m in self.metrics_history]
        disk_values = [m['disk']['percent'] for m in self.metrics_history]
        
        summary = {
            'total_records': len(self.metrics_history),
            'time_range': {
                'start': self.metrics_history[0]['timestamp'],
                'end': self.metrics_history[-1]['timestamp']
            },
            'averages': {
                'cpu_percent': np.mean(cpu_values),
                'memory_percent': np.mean(memory_values),
                'disk_percent': np.mean(disk_values)
            },
            'maxima': {
                'cpu_percent': np.max(cpu_values),
                'memory_percent': np.max(memory_values),
                'disk_percent': np.max(disk_values)
            },
            'current': self.metrics_history[-1] if self.metrics_history else None
        }
        
        return summary
    
    def save_metrics(self, filepath: str = 'logs/system_metrics.json'):
        """
        Save metrics to file.
        
        Args:
            filepath: Path to save metrics
        """
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                json.dump(self.metrics_history, f, indent=2)
            
            logger.info(f"Metrics saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving metrics: {e}")


class PerformanceAnalyzer:
    """Performance analysis utilities."""
    
    def __init__(self):
        """Initialize performance analyzer."""
        self.performance_data = []
    
    def analyze_model_performance(self, model, test_data, test_labels) -> Dict[str, Any]:
        """
        Analyze model performance.
        
        Args:
            model: Trained model
            test_data: Test data
            test_labels: Test labels
            
        Returns:
            Dict[str, Any]: Performance analysis
        """
        try:
            import torch
            from sklearn.metrics import accuracy_score, f1_score, classification_report
            
            # Get predictions
            model.eval()
            with torch.no_grad():
                predictions = model(test_data)
                predicted_labels = torch.argmax(predictions, dim=1).cpu().numpy()
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predicted_labels)
            f1_macro = f1_score(test_labels, predicted_labels, average='macro')
            f1_weighted = f1_score(test_labels, predicted_labels, average='weighted')
            
            # Classification report
            report = classification_report(test_labels, predicted_labels, output_dict=True)
            
            analysis = {
                'accuracy': accuracy,
                'f1_macro': f1_macro,
                'f1_weighted': f1_weighted,
                'classification_report': report,
                'predictions': predicted_labels.tolist(),
                'true_labels': test_labels.tolist()
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing model performance: {e}")
            return {'error': str(e)}
    
    def analyze_trading_performance(self, backtest_results) -> Dict[str, Any]:
        """
        Analyze trading performance.
        
        Args:
            backtest_results: Backtest results
            
        Returns:
            Dict[str, Any]: Trading performance analysis
        """
        try:
            if not backtest_results:
                return {'error': 'No backtest results provided'}
            
            # Basic metrics
            total_return = backtest_results.total_return
            sharpe_ratio = backtest_results.sharpe_ratio
            max_drawdown = backtest_results.max_drawdown
            win_rate = backtest_results.win_rate
            total_trades = backtest_results.total_trades
            
            # Performance classification
            if total_return > 0.2:
                performance_grade = 'Excellent'
            elif total_return > 0.1:
                performance_grade = 'Good'
            elif total_return > 0:
                performance_grade = 'Positive'
            else:
                performance_grade = 'Negative'
            
            # Risk assessment
            if max_drawdown < 0.1:
                risk_level = 'Low'
            elif max_drawdown < 0.2:
                risk_level = 'Medium'
            else:
                risk_level = 'High'
            
            analysis = {
                'performance_grade': performance_grade,
                'risk_level': risk_level,
                'total_return': total_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'risk_adjusted_return': total_return / max_drawdown if max_drawdown > 0 else 0,
                'recommendations': self._generate_trading_recommendations(
                    total_return, sharpe_ratio, max_drawdown, win_rate
                )
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing trading performance: {e}")
            return {'error': str(e)}
    
    def _generate_trading_recommendations(self, total_return, sharpe_ratio, max_drawdown, win_rate) -> List[str]:
        """Generate trading recommendations based on performance."""
        recommendations = []
        
        if total_return < 0:
            recommendations.append("Consider reviewing trading strategy - negative returns")
        
        if sharpe_ratio < 1.0:
            recommendations.append("Low Sharpe ratio - consider improving risk-adjusted returns")
        
        if max_drawdown > 0.2:
            recommendations.append("High maximum drawdown - consider reducing position sizes")
        
        if win_rate < 0.4:
            recommendations.append("Low win rate - consider improving signal quality")
        
        if total_return > 0.2 and sharpe_ratio > 1.5:
            recommendations.append("Excellent performance - consider scaling up")
        
        return recommendations


class SystemHealthChecker:
    """System health checking utilities."""
    
    def __init__(self, thresholds: Optional[Dict[str, float]] = None):
        """
        Initialize health checker.
        
        Args:
            thresholds: Health check thresholds
        """
        self.thresholds = thresholds or {
            'cpu_usage': 80.0,
            'memory_usage': 85.0,
            'disk_usage': 90.0,
            'response_time': 5.0
        }
    
    def check_system_health(self) -> Dict[str, Any]:
        """
        Check overall system health.
        
        Returns:
            Dict[str, Any]: System health status
        """
        try:
            # Collect current metrics
            metrics = SystemMetrics().collect_system_metrics()
            
            if not metrics:
                return {'status': 'error', 'message': 'Failed to collect metrics'}
            
            # Check thresholds
            health_issues = []
            
            if metrics['cpu']['percent'] > self.thresholds['cpu_usage']:
                health_issues.append(f"High CPU usage: {metrics['cpu']['percent']:.1f}%")
            
            if metrics['memory']['percent'] > self.thresholds['memory_usage']:
                health_issues.append(f"High memory usage: {metrics['memory']['percent']:.1f}%")
            
            if metrics['disk']['percent'] > self.thresholds['disk_usage']:
                health_issues.append(f"High disk usage: {metrics['disk']['percent']:.1f}%")
            
            # Determine overall status
            if not health_issues:
                status = 'healthy'
            elif len(health_issues) == 1:
                status = 'warning'
            else:
                status = 'critical'
            
            return {
                'status': status,
                'timestamp': metrics['timestamp'],
                'issues': health_issues,
                'metrics': metrics,
                'recommendations': self._generate_health_recommendations(health_issues)
            }
            
        except Exception as e:
            logger.error(f"Error checking system health: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _generate_health_recommendations(self, issues: List[str]) -> List[str]:
        """Generate health recommendations based on issues."""
        recommendations = []
        
        for issue in issues:
            if 'CPU' in issue:
                recommendations.append("Consider closing unnecessary applications or upgrading CPU")
            elif 'memory' in issue:
                recommendations.append("Consider closing applications or adding more RAM")
            elif 'disk' in issue:
                recommendations.append("Consider cleaning up disk space or adding storage")
        
        return recommendations


class DataQualityChecker:
    """Data quality checking utilities."""
    
    def __init__(self):
        """Initialize data quality checker."""
        self.quality_rules = {
            'missing_threshold': 0.05,  # 5% missing data threshold
            'outlier_threshold': 3.0,   # 3 standard deviations
            'correlation_threshold': 0.95  # High correlation threshold
        }
    
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data quality.
        
        Args:
            data: DataFrame to check
            
        Returns:
            Dict[str, Any]: Data quality report
        """
        try:
            issues = []
            quality_score = 100.0
            
            # Check for missing values
            missing_percent = data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            if missing_percent > self.quality_rules['missing_threshold'] * 100:
                issues.append(f"High missing data: {missing_percent:.2f}%")
                quality_score -= 20
            
            # Check for duplicates
            duplicate_count = data.duplicated().sum()
            if duplicate_count > 0:
                issues.append(f"Duplicate rows: {duplicate_count}")
                quality_score -= 10
            
            # Check for outliers
            numeric_cols = data.select_dtypes(include=[np.number]).columns
            outlier_count = 0
            for col in numeric_cols:
                if data[col].std() > 0:  # Avoid division by zero
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outlier_count += (z_scores > self.quality_rules['outlier_threshold']).sum()
            
            if outlier_count > len(data) * 0.1:  # More than 10% outliers
                issues.append(f"High outlier count: {outlier_count}")
                quality_score -= 15
            
            # Check for high correlations
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_corr_pairs = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        if abs(corr_matrix.iloc[i, j]) > self.quality_rules['correlation_threshold']:
                            high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
                
                if high_corr_pairs:
                    issues.append(f"High correlations: {len(high_corr_pairs)} pairs")
                    quality_score -= 5
            
            # Determine quality grade
            if quality_score >= 90:
                grade = 'Excellent'
            elif quality_score >= 80:
                grade = 'Good'
            elif quality_score >= 70:
                grade = 'Fair'
            else:
                grade = 'Poor'
            
            return {
                'quality_score': quality_score,
                'grade': grade,
                'issues': issues,
                'missing_percent': missing_percent,
                'duplicate_count': duplicate_count,
                'outlier_count': outlier_count,
                'high_corr_pairs': len(high_corr_pairs) if 'high_corr_pairs' in locals() else 0,
                'recommendations': self._generate_data_recommendations(issues)
            }
            
        except Exception as e:
            logger.error(f"Error checking data quality: {e}")
            return {'error': str(e)}
    
    def _generate_data_recommendations(self, issues: List[str]) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        for issue in issues:
            if 'missing' in issue:
                recommendations.append("Consider imputing missing values or removing incomplete records")
            elif 'duplicate' in issue:
                recommendations.append("Remove duplicate rows")
            elif 'outlier' in issue:
                recommendations.append("Investigate and handle outliers appropriately")
            elif 'correlation' in issue:
                recommendations.append("Consider removing highly correlated features")
        
        return recommendations


class LogAnalyzer:
    """Log analysis utilities."""
    
    def __init__(self, log_file: str = 'logs/trading_system.log'):
        """
        Initialize log analyzer.
        
        Args:
            log_file: Path to log file
        """
        self.log_file = log_file
    
    def analyze_logs(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze system logs.
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dict[str, Any]: Log analysis results
        """
        try:
            if not os.path.exists(self.log_file):
                return {'error': 'Log file not found'}
            
            # Read log file
            with open(self.log_file, 'r') as f:
                lines = f.readlines()
            
            # Filter recent logs
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_lines = []
            
            for line in lines:
                try:
                    # Extract timestamp from log line
                    timestamp_str = line.split(' - ')[0]
                    timestamp = datetime.fromisoformat(timestamp_str.replace(' ', 'T'))
                    
                    if timestamp >= cutoff_time:
                        recent_lines.append(line)
                except:
                    continue
            
            # Analyze log levels
            error_count = sum(1 for line in recent_lines if 'ERROR' in line)
            warning_count = sum(1 for line in recent_lines if 'WARNING' in line)
            info_count = sum(1 for line in recent_lines if 'INFO' in line)
            
            # Analyze error patterns
            error_patterns = {}
            for line in recent_lines:
                if 'ERROR' in line:
                    error_msg = line.split('ERROR - ')[-1].strip()
                    error_patterns[error_msg] = error_patterns.get(error_msg, 0) + 1
            
            # Calculate error rate
            total_logs = len(recent_lines)
            error_rate = error_count / total_logs if total_logs > 0 else 0
            
            return {
                'total_logs': total_logs,
                'error_count': error_count,
                'warning_count': warning_count,
                'info_count': info_count,
                'error_rate': error_rate,
                'error_patterns': error_patterns,
                'time_range': f"Last {hours} hours",
                'recommendations': self._generate_log_recommendations(error_rate, error_patterns)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing logs: {e}")
            return {'error': str(e)}
    
    def _generate_log_recommendations(self, error_rate: float, error_patterns: Dict[str, int]) -> List[str]:
        """Generate recommendations based on log analysis."""
        recommendations = []
        
        if error_rate > 0.1:  # More than 10% errors
            recommendations.append("High error rate - investigate system stability")
        
        if error_patterns:
            most_common_error = max(error_patterns, key=error_patterns.get)
            recommendations.append(f"Most common error: {most_common_error}")
        
        return recommendations


def create_system_report(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create comprehensive system report.
    
    Args:
        config: System configuration
        
    Returns:
        Dict[str, Any]: System report
    """
    try:
        report = {
            'timestamp': datetime.now().isoformat(),
            'system_info': {
                'python_version': sys.version,
                'platform': os.name,
                'working_directory': os.getcwd()
            },
            'health_check': SystemHealthChecker().check_system_health(),
            'metrics_summary': SystemMetrics().get_metrics_summary(),
            'log_analysis': LogAnalyzer().analyze_logs(),
            'recommendations': []
        }
        
        # Add recommendations
        if report['health_check']['status'] != 'healthy':
            report['recommendations'].extend(report['health_check']['recommendations'])
        
        if report['log_analysis'].get('error_rate', 0) > 0.05:
            report['recommendations'].extend(report['log_analysis']['recommendations'])
        
        return report
        
    except Exception as e:
        logger.error(f"Error creating system report: {e}")
        return {'error': str(e)}


def save_system_report(report: Dict[str, Any], filepath: str = 'reports/system_report.json'):
    """
    Save system report to file.
    
    Args:
        report: System report
        filepath: Path to save report
    """
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"System report saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving system report: {e}")
