"""
Main system runner for Crypto Futures Trading System.

This script provides the main entry point for running the complete
trading system with various modes and configurations.
"""

import sys
import os
import argparse
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from main import TradingSystem
from system_integration import SystemMonitor, SystemIntegrator
from utils.system_utils import create_system_report, save_system_report

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/trading_system.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)


def create_default_config():
    """Create default system configuration."""
    return {
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


def run_complete_pipeline(args):
    """Run complete trading system pipeline."""
    print("🚀 Starting Complete Trading System Pipeline...")
    print("=" * 60)
    
    # Create configuration
    config = create_default_config()
    
    # Override with command line arguments
    if args.symbol:
        config['symbol'] = args.symbol
    if args.timeframe:
        config['timeframe'] = args.timeframe
    if args.days:
        config['days'] = args.days
    if args.model_type:
        config['model_type'] = args.model_type
    
    # Initialize trading system
    trading_system = TradingSystem(config, args.model_path)
    
    try:
        # Run complete pipeline
        results = trading_system.run_complete_pipeline(
            symbol=args.symbol or "BTC/USDT",
            timeframe=args.timeframe or "30m",
            days=args.days or 365,
            model_type=args.model_type or "cnn_lstm"
        )
        
        if results['success']:
            print("✅ Pipeline completed successfully!")
            print(f"📊 Performance Metrics:")
            print(f"  • Total Return: {results['performance_metrics']['total_return']:.2%}")
            print(f"  • Sharpe Ratio: {results['performance_metrics']['sharpe_ratio']:.2f}")
            print(f"  • Max Drawdown: {results['performance_metrics']['max_drawdown']:.2%}")
            print(f"  • Win Rate: {results['performance_metrics']['win_rate']:.2%}")
            print(f"  • Total Trades: {results['performance_metrics']['total_trades']}")
            
            # Start dashboard if requested
            if args.dashboard:
                print("\n🌐 Starting Dashboard...")
                trading_system.start_dashboard(host=args.host, port=args.port)
        else:
            print(f"❌ Pipeline failed: {results['error']}")
            return False
            
    except Exception as e:
        print(f"❌ Error running pipeline: {e}")
        logger.error(f"Pipeline error: {e}")
        return False
    
    return True


def run_health_check(args):
    """Run system health check."""
    print("🏥 Running System Health Check...")
    print("=" * 60)
    
    config = create_default_config()
    monitor = SystemMonitor(config)
    
    try:
        # Run health check
        health = monitor.run_health_check()
        
        print(f"🏥 System Health: {health.overall_status.value.upper()}")
        print(f"📊 Components Checked: {len(health.components)}")
        print(f"⏱️  Response Time: {health.uptime:.2f}s")
        
        # Print component status
        for comp in health.components:
            status_emoji = "✅" if comp.status.value == "healthy" else "⚠️" if comp.status.value == "warning" else "❌"
            print(f"  {status_emoji} {comp.name}: {comp.status.value}")
            if comp.error_message:
                print(f"    Error: {comp.error_message}")
        
        # Get health summary
        summary = monitor.get_health_summary()
        print(f"\n📈 Health Summary:")
        print(f"  • Current Status: {summary['current_status']}")
        print(f"  • History Length: {summary['history_length']}")
        
        return health.overall_status.value == "healthy"
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        logger.error(f"Health check error: {e}")
        return False


def run_integration_tests(args):
    """Run system integration tests."""
    print("🔗 Running System Integration Tests...")
    print("=" * 60)
    
    config = create_default_config()
    integrator = SystemIntegrator(config)
    
    try:
        # Run integration tests
        integration_results = integrator.run_integration_tests()
        performance_results = integrator.run_performance_tests()
        
        print(f"📊 Integration Tests:")
        print(f"  • Total Tests: {integration_results['summary']['total_tests']}")
        print(f"  • Passed: {integration_results['summary']['passed_tests']}")
        print(f"  • Failed: {integration_results['summary']['failed_tests']}")
        print(f"  • Success Rate: {integration_results['summary']['success_rate']:.2%}")
        
        print(f"\n📊 Performance Tests:")
        print(f"  • Total Tests: {performance_results['summary']['total_tests']}")
        print(f"  • Passed: {performance_results['summary']['passed_tests']}")
        print(f"  • Failed: {performance_results['summary']['failed_tests']}")
        print(f"  • Success Rate: {performance_results['summary']['success_rate']:.2%}")
        
        # Generate integration report
        report = integrator.generate_integration_report()
        print(f"\n📄 Integration report saved to: reports/integration_report.json")
        
        return integration_results['overall_success']
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        logger.error(f"Integration test error: {e}")
        return False


def run_monitoring(args):
    """Run system monitoring."""
    print("📊 Starting System Monitoring...")
    print("=" * 60)
    
    config = create_default_config()
    monitor = SystemMonitor(config)
    
    try:
        # Start monitoring
        monitor.start_monitoring(
            interval=args.interval or 60,
            model_path=args.model_path
        )
        
        print(f"📊 Monitoring started with {args.interval or 60}s interval")
        print("Press Ctrl+C to stop monitoring...")
        
        # Keep running until interrupted
        import time
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping monitoring...")
            monitor.stop_monitoring()
            print("✅ Monitoring stopped")
        
        return True
        
    except Exception as e:
        print(f"❌ Monitoring failed: {e}")
        logger.error(f"Monitoring error: {e}")
        return False


def run_system_report(args):
    """Generate system report."""
    print("📄 Generating System Report...")
    print("=" * 60)
    
    config = create_default_config()
    
    try:
        # Create system report
        report = create_system_report(config)
        
        # Save report
        save_system_report(report, f'reports/system_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json')
        
        print("✅ System report generated successfully!")
        print(f"📄 Report saved to: reports/system_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Print summary
        if 'health_check' in report:
            health = report['health_check']
            print(f"\n🏥 System Health: {health.get('status', 'unknown').upper()}")
            
            if 'issues' in health:
                print(f"⚠️  Issues: {len(health['issues'])}")
                for issue in health['issues']:
                    print(f"  • {issue}")
        
        if 'recommendations' in report:
            print(f"\n💡 Recommendations: {len(report['recommendations'])}")
            for rec in report['recommendations']:
                print(f"  • {rec}")
        
        return True
        
    except Exception as e:
        print(f"❌ Report generation failed: {e}")
        logger.error(f"Report generation error: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Crypto Futures Trading System')
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Complete pipeline command
    pipeline_parser = subparsers.add_parser('pipeline', help='Run complete trading pipeline')
    pipeline_parser.add_argument('--symbol', default='BTC/USDT', help='Trading symbol')
    pipeline_parser.add_argument('--timeframe', default='30m', help='Data timeframe')
    pipeline_parser.add_argument('--days', type=int, default=365, help='Number of days to collect')
    pipeline_parser.add_argument('--model-type', default='cnn_lstm', help='Model type')
    pipeline_parser.add_argument('--model-path', help='Path to pre-trained model')
    pipeline_parser.add_argument('--dashboard', action='store_true', help='Start dashboard after pipeline')
    pipeline_parser.add_argument('--host', default='127.0.0.1', help='Dashboard host')
    pipeline_parser.add_argument('--port', type=int, default=8050, help='Dashboard port')
    
    # Health check command
    health_parser = subparsers.add_parser('health', help='Run system health check')
    
    # Integration tests command
    integration_parser = subparsers.add_parser('integration', help='Run integration tests')
    
    # Monitoring command
    monitoring_parser = subparsers.add_parser('monitor', help='Run system monitoring')
    monitoring_parser.add_argument('--interval', type=int, default=60, help='Monitoring interval in seconds')
    monitoring_parser.add_argument('--model-path', help='Path to model file')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate system report')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create necessary directories
    os.makedirs('logs', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    
    # Run command
    success = False
    
    if args.command == 'pipeline':
        success = run_complete_pipeline(args)
    elif args.command == 'health':
        success = run_health_check(args)
    elif args.command == 'integration':
        success = run_integration_tests(args)
    elif args.command == 'monitor':
        success = run_monitoring(args)
    elif args.command == 'report':
        success = run_system_report(args)
    else:
        print(f"❌ Unknown command: {args.command}")
        return 1
    
    return 0 if success else 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
