"""
Test runner for Crypto Futures Trading System.

This script runs all tests and generates comprehensive test reports.
"""

import pytest
import sys
import os
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def run_all_tests():
    """Run all tests and generate report."""
    print("🧪 Running Crypto Futures Trading System Tests...")
    print("=" * 60)
    
    # Test directories
    test_dirs = [
        "tests/test_data_pipeline.py",
        "tests/test_models.py", 
        "tests/test_trading_system.py",
        "tests/test_system_integration.py"
    ]
    
    # Run tests
    test_results = {}
    overall_success = True
    
    for test_dir in test_dirs:
        print(f"\n📋 Running {test_dir}...")
        
        try:
            # Run pytest
            result = pytest.main([
                test_dir,
                "-v",
                "--tb=short",
                "--junitxml=reports/test_results.xml"
            ])
            
            test_results[test_dir] = {
                'status': 'passed' if result == 0 else 'failed',
                'exit_code': result
            }
            
            if result != 0:
                overall_success = False
                
        except Exception as e:
            print(f"❌ Error running {test_dir}: {e}")
            test_results[test_dir] = {
                'status': 'error',
                'error': str(e)
            }
            overall_success = False
    
    # Generate test report
    report = {
        'timestamp': datetime.now().isoformat(),
        'overall_success': overall_success,
        'test_results': test_results,
        'summary': {
            'total_tests': len(test_dirs),
            'passed_tests': len([r for r in test_results.values() if r['status'] == 'passed']),
            'failed_tests': len([r for r in test_results.values() if r['status'] == 'failed']),
            'error_tests': len([r for r in test_results.values() if r['status'] == 'error'])
        }
    }
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    with open('reports/test_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Overall Success: {'PASSED' if overall_success else 'FAILED'}")
    print(f"📈 Total Tests: {report['summary']['total_tests']}")
    print(f"✅ Passed: {report['summary']['passed_tests']}")
    print(f"❌ Failed: {report['summary']['failed_tests']}")
    print(f"⚠️  Errors: {report['summary']['error_tests']}")
    
    if overall_success:
        print("\n🎉 All tests passed! System is ready for deployment.")
    else:
        print("\n⚠️  Some tests failed. Please review the results.")
    
    print(f"\n📄 Detailed report saved to: reports/test_report.json")
    
    return overall_success

def run_integration_tests():
    """Run integration tests specifically."""
    print("🔗 Running Integration Tests...")
    
    try:
        from src.system_integration import SystemIntegrator
        
        # Create test configuration
        config = {
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
        
        # Run integration tests
        integrator = SystemIntegrator(config)
        integration_results = integrator.run_integration_tests()
        
        # Run performance tests
        performance_results = integrator.run_performance_tests()
        
        # Generate integration report
        report = integrator.generate_integration_report()
        
        print("✅ Integration tests completed!")
        print(f"📊 Integration Success Rate: {integration_results['summary']['success_rate']:.2%}")
        print(f"📊 Performance Success Rate: {performance_results['summary']['success_rate']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Integration tests failed: {e}")
        return False

def run_system_health_check():
    """Run system health check."""
    print("🏥 Running System Health Check...")
    
    try:
        from src.system_integration import SystemMonitor
        
        # Create test configuration
        config = {
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
        
        # Run health check
        monitor = SystemMonitor(config)
        health = monitor.run_health_check()
        
        print(f"🏥 System Health: {health.overall_status.value.upper()}")
        print(f"📊 Components Checked: {len(health.components)}")
        print(f"⏱️  Response Time: {health.uptime:.2f}s")
        
        # Print component status
        for comp in health.components:
            status_emoji = "✅" if comp.status.value == "healthy" else "⚠️" if comp.status.value == "warning" else "❌"
            print(f"  {status_emoji} {comp.name}: {comp.status.value}")
        
        return health.overall_status.value == "healthy"
        
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False

def main():
    """Main test runner."""
    print("🚀 Crypto Futures Trading System - Test Suite")
    print("=" * 60)
    
    # Run all tests
    test_success = run_all_tests()
    
    # Run integration tests
    integration_success = run_integration_tests()
    
    # Run health check
    health_success = run_system_health_check()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL SUMMARY")
    print("=" * 60)
    print(f"🧪 Unit Tests: {'✅ PASSED' if test_success else '❌ FAILED'}")
    print(f"🔗 Integration Tests: {'✅ PASSED' if integration_success else '❌ FAILED'}")
    print(f"🏥 Health Check: {'✅ HEALTHY' if health_success else '❌ UNHEALTHY'}")
    
    overall_success = test_success and integration_success and health_success
    
    if overall_success:
        print("\n🎉 ALL SYSTEMS GO! System is ready for production.")
        return 0
    else:
        print("\n⚠️  SYSTEM ISSUES DETECTED! Please review and fix.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
