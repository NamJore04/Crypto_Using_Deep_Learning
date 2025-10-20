# Crypto Futures Trading System with Deep Learning

A comprehensive automated trading system using CNN-LSTM and CNN-Transformer models for market regime classification and trading signal generation.

## 🎯 Project Overview

**Project Name**: Crypto Futures Trading System with Deep Learning  
**Timeline**: 4 weeks (28 days)  
**Team Size**: 1-2 people  
**Purpose**: Research and educational use only - NOT financial advice

## 🏗️ System Architecture

### 3-Layer Architecture
```
Data Layer:     Binance API → MongoDB → Feature Engineering
AI Layer:       CNN-LSTM + CNN-Transformer + Ensemble
Trading Layer:  Signal Generation → Risk Management → Backtest
Visualization:  Interactive Dashboard → Performance Analytics
```

### Key Components
- **Data Pipeline**: Real-time data collection from Binance Futures
- **AI Models**: CNN-LSTM and CNN-Transformer for market regime classification
- **Trading System**: Backtest engine with comprehensive risk management
- **Visualization**: Interactive dashboard with real-time monitoring
- **System Integration**: Complete end-to-end pipeline with health monitoring

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd crypto_trading_system

# Create virtual environment
python -m venv crypto_trading_env
source crypto_trading_env/bin/activate  # Linux/Mac
# or
crypto_trading_env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create `.env` file:
```env
MONGO_HOST=localhost
MONGO_PORT=27017
MONGO_DB=crypto_trading
BINANCE_API_KEY=your_api_key
BINANCE_SECRET_KEY=your_secret_key
```

### 3. Run System

```bash
# Run complete pipeline
python run_system.py pipeline --symbol BTC/USDT --timeframe 30m --days 365

# Run health check
python run_system.py health

# Run integration tests
python run_system.py integration

# Start monitoring
python run_system.py monitor --interval 60

# Generate system report
python run_system.py report
```

## 📊 Features

### Data Pipeline
- **Real-time Data Collection**: Binance Futures API integration
- **Technical Indicators**: RSI, MACD, Bollinger Bands, ATR, OBV, CMF
- **Breakout Detection**: Quantitative breakout detection algorithm
- **Data Storage**: MongoDB with efficient data management

### AI Models
- **CNN-LSTM**: CNN feature extraction + LSTM sequence modeling
- **CNN-Transformer**: CNN features + Transformer attention mechanism
- **Ensemble Methods**: Weighted averaging, voting, adaptive ensemble
- **Market Regimes**: Sideway, Uptrend, Downtrend, Breakout classification

### Trading System
- **Backtest Engine**: Comprehensive backtesting with realistic assumptions
- **Risk Management**: ATR-based stop loss, position sizing, drawdown control
- **Signal Generation**: Model prediction to trading signal conversion
- **Performance Metrics**: Sharpe ratio, Max drawdown, Win rate, VaR

### Visualization
- **Interactive Dashboard**: Real-time monitoring with Dash framework
- **Performance Charts**: Equity curve, drawdown, returns distribution
- **Signal Analysis**: Trading signals visualization
- **Risk Metrics**: Comprehensive risk analysis

### System Integration
- **Health Monitoring**: System component health checks
- **Performance Testing**: Comprehensive performance analysis
- **Integration Testing**: End-to-end system validation
- **System Reporting**: Detailed system status reports

## 🧪 Testing

### Run All Tests
```bash
python run_tests.py
```

### Individual Test Suites
```bash
# Data pipeline tests
pytest tests/test_data_pipeline.py -v

# Model tests
pytest tests/test_models.py -v

# Trading system tests
pytest tests/test_trading_system.py -v

# Integration tests
pytest tests/test_system_integration.py -v
```

### Test Coverage
- **Unit Tests**: Individual component testing
- **Integration Tests**: Component interaction testing
- **Performance Tests**: System performance validation
- **Health Checks**: System health monitoring

## 📈 Performance Targets

### Model Performance
- **Accuracy**: > 60% overall accuracy
- **F1-Score**: > 0.6 macro F1-score
- **Breakout Recall**: > 50% recall for breakout detection

### Trading Performance
- **Sharpe Ratio**: > 1.0 in backtest
- **Max Drawdown**: < 20% maximum drawdown
- **Win Rate**: > 40% trade success rate

## 🔧 Configuration

### Trading Configuration
```python
trading_config = {
    'initial_capital': 10000.0,
    'commission': 0.001,
    'max_position_size': 0.1,
    'stop_loss_atr': 2.0,
    'take_profit_atr': 3.0,
    'max_drawdown': 0.2,
    'risk_per_trade': 0.02
}
```

### Model Configuration
```python
model_config = {
    'input_dim': 20,
    'sequence_length': 60,
    'hidden_dim': 64,
    'num_layers': 2,
    'num_classes': 4,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 100,
    'dropout_rate': 0.3
}
```

## 📁 Project Structure

```
crypto_trading_system/
├── src/                    # Source code
│   ├── config/            # Configuration
│   ├── data/              # Data pipeline
│   │   ├── collectors/    # Data collection
│   │   ├── processors/    # Data processing
│   │   └── storage/       # Data storage
│   ├── models/            # AI models
│   │   ├── cnn_lstm/      # CNN-LSTM model
│   │   ├── transformer/   # CNN-Transformer model
│   │   └── ensemble/      # Ensemble methods
│   ├── trading/           # Trading system
│   │   ├── backtest/      # Backtest engine
│   │   ├── risk_management/ # Risk management
│   │   └── signals/       # Signal generation
│   ├── visualization/     # Visualization
│   │   ├── dashboard.py   # Interactive dashboard
│   │   └── chart_utils.py # Chart utilities
│   ├── utils/             # Utilities
│   ├── main.py            # Main system
│   └── system_integration.py # System integration
├── tests/                 # Test suites
├── data/                  # Data storage
├── models/                # Trained models
├── logs/                  # System logs
├── reports/               # System reports
├── requirements.txt       # Dependencies
├── run_system.py         # System runner
├── run_tests.py          # Test runner
└── README.md             # This file
```

## 🛡️ Risk Management

### Financial Risks
- **Position Sizing**: Maximum 10% capital per trade
- **Stop Loss**: ATR-based stop loss (2x ATR)
- **Take Profit**: Risk-reward ratio 1:2
- **Maximum Drawdown**: Alert when > 20%
- **No Leverage**: No leverage used in backtest

### Technical Risks
- **Model Overfitting**: Early stopping and validation
- **Data Quality**: Comprehensive data validation
- **System Reliability**: Health monitoring and alerts
- **Performance**: Continuous performance monitoring

## 📚 Documentation

### Technical Documentation
- **API Documentation**: All APIs documented
- **Code Documentation**: Comprehensive code comments
- **Architecture Documentation**: System architecture details
- **Deployment Documentation**: Deployment procedures

### User Documentation
- **User Guide**: System usage instructions
- **Installation Guide**: Setup and installation
- **Troubleshooting**: Common issues and solutions
- **FAQ**: Frequently asked questions

## ⚠️ Important Disclaimers

### Legal Compliance
- **Research Only**: This system is for research and educational purposes only
- **Not Financial Advice**: Not intended as financial advice
- **Risk Warning**: Trading involves significant financial risk
- **Compliance**: Follow local regulations and laws

### Ethical Considerations
- **Transparency**: Open source and transparent methodology
- **Responsibility**: Use responsibly and ethically
- **Education**: Focus on learning and research
- **Community**: Contribute to the community

## 🚀 Future Development

### Planned Features
- **Live Trading Integration**: Real-time trading capabilities
- **Multi-Asset Support**: Support for multiple cryptocurrencies
- **Advanced Risk Management**: Portfolio-level risk controls
- **Machine Learning Pipeline**: Automated model retraining
- **Real-time Monitoring**: Live system monitoring and alerts

### Performance Improvements
- **Model Optimization**: Improved model architectures
- **Feature Engineering**: Advanced feature extraction
- **Risk Management**: Enhanced risk controls
- **Performance**: System performance optimization

## 🤝 Contributing

### Development Guidelines
- **Code Quality**: Follow PEP 8 standards
- **Testing**: Comprehensive test coverage
- **Documentation**: Clear documentation
- **Version Control**: Git workflow with feature branches

### Contribution Process
1. Fork the repository
2. Create feature branch
3. Make changes with tests
4. Submit pull request
5. Code review and merge

## 📞 Support

### Getting Help
- **Documentation**: Check documentation first
- **Issues**: Report issues on GitHub
- **Discussions**: Use GitHub discussions
- **Community**: Join the community

### Contact
- **GitHub**: [Repository URL]
- **Email**: [Contact Email]
- **Discord**: [Community Discord]

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Binance**: For providing market data API
- **PyTorch**: For deep learning framework
- **MongoDB**: For data storage
- **Community**: For contributions and feedback

---

**⚠️ DISCLAIMER**: This system is for research and educational purposes only. It is not financial advice. Trading involves significant financial risk. Use at your own risk and ensure compliance with local regulations.

**📊 STATUS**: System is in active development. Current version: 1.0.0

**🔄 LAST UPDATED**: [Current Date]

---

*Built with ❤️ for the crypto trading community*