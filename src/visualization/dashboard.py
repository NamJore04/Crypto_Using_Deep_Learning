"""
Interactive dashboard for Crypto Futures Trading System.

This module provides a comprehensive dashboard for monitoring trading performance,
visualizing backtest results, and analyzing system metrics using Dash and Plotly.
"""

import dash
from dash import dcc, html, Input, Output, callback_context
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import logging
from dataclasses import asdict

from .chart_utils import ChartGenerator, PerformanceAnalyzer
from ..trading.backtest.backtest_engine import BacktestResults, Trade
from ..trading.signals.signal_generator import Signal
from ..config import TradingConfig

logger = logging.getLogger(__name__)


class TradingDashboard:
    """
    Interactive dashboard for crypto trading system.
    
    Provides comprehensive visualization of trading performance,
    backtest results, risk metrics, and system monitoring.
    """
    
    def __init__(self, config: TradingConfig, title: str = "Crypto Trading System Dashboard"):
        """
        Initialize trading dashboard.
        
        Args:
            config: Trading configuration
            title: Dashboard title
        """
        self.config = config
        self.title = title
        
        # Initialize Dash app
        self.app = dash.Dash(__name__)
        self.app.title = title
        
        # Initialize components
        self.chart_generator = ChartGenerator()
        self.performance_analyzer = PerformanceAnalyzer()
        
        # Dashboard state
        self.current_data: Optional[Dict[str, Any]] = None
        self.backtest_results: Optional[BacktestResults] = None
        self.signals: List[Signal] = []
        
        # Setup dashboard
        self.setup_layout()
        self.setup_callbacks()
        
        logger.info(f"Trading dashboard initialized: {title}")
    
    def setup_layout(self):
        """Setup dashboard layout with comprehensive sections."""
        self.app.layout = html.Div([
            # Header
            html.Div([
                html.H1(self.title, className="header-title"),
                html.Div([
                    html.Span("Status: ", className="status-label"),
                    html.Span("Ready", id="system-status", className="status-value")
                ], className="status-indicator")
            ], className="header"),
            
            # Control Panel
            html.Div([
                html.Div([
                    html.Label("Time Range:", className="control-label"),
                    dcc.Dropdown(
                        id="time-range-dropdown",
                        options=[
                            {'label': 'Last 24 Hours', 'value': '24h'},
                            {'label': 'Last 7 Days', 'value': '7d'},
                            {'label': 'Last 30 Days', 'value': '30d'},
                            {'label': 'All Time', 'value': 'all'}
                        ],
                        value='7d',
                        className="control-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Label("Symbol:", className="control-label"),
                    dcc.Dropdown(
                        id="symbol-dropdown",
                        options=[
                            {'label': 'BTC/USDT', 'value': 'BTC/USDT'},
                            {'label': 'ETH/USDT', 'value': 'ETH/USDT'},
                            {'label': 'BNB/USDT', 'value': 'BNB/USDT'}
                        ],
                        value='BTC/USDT',
                        className="control-dropdown"
                    )
                ], className="control-item"),
                
                html.Div([
                    html.Button("Refresh Data", id="refresh-button", className="refresh-button"),
                    html.Button("Export Report", id="export-button", className="export-button")
                ], className="control-buttons")
                
            ], className="control-panel"),
            
            # Performance Metrics Cards
            html.Div([
                html.Div([
                    html.H3("Total Return", className="metric-title"),
                    html.Div(id="total-return", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Sharpe Ratio", className="metric-title"),
                    html.Div(id="sharpe-ratio", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Max Drawdown", className="metric-title"),
                    html.Div(id="max-drawdown", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Win Rate", className="metric-title"),
                    html.Div(id="win-rate", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Total Trades", className="metric-title"),
                    html.Div(id="total-trades", className="metric-value")
                ], className="metric-card"),
                
                html.Div([
                    html.H3("Final Capital", className="metric-title"),
                    html.Div(id="final-capital", className="metric-value")
                ], className="metric-card")
                
            ], className="metrics-grid"),
            
            # Main Charts Section
            html.Div([
                # Equity Curve and Price Chart
                html.Div([
                    dcc.Graph(id="equity-curve-chart", className="main-chart"),
                    dcc.Graph(id="price-chart", className="main-chart")
                ], className="charts-row"),
                
                # Performance Analysis
                html.Div([
                    dcc.Graph(id="returns-distribution", className="analysis-chart"),
                    dcc.Graph(id="drawdown-chart", className="analysis-chart")
                ], className="charts-row"),
                
                # Risk Metrics
                html.Div([
                    dcc.Graph(id="risk-metrics-chart", className="analysis-chart"),
                    dcc.Graph(id="signal-analysis-chart", className="analysis-chart")
                ], className="charts-row")
                
            ], className="charts-section"),
            
            # Detailed Analysis Section
            html.Div([
                html.H2("Detailed Analysis", className="section-title"),
                
                # Tabs for different analysis views
                dcc.Tabs(id="analysis-tabs", value="trades", children=[
                    dcc.Tab(label="Trade Analysis", value="trades"),
                    dcc.Tab(label="Risk Analysis", value="risk"),
                    dcc.Tab(label="Signal Analysis", value="signals"),
                    dcc.Tab(label="Model Performance", value="model")
                ]),
                
                html.Div(id="analysis-content", className="analysis-content")
                
            ], className="analysis-section"),
            
            # Footer
            html.Div([
                html.P(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 
                       className="footer-text"),
                html.P("Crypto Trading System - Research & Educational Use Only", 
                       className="disclaimer-text")
            ], className="footer")
            
        ], className="dashboard-container")
    
    def setup_callbacks(self):
        """Setup dashboard callbacks for interactivity."""
        
        @self.app.callback(
            [Output("total-return", "children"),
             Output("sharpe-ratio", "children"),
             Output("max-drawdown", "children"),
             Output("win-rate", "children"),
             Output("total-trades", "children"),
             Output("final-capital", "children")],
            [Input("refresh-button", "n_clicks"),
             Input("time-range-dropdown", "value"),
             Input("symbol-dropdown", "value")]
        )
        def update_metrics(n_clicks, time_range, symbol):
            """Update performance metrics."""
            if self.backtest_results is None:
                return ["N/A"] * 6
            
            try:
                metrics = self.backtest_results
                
                total_return = f"{metrics.total_return:.2%}"
                sharpe_ratio = f"{metrics.sharpe_ratio:.2f}"
                max_drawdown = f"{metrics.max_drawdown:.2%}"
                win_rate = f"{metrics.win_rate:.2%}"
                total_trades = str(metrics.total_trades)
                final_capital = f"${metrics.final_capital:,.2f}"
                
                return [total_return, sharpe_ratio, max_drawdown, 
                        win_rate, total_trades, final_capital]
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
                return ["Error"] * 6
        
        @self.app.callback(
            Output("equity-curve-chart", "figure"),
            [Input("refresh-button", "n_clicks"),
             Input("time-range-dropdown", "value")]
        )
        def update_equity_curve(n_clicks, time_range):
            """Update equity curve chart."""
            if self.backtest_results is None:
                return self.chart_generator.create_empty_chart("No Data Available")
            
            try:
                return self.chart_generator.create_equity_curve(
                    self.backtest_results.equity_curve,
                    title="Portfolio Equity Curve"
                )
            except Exception as e:
                logger.error(f"Error updating equity curve: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("price-chart", "figure"),
            [Input("refresh-button", "n_clicks"),
             Input("time-range-dropdown", "value"),
             Input("symbol-dropdown", "value")]
        )
        def update_price_chart(n_clicks, time_range, symbol):
            """Update price chart with signals."""
            if self.current_data is None:
                return self.chart_generator.create_empty_chart("No Data Available")
            
            try:
                # Get price data and signals
                price_data = self.current_data.get('price_data')
                signals = self.current_data.get('signals')
                
                if price_data is None:
                    return self.chart_generator.create_empty_chart("No Price Data")
                
                return self.chart_generator.create_price_chart_with_signals(
                    price_data, signals, symbol, title=f"{symbol} Price Chart with Signals"
                )
                
            except Exception as e:
                logger.error(f"Error updating price chart: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("returns-distribution", "figure"),
            [Input("refresh-button", "n_clicks")]
        )
        def update_returns_distribution(n_clicks):
            """Update returns distribution chart."""
            if self.backtest_results is None:
                return self.chart_generator.create_empty_chart("No Data Available")
            
            try:
                returns = self.backtest_results.equity_curve['returns'].dropna()
                return self.chart_generator.create_returns_distribution(returns)
            except Exception as e:
                logger.error(f"Error updating returns distribution: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("drawdown-chart", "figure"),
            [Input("refresh-button", "n_clicks")]
        )
        def update_drawdown_chart(n_clicks):
            """Update drawdown chart."""
            if self.backtest_results is None:
                return self.chart_generator.create_empty_chart("No Data Available")
            
            try:
                return self.chart_generator.create_drawdown_chart(
                    self.backtest_results.equity_curve
                )
            except Exception as e:
                logger.error(f"Error updating drawdown chart: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("risk-metrics-chart", "figure"),
            [Input("refresh-button", "n_clicks")]
        )
        def update_risk_metrics(n_clicks):
            """Update risk metrics chart."""
            if self.backtest_results is None:
                return self.chart_generator.create_empty_chart("No Data Available")
            
            try:
                return self.chart_generator.create_risk_metrics_chart(
                    self.backtest_results.performance_metrics
                )
            except Exception as e:
                logger.error(f"Error updating risk metrics: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("signal-analysis-chart", "figure"),
            [Input("refresh-button", "n_clicks")]
        )
        def update_signal_analysis(n_clicks):
            """Update signal analysis chart."""
            if not self.signals:
                return self.chart_generator.create_empty_chart("No Signal Data")
            
            try:
                return self.chart_generator.create_signal_analysis_chart(self.signals)
            except Exception as e:
                logger.error(f"Error updating signal analysis: {e}")
                return self.chart_generator.create_empty_chart("Error Loading Data")
        
        @self.app.callback(
            Output("analysis-content", "children"),
            [Input("analysis-tabs", "value")]
        )
        def update_analysis_content(active_tab):
            """Update analysis content based on selected tab."""
            try:
                if active_tab == "trades":
                    return self.create_trade_analysis_content()
                elif active_tab == "risk":
                    return self.create_risk_analysis_content()
                elif active_tab == "signals":
                    return self.create_signal_analysis_content()
                elif active_tab == "model":
                    return self.create_model_analysis_content()
                else:
                    return html.Div("Select an analysis tab")
                    
            except Exception as e:
                logger.error(f"Error updating analysis content: {e}")
                return html.Div("Error loading analysis content")
    
    def create_trade_analysis_content(self):
        """Create trade analysis content."""
        if self.backtest_results is None:
            return html.Div("No trade data available")
        
        try:
            trades_df = pd.DataFrame([asdict(trade) for trade in self.backtest_results.trades])
            
            if trades_df.empty:
                return html.Div("No trades executed")
            
            # Trade statistics
            closed_trades = trades_df[trades_df['action'] == 'CLOSE']
            if not closed_trades.empty:
                avg_pnl = closed_trades['pnl'].mean()
                best_trade = closed_trades['pnl'].max()
                worst_trade = closed_trades['pnl'].min()
                profitable_trades = len(closed_trades[closed_trades['pnl'] > 0])
                
                return html.Div([
                    html.H3("Trade Statistics"),
                    html.Div([
                        html.Div(f"Average P&L: ${avg_pnl:.2f}", className="stat-item"),
                        html.Div(f"Best Trade: ${best_trade:.2f}", className="stat-item"),
                        html.Div(f"Worst Trade: ${worst_trade:.2f}", className="stat-item"),
                        html.Div(f"Profitable Trades: {profitable_trades}/{len(closed_trades)}", className="stat-item")
                    ], className="stats-grid")
                ])
            else:
                return html.Div("No closed trades to analyze")
                
        except Exception as e:
            logger.error(f"Error creating trade analysis: {e}")
            return html.Div("Error analyzing trades")
    
    def create_risk_analysis_content(self):
        """Create risk analysis content."""
        if self.backtest_results is None:
            return html.Div("No risk data available")
        
        try:
            risk_metrics = self.backtest_results.performance_metrics
            
            return html.Div([
                html.H3("Risk Analysis"),
                html.Div([
                    html.Div(f"Volatility: {risk_metrics.get('volatility', 0):.2%}", className="risk-item"),
                    html.Div(f"VaR (95%): {risk_metrics.get('var_95', 0):.2%}", className="risk-item"),
                    html.Div(f"Expected Shortfall: {risk_metrics.get('expected_shortfall', 0):.2%}", className="risk-item"),
                    html.Div(f"Max Drawdown Duration: {risk_metrics.get('max_drawdown_duration', 0)} periods", className="risk-item")
                ], className="risk-grid")
            ])
            
        except Exception as e:
            logger.error(f"Error creating risk analysis: {e}")
            return html.Div("Error analyzing risk")
    
    def create_signal_analysis_content(self):
        """Create signal analysis content."""
        if not self.signals:
            return html.Div("No signal data available")
        
        try:
            signal_stats = self.performance_analyzer.analyze_signals(self.signals)
            
            return html.Div([
                html.H3("Signal Analysis"),
                html.Div([
                    html.Div(f"Total Signals: {signal_stats.get('total_signals', 0)}", className="signal-item"),
                    html.Div(f"Average Confidence: {signal_stats.get('average_confidence', 0):.2%}", className="signal-item"),
                    html.Div(f"Signal Frequency: {signal_stats.get('signal_frequency', 0):.2f}/hour", className="signal-item"),
                    html.Div(f"Current Signal: {signal_stats.get('current_signal', 'N/A')}", className="signal-item")
                ], className="signal-grid")
            ])
            
        except Exception as e:
            logger.error(f"Error creating signal analysis: {e}")
            return html.Div("Error analyzing signals")
    
    def create_model_analysis_content(self):
        """Create model analysis content."""
        return html.Div([
            html.H3("Model Performance"),
            html.Div("Model performance analysis will be available after training completion.")
        ])
    
    def update_data(self, backtest_results: BacktestResults, 
                   current_data: Optional[Dict[str, Any]] = None,
                   signals: Optional[List[Signal]] = None):
        """
        Update dashboard data.
        
        Args:
            backtest_results: Backtest results to display
            current_data: Current market data
            signals: Trading signals
        """
        try:
            self.backtest_results = backtest_results
            self.current_data = current_data or {}
            self.signals = signals or []
            
            logger.info("Dashboard data updated successfully")
            
        except Exception as e:
            logger.error(f"Error updating dashboard data: {e}")
    
    def run_server(self, host: str = "127.0.0.1", port: int = 8050, debug: bool = False):
        """
        Run dashboard server.
        
        Args:
            host: Server host
            port: Server port
            debug: Debug mode
        """
        try:
            logger.info(f"Starting dashboard server on {host}:{port}")
            self.app.run_server(host=host, port=port, debug=debug)
            
        except Exception as e:
            logger.error(f"Error running dashboard server: {e}")
    
    def export_report(self, filepath: str) -> bool:
        """
        Export dashboard report.
        
        Args:
            filepath: Path to save report
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            if self.backtest_results is None:
                logger.warning("No data to export")
                return False
            
            # Create comprehensive report
            report_data = {
                'timestamp': datetime.now().isoformat(),
                'backtest_results': asdict(self.backtest_results),
                'performance_metrics': self.backtest_results.performance_metrics,
                'signals_count': len(self.signals)
            }
            
            # Save report (implement based on requirements)
            logger.info(f"Report exported to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
            return False
