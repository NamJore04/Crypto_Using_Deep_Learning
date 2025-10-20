"""
Chart utilities for Crypto Futures Trading System.

This module provides comprehensive chart generation utilities for
visualizing trading performance, risk metrics, and system analysis.
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple, Union
import logging

from ..trading.signals.signal_generator import Signal, SignalType, MarketRegime

logger = logging.getLogger(__name__)


class ChartGenerator:
    """
    Chart generation utilities for trading system visualization.
    
    Provides comprehensive chart generation for performance analysis,
    risk visualization, and trading signal analysis.
    """
    
    def __init__(self):
        """Initialize chart generator."""
        self.color_scheme = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e',
            'success': '#2ca02c',
            'danger': '#d62728',
            'warning': '#ff7f0e',
            'info': '#17a2b8',
            'light': '#f8f9fa',
            'dark': '#343a40'
        }
        
        self.chart_theme = {
            'layout': {
                'font': {'family': 'Arial, sans-serif', 'size': 12},
                'plot_bgcolor': 'white',
                'paper_bgcolor': 'white',
                'margin': {'l': 50, 'r': 50, 't': 50, 'b': 50}
            }
        }
        
        logger.info("Chart generator initialized")
    
    def create_equity_curve(self, equity_data: pd.DataFrame, 
                          title: str = "Portfolio Equity Curve") -> go.Figure:
        """
        Create equity curve chart.
        
        Args:
            equity_data: DataFrame with equity curve data
            title: Chart title
            
        Returns:
            go.Figure: Equity curve chart
        """
        try:
            fig = go.Figure()
            
            # Main equity curve
            fig.add_trace(go.Scatter(
                x=equity_data.index,
                y=equity_data['portfolio_value'],
                mode='lines',
                name='Portfolio Value',
                line=dict(color=self.color_scheme['primary'], width=2),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                             'Time: %{x}<br>' +
                             'Value: $%{y:,.2f}<extra></extra>'
            ))
            
            # Add initial capital line
            if 'portfolio_value' in equity_data.columns and not equity_data.empty:
                initial_capital = equity_data['portfolio_value'].iloc[0]
                fig.add_hline(
                    y=initial_capital,
                    line_dash="dash",
                    line_color=self.color_scheme['secondary'],
                    annotation_text="Initial Capital"
                )
            
            # Add drawdown shading
            if 'drawdown' in equity_data.columns:
                fig.add_trace(go.Scatter(
                    x=equity_data.index,
                    y=equity_data['drawdown'],
                    fill='tonexty',
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.color_scheme['danger'], width=0),
                    fillcolor='rgba(214, 39, 40, 0.3)',
                    hovertemplate='<b>Drawdown</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Drawdown: %{y:.2%}<extra></extra>'
                ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Portfolio Value ($)",
                hovermode='x unified',
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating equity curve: {e}")
            return self.create_empty_chart("Error creating equity curve")
    
    def create_price_chart_with_signals(self, price_data: pd.DataFrame, 
                                      signals: Optional[List[Signal]] = None,
                                      symbol: str = "BTC/USDT",
                                      title: str = "Price Chart with Signals") -> go.Figure:
        """
        Create price chart with trading signals.
        
        Args:
            price_data: DataFrame with OHLCV data
            signals: List of trading signals
            symbol: Trading symbol
            title: Chart title
            
        Returns:
            go.Figure: Price chart with signals
        """
        try:
            fig = go.Figure()
            
            # Candlestick chart
            fig.add_trace(go.Candlestick(
                x=price_data.index,
                open=price_data['open'],
                high=price_data['high'],
                low=price_data['low'],
                close=price_data['close'],
                name=symbol,
                increasing_line_color=self.color_scheme['success'],
                decreasing_line_color=self.color_scheme['danger']
            ))
            
            # Add trading signals if available
            if signals:
                signal_df = pd.DataFrame([{
                    'timestamp': s.timestamp,
                    'signal_type': s.signal_type.value,
                    'confidence': s.confidence,
                    'price': s.price
                } for s in signals])
                
                # Buy signals (LONG)
                buy_signals = signal_df[signal_df['signal_type'] == 'LONG']
                if not buy_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=buy_signals['timestamp'],
                        y=buy_signals['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-up',
                            size=12,
                            color=self.color_scheme['success'],
                            line=dict(width=2, color='white')
                        ),
                        name='Buy Signal',
                        hovertemplate='<b>Buy Signal</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Price: $%{y:,.2f}<br>' +
                                     'Confidence: %{customdata:.2%}<extra></extra>',
                        customdata=buy_signals['confidence']
                    ))
                
                # Sell signals (SHORT)
                sell_signals = signal_df[signal_df['signal_type'] == 'SHORT']
                if not sell_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=sell_signals['timestamp'],
                        y=sell_signals['price'],
                        mode='markers',
                        marker=dict(
                            symbol='triangle-down',
                            size=12,
                            color=self.color_scheme['danger'],
                            line=dict(width=2, color='white')
                        ),
                        name='Sell Signal',
                        hovertemplate='<b>Sell Signal</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Price: $%{y:,.2f}<br>' +
                                     'Confidence: %{customdata:.2%}<extra></extra>',
                        customdata=sell_signals['confidence']
                    ))
                
                # Close signals
                close_signals = signal_df[signal_df['signal_type'] == 'CLOSE']
                if not close_signals.empty:
                    fig.add_trace(go.Scatter(
                        x=close_signals['timestamp'],
                        y=close_signals['price'],
                        mode='markers',
                        marker=dict(
                            symbol='x',
                            size=10,
                            color=self.color_scheme['warning'],
                            line=dict(width=2, color='white')
                        ),
                        name='Close Signal',
                        hovertemplate='<b>Close Signal</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Price: $%{y:,.2f}<extra></extra>'
                    ))
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Price ($)",
                hovermode='x unified',
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating price chart: {e}")
            return self.create_empty_chart("Error creating price chart")
    
    def create_returns_distribution(self, returns: pd.Series, 
                                  title: str = "Returns Distribution") -> go.Figure:
        """
        Create returns distribution histogram.
        
        Args:
            returns: Series of returns
            title: Chart title
            
        Returns:
            go.Figure: Returns distribution chart
        """
        try:
            fig = go.Figure()
            
            # Histogram
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns',
                marker_color=self.color_scheme['primary'],
                opacity=0.7,
                hovertemplate='<b>Returns</b><br>' +
                             'Value: %{x:.2%}<br>' +
                             'Count: %{y}<extra></extra>'
            ))
            
            # Add mean line
            mean_return = returns.mean()
            fig.add_vline(
                x=mean_return,
                line_dash="dash",
                line_color=self.color_scheme['secondary'],
                annotation_text=f"Mean: {mean_return:.2%}"
            )
            
            # Add zero line
            fig.add_vline(
                x=0,
                line_dash="dot",
                line_color=self.color_scheme['dark'],
                annotation_text="Zero"
            )
            
            fig.update_layout(
                title=title,
                xaxis_title="Returns",
                yaxis_title="Frequency",
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating returns distribution: {e}")
            return self.create_empty_chart("Error creating returns distribution")
    
    def create_drawdown_chart(self, equity_data: pd.DataFrame,
                            title: str = "Drawdown Analysis") -> go.Figure:
        """
        Create drawdown analysis chart.
        
        Args:
            equity_data: DataFrame with equity curve data
            title: Chart title
            
        Returns:
            go.Figure: Drawdown chart
        """
        try:
            fig = go.Figure()
            
            if 'drawdown' in equity_data.columns:
                # Drawdown line
                fig.add_trace(go.Scatter(
                    x=equity_data.index,
                    y=equity_data['drawdown'],
                    mode='lines',
                    name='Drawdown',
                    line=dict(color=self.color_scheme['danger'], width=2),
                    fill='tonexty',
                    fillcolor='rgba(214, 39, 40, 0.3)',
                    hovertemplate='<b>Drawdown</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Drawdown: %{y:.2%}<extra></extra>'
                ))
                
                # Add zero line
                fig.add_hline(
                    y=0,
                    line_dash="dash",
                    line_color=self.color_scheme['dark'],
                    annotation_text="Zero Drawdown"
                )
            
            fig.update_layout(
                title=title,
                xaxis_title="Time",
                yaxis_title="Drawdown (%)",
                hovermode='x unified',
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating drawdown chart: {e}")
            return self.create_empty_chart("Error creating drawdown chart")
    
    def create_risk_metrics_chart(self, risk_metrics: Dict[str, Any],
                                 title: str = "Risk Metrics") -> go.Figure:
        """
        Create risk metrics visualization.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            title: Chart title
            
        Returns:
            go.Figure: Risk metrics chart
        """
        try:
            # Extract relevant metrics
            metrics = ['volatility', 'var_95', 'expected_shortfall', 'max_drawdown']
            values = [risk_metrics.get(metric, 0) for metric in metrics]
            labels = ['Volatility', 'VaR (95%)', 'Expected Shortfall', 'Max Drawdown']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=[self.color_scheme['primary'], 
                                 self.color_scheme['danger'],
                                 self.color_scheme['warning'],
                                 self.color_scheme['secondary']],
                    hovertemplate='<b>%{x}</b><br>' +
                                 'Value: %{y:.2%}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Risk Metrics",
                yaxis_title="Value (%)",
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating risk metrics chart: {e}")
            return self.create_empty_chart("Error creating risk metrics chart")
    
    def create_signal_analysis_chart(self, signals: List[Signal],
                                   title: str = "Signal Analysis") -> go.Figure:
        """
        Create signal analysis chart.
        
        Args:
            signals: List of trading signals
            title: Chart title
            
        Returns:
            go.Figure: Signal analysis chart
        """
        try:
            if not signals:
                return self.create_empty_chart("No signals available")
            
            # Convert signals to DataFrame
            signal_data = []
            for signal in signals:
                signal_data.append({
                    'timestamp': signal.timestamp,
                    'signal_type': signal.signal_type.value,
                    'confidence': signal.confidence,
                    'market_regime': signal.market_regime.value,
                    'price': signal.price
                })
            
            signal_df = pd.DataFrame(signal_data)
            
            # Create subplot with secondary y-axis
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=('Signal Types Over Time', 'Signal Confidence'),
                vertical_spacing=0.1
            )
            
            # Signal types over time
            for signal_type in signal_df['signal_type'].unique():
                type_data = signal_df[signal_df['signal_type'] == signal_type]
                fig.add_trace(
                    go.Scatter(
                        x=type_data['timestamp'],
                        y=type_data['signal_type'],
                        mode='markers',
                        name=signal_type,
                        marker=dict(size=8),
                        hovertemplate='<b>%{y}</b><br>' +
                                     'Time: %{x}<br>' +
                                     'Price: $%{customdata:,.2f}<extra></extra>',
                        customdata=type_data['price']
                    ),
                    row=1, col=1
                )
            
            # Signal confidence over time
            fig.add_trace(
                go.Scatter(
                    x=signal_df['timestamp'],
                    y=signal_df['confidence'],
                    mode='lines+markers',
                    name='Confidence',
                    line=dict(color=self.color_scheme['primary']),
                    hovertemplate='<b>Confidence</b><br>' +
                                 'Time: %{x}<br>' +
                                 'Confidence: %{y:.2%}<extra></extra>'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                title=title,
                height=600,
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating signal analysis chart: {e}")
            return self.create_empty_chart("Error creating signal analysis chart")
    
    def create_performance_comparison(self, results: Dict[str, Any],
                                    title: str = "Performance Comparison") -> go.Figure:
        """
        Create performance comparison chart.
        
        Args:
            results: Dictionary of performance results
            title: Chart title
            
        Returns:
            go.Figure: Performance comparison chart
        """
        try:
            metrics = ['total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate']
            values = [results.get(metric, 0) for metric in metrics]
            labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown', 'Win Rate']
            
            fig = go.Figure(data=[
                go.Bar(
                    x=labels,
                    y=values,
                    marker_color=self.color_scheme['primary'],
                    hovertemplate='<b>%{x}</b><br>' +
                                 'Value: %{y:.2f}<extra></extra>'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Performance Metrics",
                yaxis_title="Value",
                **self.chart_theme['layout']
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Error creating performance comparison: {e}")
            return self.create_empty_chart("Error creating performance comparison")
    
    def create_empty_chart(self, message: str = "No data available") -> go.Figure:
        """
        Create empty chart with message.
        
        Args:
            message: Message to display
            
        Returns:
            go.Figure: Empty chart with message
        """
        fig = go.Figure()
        fig.add_annotation(
            text=message,
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            showarrow=False,
            font=dict(size=16, color=self.color_scheme['dark'])
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False, showticklabels=False),
            **self.chart_theme['layout']
        )
        return fig


class PerformanceAnalyzer:
    """
    Performance analysis utilities for trading system.
    
    Provides comprehensive performance analysis and metrics calculation.
    """
    
    def __init__(self):
        """Initialize performance analyzer."""
        logger.info("Performance analyzer initialized")
    
    def analyze_signals(self, signals: List[Signal]) -> Dict[str, Any]:
        """
        Analyze trading signals.
        
        Args:
            signals: List of trading signals
            
        Returns:
            Dict[str, Any]: Signal analysis results
        """
        try:
            if not signals:
                return {}
            
            # Basic statistics
            total_signals = len(signals)
            signal_types = {}
            for signal in signals:
                signal_type = signal.signal_type.value
                signal_types[signal_type] = signal_types.get(signal_type, 0) + 1
            
            # Confidence analysis
            confidences = [s.confidence for s in signals]
            avg_confidence = np.mean(confidences) if confidences else 0
            
            # Time analysis
            if len(signals) > 1:
                time_span = (signals[-1].timestamp - signals[0].timestamp).total_seconds()
                signal_frequency = len(signals) / (time_span / 3600)  # signals per hour
            else:
                signal_frequency = 0
            
            return {
                'total_signals': total_signals,
                'signal_types': signal_types,
                'average_confidence': avg_confidence,
                'signal_frequency': signal_frequency,
                'current_signal': signals[-1].signal_type.value if signals else 'N/A'
            }
            
        except Exception as e:
            logger.error(f"Error analyzing signals: {e}")
            return {}
    
    def calculate_rolling_metrics(self, equity_data: pd.DataFrame, 
                                window: int = 30) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            equity_data: DataFrame with equity curve data
            window: Rolling window size
            
        Returns:
            pd.DataFrame: Rolling metrics
        """
        try:
            if 'portfolio_value' not in equity_data.columns:
                return pd.DataFrame()
            
            # Calculate returns
            returns = equity_data['portfolio_value'].pct_change()
            
            # Rolling metrics
            rolling_metrics = pd.DataFrame(index=equity_data.index)
            rolling_metrics['rolling_return'] = returns.rolling(window).mean()
            rolling_metrics['rolling_volatility'] = returns.rolling(window).std()
            rolling_metrics['rolling_sharpe'] = rolling_metrics['rolling_return'] / rolling_metrics['rolling_volatility']
            
            return rolling_metrics
            
        except Exception as e:
            logger.error(f"Error calculating rolling metrics: {e}")
            return pd.DataFrame()
    
    def generate_performance_report(self, backtest_results: Any) -> Dict[str, Any]:
        """
        Generate comprehensive performance report.
        
        Args:
            backtest_results: Backtest results object
            
        Returns:
            Dict[str, Any]: Performance report
        """
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'total_return': backtest_results.total_return,
                'sharpe_ratio': backtest_results.sharpe_ratio,
                'max_drawdown': backtest_results.max_drawdown,
                'win_rate': backtest_results.win_rate,
                'total_trades': backtest_results.total_trades,
                'final_capital': backtest_results.final_capital
            }
            
            # Add additional metrics if available
            if hasattr(backtest_results, 'performance_metrics'):
                report.update(backtest_results.performance_metrics)
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating performance report: {e}")
            return {}
