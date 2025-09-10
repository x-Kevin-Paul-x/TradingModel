import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
import os

class VisualizationUtils:
    @staticmethod
    def setup_style():
        """Set up plot style"""
        sns.set_theme(style="darkgrid")
        plt.rcParams['figure.figsize'] = (12, 6)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '--'
        plt.rcParams['grid.alpha'] = 0.5
        
    def plot_feature_importance(self, feature_importance, title='Feature Importance'):
        """Plot feature importance from ML model and return figure."""
        self.setup_style()
        
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15), ax=ax)
        ax.set_title(title)
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        
        return fig

    def plot_model_performance(self, predictions, actual, title='Model Performance'):
        """Plot ML model predictions vs actual values and return figure."""
        self.setup_style()
        
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(actual.index, actual, label='Actual', alpha=0.7)
        ax.plot(actual.index, predictions, label='Predicted', alpha=0.7)
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Value')
        ax.legend()
        
        return fig

import plotly.graph_objects as go
from plotly.subplots import make_subplots

    def plot_trading_performance(self, price_data, trades_history=None, title='Trading Performance'):
        """Create an interactive candlestick chart with trade overlays."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.1, row_heights=[0.7, 0.3])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=price_data.index,
                                     open=price_data['open'],
                                     high=price_data['high'],
                                     low=price_data['low'],
                                     close=price_data['close'],
                                     name='Price'), row=1, col=1)

        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                sell_trades = trades_df[trades_df['action'] == 'SELL']

                fig.add_trace(go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'],
                                         mode='markers', marker_symbol='triangle-up',
                                         marker_color='green', marker_size=10,
                                         name='Buy Signal'), row=1, col=1)

                fig.add_trace(go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'],
                                         mode='markers', marker_symbol='triangle-down',
                                         marker_color='red', marker_size=10,
                                         name='Sell Signal'), row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=price_data.index, y=price_data['volume'], name='Volume'), row=2, col=1)

        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            template='plotly_dark'
        )
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def plot_portfolio_value(self, trades_history, title='Portfolio Value Over Time'):
        """Plot portfolio value evolution and return figure."""
        self.setup_style()
        fig, ax = plt.subplots(figsize=(15, 6))
        
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns and 'capital' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                ax.plot(trades_df['timestamp'], trades_df['capital'], label='Portfolio Value')
        
        ax.set_title(title)
        ax.set_xlabel('Date')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        
        return fig

    def plot_returns_distribution(self, trades_history, title='Returns Distribution'):
        """Plot distribution of returns and return figure."""
        self.setup_style()
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'capital' in trades_df.columns:
                returns = pd.Series(trades_df['capital']).pct_change().dropna()
                if not returns.empty:
                    sns.histplot(returns, kde=True, ax=ax)
        
        ax.set_title(title)
        ax.set_xlabel('Return')
        ax.set_ylabel('Frequency')
        
        return fig

    def plot_comparison(self, ml_performance, bot_performance, title='ML Model vs Trading Bot Performance'):
        """Plot performance comparison and return figure."""
        self.setup_style()
        fig, ax = plt.subplots(figsize=(15, 8))
        
        ml_performance = pd.Series(ml_performance)
        bot_performance = pd.Series(bot_performance)
        
        if len(ml_performance) > len(bot_performance):
            ml_performance = ml_performance.iloc[:len(bot_performance)]
        else:
            bot_performance = bot_performance.iloc[:len(ml_performance)]
        
        ax.plot(range(len(ml_performance)), ml_performance.values,
                label='ML Model', alpha=0.7)
        ax.plot(range(len(bot_performance)), bot_performance.values,
                label='Trading Bot', alpha=0.7)
        
        ax.set_title(title)
        ax.set_xlabel('Trading Days')
        ax.set_ylabel('Portfolio Value ($)')
        ax.legend()
        
        ml_return = (ml_performance.iloc[-1] - ml_performance.iloc[0]) / ml_performance.iloc[0] * 100
        bot_return = (bot_performance.iloc[-1] - bot_performance.iloc[0]) / bot_performance.iloc[0] * 100
        
        fig.text(0.02, 0.02,
                   f'ML Model Return: {ml_return:.2f}%\n'
                   f'Trading Bot Return: {bot_return:.2f}%',
                   fontsize=10)
        
        return fig

    def create_performance_dashboard(self, ml_model, trading_bot, price_data):
        """Create and return a dictionary of all performance plots."""
        figs = {}

        figs['feature_importance'] = self.plot_feature_importance(ml_model.get_top_features())
        
        # Since plot_trading_performance now returns a Plotly figure, we handle it differently
        figs['trading_performance'] = self.plot_trading_performance(price_data, trading_bot.trades_history)
        
        figs['portfolio_value'] = self.plot_portfolio_value(trading_bot.trades_history)
        
        figs['returns_distribution'] = self.plot_returns_distribution(trading_bot.trades_history)
        
        if hasattr(ml_model, 'performance_history') and trading_bot.trades_history:
            ml_performance = pd.Series(ml_model.performance_history)
            bot_performance = pd.Series([t['capital'] for t in trading_bot.trades_history])
            figs['comparison'] = self.plot_comparison(ml_performance, bot_performance)

        return figs
