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
        
    @staticmethod
    def save_plot(plt, name):
        """Save plot to visualizations directory"""
        os.makedirs('visualizations', exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plt.savefig(f'visualizations/{name}_{timestamp}.png', bbox_inches='tight', dpi=300)
        plt.close()

    def plot_feature_importance(self, feature_importance, title='Feature Importance'):
        """Plot feature importance from ML model"""
        self.setup_style()
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_importance.head(15))
        plt.title(title)
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        
        self.save_plot(plt, 'feature_importance')

    def plot_model_performance(self, predictions, actual, title='Model Performance'):
        """Plot ML model predictions vs actual values"""
        self.setup_style()
        
        plt.figure(figsize=(15, 6))
        plt.plot(actual.index, actual, label='Actual', alpha=0.7)
        plt.plot(actual.index, predictions, label='Predicted', alpha=0.7)
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        
        self.save_plot(plt, 'model_performance')

    def plot_trading_performance(self, trades_history, price_data, title='Trading Performance'):
        """Plot trading bot performance with buy/sell signals"""
        self.setup_style()
        plt.figure(figsize=(15, 8))
        
        # Plot price
        plt.plot(price_data.index, price_data['close'], label='Price', alpha=0.7)
        
        if trades_history:  # Only plot trades if there are any
            # Create DataFrame from trades history
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                
                # Plot buy signals
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                if not buy_trades.empty:
                    plt.scatter(buy_trades['timestamp'], buy_trades['price'],
                              marker='^', color='g', label='Buy', s=100)
                
                # Plot sell signals
                sell_trades = trades_df[trades_df['action'] == 'SELL']
                if not sell_trades.empty:
                    plt.scatter(sell_trades['timestamp'], sell_trades['price'],
                              marker='v', color='r', label='Sell', s=100)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        
        self.save_plot(plt, 'trading_performance')

    def plot_portfolio_value(self, trades_history, title='Portfolio Value Over Time'):
        """Plot portfolio value evolution"""
        self.setup_style()
        plt.figure(figsize=(15, 6))
        
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns and 'capital' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                plt.plot(trades_df['timestamp'], trades_df['capital'], label='Portfolio Value')
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        
        self.save_plot(plt, 'portfolio_value')

    def plot_returns_distribution(self, trades_history, title='Returns Distribution'):
        """Plot distribution of returns"""
        self.setup_style()
        plt.figure(figsize=(12, 6))
        
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'capital' in trades_df.columns:
                returns = pd.Series(trades_df['capital']).pct_change().dropna()
                if not returns.empty:
                    sns.histplot(returns, kde=True)
        
        plt.title(title)
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        
        self.save_plot(plt, 'returns_distribution')

    def plot_comparison(self, ml_performance, bot_performance, title='ML Model vs Trading Bot Performance'):
        """Plot performance comparison between ML model and trading bot"""
        self.setup_style()
        
        plt.figure(figsize=(15, 8))
        
        # Convert to pandas Series if they aren't already
        ml_performance = pd.Series(ml_performance)
        bot_performance = pd.Series(bot_performance)
        
        # Create a common date range index
        if len(ml_performance) > len(bot_performance):
            ml_performance = ml_performance.iloc[:len(bot_performance)]
        else:
            bot_performance = bot_performance.iloc[:len(ml_performance)]
        
        # Plot performances
        plt.plot(range(len(ml_performance)), ml_performance.values,
                label='ML Model', alpha=0.7)
        plt.plot(range(len(bot_performance)), bot_performance.values,
                label='Trading Bot', alpha=0.7)
        
        plt.title(title)
        plt.xlabel('Trading Days')
        plt.ylabel('Portfolio Value ($)')
        plt.legend()
        
        # Add performance metrics
        ml_return = (ml_performance.iloc[-1] - ml_performance.iloc[0]) / ml_performance.iloc[0] * 100
        bot_return = (bot_performance.iloc[-1] - bot_performance.iloc[0]) / bot_performance.iloc[0] * 100
        
        plt.figtext(0.02, 0.02,
                   f'ML Model Return: {ml_return:.2f}%\n'
                   f'Trading Bot Return: {bot_return:.2f}%',
                   fontsize=10)
        
        self.save_plot(plt, 'performance_comparison')

    def create_performance_dashboard(self, ml_model, trading_bot, price_data):
        """Create a comprehensive performance dashboard"""
        # Plot feature importance
        self.plot_feature_importance(ml_model.get_top_features())
        
        # Plot trading performance
        self.plot_trading_performance(trading_bot.trades_history, price_data)
        
        # Plot portfolio value
        self.plot_portfolio_value(trading_bot.trades_history)
        
        # Plot returns distribution
        self.plot_returns_distribution(trading_bot.trades_history)
        
        # Create comparison plot if both systems have performance data
        if hasattr(ml_model, 'performance_history') and trading_bot.trades_history:
            ml_performance = pd.Series(ml_model.performance_history)
            bot_performance = pd.Series([t['capital'] for t in trading_bot.trades_history])
            self.plot_comparison(ml_performance, bot_performance)
