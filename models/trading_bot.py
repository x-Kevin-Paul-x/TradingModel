import numpy as np
import pandas as pd
from datetime import datetime
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

class TradingBot:
    def __init__(self, initial_capital=config.INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = {}
        self.trades_history = []
        self.data = {}  # Store market data for each symbol

    def update_market_data(self, symbol, data):
        """Update market data for a symbol"""
        self.data[symbol] = data
        
    def reset(self):
        """Reset the bot to initial state"""
        self.capital = self.initial_capital
        self.positions = {}
        self.trades_history = []
        
    def calculate_position_size(self, price):
        """Calculate position size based on risk management rules"""
        max_position_value = self.capital * config.MAX_POSITION_SIZE
        return min(max_position_value / price, self.capital / price)
    
    def execute_trade(self, symbol, price, action, size, timestamp, strategy):
        """Execute a trade and update positions with enhanced risk management"""
        # Check volume threshold
        if 'volume_ma_ratio' in self.data[symbol]:
            if self.data[symbol]['volume_ma_ratio'].iloc[-1] < config.MIN_VOL_THRESHOLD:
                return False

        if action == 'BUY':
            # Check position limits
            if len(self.positions) >= config.MAX_HOLDINGS:
                return False
                
            # Check if we have enough capital
            cost = price * size
            if cost > self.capital:
                return False
                
            # Check expected profit potential
            if 'atr' in self.data[symbol]:
                expected_profit = (self.data[symbol]['atr'].iloc[-1] / price)
                if expected_profit < config.MIN_PROFIT_THRESHOLD:
                    return False
                
            self.capital -= cost
            if symbol in self.positions:
                self.positions[symbol]['size'] += size
                self.positions[symbol]['avg_price'] = (
                    (self.positions[symbol]['avg_price'] * (self.positions[symbol]['size'] - size) +
                     price * size) / self.positions[symbol]['size']
                )
            else:
                self.positions[symbol] = {
                    'size': size,
                    'avg_price': price,
                    'stop_loss': price * (1 - config.STOP_LOSS_PCT),
                    'take_profit': price * (1 + config.TAKE_PROFIT_PCT),
                    'entry_time': timestamp
                }
                
        elif action == 'SELL':
            if symbol not in self.positions or self.positions[symbol]['size'] < size:
                return False
                
            profit = (price - self.positions[symbol]['avg_price']) * size
            self.capital += price * size
            self.positions[symbol]['size'] -= size
            
            if self.positions[symbol]['size'] <= 0:
                del self.positions[symbol]
                
        # Record trade
        self.trades_history.append({
            'timestamp': timestamp,
            'symbol': symbol,
            'action': action,
            'price': price,
            'size': size,
            'capital': self.capital,
            'strategy': strategy
        })
        
        return True
        
    def check_exit_signals(self, symbol, current_price, timestamp):
        """Check if any exit signals are triggered"""
        if symbol in self.positions:
            position = self.positions[symbol]
            
            # Check stop loss
            if current_price <= position['stop_loss']:
                self.execute_trade(
                    symbol, current_price, 'SELL',
                    position['size'], timestamp,
                    'stop_loss'
                )
                return True
                
            # Check take profit
            if current_price >= position['take_profit']:
                self.execute_trade(
                    symbol, current_price, 'SELL',
                    position['size'], timestamp,
                    'take_profit'
                )
                return True
                
        return False
        
    def moving_average_strategy(self, data, ml_predictions=None):
        """Enhanced Moving Average Crossover Strategy with RSI Filter and optional ML confirmation."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Moving Average Crossover
        ma_crossover = (data['sma_20'] > data['sma_50']).astype(int)
        ma_crossunder = (data['sma_20'] < data['sma_50']).astype(int)
        
        # RSI Filter
        rsi_oversold = (data['rsi'] < config.MA_RSI_OVERSOLD).astype(int)
        rsi_overbought = (data['rsi'] > config.MA_RSI_OVERBOUGHT).astype(int)
        
        # Momentum Filter
        momentum_filter = (data['momentum_5d'] * data['momentum_1d'] > 0).astype(int)
        
        # Define base buy and sell conditions
        buy_conditions = (
            (ma_crossover & rsi_oversold & momentum_filter) | 
            (data['close'] < data['bbands_lower'])
        )
        sell_conditions = (
            (ma_crossunder & rsi_overbought) |
            (data['close'] > data['bbands_upper'])
        )
        
        # Integrate ML predictions if available
        if ml_predictions is not None:
            buy_conditions &= (ml_predictions == 1)
            sell_conditions &= (ml_predictions == 0) # Assuming 0 is a hold/sell signal

        # Generate signals
        signals['signal'] = np.where(buy_conditions, 1, np.where(sell_conditions, -1, 0))

        # Only get signal changes and filter out weak signals
        signals['position'] = signals['signal'].diff()
        signals.loc[abs(data['atr']) < data['atr'].mean(), 'position'] = 0
        
        return signals
        
    def rsi_strategy(self, data, ml_predictions=None):
        """Enhanced RSI Strategy with Multiple Confirmations and optional ML integration."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Dynamic RSI thresholds based on volatility
        atr_percentile = data['atr'].rolling(window=config.RSI_ATR_WINDOW).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
        rsi_oversold = np.where(atr_percentile > config.RSI_ATR_PERCENTILE_THRESHOLD, config.RSI_OVERSOLD_VOLATILITY_ADJUSTED, config.RSI_OVERSOLD_BASE)
        rsi_overbought = np.where(atr_percentile > config.RSI_ATR_PERCENTILE_THRESHOLD, config.RSI_OVERBOUGHT_VOLATILITY_ADJUSTED, config.RSI_OVERBOUGHT_BASE)
        
        # Volume conditions
        volume_active = data['volume'] > data['volume'].rolling(window=config.RSI_VOLUME_WINDOW).mean()
        
        # Trend conditions with momentum
        price_above_ema = data['close'] > data['ema_20']
        momentum_positive = data['momentum_5d'] > 0
        macd_signal = data['macd'] > 0
        
        # Base conditions for entry
        buy_setup = (
            (data['rsi'] < rsi_oversold) &
            (data['close'] > data['bbands_lower'])
        )
        
        sell_setup = (
            (data['rsi'] > rsi_overbought) &
            (data['close'] < data['bbands_upper'])
        )
        
        # Entry conditions with trend and momentum confirmation
        buy_setup &= (
            (price_above_ema & momentum_positive) |
            (macd_signal & data['rsi'].rolling(window=config.RSI_COUNTER_TREND_WINDOW).mean() < config.RSI_COUNTER_TREND_OVERSOLD)
        )
        
        sell_setup &= (
            (~price_above_ema & ~momentum_positive) |
            (~macd_signal & data['rsi'].rolling(window=config.RSI_COUNTER_TREND_WINDOW).mean() > config.RSI_COUNTER_TREND_OVERBOUGHT)
        )
        
        # Integrate ML predictions if available
        if ml_predictions is not None:
            buy_setup &= (ml_predictions == 1)
            sell_setup &= (ml_predictions == 0)

        # Generate base signals
        signals.loc[buy_setup, 'signal'] = 1
        signals.loc[sell_setup, 'signal'] = -1
        
        # Process exit signals
        long_exit = (
            (data['rsi'] > config.RSI_EXIT_LONG) |
            (data['close'] < data['sma_20']) |
            (data['momentum_1d'] < 0)
        )
        
        short_exit = (
            (data['rsi'] < config.RSI_EXIT_SHORT) |
            (data['close'] > data['sma_20']) |
            (data['momentum_1d'] > 0)
        )
        
        # Apply exit signals
        signals.loc[long_exit & (signals['signal'].shift(1) == 1), 'signal'] = -1
        signals.loc[short_exit & (signals['signal'].shift(1) == -1), 'signal'] = 1
        
        # Calculate final position changes
        signals['position'] = signals['signal'].diff()
        
        # Filter out weak signals
        signals.loc[abs(data['atr']) < data['atr'].mean() * config.RSI_ATR_FILTER_FACTOR, 'position'] = 0
        
        return signals
        
    def momentum_strategy(self, data, ml_predictions=None):
        """Price Momentum Strategy with optional ML confirmation."""
        signals = pd.DataFrame(index=data.index)
        signals['signal'] = 0
        
        # Define base buy and sell conditions
        buy_conditions = (data['momentum_5d'] > 0) & (data['momentum_1d'] > data['momentum_5d'])
        sell_conditions = (data['momentum_5d'] < 0) & (data['momentum_1d'] < data['momentum_5d'])
        
        # Integrate ML predictions if available
        if ml_predictions is not None:
            buy_conditions &= (ml_predictions == 1)
            sell_conditions &= (ml_predictions == 0)

        # Apply signals
        signals.loc[buy_conditions, 'signal'] = 1
        signals.loc[sell_conditions, 'signal'] = -1
        
        signals['position'] = signals['signal'].diff()
        
        return signals
        
    def run_strategy(self, data, strategy='moving_average', ml_predictions=None):
        """Run specified trading strategy"""
        if not isinstance(data, pd.DataFrame):
            print(f"Warning: Invalid data format for strategy execution")
            return
            
        symbol = data.name if hasattr(data, 'name') else 'UNKNOWN'
        self.update_market_data(symbol, data)  # Store the data
        
        # Generate signals based on strategy
        if strategy == 'moving_average':
            signals = self.moving_average_strategy(data, ml_predictions)
        elif strategy == 'rsi':
            signals = self.rsi_strategy(data, ml_predictions)
        elif strategy == 'momentum':
            signals = self.momentum_strategy(data, ml_predictions)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
            
        # Execute trades based on signals
        for timestamp, row in data.iterrows():
            
            # Check exit conditions first
            self.check_exit_signals(symbol, row['close'], timestamp)
            
            # Execute new trades based on signals
            if signals.loc[timestamp, 'position'] == 1:  # Buy signal
                size = self.calculate_position_size(row['close'])
                if size > 0:
                    self.execute_trade(symbol, row['close'], 'BUY',
                                    size, timestamp, strategy)
                    
            elif signals.loc[timestamp, 'position'] == -1:  # Sell signal
                if symbol in self.positions:
                    self.execute_trade(symbol, row['close'], 'SELL',
                                    self.positions[symbol]['size'],
                                    timestamp, strategy)
                    
    def get_performance_metrics(self):
        """Calculate performance metrics"""
        # Initialize default metrics
        default_metrics = {
            'total_return': 0,
            'total_trades': 0,
            'winning_trades': 0,
            'final_capital': self.initial_capital,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }
        
        if not self.trades_history:
            return default_metrics
            
        try:
            trades_df = pd.DataFrame(self.trades_history)
            trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
            
            # Calculate final capital including open positions
            final_capital = self.capital
            for symbol, position in self.positions.items():
                final_capital += position['size'] * position['avg_price']
            
            # Calculate metrics
            metrics = {
                'total_return': (final_capital - self.initial_capital) / self.initial_capital,
                'total_trades': len(trades_df),
                'winning_trades': len(trades_df[trades_df['capital'] > trades_df['capital'].shift(1)]),
                'final_capital': final_capital,
                'max_drawdown': self.calculate_max_drawdown(trades_df),
                'sharpe_ratio': self.calculate_sharpe_ratio(trades_df)
            }
            
            return metrics
            
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return default_metrics
        
    def calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        if trades_df.empty:
            return 0
            
        capital_series = trades_df['capital']
        rolling_max = capital_series.expanding(min_periods=1).max()
        drawdowns = capital_series / rolling_max - 1
        
        return abs(drawdowns.min())
        
    def calculate_sharpe_ratio(self, trades_df):
        """Calculate Sharpe ratio"""
        if len(trades_df) < 2:
            return 0
            
        capital_series = trades_df['capital']
        returns = capital_series.pct_change().dropna()
        
        if returns.empty:
            return 0
            
        # Assuming 252 trading days in a year
        sharpe_ratio = np.sqrt(252) * returns.mean() / returns.std()
        
        return sharpe_ratio
