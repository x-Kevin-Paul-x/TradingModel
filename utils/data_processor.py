import pandas as pd
import numpy as np
from polygon import RESTClient
from datetime import datetime, timedelta
import ta
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *

class DataProcessor:
    def __init__(self):
        self.client = RESTClient(api_key=POLYGON_API_KEY)
        
    def fetch_historical_data(self, symbol, start_date, end_date):
        """Fetch historical data from Polygon.io with retry mechanism"""
        import time
        from requests.exceptions import RequestException
        
        max_retries = 5
        base_delay = 12  # Base delay in seconds
        
        for attempt in range(max_retries):
            try:
                # Add exponential backoff delay
                if attempt > 0:
                    delay = base_delay * (2 ** (attempt - 1))
                    print(f"Retrying {symbol} after {delay} seconds (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                
                aggs = []
                for agg in self.client.list_aggs(
                    ticker=symbol,
                    multiplier=1,
                    timespan="day",
                    from_=start_date,
                    to=end_date,
                    limit=50000
                ):
                    aggs.append({
                        'timestamp': agg.timestamp,
                        'open': agg.open,
                        'high': agg.high,
                        'low': agg.low,
                        'close': agg.close,
                        'volume': agg.volume,
                        'vwap': agg.vwap
                    })
                
                if not aggs:
                    print(f"Warning: No data returned for {symbol}")
                    return None
                
                df = pd.DataFrame(aggs)
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
                
                # Add delay between successful requests
                time.sleep(base_delay)
                
                return df
                
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed to fetch data for {symbol} after {max_retries} attempts: {str(e)}")
                    return None
                continue

    def add_technical_indicators(self, df):
        """Add technical indicators to the dataframe"""
        # Price-based indicators
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close']).diff()
        
        # Trend indicators
        df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
        df['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
        df['ema_20'] = ta.trend.ema_indicator(df['close'], window=20)
        
        # Momentum indicators
        df['rsi'] = ta.momentum.rsi(df['close'], window=14)
        df['macd'] = ta.trend.macd_diff(df['close'])
        df['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
        
        # Volatility indicators
        df['bbands_upper'] = ta.volatility.bollinger_hband(df['close'])
        df['bbands_lower'] = ta.volatility.bollinger_lband(df['close'])
        df['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
        
        # Volume indicators
        df['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
        df['mfi'] = ta.volume.money_flow_index(df['high'], df['low'], df['close'], df['volume'])
        
        # Custom features
        df['price_range'] = (df['high'] - df['low']) / df['close']
        df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(window=20).mean()
        
        return df

    def create_features(self, df):
        """Create additional features for ML model"""
        # Price momentum
        df['momentum_1d'] = df['returns'].shift(1)
        df['momentum_5d'] = df['close'].pct_change(5)
        df['momentum_20d'] = df['close'].pct_change(20)
        
        # Volatility features
        df['volatility_5d'] = df['returns'].rolling(window=5).std()
        df['volatility_20d'] = df['returns'].rolling(window=20).std()
        
        # Price relative to moving averages
        df['price_sma20_ratio'] = df['close'] / df['sma_20']
        df['price_sma50_ratio'] = df['close'] / df['sma_50']
        
        # Volume features
        df['volume_momentum'] = df['volume'].pct_change()
        df['volume_volatility'] = df['volume'].rolling(window=5).std()
        
        # Target variable (Next day return > 0)
        df['target'] = (df['returns'].shift(-1) > 0).astype(int)
        
        return df

    def prepare_data(self, symbols, start_date, end_date):
        """Prepare data for all symbols"""
        all_data = {}
        print("Fetching and processing data...")
        for symbol in tqdm(symbols):
            # Fetch data
            df = self.fetch_historical_data(symbol, start_date, end_date)
            if df is not None:
                # Process data
                df = self.add_technical_indicators(df)
                df = self.create_features(df)
                
                # Drop rows with NaN values
                df.dropna(inplace=True)
                
                all_data[symbol] = df
        
        return all_data

    def split_data(self, data, test_size=0.2, validation_size=0.2):
        """Split data into train, validation, and test sets"""
        train_data = {}
        val_data = {}
        test_data = {}
        
        for symbol, df in data.items():
            # Calculate split points
            total_size = len(df)
            test_index = int(total_size * (1 - test_size))
            val_index = int(test_index * (1 - validation_size))
            
            # Split data
            train_data[symbol] = df[:val_index]
            val_data[symbol] = df[val_index:test_index]
            test_data[symbol] = df[test_index:]
        
        return train_data, val_data, test_data

    def prepare_features(self, data):
        """Prepare features for ML model"""
        feature_columns = [
            'returns', 'log_returns', 'rsi', 'macd', 'williams_r',
            'price_range', 'volume_ma_ratio', 'momentum_1d', 'momentum_5d',
            'momentum_20d', 'volatility_5d', 'volatility_20d',
            'price_sma20_ratio', 'price_sma50_ratio', 'volume_momentum',
            'volume_volatility', 'mfi'
        ]
        
        X = []
        y = []
        
        for symbol, df in data.items():
            # Ensure all required features are present
            missing_features = set(feature_columns) - set(df.columns)
            if missing_features:
                print(f"Warning: Missing features for {symbol}: {missing_features}")
                continue
                
            # Select only the specified features
            X.append(df[feature_columns])
            y.append(df['target'])
        
        if not X:
            raise ValueError("No valid data found with required features")
            
        return np.vstack(X), np.hstack(y)
