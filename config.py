import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Configuration
POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")

# Data Configuration
from datetime import datetime, timedelta

# Calculate dates based on 2-year limitation
END_DATE = datetime.now().strftime("%Y-%m-%d")
START_DATE = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")  # 2 years ago
TEST_START_DATE = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")  # Last 90 days for testing

# Use these for data fetching
TRAINING_START_DATE = START_DATE
TRAINING_END_DATE = TEST_START_DATE
TEST_END_DATE = END_DATE

# Trading Configuration
INITIAL_CAPITAL = 100000
TRADING_SYMBOLS = ["AAPL", "GOOGL"]
TIMEFRAME = "1d"  # Daily timeframe

# ML Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Trading Bot Configuration
STOP_LOSS_PCT = 0.03  # 3% stop loss
TAKE_PROFIT_PCT = 0.06  # 6% take profit
MAX_POSITION_SIZE = 0.15  # Maximum 15% of portfolio in single position

# Additional Trading Parameters
MIN_VOL_THRESHOLD = 0.7  # Minimum volume ratio threshold - stricter volume requirement
MIN_PROFIT_THRESHOLD = 0.002  # Minimum expected profit threshold (0.3%)
MAX_HOLDINGS = 2  # Maximum number of simultaneous positions - more concentrated

# ML Model Hyperparameter Grid for XGBoost (a small grid for efficient tuning)
XGB_PARAM_GRID = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.1],
    'subsample': [0.8]
}

# Strategy Configuration
# Moving Average Strategy
MA_RSI_OVERSOLD = 30
MA_RSI_OVERBOUGHT = 70

# RSI Strategy
RSI_OVERSOLD_BASE = 40
RSI_OVERBOUGHT_BASE = 60
RSI_ATR_WINDOW = 20
RSI_ATR_PERCENTILE_THRESHOLD = 0.7
RSI_OVERSOLD_VOLATILITY_ADJUSTED = 40
RSI_OVERBOUGHT_VOLATILITY_ADJUSTED = 60
RSI_VOLUME_WINDOW = 10
RSI_MOMENTUM_THRESHOLD = 0.02
RSI_COUNTER_TREND_WINDOW = 2
RSI_COUNTER_TREND_OVERSOLD = 40
RSI_COUNTER_TREND_OVERBOUGHT = 60
RSI_EXIT_LONG = 60
RSI_EXIT_SHORT = 40
RSI_ATR_FILTER_FACTOR = 0.5

# Feature Engineering Configuration
FEATURE_COLUMNS = [
    'returns', 'log_returns', 'rsi', 'macd', 'williams_r',
    'price_range', 'volume_ma_ratio', 'momentum_1d', 'momentum_5d',
    'momentum_20d', 'volatility_5d', 'volatility_20d',
    'price_sma20_ratio', 'price_sma50_ratio', 'volume_momentum',
    'volume_volatility', 'mfi'
]
