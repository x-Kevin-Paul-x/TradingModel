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
INITIAL_CAPITAL = 100
TRADING_SYMBOLS = ["AAPL", "GOOGL", "MSFT", "AMZN", "META"]  # Example symbols
TIMEFRAME = "1d"  # Daily timeframe

# ML Model Configuration
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.2

# Trading Bot Configuration
STOP_LOSS_PCT = 0.02  # 2% stop loss - slightly wider for RSI strategy
TAKE_PROFIT_PCT = 0.04  # 4% take profit - higher reward/risk ratio
MAX_POSITION_SIZE = 0.15  # Maximum 15% of portfolio in single position

# Additional Trading Parameters
MIN_VOL_THRESHOLD = 0.8  # Minimum volume ratio threshold - stricter volume requirement
MIN_PROFIT_THRESHOLD = 0.003  # Minimum expected profit threshold (0.3%)
MAX_HOLDINGS = 2  # Maximum number of simultaneous positions - more concentrated
