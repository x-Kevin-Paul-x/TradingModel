# RSI Strategy Configuration - Baseline parameters

# Signal Generation Parameters
RSI_THRESHOLDS = {
    'oversold_base': 35,
    'overbought_base': 65,
    'oversold_high_vol': 40,
    'overbought_high_vol': 60
}

# Confirmation Signals
VOLUME_WINDOW = 10
MOMENTUM_THRESHOLD = 0.02
RSI_EXIT_LEVELS = {
    'high': 60,
    'low': 40
}

# Risk Management
STOP_LOSS = 0.02  # 2%
TAKE_PROFIT = 0.04  # 4%
MAX_POSITION_SIZE = 0.15  # 15%
MAX_HOLDINGS = 2

# Additional Filters
MIN_VOLUME_THRESHOLD = 0.8
MIN_PROFIT_THRESHOLD = 0.003
ATR_PERCENTILE_THRESHOLD = 0.7
ATR_SIGNAL_FILTER = 0.5  # Filter out signals when ATR is below 50% of mean

# Performance Metrics Achieved
BASELINE_METRICS = {
    'total_return': 0.6039,  # 60.39%
    'sharpe_ratio': 4.24,
    'max_drawdown': 0.155,  # 15.50%
    'win_rate': 0.47  # 47%
}
