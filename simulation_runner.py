import pandas as pd
from utils.data_processor import DataProcessor
from models.ml_model import MLModel
from models.trading_bot import TradingBot
from config import *

def run_simulation(symbols, start_date, end_date, initial_capital=INITIAL_CAPITAL):
    """
    Runs the trading simulation for the given symbols and date range.
    """
    print("Starting Trading Simulation...")

    # 1. Initialization
    print("\nInitializing components...")
    data_processor = DataProcessor()
    model = MLModel(algorithm='xgboost')
    trading_bot = TradingBot(initial_capital=initial_capital)

    # 2. Data Fetching and Preparation
    print("\nFetching and processing data...")
    data = data_processor.prepare_data(
        symbols,
        start_date,
        end_date
    )

    if not data:
        print("Error: No data available. Please check your API key and dates.")
        return None

    # Split data into train/validation/test sets
    train_data, val_data, test_data = data_processor.split_data(
        data,
        test_size=TEST_SIZE,
        validation_size=VALIDATION_SIZE
    )

    # Prepare features for ML model
    print("\nPreparing features for ML model...")
    X_train, y_train = data_processor.prepare_features(train_data)
    X_val, y_val = data_processor.prepare_features(val_data)
    X_test, y_test = data_processor.prepare_features(test_data)

    # 3. ML Model Training with Hyperparameter Tuning
    print("\nEnhancing ML model with hyperparameter tuning...")
    from config import XGB_PARAM_GRID, FEATURE_COLUMNS
    tuning_results = model.tune_hyperparameters(
        pd.DataFrame(X_train, columns=FEATURE_COLUMNS),
        y_train,
        param_grid=XGB_PARAM_GRID,
        cv=3,
        scoring='f1'
    )
    print(f"Best hyperparameters found: {tuning_results['best_params']}")
    print(f"Best F1-score from tuning: {tuning_results['best_score']:.4f}")

    # 4. Backtest a pure ML-based strategy
    print("\nSimulating ML model trading...")
    ml_bot = TradingBot(initial_capital=initial_capital)
    for symbol, sym_data in test_data.items():
        symbol_X_test, _ = data_processor.prepare_features({symbol: sym_data})
        symbol_predictions = model.predict(pd.DataFrame(symbol_X_test, columns=FEATURE_COLUMNS))

        signals = pd.DataFrame(index=sym_data.iloc[:len(symbol_predictions)].index)
        signals['signal'] = symbol_predictions
        signals['position'] = signals['signal'].diff()

        ml_bot.update_market_data(symbol, sym_data)
        for timestamp, row in sym_data.iterrows():
            ml_bot.check_exit_signals(symbol, row['close'], timestamp)
            if timestamp in signals.index:
                if signals.loc[timestamp, 'position'] == 1:
                    size = ml_bot.calculate_position_size(row['close'])
                    if size > 0:
                        ml_bot.execute_trade(symbol, row['close'], 'BUY', size, timestamp, 'ml_strategy')
                elif signals.loc[timestamp, 'position'] == -1:
                    if symbol in ml_bot.positions:
                        ml_bot.execute_trade(symbol, row['close'], 'SELL', ml_bot.positions[symbol]['size'], timestamp, 'ml_strategy')

    # 5. Backtest rule-based strategies enhanced with ML predictions
    print("\nGenerating ML predictions for strategy integration...")
    all_predictions = {}
    for symbol, sym_data in test_data.items():
        symbol_X_test, _ = data_processor.prepare_features({symbol: sym_data})
        if symbol_X_test.size > 0:
            symbol_predictions = model.predict(pd.DataFrame(symbol_X_test, columns=FEATURE_COLUMNS))
            all_predictions[symbol] = pd.Series(symbol_predictions, index=sym_data.iloc[:len(symbol_predictions)].index, name="ml_prediction")

    print("\nRunning trading bot strategies with ML integration...")
    strategies = ['moving_average', 'rsi', 'momentum']
    best_strategy = None
    best_return = float('-inf')

    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        trading_bot.reset()

        for symbol, sym_data in test_data.items():
            if symbol in all_predictions:
                trading_bot.run_strategy(sym_data, strategy=strategy, ml_predictions=all_predictions[symbol])

        metrics = trading_bot.get_performance_metrics()
        returns = metrics['total_return']

        if returns > best_return:
            best_return = returns
            best_strategy = strategy

    print(f"\nRunning final comparison with best strategy: {best_strategy}")
    trading_bot.reset()

    for symbol, sym_data in test_data.items():
        if symbol in all_predictions:
            trading_bot.run_strategy(sym_data, strategy=best_strategy, ml_predictions=all_predictions[symbol])

    # 6. Return results
    ml_bot_metrics = ml_bot.get_performance_metrics()
    bot_metrics = trading_bot.get_performance_metrics()

    return {
        'ml_bot_metrics': ml_bot_metrics,
        'bot_metrics': bot_metrics,
        'trading_bot': trading_bot,
        'ml_model': model,
        'test_data': test_data,
        'best_strategy': best_strategy
    }
