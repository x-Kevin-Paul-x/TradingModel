import pandas as pd
import numpy as np
from datetime import datetime
import os

from utils.data_processor import DataProcessor
from models.ml_model import MLModel
from models.trading_bot import TradingBot
from utils.visualization import VisualizationUtils
from config import *

def main():
    """
    Main function to run the entire trading simulation pipeline.
    This includes data preparation, model training, backtesting, and result visualization.
    """
    print("Starting Trading Simulation...")
    
    # 1. Initialization
    print("\nInitializing components...")
    data_processor = DataProcessor()
    model = MLModel(algorithm='xgboost')
    trading_bot = TradingBot(initial_capital=INITIAL_CAPITAL)
    visualizer = VisualizationUtils()
    
    # 2. Data Fetching and Preparation
    print("\nFetching and processing data...")
    data = data_processor.prepare_data(
        TRADING_SYMBOLS,
        TRAINING_START_DATE,
        TEST_END_DATE
    )
    
    if not data:
        print("Error: No data available. Please check your API key and dates.")
        return
    
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
        cv=3,  # Using 3-fold CV for speed in this environment
        scoring='f1'
    )
    print(f"Best hyperparameters found: {tuning_results['best_params']}")
    print(f"Best F1-score from tuning: {tuning_results['best_score']:.4f}")

    # 4. Backtest a pure ML-based strategy
    print("\nSimulating ML model trading with realistic portfolio management...")
    ml_bot = TradingBot(initial_capital=INITIAL_CAPITAL)
    for symbol, data in test_data.items():
        # Align predictions with the correct data
        symbol_X_test, _ = data_processor.prepare_features({symbol: data})
        symbol_predictions = model.predict(pd.DataFrame(symbol_X_test, columns=FEATURE_COLUMNS))

        # Create a signals DataFrame for the bot
        signals = pd.DataFrame(index=data.iloc[:len(symbol_predictions)].index)
        signals['signal'] = symbol_predictions
        signals['position'] = signals['signal'].diff()

        # Run the bot with these signals
        ml_bot.update_market_data(symbol, data)
        for timestamp, row in data.iterrows():
            ml_bot.check_exit_signals(symbol, row['close'], timestamp)
            if timestamp in signals.index:
                if signals.loc[timestamp, 'position'] == 1:  # Buy signal
                    size = ml_bot.calculate_position_size(row['close'])
                    if size > 0:
                        ml_bot.execute_trade(symbol, row['close'], 'BUY', size, timestamp, 'ml_strategy')
                elif signals.loc[timestamp, 'position'] == -1:  # Sell signal
                    if symbol in ml_bot.positions:
                        ml_bot.execute_trade(symbol, row['close'], 'SELL', ml_bot.positions[symbol]['size'], timestamp, 'ml_strategy')

    # 5. Backtest rule-based strategies enhanced with ML predictions
    print("\nGenerating ML predictions for strategy integration...")
    all_predictions = {}
    for symbol, data in test_data.items():
        symbol_X_test, _ = data_processor.prepare_features({symbol: data})
        if symbol_X_test.size > 0:
            symbol_predictions = model.predict(pd.DataFrame(symbol_X_test, columns=FEATURE_COLUMNS))
            all_predictions[symbol] = pd.Series(symbol_predictions, index=data.iloc[:len(symbol_predictions)].index, name="ml_prediction")

    # Run trading bot with different strategies, now integrated with ML predictions
    print("\nRunning trading bot strategies with ML integration...")
    strategies = ['moving_average', 'rsi', 'momentum']
    best_strategy = None
    best_return = float('-inf')
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        trading_bot.reset()
        
        for symbol, data in test_data.items():
            if symbol in all_predictions:
                trading_bot.run_strategy(data, strategy=strategy, ml_predictions=all_predictions[symbol])
        
        metrics = trading_bot.get_performance_metrics()
        returns = metrics['total_return']
        
        print(f"Strategy: {strategy}")
        print(f"Total Return: {returns*100:.2f}%")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
        
        if returns > best_return:
            best_return = returns
            best_strategy = strategy
    
    # Run final comparison with best strategy
    print(f"\nRunning final comparison with best strategy: {best_strategy}")
    trading_bot.reset()
    
    for symbol, data in test_data.items():
        if symbol in all_predictions:
            trading_bot.run_strategy(data, strategy=best_strategy, ml_predictions=all_predictions[symbol])
    
    # 6. Generate and display final results and visualizations
    print("\nGenerating performance visualizations...")
    for symbol, data in test_data.items():
        visualizer.create_performance_dashboard(model, trading_bot, data)
    
    # Print final results
    ml_bot_metrics = ml_bot.get_performance_metrics()
    bot_metrics = trading_bot.get_performance_metrics()
    
    print("\nFinal Results:")
    print("-" * 50)
    print("ML Model Strategy Performance:")
    print(f"Total Return: {ml_bot_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {ml_bot_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {ml_bot_metrics['max_drawdown']*100:.2f}%")
    print(f"Final Capital: ${ml_bot_metrics['final_capital']:.2f}")
    
    print("\nBest Trading Bot Strategy Performance:")
    print(f"Total Return: {bot_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {bot_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {bot_metrics['max_drawdown']*100:.2f}%")
    print(f"Total Trades: {bot_metrics['total_trades']}")
    print(f"Winning Trades: {bot_metrics['winning_trades']}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital: ${bot_metrics['final_capital']:.2f}")

if __name__ == "__main__":
    main()
