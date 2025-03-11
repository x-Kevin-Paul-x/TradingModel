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
    print("Starting Trading Simulation...")
    
    # Initialize components
    data_processor = DataProcessor()
    model = MLModel(algorithm='xgboost')
    trading_bot = TradingBot(initial_capital=INITIAL_CAPITAL)
    visualizer = VisualizationUtils()
    
    # Fetch and process data
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
    
    # Train ML model
    print("\nTraining ML model...")
    model.train(
        pd.DataFrame(X_train),
        y_train,
        pd.DataFrame(X_val),
        y_val
    )

    # Get ML model predictions
    print("\nGenerating ML model predictions...")
    test_predictions = model.predict(pd.DataFrame(X_test))
    
    # Initialize performance tracking for ML model
    model.performance_history = []
    current_capital = INITIAL_CAPITAL
    
    # Simulate ML model trading
    print("\nSimulating ML model trading...")
    for symbol in test_data:
        symbol_data = test_data[symbol]
        
        # Prepare features for prediction
        features = pd.DataFrame(symbol_data[:-1])  # Exclude last day
        try:
            feature_data, _ = data_processor.prepare_features({symbol: features})
            symbol_predictions = model.predict(pd.DataFrame(feature_data))
            
            for i, prediction in enumerate(symbol_predictions):
                price = symbol_data.iloc[i]['close']
                next_price = symbol_data.iloc[i+1]['close']
                
                if prediction == 1:  # Buy signal
                    position_size = current_capital / price
                    profit = position_size * (next_price - price)
                    current_capital += profit
                
                model.performance_history.append(current_capital)
        except Exception as e:
            print(f"Error processing {symbol}: {str(e)}")
            continue
    
    # Run trading bot with different strategies
    print("\nRunning trading bot strategies...")
    strategies = ['moving_average', 'rsi', 'momentum']
    best_strategy = None
    best_return = float('-inf')
    
    for strategy in strategies:
        print(f"\nTesting {strategy} strategy...")
        trading_bot.reset()
        
        for symbol, data in test_data.items():
            trading_bot.run_strategy(data, strategy=strategy)
        
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
        trading_bot.run_strategy(data, strategy=best_strategy)
    
    # Create performance visualizations
    print("\nGenerating performance visualizations...")
    for symbol, data in test_data.items():
        visualizer.create_performance_dashboard(model, trading_bot, data)
    
    # Print final results
    ml_final_return = (model.performance_history[-1] - INITIAL_CAPITAL) / INITIAL_CAPITAL
    bot_metrics = trading_bot.get_performance_metrics()
    
    print("\nFinal Results:")
    print("-" * 50)
    print("ML Model Performance:")
    print(f"Total Return: {ml_final_return*100:.2f}%")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital: ${model.performance_history[-1]:.2f}")
    
    print("\nTrading Bot Performance:")
    print(f"Total Return: {bot_metrics['total_return']*100:.2f}%")
    print(f"Sharpe Ratio: {bot_metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: {bot_metrics['max_drawdown']*100:.2f}%")
    print(f"Total Trades: {bot_metrics['total_trades']}")
    print(f"Winning Trades: {bot_metrics['winning_trades']}")
    print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
    print(f"Final Capital: ${bot_metrics['final_capital']:.2f}")

if __name__ == "__main__":
    main()
