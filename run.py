from simulation_runner import run_simulation
from config import *

def main():
    """
    Main function to run the entire trading simulation pipeline.
    """
    results = run_simulation(
        TRADING_SYMBOLS,
        TRAINING_START_DATE,
        TEST_END_DATE
    )

    if results:
        ml_bot_metrics = results['ml_bot_metrics']
        bot_metrics = results['bot_metrics']

        print("\nFinal Results:")
        print("-" * 50)
        print("ML Model Strategy Performance:")
        print(f"Total Return: {ml_bot_metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {ml_bot_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {ml_bot_metrics['max_drawdown']*100:.2f}%")
        print(f"Final Capital: ${ml_bot_metrics['final_capital']:.2f}")

        print("\nBest Trading Bot Strategy Performance:")
        print(f"Best Strategy: {results['best_strategy']}")
        print(f"Total Return: {bot_metrics['total_return']*100:.2f}%")
        print(f"Sharpe Ratio: {bot_metrics['sharpe_ratio']:.2f}")
        print(f"Max Drawdown: {bot_metrics['max_drawdown']*100:.2f}%")
        print(f"Total Trades: {bot_metrics['total_trades']}")
        print(f"Winning Trades: {bot_metrics['winning_trades']}")
        print(f"Initial Capital: ${INITIAL_CAPITAL:.2f}")
        print(f"Final Capital: ${bot_metrics['final_capital']:.2f}")

if __name__ == "__main__":
    main()
