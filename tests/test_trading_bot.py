import unittest
import pandas as pd
import sys
import os

# Add the parent directory to the sys.path to allow imports from the main project
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.trading_bot import TradingBot
import config

class TestTradingBot(unittest.TestCase):

    def setUp(self):
        """Set up a new TradingBot instance before each test."""
        self.bot = TradingBot(initial_capital=10000)

        # Create mock data for a symbol
        mock_data = {
            'close': [100, 102, 105, 103, 98],
            'volume': [1000, 1100, 1200, 1050, 1300],
            'volume_ma_ratio': [1.0, 1.1, 1.2, 1.05, 1.3],
            'atr': [2.0, 2.1, 2.2, 2.1, 2.3]
        }
        self.mock_df = pd.DataFrame(mock_data, index=pd.to_datetime(['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04', '2023-01-05']))
        self.symbol = 'TEST'
        self.bot.update_market_data(self.symbol, self.mock_df)

    def test_initial_state(self):
        """Test the initial state of the TradingBot."""
        self.assertEqual(self.bot.capital, 10000)
        self.assertEqual(self.bot.initial_capital, 10000)
        self.assertEqual(len(self.bot.positions), 0)
        self.assertEqual(len(self.bot.trades_history), 0)

    def test_calculate_position_size(self):
        """Test the position size calculation."""
        # With default MAX_POSITION_SIZE = 0.15
        # 10000 * 0.15 = 1500. At price 100, size should be 15.
        size = self.bot.calculate_position_size(price=100)
        self.assertEqual(size, 15)

        # Test with capital being the limiting factor
        self.bot.capital = 1000
        # 1000 * 0.15 = 150. At price 100, size is 1.5.
        # min(1.5, 1000/100=10) = 1.5
        size = self.bot.calculate_position_size(price=100)
        self.assertEqual(size, 1.5)

    def test_execute_buy_trade(self):
        """Test the execution of a BUY trade."""
        price = 100
        size = 10
        timestamp = pd.to_datetime('2023-01-01')

        trade_executed = self.bot.execute_trade(self.symbol, price, 'BUY', size, timestamp, 'test_strategy')

        self.assertTrue(trade_executed)
        self.assertEqual(self.bot.capital, 9000) # 10000 - 10*100
        self.assertIn(self.symbol, self.bot.positions)
        self.assertEqual(self.bot.positions[self.symbol]['size'], 10)
        self.assertEqual(self.bot.positions[self.symbol]['avg_price'], 100)
        self.assertEqual(len(self.bot.trades_history), 1)
        self.assertEqual(self.bot.trades_history[0]['action'], 'BUY')

    def test_execute_sell_trade(self):
        """Test the execution of a SELL trade."""
        # First, execute a buy trade to have a position
        self.bot.execute_trade(self.symbol, 100, 'BUY', 10, pd.to_datetime('2023-01-01'), 'test_strategy')

        # Now, sell it
        price = 105
        size = 10
        timestamp = pd.to_datetime('2023-01-02')

        trade_executed = self.bot.execute_trade(self.symbol, price, 'SELL', size, timestamp, 'test_strategy')

        self.assertTrue(trade_executed)
        self.assertEqual(self.bot.capital, 10050) # 9000 + 10*105
        self.assertNotIn(self.symbol, self.bot.positions)
        self.assertEqual(len(self.bot.trades_history), 2)
        self.assertEqual(self.bot.trades_history[1]['action'], 'SELL')

    def test_stop_loss_trigger(self):
        """Test that the stop-loss mechanism triggers a sell."""
        # Buy at 100, stop-loss at 100 * (1 - 0.03) = 97
        self.bot.execute_trade(self.symbol, 100, 'BUY', 10, pd.to_datetime('2023-01-01'), 'test_strategy')

        # Price drops to 95, should trigger stop-loss
        triggered = self.bot.check_exit_signals(self.symbol, 95, pd.to_datetime('2023-01-02'))

        self.assertTrue(triggered)
        self.assertNotIn(self.symbol, self.bot.positions)
        self.assertEqual(self.bot.capital, 9950) # 9000 from buy + 10*95 from sell
        self.assertEqual(len(self.bot.trades_history), 2)
        self.assertEqual(self.bot.trades_history[1]['strategy'], 'stop_loss')

    def test_take_profit_trigger(self):
        """Test that the take-profit mechanism triggers a sell."""
        # Buy at 100, take-profit at 100 * (1 + 0.06) = 106
        self.bot.execute_trade(self.symbol, 100, 'BUY', 10, pd.to_datetime('2023-01-01'), 'test_strategy')

        # Price rises to 107, should trigger take-profit
        triggered = self.bot.check_exit_signals(self.symbol, 107, pd.to_datetime('2023-01-02'))

        self.assertTrue(triggered)
        self.assertNotIn(self.symbol, self.bot.positions)
        self.assertEqual(self.bot.capital, 10070) # 9000 from buy + 10*107 from sell
        self.assertEqual(len(self.bot.trades_history), 2)
        self.assertEqual(self.bot.trades_history[1]['strategy'], 'take_profit')

if __name__ == '__main__':
    unittest.main()
