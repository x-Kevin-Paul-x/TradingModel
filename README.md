# MEtrad - Algorithmic Trading System

MEtrad is an algorithmic trading system that utilizes machine learning and technical analysis to generate trading signals and manage a portfolio of stocks. It leverages the Polygon.io API for historical market data, employs an XGBoost model for predictions, and implements various trading strategies with risk management features.

## Table of Contents
- [Project Description](#project-description)
- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Strategies](#strategies)
- [ML Model](#ml-model)
- [Data](#data)
- [Risk Management](#risk-management)
- [Contributing](#contributing)
- [License](#license)

## Project Description
This project aims to develop and evaluate an algorithmic trading system capable of making profitable trades in the stock market. It combines machine learning techniques with technical analysis indicators to generate buy and sell signals. The system includes a trading bot that executes trades based on these signals and incorporates risk management strategies to protect the portfolio. The performance of the system is evaluated using metrics such as total return, Sharpe ratio, and maximum drawdown.

## Features

*   **Data Fetching:** Fetches historical stock data from the Polygon.io API.
*   **Technical Analysis:** Calculates various technical indicators using the `ta` library.
*   **Machine Learning:** Employs an XGBoost model for predicting stock price movements.
*   **Trading Strategies:** Implements multiple trading strategies, including:
    *   Moving Average Crossover
    *   RSI (Relative Strength Index)
    *   Price Momentum
*   **Risk Management:** Includes features like stop-loss, take-profit, and position sizing.
*   **Performance Evaluation:** Calculates metrics such as total return, Sharpe ratio, and maximum drawdown.
*   **Visualization:** Generates performance dashboards for visual analysis.
*   **Configurable:** Parameters like API keys, trading symbols, and strategy settings can be configured.

## Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/x-Kevin-Paul-x/TradingModel.git
    cd TradingModel
    ```

2.  **Install dependencies:**

    ```bash
    pip install -r requirements.txt
    ```

## Configuration

1.  **Environment Variables:**

    Create a `.env` file in the project root directory and add your Polygon.io API key:

    ```
    POLYGON_API_KEY=your_polygon_api_key
    ```
    Replace `your_polygon_api_key` with your actual API key. The `.env.example` file provides a template.

2.  **Configuration File:**

    The `config.py` file contains various configuration parameters, including:

    *   `POLYGON_API_KEY`: Your Polygon.io API key (loaded from the `.env` file).
    *   `TRAINING_START_DATE`, `TRAINING_END_DATE`, `TEST_END_DATE`: Date ranges for training and testing.
    *   `TRADING_SYMBOLS`: List of stock symbols to trade.
    *   `TIMEFRAME`: Timeframe for data (e.g., "1d" for daily).
    *   `INITIAL_CAPITAL`: Initial trading capital.
    *   `RANDOM_STATE`: Random seed for reproducibility.
    *   `TEST_SIZE`, `VALIDATION_SIZE`: Data split ratios.
    *   `STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`, `MAX_POSITION_SIZE`: Risk management parameters.
    *   `MIN_VOL_THRESHOLD`, `MIN_PROFIT_THRESHOLD`, `MAX_HOLDINGS`: Additional trading parameters.

    Adjust these parameters as needed for your trading strategy and risk tolerance.

## Usage

To run the trading simulation, execute the `run.py` script:

```bash
python run.py
```

This will:

1.  Fetch historical data from Polygon.io.
2.  Preprocess the data and add technical indicators.
3.  Split the data into training, validation, and test sets.
4.  Train the XGBoost ML model.
5.  Simulate trading using the ML model's predictions.
6.  Run the trading bot with different strategies (moving average, RSI, momentum).
7.  Generate performance visualizations.
8.  Print the final results, including performance metrics for both the ML model and the trading bot.

## Strategies

The trading bot implements the following strategies:

*   **Moving Average Crossover:** Generates buy signals when the short-term moving average crosses above the long-term moving average and sell signals when the opposite occurs. It also incorporates RSI and momentum filters.
*   **RSI (Relative Strength Index):** Uses RSI to identify overbought and oversold conditions, with additional confirmations based on volume, trend, and momentum.
*   **Price Momentum:** Buys when momentum is positive and increasing, and sells when momentum is negative and decreasing.

## ML Model

The machine learning model used for predictions is an XGBoost classifier (`XGBClassifier`). It is trained on historical data with various technical indicators as features. The model's hyperparameters are defined in `config.py` and can be tuned for optimal performance. The model is saved to the `models/saved` directory after training, along with the feature scaler and feature importance data.

## Data

The project uses historical stock data from the [Polygon.io](https://polygon.io/) API. You will need a valid API key to fetch data. The `DataProcessor` class in `utils/data_processor.py` handles data fetching, preprocessing, and feature engineering. It includes a retry mechanism for handling API request failures and adds a wide range of technical indicators using the `ta` library.

## Risk Management

The trading bot incorporates several risk management features:

*   **Stop-Loss:** Automatically sells a position if the price falls below a certain percentage (defined by `STOP_LOSS_PCT`).
*   **Take-Profit:** Automatically sells a position if the price reaches a certain profit target (defined by `TAKE_PROFIT_PCT`).
*   **Position Sizing:** Calculates the position size based on the available capital and a maximum position size limit (defined by `MAX_POSITION_SIZE`).
* **Maximum Holdings:** Limits the number of simultaneous positions (defined by `MAX_HOLDINGS`).
* **Minimum Volume Threshold:** Avoids trading when volume is too low (defined by `MIN_VOL_THRESHOLD`).
* **Minimum Profit Threshold:** Avoids trades with low expected profit (defined by `MIN_PROFIT_THRESHOLD`).

## Contributing

Contributions are welcome! If you'd like to contribute to this project, please follow these guidelines:

1.  Fork the repository.
2.  Create a new branch for your feature or bug fix.
3.  Make your changes and commit them with clear and descriptive messages.
4.  Push your branch to your forked repository.
5.  Submit a pull request to the main repository.

## License

This project does not have a license file. Therefore, it is under exclusive copyright by default.
