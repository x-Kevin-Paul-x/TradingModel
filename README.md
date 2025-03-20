# MEtrad - Advanced Algorithmic Trading System

MEtrad is a algorithmic trading simulation that combines machine learning with technical analysis to generate high-quality trading signals and effectively manage stock portfolios. The system leverages Polygon.io's market data API, employs an optimized XGBoost prediction model, and implements multiple trading strategies with robust risk management features.

## Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
- [Using MEtrad](#using-metrad)
- [Trading Strategies](#trading-strategies)
- [Machine Learning Model](#machine-learning-model)
- [Data Processing](#data-processing)
- [Risk Management System](#risk-management-system)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)

## Overview

MEtrad is designed to help traders and investors make data-driven decisions in the stock market. By combining cutting-edge machine learning techniques with established technical analysis indicators, the system generates actionable buy and sell signals. The integrated trading bot executes these signals while implementing comprehensive risk management strategies to protect your portfolio. The system's performance is continuously evaluated using industry-standard metrics like total return, Sharpe ratio, and maximum drawdown.

## Key Features

**Data & Analysis:**
- **Real-time Data Integration:** Seamless fetching of historical and current stock data via Polygon.io API
- **Comprehensive Technical Analysis:** Calculation of 15+ technical indicators using the `ta` library
- **Advanced Feature Engineering:** Creation of custom predictive features for machine learning models

**Trading Intelligence:**
- **Machine Learning Prediction:** XGBoost model optimized for stock price movement prediction
- **Multi-Strategy Framework:** Implementation of three distinct trading strategies:
  - Moving Average Crossover with RSI and momentum filters
  - Enhanced RSI with volume and trend confirmations
  - Dynamic Price Momentum with volatility adaptation
- **Backtesting Engine:** Thorough strategy evaluation against historical data

**Portfolio Management:**
- **Intelligent Risk Management:** Stop-loss, take-profit, and position sizing
- **Capital Preservation:** Maximum holdings limits and volume filters
- **Performance Analytics:** Detailed metrics and visual performance dashboards

**System Design:**
- **Highly Configurable:** Customizable parameters for strategies and risk tolerance
- **Modular Architecture:** Easily extendable for new strategies and features
- **Visualization Tools:** Performance dashboards for intuitive analysis

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Polygon.io API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/x-Kevin-Paul-x/TradingModel.git
   cd TradingModel
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Environment Variables:**
   Create a `.env` file in the project root with your API credentials:
   ```
   POLYGON_API_KEY=your_polygon_api_key
   ```
   Use the provided `.env.example` as a template.

2. **Strategy Parameters:**
   Modify the `config.py` file to adjust:
   - Data ranges (`TRAINING_START_DATE`, `TRAINING_END_DATE`)
   - Trading universe (`TRADING_SYMBOLS`)
   - Capital allocation (`INITIAL_CAPITAL`)
   - Risk parameters (`STOP_LOSS_PCT`, `TAKE_PROFIT_PCT`)
   - Position sizing (`MAX_POSITION_SIZE`, `MAX_HOLDINGS`)

## Using MEtrad

Run the trading simulation with a single command:

```bash
python run.py
```

This initiates the complete workflow:
1. Data acquisition and preprocessing
2. Feature engineering and technical indicator calculation
3. Model training and validation
4. Strategy backtesting and comparison
5. Performance visualization and reporting

Detailed results including total returns, Sharpe ratio, and win rates are provided at the end of the simulation.

## Trading Strategies

MEtrad implements three sophisticated trading strategies:

**Moving Average Crossover (Enhanced)**
- Uses SMA20 and SMA50 crossovers as primary signals
- Applies RSI filters to avoid false breakouts
- Incorporates momentum confirmation for trade direction
- Uses Bollinger Bands for additional entry/exit points

**RSI Strategy (Advanced)**
- Dynamic RSI thresholds that adapt to market volatility
- Volume confirmation requirements for trade execution
- Trend-following and counter-trend modes for different market conditions
- Multiple exit conditions for profit protection

**Momentum Strategy**
- Multi-timeframe momentum analysis (1-day, 5-day, 20-day)
- Acceleration and deceleration measurements
- Volume-weighted momentum calculation
- Volatility normalization for consistent signal generation

## Machine Learning Model

The XGBoost classifier model lies at the heart of MEtrad's predictive capabilities:

- Feature-rich training on 15+ technical indicators
- Automatic hyperparameter optimization
- Feature importance analysis for model interpretability
- SHAP value explanations for prediction transparency
- Cross-validation for robust performance estimation

Models are saved after training, along with feature scalers and importance data.

## Data Processing

MEtrad processes market data through several sophisticated stages:

1. **Acquisition:** Fetching from Polygon.io with smart retry mechanisms
2. **Cleaning:** Handling missing values and outliers
3. **Feature Engineering:** Creating predictive technical indicators
4. **Normalization:** Scaling features for optimal model performance
5. **Splitting:** Creating appropriate train/validation/test datasets

## Risk Management System

MEtrad implements a multi-layered risk management approach:

- **Position-Level Protection:**
  - Stop-Loss orders (configurable percentage)
  - Take-Profit targets (customizable risk/reward ratio)
  - Trailing stops for trend-following strategies

- **Portfolio-Level Protection:**
  - Maximum position sizing (% of capital per position)
  - Diversification rules (maximum holdings per sector)
  - Capital allocation limits

- **Market-Condition Filters:**
  - Minimum volume requirements
  - Volatility-based position sizing
  - Expected return thresholds

## Performance Metrics

MEtrad evaluates trading performance using industry-standard metrics:

- Total Return (%)
- Sharpe Ratio
- Maximum Drawdown
- Win/Loss Ratio
- Average Profit per Trade
- Risk-Adjusted Return

## Contributing

Contributions are welcome! To contribute to MEtrad:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to your branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code follows the project's style guidelines and includes appropriate tests.

