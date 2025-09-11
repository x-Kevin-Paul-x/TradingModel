import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

class VisualizationUtils:
    def _get_dark_template(self):
        """A custom dark theme for Plotly charts."""
        layout = go.Layout(
            template="plotly_dark",
            paper_bgcolor="#0E1117",
            plot_bgcolor="#262730",
            font_color="#FFFFFF",
            title_font_color="#FFFFFF",
            legend_font_color="#FFFFFF",
            xaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
            yaxis=dict(gridcolor='rgba(255, 255, 255, 0.1)'),
        )
        return layout

    def plot_feature_importance(self, feature_importance, title='Feature Importance'):
        """Plot feature importance using Plotly."""
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=feature_importance['importance'],
            y=feature_importance['feature'],
            orientation='h',
            marker_color='#4F8BF9'
        ))
        fig.update_layout(
            title=title,
            xaxis_title='Importance',
            yaxis_title='Feature',
            yaxis=dict(autorange="reversed"),
            **self._get_dark_template().to_plotly_json()
        )
        return fig

    def plot_trading_performance(self, price_data, trades_history=None, title='Trading Performance'):
        """Create an interactive candlestick chart with trade overlays using Plotly."""
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           vertical_spacing=0.05, row_heights=[0.8, 0.2])

        # Candlestick chart
        fig.add_trace(go.Candlestick(x=price_data.index,
                                     open=price_data['open'],
                                     high=price_data['high'],
                                     low=price_data['low'],
                                     close=price_data['close'],
                                     name='Price'), row=1, col=1)

        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                buy_trades = trades_df[trades_df['action'] == 'BUY']
                sell_trades = trades_df[trades_df['action'] == 'SELL']

                fig.add_trace(go.Scatter(x=buy_trades['timestamp'], y=buy_trades['price'],
                                         mode='markers', marker_symbol='triangle-up',
                                         marker=dict(color='#26A69A', size=10, line=dict(width=1, color='DarkSlateGrey')),
                                         name='Buy Signal'), row=1, col=1)

                fig.add_trace(go.Scatter(x=sell_trades['timestamp'], y=sell_trades['price'],
                                         mode='markers', marker_symbol='triangle-down',
                                         marker=dict(color='#EF5350', size=10, line=dict(width=1, color='DarkSlateGrey')),
                                         name='Sell Signal'), row=1, col=1)

        # Volume chart
        fig.add_trace(go.Bar(x=price_data.index, y=price_data['volume'], name='Volume', marker_color='#6A4C93'), row=2, col=1)

        fig.update_layout(
            title=title,
            xaxis_title=None,
            yaxis_title='Price',
            xaxis_rangeslider_visible=False,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            **self._get_dark_template().to_plotly_json()
        )
        fig.update_xaxes(showticklabels=False, row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        return fig

    def plot_portfolio_value(self, trades_history, title='Portfolio Value Over Time'):
        """Plot portfolio value evolution using Plotly."""
        fig = go.Figure()
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'timestamp' in trades_df.columns and 'capital' in trades_df.columns:
                trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
                fig.add_trace(go.Scatter(x=trades_df['timestamp'], y=trades_df['capital'],
                                         mode='lines', line=dict(color='#4F8BF9'),
                                         name='Portfolio Value'))
        
        fig.update_layout(
            title=title,
            xaxis_title='Date',
            yaxis_title='Portfolio Value ($)',
            **self._get_dark_template().to_plotly_json()
        )
        return fig

    def plot_returns_distribution(self, trades_history, title='Returns Distribution'):
        """Plot distribution of returns using Plotly."""
        fig = go.Figure()
        if trades_history:
            trades_df = pd.DataFrame(trades_history)
            if 'capital' in trades_df.columns:
                returns = pd.Series(trades_df['capital']).pct_change().dropna()
                if not returns.empty:
                    fig.add_trace(go.Histogram(x=returns, nbinsx=50, name='Returns', marker_color='#4F8BF9'))
        
        fig.update_layout(
            title=title,
            xaxis_title='Return',
            yaxis_title='Frequency',
            **self._get_dark_template().to_plotly_json()
        )
        return fig
