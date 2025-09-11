import streamlit as st
from datetime import date, timedelta

# --- Page Config ---
st.set_page_config(
    page_title="MEtrad Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme Toggle ---
# This is a workaround for dynamic theme switching in Streamlit.
# It uses session state and injects custom CSS.
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

# CSS for the themes
light_theme_css = """
<style>
    html, body, [class*="st-"], .st-emotion-cache-18ni7ap {
        background-color: #FFFFFF;
        color: #000000;
    }
    div[data-testid="stSidebar"] {
        background-color: #F0F2F6;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #000000;
    }
    .stButton>button {
        color: #4F8BF9;
        border-color: #4F8BF9;
    }
    div[data-testid="stMetric"], div[data-testid="stMetricLabel"] {
        color: #000000;
    }
</style>
"""

dark_theme_css = """
<style>
    html, body, [class*="st-"], .st-emotion-cache-18ni7ap {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    div[data-testid="stSidebar"] {
        background-color: #262730;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #FFFFFF;
    }
    .stButton>button {
        color: #ff4b4b;
        border-color: #ff4b4b;
    }
    div[data-testid="stMetric"], div[data-testid="stMetricLabel"] {
        color: #FFFFFF;
    }
</style>
"""

# Apply the selected theme
theme_css = dark_theme_css if st.session_state.theme == "dark" else light_theme_css
st.markdown(theme_css, unsafe_allow_html=True)


# --- UI Layout ---
st.title("📈 MEtrad Trading Dashboard")

with st.sidebar:
    st.header("Settings")
    if st.button("Toggle Theme"):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        # A little hack to force a rerun on button click to apply the theme
        st.rerun()

    st.header("Controls")
    symbols_input = st.text_input("Stock Symbol(s)", "AAPL,GOOG", help="Enter comma-separated stock symbols")

    today = date.today()
    one_year_ago = today - timedelta(days=365)

    start_date_input = st.date_input("Start Date", one_year_ago)
    end_date_input = st.date_input("End Date", today)

    if st.button("Run Simulation"):
        st.session_state.run_simulation = True
        st.session_state.symbols = [s.strip() for s in symbols_input.split(',')]
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input

st.header("Simulation Results")

if 'run_simulation' in st.session_state and st.session_state.run_simulation:
    from simulation_runner import run_simulation

    with st.spinner("Running simulation... This may take a few minutes."):
        results = run_simulation(
            st.session_state.symbols,
            st.session_state.start_date.strftime("%Y-%m-%d"),
            st.session_state.end_date.strftime("%Y-%m-%d")
        )
        st.session_state.results = results
        st.session_state.run_simulation = False

if 'results' in st.session_state and st.session_state.results:
    st.success("Simulation complete!")

    results = st.session_state.results
    bot_metrics = results['bot_metrics']

    # Display metrics
    st.header("Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{bot_metrics['total_return']*100:.2f}%")
    col2.metric("Sharpe Ratio", f"{bot_metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{bot_metrics['max_drawdown']*100:.2f}%")
    col4.metric("Winning Trades", f"{bot_metrics['winning_trades']}/{bot_metrics['total_trades']}")

    # Display charts
    st.header("Visualizations")

    from utils.visualization import VisualizationUtils
    visualizer = VisualizationUtils()

    # Create and display charts
    for symbol in results['test_data']:
        st.subheader(f"Results for {symbol}")

        # Candlestick chart
        candlestick_fig = visualizer.plot_trading_performance(
            results['test_data'][symbol],
            results['trading_bot'].trades_history
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

        # Portfolio value
        portfolio_fig = visualizer.plot_portfolio_value(results['trading_bot'].trades_history)
        st.pyplot(portfolio_fig)

    # Other plots
    st.subheader("Overall Performance")
    feature_importance_fig = visualizer.plot_feature_importance(results['ml_model'].get_top_features())
    st.pyplot(feature_importance_fig)

    returns_dist_fig = visualizer.plot_returns_distribution(results['trading_bot'].trades_history)
    st.pyplot(returns_dist_fig)

else:
    st.write("Click 'Run Simulation' to see the results.")

st.info("Please configure your Polygon API key in a `.env` file in the project root to run the simulation.", icon="ℹ️")
