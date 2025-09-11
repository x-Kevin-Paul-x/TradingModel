import streamlit as st
from datetime import date, timedelta
import time

# --- Page Config ---
st.set_page_config(
    page_title="MEtrad Trading Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Theme ---
if "theme" not in st.session_state:
    st.session_state.theme = "dark"

dark_theme_css = """
<style>
    /* Base */
    html, body, [class*="st-"], .st-emotion-cache-18ni7ap {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    /* Sidebar */
    div[data-testid="stSidebar"] {
        background-color: #161b22;
    }
    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #c9d1d9;
    }
    /* Buttons */
    .stButton>button {
        color: #58a6ff;
        border: 1px solid #58a6ff;
        background-color: transparent;
    }
    .stButton>button:hover {
        background-color: rgba(88, 166, 255, 0.1);
    }
    /* Metrics */
    div[data-testid="stMetric"] {
        background-color: #161b22;
        border-radius: 8px;
        padding: 15px;
    }
    div[data-testid="stMetricLabel"] {
        color: #8b949e;
    }
    /* Expander */
    .st-emotion-cache-1h9usn1 {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
    }
</style>
"""

# Only dark theme is applied as per the new design.
st.markdown(dark_theme_css, unsafe_allow_html=True)

# --- UI Layout ---
# Header
col1, col2 = st.columns([1, 10])
with col1:
    st.image("https://emojicdn.elk.sh/📈", width=60)
with col2:
    st.title("MEtrad Trading Dashboard")


# Sidebar
with st.sidebar:
    st.header("Settings")
    if st.button("Toggle Theme", help="Theme is currently locked to dark for this design."):
        # Theme toggle is disabled to match the single design
        st.toast("Theme is locked to dark mode.", icon="🎨")

    st.markdown("---")
    st.header("Controls")
    symbols_input = st.text_input("Stock(s)", "AAPL,GOOG", help="Enter comma-separated stock symbols, e.g., AAPL,GOOG")

    today = date.today()
    one_year_ago = today - timedelta(days=365)

    start_date_input = st.date_input("Start Date", one_year_ago)
    end_date_input = st.date_input("End Date", today)

    st.markdown("---")
    if st.button("Run Simulation", use_container_width=True):
        st.session_state.run_simulation = True
        st.session_state.symbols = [s.strip().upper() for s in symbols_input.split(',')]
        st.session_state.start_date = start_date_input
        st.session_state.end_date = end_date_input

# --- Main Content ---
st.header("Simulation Results")

# Simulation Log Expander
log_expander = st.expander("Simulation Log", expanded=True)

if 'run_simulation' in st.session_state and st.session_state.run_simulation:
    from simulation_runner import run_simulation

    log_expander.info("Starting simulation...")
    progress_bar = log_expander.progress(0)

    with st.spinner("Running simulation... This may take a few minutes."):
        # Simulate progress
        for i in range(100):
            time.sleep(0.02) # Simulate work
            progress_bar.progress(i + 1)

        results = run_simulation(
            st.session_state.symbols,
            st.session_state.start_date.strftime("%Y-%m-%d"),
            st.session_state.end_date.strftime("%Y-%m-%d")
        )
        st.session_state.results = results
        st.session_state.run_simulation = False
        log_expander.success("Simulation completed!")

if 'results' in st.session_state and st.session_state.results:
    if not ('run_simulation' in st.session_state and st.session_state.run_simulation):
        log_expander.success("Simulation complete!")

    results = st.session_state.results
    bot_metrics = results['bot_metrics']

    # --- Key Performance Metrics ---
    st.markdown("---")
    st.header("Key Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Return", f"{bot_metrics['total_return']*100:.2f}%")
    col2.metric("Sharpe Ratio", f"{bot_metrics['sharpe_ratio']:.2f}")
    col3.metric("Max Drawdown", f"{bot_metrics['max_drawdown']*100:.2f}%")
    col4.metric("Winning Trades", f"{bot_metrics['winning_trades']}/{bot_metrics['total_trades']}")

    # --- Visualizations ---
    st.markdown("---")
    st.header("Visualizations")

    from utils.visualization import VisualizationUtils
    visualizer = VisualizationUtils()

    # Create and display charts
    for symbol in results['test_data']:
        st.subheader(f"Results for {symbol}")

        candlestick_fig = visualizer.plot_trading_performance(
            results['test_data'][symbol],
            results['trading_bot'].trades_history
        )
        st.plotly_chart(candlestick_fig, use_container_width=True)

    st.subheader("Overall Performance")

    # Portfolio value
    portfolio_fig = visualizer.plot_portfolio_value(results['trading_bot'].trades_history)
    st.plotly_chart(portfolio_fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        feature_importance_fig = visualizer.plot_feature_importance(results['ml_model'].get_top_features())
        st.plotly_chart(feature_importance_fig, use_container_width=True)

    with col2:
        returns_dist_fig = visualizer.plot_returns_distribution(results['trading_bot'].trades_history)
        st.plotly_chart(returns_dist_fig, use_container_width=True)

else:
    log_expander.write("Click 'Run Simulation' to see the results.")

st.sidebar.info("Please configure your Polygon API key in a `.env` file in the project root to run the simulation.", icon="ℹ️")
