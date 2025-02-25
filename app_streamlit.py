import streamlit as st
import io
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import pandas as pd
import pickle
import os
import base64
import hashlib
from datetime import datetime
import csv  # Added for accuracy loading

# Define SECRET_VAULT to match train_model.py
SECRET_VAULT = "my_star_vault"

# Clear Streamlit cache to ensure new graph loads
st.cache_data.clear()

# Your personal proprietary key
MY_PERSONAL_SECRET = hashlib.sha256("5YearOldCoderStar".encode()).hexdigest()[:8]  # "d8e8fca2"

# Load external files safely (like styles.css)
def load_external_file(file_name):
    try:
        if os.path.exists(file_name):
            with open(file_name) as f:
                return f.read()
        else:
            st.warning(f"Couldn’t find {file_name}. Using default style!")
            return ""
    except Exception as e:
        st.error(f"Oops! Something went wrong with {file_name}: {e}")
        return ""

# Load your styles.css or use default
css = load_external_file('styles.css')
if css:
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
else:
    st.markdown("""
    <style>
        h1 {color: purple; font-size: 36px;}
        h2 {color: orange; font-size: 24px;}
        .stMetric {border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px; padding: 15px; background: #ffffff; border: 1px solid #e0e0e0;}
        .stMetric > div {font-size: 12px; color: #333;}
        .stMetric > div > span {font-size: 24px; color: #000;}
        .stImage > img {border-radius: 12px; box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1); margin: 10px;}
        .main {padding: 20px;}
    </style>
    """, unsafe_allow_html=True)

# Available Stocks and Forex Pairs
stocks = {
    "AAPL": "Apple Inc.",
    "GOOGL": "Alphabet Inc.",
    "MSFT": "Microsoft Corp.",
    "ZOMATO.NS": "Zomato Ltd.",
    "360ONE.NS": "360 One Wam Limited."
}

forex = {
    "EURUSD=X": "Euro / US Dollar",
    "GBPUSD=X": "British Pound / US Dollar",
    "USDJPY=X": "US Dollar / Japanese Yen",
    "USDINR=X": "US Dollar / Indian Rupee"
}

# Time Periods
time_periods = {15: "15d", 30: "30d", 180: "180d", 365: "1y"}  # Adjusted for display

# Decrypt your StarForge models
def decrypt_star_model(filename):
    try:
        with open(filename, "rb") as f:
            encrypted = base64.b64decode(f.read())
            key_bytes = bytes.fromhex(MY_PERSONAL_SECRET)
            decrypted = bytes(a ^ b for a, b in zip(encrypted, key_bytes * (len(encrypted) // len(key_bytes) + 1)))
            return pickle.loads(decrypted)  # Use pickle.loads() for bytes
    except Exception as e:
        st.error(f"Can’t unlock StarForge model: {e}")
        return None

# Load trained models with accuracy from training
def load_trained_models():
    horizons = [15, 30, 180, 365]
    stock_models = {}
    forex_models = {}
    stock_accuracies = {}  # Store stock accuracies from training
    forex_accuracies = {}  # Store forex accuracies from training
    for h in horizons:
        stock_models[h] = decrypt_star_model(f"{SECRET_VAULT}/starforge_stock_h{h}.pkl")
        forex_models[h] = decrypt_star_model(f"{SECRET_VAULT}/starforge_forex_h{h}.pkl")
        # Load separate accuracies for stock and forex from training output
        stock_accuracies[h] = get_accuracy_from_training("stock", h)  # Get stock accuracy
        forex_accuracies[h] = get_accuracy_from_training("forex", h)  # Get forex accuracy
    return stock_models, forex_models, stock_accuracies, forex_accuracies

# Get accuracy from training with better error handling
def get_accuracy_from_training(asset_type, horizon):
    try:
        if os.path.exists(f"{SECRET_VAULT}/accuracies.csv"):
            with open(f"{SECRET_VAULT}/accuracies.csv", "r", encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if int(row["Horizon"]) == horizon:
                        accuracy = float(row["Stock_Accuracy"]) if asset_type == "stock" else float(row["Forex_Accuracy"])
                        return accuracy if not np.isnan(accuracy) else "N/A"
            st.warning(f"No accuracy found for horizon {horizon} in {asset_type}. Using defaults.")
        else:
            st.warning(f"Accuracies file not found at {SECRET_VAULT}/accuracies.csv. Using defaults.")
    except (FileNotFoundError, PermissionError, ValueError) as e:
        st.warning(f"Error loading accuracies: {e}. Using defaults.")
    
    # Default accuracies (update with real values if needed)
    stock_accuracies = {
        15: 85.5,  # Replace with real Stock accuracy for 15 days
        30: 82.3,  # Replace with real Stock accuracy for 30 days
        180: 78.9, # Replace with real Stock accuracy for 180 days
        365: 75.2  # Replace with real Stock accuracy for 365 days
    }
    forex_accuracies = {
        15: 88.2,  # Replace with real Forex accuracy for 15 days
        30: 84.7,  # Replace with real Forex accuracy for 30 days
        180: 80.5, # Replace with real Forex accuracy for 180 days
        365: 77.1  # Replace with real Forex accuracy for 365 days
    }
    return stock_accuracies.get(horizon, "N/A") if asset_type == "stock" else forex_accuracies.get(horizon, "N/A")

# Get USD to INR exchange rate (fix warning, realistic conversion, fix scaling, handle Series ambiguity)
def get_usd_to_inr():
    try:
        data = yf.download("USDINR=X", period="1d", progress=False, auto_adjust=True)
        # Ensure data is a DataFrame and handle empty or missing data
        if data is None or data.empty:  # Explicitly check if data is None or empty
            return 83.50  # Default without warning
        if "Close" not in data.columns:  # Check if 'Close' column exists
            return 83.50  # Default without warning
        rate = float(data["Close"].iloc[-1].item())  # Use latest close for current price, avoiding FutureWarning
        if rate < 50 or rate > 100:  # Sanity check for unrealistic rates
            return 83.50  # Default without warning
        return rate
    except Exception as e:
        return 83.50  # Default without warning

# Fetch real-time data with current price (fix warning, no twist for forex, fix scaling, handle Series ambiguity)
def get_real_time_data(symbol, period="5y"):
    try:
        data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
        # Ensure data is a DataFrame and handle empty or missing data
        if data is None or data.empty:  # Explicitly check if data is None or empty
            return None, None, []
        if "Close" not in data.columns or len(data) < 200:  # Check if 'Close' column exists and has enough data
            return None, None, []
        # No twist for forex (USDINR=X) to keep realistic, twist for stocks (GOOGL/ZOMATO.NS)
        if "USDINR=X" in symbol or any(fx in symbol for fx in ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]):
            data["Close"] = data["Close"]  # No twist for forex
        else:
            data["Close"] = data["Close"] * (1 + np.sin(len(data) % 5) * 0.05)  # Your twist for stocks
        current_price = float(data["Close"].iloc[-1].item())  # Use latest close for current price, avoiding FutureWarning
        if "USDINR=X" in symbol and (current_price < 50 or current_price > 100):  # Sanity check for forex
            current_price = get_usd_to_inr()  # Use current USD to INR rate silently
        return current_price, float(data["Close"].iloc[-1].item()), data["Close"].values.tolist()  # Use latest close for current price, return scalar current price, latest close, and price history as list
    except Exception as e:
        return None, None, []

# Fetch additional fundamental data (PE, EPS) using yfinance for prediction
def fetch_fundamental_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        pe_ratio = info.get('trailingPE', np.nan)  # Trailing PE ratio
        eps = info.get('trailingEps', np.nan)  # Trailing 12-month EPS
        return pd.Series({'PE_Ratio': pe_ratio, 'EPS_TTM': eps})
    except Exception as e:
        return pd.Series({'PE_Ratio': np.nan, 'EPS_TTM': np.nan})

# Calculate technical indicators (Support/Resistance, RSI) for prediction
def calculate_technical_indicators(data):
    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        close_col = ('Close', data.columns.get_level_values(1)[0]) if ('Close', data.columns.get_level_values(1)[0]) in data.columns else 'Close'
    else:
        close_col = 'Close'

    # Ensure Close column is numeric
    if not np.issubdtype(data[close_col].dtype, np.number):
        data[close_col] = pd.to_numeric(data[close_col], errors='coerce').astype(float)

    # Support and Resistance (simple moving average-based)
    data['MA20'] = data[close_col].rolling(window=20).mean()  # 20-day moving average for support/resistance
    data['MA50'] = data[close_col].rolling(window=50).mean()  # 50-day moving average
    support = data['MA20'].rolling(window=5).min().shift(1)
    resistance = data['MA50'].rolling(window=5).max().shift(1)
    data['Support'] = support
    data['Resistance'] = resistance

    # RSI (Relative Strength Index, 14-day period)
    delta = data[close_col].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))
    data['RSI'] = data['RSI'].fillna(50)  # Fill NaN with neutral RSI (50)

    return data

# Simulate buyer/seller activity (volume-based proxy) for prediction
def calculate_buyer_seller_activity(data):
    # Handle multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        volume_col = ('Volume', data.columns.get_level_values(1)[0]) if ('Volume', data.columns.get_level_values(1)[0]) in data.columns else 'Volume'
    else:
        volume_col = 'Volume'

    if volume_col not in data.columns:
        return data.assign(Buyer_Seller=np.nan)  # Default to NaN if Volume is missing

    # Ensure Volume column is numeric
    if not np.issubdtype(data[volume_col].dtype, np.number):
        data[volume_col] = pd.to_numeric(data[volume_col], errors='coerce').astype(float)

    # Align Volume and Volume_MA10 before comparison
    data['Volume_MA10'] = data[volume_col].rolling(window=10, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
    volume_aligned, volume_ma10_aligned = data[volume_col].align(data['Volume_MA10'], join='inner', axis=0)

    # Ensure 1-dimensional Series by squeezing
    volume_aligned = volume_aligned.squeeze()
    volume_ma10_aligned = volume_ma10_aligned.squeeze()

    # Drop NaN values and create Buyer_Seller
    clean_data = pd.DataFrame({
        'Volume': volume_aligned,
        'Volume_MA10': volume_ma10_aligned
    }).dropna()
    
    valid_mask = pd.notna(clean_data['Volume']) & pd.notna(clean_data['Volume_MA10'])
    comparison = clean_data['Volume'] > clean_data['Volume_MA10']
    
    # Create Buyer_Seller column with aligned data
    buyer_seller = pd.Series(np.where(valid_mask & comparison, 1, -1), index=clean_data.index).reindex(data.index, fill_value=-1)
    data['Buyer_Seller'] = buyer_seller
    
    return data

# Prepare features (matches train_model.py, ensure no scaling issues, updated for new features and multi-index)
def prepare_prediction_data(prices, symbol):
    if not isinstance(prices, (list, np.ndarray)):
        return None
    df = pd.DataFrame(prices, columns=["Close"])
    if len(df) < 200:
        return None
    
    # Fetch additional data for fundamental and technical indicators
    full_data = yf.download(symbol, period="5y", progress=False, auto_adjust=True)
    if full_data is None or full_data.empty or len(full_data) < 200 or "Close" not in full_data.columns:
        return None
    
    # Handle multi-index columns
    if isinstance(full_data.columns, pd.MultiIndex):
        close_col = ('Close', symbol) if ('Close', symbol) in full_data.columns else 'Close'
    else:
        close_col = 'Close'

    # Ensure Close column is numeric and handle any non-numeric values
    if not np.issubdtype(full_data[close_col].dtype, np.number):
        full_data[close_col] = pd.to_numeric(full_data[close_col], errors='coerce').astype(float)

    # Apply the same twist as in train_model.py for stocks, not forex
    if "USDINR=X" not in symbol and not any(fx in symbol for fx in ["EURUSD=X", "GBPUSD=X", "USDAJPY=X"]):
        data_length = int(len(full_data))
        twist_factor = 1 + float(np.sin(data_length % 5) * 0.05)
        full_data[close_col] = full_data[close_col] * twist_factor
    
    # Add fundamental data
    fundamentals = fetch_fundamental_data(symbol)
    if isinstance(full_data.columns, pd.MultiIndex):
        full_data[('PE_Ratio', '')] = fundamentals['PE_Ratio']
        full_data[('EPS_TTM', '')] = fundamentals['EPS_TTM']
    else:
        full_data['PE_Ratio'] = fundamentals['PE_Ratio']
        full_data['EPS_TTM'] = fundamentals['EPS_TTM']
    
    # Add technical indicators
    full_data = calculate_technical_indicators(full_data)
    full_data = calculate_buyer_seller_activity(full_data)

    # Calculate proprietary features, handling multi-index
    seasonal_factor = np.cos(len(full_data) % 365 / 365 * 2 * np.pi)
    if isinstance(full_data.columns, pd.MultiIndex):
        full_data[('StarReturn', '')] = full_data[close_col].pct_change(periods=10) * seasonal_factor
        full_data[('OrbitVol', '')] = full_data[close_col].rolling(window=30).std() / full_data[close_col].rolling(window=90).mean()
        full_data[('GalaxyMomentum', '')] = (full_data[close_col] - full_data[close_col].shift(60)) * (full_data[close_col].shift(5) / full_data[close_col].shift(10))
        trend = ((full_data[close_col] / full_data[close_col].rolling(window=200).mean()) ** 5).squeeze()
        momentum = np.tanh(full_data[close_col].pct_change().rolling(window=15).sum()).squeeze()
        cycle = pd.Series((pd.to_datetime(full_data.index).month % 5) + 1, index=full_data.index)
        five_magic = (trend * momentum * cycle).rename("FiveMagic")
        full_data[('FiveMagic', '')] = five_magic
    else:
        full_data["StarReturn"] = full_data["Close"].pct_change(periods=10) * seasonal_factor
        full_data["OrbitVol"] = full_data["Close"].rolling(window=30).std() / full_data["Close"].rolling(window=90).mean()
        full_data["GalaxyMomentum"] = (full_data["Close"] - full_data["Close"].shift(60)) * (full_data["Close"].shift(5) / full_data["Close"].shift(10))
        trend = ((full_data["Close"] / full_data["Close"].rolling(window=200).mean()) ** 5).squeeze()
        momentum = np.tanh(full_data["Close"].pct_change().rolling(window=15).sum()).squeeze()
        cycle = pd.Series((pd.to_datetime(full_data.index).month % 5) + 1, index=full_data.index)
        five_magic = (trend * momentum * cycle).rename("FiveMagic")
        full_data["FiveMagic"] = five_magic

    # Ensure all features are present, handling multi-index
    features = ["StarReturn", "OrbitVol", "GalaxyMomentum", "FiveMagic", "PE_Ratio", "EPS_TTM", "Support", "Resistance", "RSI", "Buyer_Seller"]
    if isinstance(full_data.columns, pd.MultiIndex):
        multi_index_features = [(f, '') for f in features]
        for feature, multi_feature in zip(features, multi_index_features):
            if multi_feature not in full_data.columns:
                full_data[multi_feature] = np.nan
    else:
        for feature in features:
            if feature not in full_data.columns:
                full_data[feature] = np.nan

    # Drop NaN values and get the latest features, handling multi-index
    if isinstance(full_data.columns, pd.MultiIndex):
        existing_features = [col for col in multi_index_features if col in full_data.columns and full_data[col].notna().any()]
        if not existing_features:
            return None
        full_data = full_data.dropna(subset=existing_features)
    else:
        existing_features = [f for f in features if f in full_data.columns and full_data[f].notna().any()]
        if not existing_features:
            return None
        full_data = full_data.dropna(subset=existing_features)

    if len(full_data) < 200:
        return None

    # Prepare features for prediction (latest row), handling multi-index
    if isinstance(full_data.columns, pd.MultiIndex):
        prepared_features = full_data[multi_index_features].iloc[-1].values.reshape(1, -1)  # Reshape to match train_model.py's 10 features
    else:
        prepared_features = full_data[features].iloc[-1].values.reshape(1, -1)  # Reshape to match train_model.py's 10 features
    return prepared_features

# Predict future price with real accuracy, fix direction/recommendation, pass symbol, and show percent change
def predict_future_price(prices, model, horizon, stock_accuracies, forex_accuracies, is_stock=True, symbol=None):
    if not isinstance(prices, (list, np.ndarray)):
        return None, None, None, None
    if len(prices) < 200:
        return None, None, None, None
    try:
        features = prepare_prediction_data(prices, symbol)
        if features is None:
            return None, None, None, None
        # Ensure features shape matches the model's expectation (10 features)
        if features.shape[1] != 10:
            return None, None, None, None
        predicted_change = model.predict(features)[0]  # Predicted percent change
        if isinstance(predicted_change, np.ndarray):
            predicted_change = predicted_change.item()  # Convert ndarray to scalar
        # Ensure current_price is a scalar float from the last price in the list
        current_price = float(prices[-1]) if isinstance(prices[-1], (int, float)) else float(prices[-1][0]) if isinstance(prices[-1], (list, np.ndarray)) else float(prices[-1].item()) if isinstance(prices[-1], pd.Series) else None
        if current_price is None:
            return None, None, None, None
        direction = "Up" if predicted_change > 0 else "Down"
        recommendation = "Buy" if direction == "Up" else "Sell"
        accuracy = stock_accuracies[horizon] if is_stock else forex_accuracies[horizon]
        return predicted_change, f"{accuracy:.2f}%", direction, f"{predicted_change:.2f}%"
    except Exception as e:
        return None, None, None, None

# Generate enhanced graph with symbol (force new version)
@st.cache_data(show_spinner=False)  # Cache but disable spinner to avoid old graph
def generate_prediction_graph(prices, symbol, horizon, predicted_change):
    if not prices or len(prices) < 1:
        return None
    try:
        # Ensure prices is a numpy array for numerical operations
        prices = np.array(prices, dtype=float)
        current_price = prices[-1]
        predicted_price = current_price * (1 + predicted_change / 100)  # Convert percent change to absolute price
        plt.figure(figsize=(10, 5))  # Larger for clarity
        plt.plot(prices, marker='o', linestyle='-', color='#007AFF', linewidth=2, markersize=6, label="Historical Prices")
        # Plot current and predicted prices as horizontal lines
        plt.axhline(y=current_price, color='red', linestyle='--', label="Current Price")
        plt.axhline(y=predicted_price, color='green', linestyle='--', label="Predicted Price")
        plt.title(f"Price Trend for {symbol} (Next {horizon} Days)", fontsize=14, color='#333333', pad=15, fontweight='bold')
        plt.xlabel("Time (Days)", fontsize=12, color='#333333')
        plt.ylabel("Price (INR)" if "USDINR=X" in symbol or any(fx in symbol for fx in ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]) else "Price (INR if applicable)", fontsize=12, color='#333333')
        plt.grid(True, linestyle='--', alpha=0.7, color='#cccccc')
        plt.legend()
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', transparent=False, dpi=150, facecolor='white')  # High resolution, white background
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        return None

# App Title (Reverted to Original Purple/Orange)
st.markdown("<h1 style='font-size: 36px; color: #4c00b0;'>StarForge</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='font-size: 24px; color: Black;'>Predict the Future - Stock and Forex predictor</h2>", unsafe_allow_html=True)
# st.markdown("<div class='rounded-body'>", unsafe_allow_html=True)  # Use rounded-body class for responsive, rounded content

# Fetch USD to INR
usd_to_inr = get_usd_to_inr()

# Sidebar
st.sidebar.title("Click to Select Stock or Forex")
stock_options = [f"{symbol} - {name}" for symbol, name in stocks.items()]
selected_stock = st.sidebar.selectbox("Select a Stock:", [""] + stock_options)
forex_options = [f"{symbol.replace('=X', '')} - {name}" for symbol, name in forex.items()]
selected_forex = st.sidebar.selectbox("Pick a Forex:", [""] + forex_options)
selected_period = st.sidebar.selectbox("Predict How Far?", list(time_periods.keys()), format_func=lambda x: f"{x} days")

# Load models and accuracies
stock_model, forex_model, stock_accuracies, forex_accuracies = load_trained_models()

# Predict button
if st.sidebar.button("Predict"):
    if not selected_stock and not selected_forex:
        st.warning("Pick at least one stock or forex pair to predict!")
    else:
        with st.spinner("Forging predictions..."):
            horizon = selected_period
            show_stock = False
            show_forex = False

            # Process and display stock prediction if selected
            if selected_stock and stock_model.get(horizon):
                stock_symbol = selected_stock.split(" - ")[0]
                current_price, _, stock_prices = get_real_time_data(stock_symbol)
                if stock_prices:
                    predicted_change, accuracy, direction, percent_change = predict_future_price(stock_prices, stock_model[horizon], horizon, stock_accuracies, forex_accuracies, is_stock=True, symbol=stock_symbol)
                    if predicted_change is not None:
                        st.markdown(f"<h3 style='color: #2ecc71;'>📈 Stock Prediction ({horizon} days) for {stock_symbol}</h3>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        col1.markdown(f"<div class='stMetric'><div>Current Price</div><span>₹{current_price:.2f}</span></div>", unsafe_allow_html=True)
                        col1.markdown(f"<div class='stMetric'><div>Predicted Change</div><span>{percent_change}</span></div>", unsafe_allow_html=True)
                        col2.markdown(f"<div class='stMetric'><div>Accuracy</div><span>{accuracy}</span></div>", unsafe_allow_html=True)
                        col3.markdown(f"<div class='stMetric'><div>Direction</div><span>{direction}</span></div>", unsafe_allow_html=True)
                        col4.markdown(f"<div class='stMetric'><div>Recommendation</div><span>{'Buy' if direction == 'Up' else 'Sell'}</span></div>", unsafe_allow_html=True)
                        with st.expander(f"Raw Data for {stock_symbol}"):
                            st.write(f"Raw data for {stock_symbol}:")
                            st.write(pd.DataFrame({"Date": pd.date_range(end=datetime.today(), periods=len(stock_prices), freq="D"), "Price (INR)": stock_prices}))
                        graph = generate_prediction_graph(stock_prices, stock_symbol, horizon, predicted_change)
                        if graph:
                            st.image(graph, use_container_width=True)
                        show_stock = True

            # Process and display forex prediction if selected
            if selected_forex and forex_model.get(horizon):
                forex_symbol = selected_forex.split(" - ")[0] + "=X"
                current_price, _, forex_prices = get_real_time_data(forex_symbol)
                if forex_prices:
                    predicted_change, accuracy, direction, percent_change = predict_future_price(forex_prices, forex_model[horizon], horizon, stock_accuracies, forex_accuracies, is_stock=False, symbol=forex_symbol)
                    if predicted_change is not None:
                        if show_stock:
                            st.markdown("<div class='stock-forex-separator'></div>", unsafe_allow_html=True)  # Add separator after stock if forex follows
                        st.markdown(f"<h3 style='color: #2ecc71;'>💹 Forex Prediction ({horizon} days) for {forex_symbol}</h3>", unsafe_allow_html=True)
                        col1, col2, col3, col4 = st.columns(4)
                        col1.markdown(f"<div class='stMetric'><div>Current Price</div><span>₹{current_price:.2f}</span></div>", unsafe_allow_html=True)
                        col1.markdown(f"<div class='stMetric'><div>Predicted Change</div><span>{percent_change}</span></div>", unsafe_allow_html=True)
                        col2.markdown(f"<div class='stMetric'><div>Accuracy</div><span>{accuracy}</span></div>", unsafe_allow_html=True)
                        col3.markdown(f"<div class='stMetric'><div>Direction</div><span>{direction}</span></div>", unsafe_allow_html=True)
                        col4.markdown(f"<div class='stMetric'><div>Recommendation</div><span>{'Buy' if direction == 'Up' else 'Sell'}</span></div>", unsafe_allow_html=True)
                        with st.expander(f"Raw Data for {forex_symbol}"):
                            st.write(f"Raw data for {forex_symbol}:")
                            st.write(pd.DataFrame({"Date": pd.date_range(end=datetime.today(), periods=len(forex_prices), freq="D"), "Price (INR)": forex_prices}))
                        graph = generate_prediction_graph(forex_prices, forex_symbol, horizon, predicted_change)
                        if graph:
                            st.image(graph, use_container_width=True)
                        show_forex = True

            if show_stock and not show_forex:
                st.markdown("<div class='stock-forex-separator'></div>", unsafe_allow_html=True)  # Add separator after stock if no forex

        st.balloons()

st.markdown("</div>", unsafe_allow_html=True)  # Close rounded-body div

# Footer
st.markdown("---")
st.markdown("<div class='main'>", unsafe_allow_html=True)
disclaimer = load_external_file("disclaimer.html")
if disclaimer:
    st.markdown(disclaimer, unsafe_allow_html=True)
else:
    st.markdown("<p style='color: gray; font-size: 14px;'>StarForge predicts, but the future’s still a mystery! Past performance isn’t indicative of future results. Invest wisely.</p>", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.write("Enter the Stock name or Forex for which you want to see the prediction.Please enter your mobile and email for further communication on your requirement")
feedback = st.text_area("Please enter the stock or forex or both.", height=100)
if st.button("Submit Request"):
    st.write("Thank you for the input. We will Reach out soon")
