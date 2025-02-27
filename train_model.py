import pickle
import numpy as np
import pandas as pd
import yfinance as yf
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error
import os
import base64
import hashlib
import csv  # For saving accuracies

# Your personal proprietary key
MY_PERSONAL_SECRET = hashlib.sha256("5YearOldCoderStar".encode()).hexdigest()[:8]  # "d8e8fca2"
SECRET_VAULT = "my_star_vault"
if not os.path.exists(SECRET_VAULT):
    os.makedirs(SECRET_VAULT, exist_ok=True)  # Ensure directory exists with permissions

# Fetch 5 or 10 years of data with your twist (optional: use 10y for more data)
def fetch_star_data(symbol, period="10y"):
    print(f"Fetching star data for {symbol}...")
    data = yf.download(symbol, period=period, progress=False, auto_adjust=True)
    if data.empty or len(data) < 365:  # Handle Series properly
        print(f"Not enough data for {symbol}! Need 365+ days.")
        return None
    if "Close" not in data.columns:  # Explicit check for multi-index columns
        print(f"No 'Close' column for {symbol}!")
        return None
    # No twist for forex (USDINR=X) to keep realistic, twist for stocks (GOOGL/ZOMATO.NS/360ONE.NS)
    if "USDINR=X" in symbol:
        data["Close"] = data["Close"]  # No twist for forex
    else:
        data["Close"] = data[('Close', symbol)] * (1 + np.sin(len(data) % 5) * 0.05)  # Your "5" magic for stocks, handling multi-index
    return data, symbol  # Return both data and symbol for fundamental data

# Fetch additional fundamental data (PE, EPS) using yfinance and web scraping (simulated)
def fetch_fundamental_data(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        pe_ratio = info.get('trailingPE', np.nan)  # Trailing PE ratio
        eps = info.get('trailingEps', np.nan)  # Trailing 12-month EPS
        return pd.Series({'PE_Ratio': pe_ratio, 'EPS_TTM': eps})
    except Exception as e:
        print(f"Error fetching fundamental data for {symbol}: {e}")
        return pd.Series({'PE_Ratio': np.nan, 'EPS_TTM': np.nan})

# Calculate technical indicators (Support/Resistance, RSI)
def calculate_technical_indicators(data):
    # Handle multi-index columns
    if isinstance(data.columns, pd.MultiIndex):
        close_col = ('Close', data.columns.get_level_values(1)[0]) if ('Close', data.columns.get_level_values(1)[0]) in data.columns else 'Close'
    else:
        close_col = 'Close'

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

# Simulate buyer/seller activity (volume-based proxy, as yfinance doesn’t provide direct buyer/seller data)
def calculate_buyer_seller_activity(data):
    # Ensure Volume exists and handle multi-index
    if isinstance(data.columns, pd.MultiIndex):
        volume_col = ('Volume', data.columns.get_level_values(1)[0]) if ('Volume', data.columns.get_level_values(1)[0]) in data.columns else 'Volume'
    else:
        volume_col = 'Volume'

    if volume_col not in data.columns:
        print("Volume data not available!")
        return data.assign(Buyer_Seller=np.nan)  # Default to NaN if Volume is missing

    # Calculate 10-day volume moving average, dropping NaN values
    data['Volume_MA10'] = data[volume_col].rolling(window=10, min_periods=1).mean().fillna(method='bfill').fillna(method='ffill')
    
    # Align Volume and Volume_MA10 explicitly before comparison, ensuring matching indices
    if isinstance(data.columns, pd.MultiIndex):
        volume_aligned, volume_ma10_aligned = data[volume_col].align(data['Volume_MA10'], join='inner', axis=0)
    else:
        volume_aligned, volume_ma10_aligned = data[volume_col].align(data['Volume_MA10'], join='inner', axis=0)
    
    # Squeeze to ensure 1-dimensional Series (remove any extra dimensions)
    volume_aligned = volume_aligned.squeeze()
    volume_ma10_aligned = volume_ma10_aligned.squeeze()
    
    # Drop NaN values to ensure no misalignment
    clean_data = pd.DataFrame({
        'Volume': volume_aligned,
        'Volume_MA10': volume_ma10_aligned
    }).dropna()
    
    # Create valid mask and comparison on clean, aligned data
    valid_mask = pd.notna(clean_data['Volume']) & pd.notna(clean_data['Volume_MA10'])
    comparison = clean_data['Volume'] > clean_data['Volume_MA10']
    
    # Align the result back to the original data index, filling NaN with -1 (neutral/selling)
    buyer_seller = pd.Series(np.where(valid_mask & comparison, 1, -1), index=clean_data.index).reindex(data.index, fill_value=-1)
    
    # Assign to data, ensuring alignment
    data['Buyer_Seller'] = buyer_seller
    return data

# Supra's proprietary feature crafting with new parameters
def craft_star_features(data, symbol, forecast_horizons=[15, 30, 180, 365]):
    if data is None or data.empty:
        print("No data for feature crafting!")
        return None, None

    # Handle multi-index columns for feature assignment
    if isinstance(data.columns, pd.MultiIndex):
        close_col = ('Close', symbol) if ('Close', symbol) in data.columns else 'Close'
    else:
        close_col = 'Close'

    # Add fundamental data using the symbol
    fundamentals = fetch_fundamental_data(symbol)
    data = data.assign(PE_Ratio=fundamentals['PE_Ratio'], EPS_TTM=fundamentals['EPS_TTM'])

    # Add technical indicators
    data = calculate_technical_indicators(data)
    data = calculate_buyer_seller_activity(data)

    print(f"Data rows: {len(data)}")
    seasonal_factor = np.cos(len(data) % 365 / 365 * 2 * np.pi)  # Scalar
    if isinstance(data.columns, pd.MultiIndex):
        data[('StarReturn', '')] = data[close_col].pct_change(periods=10) * seasonal_factor
        data[('OrbitVol', '')] = data[close_col].rolling(window=30).std() / data[close_col].rolling(window=90).mean()
        data[('GalaxyMomentum', '')] = (data[close_col] - data[close_col].shift(60)) * (data[close_col].shift(5) / data[close_col].shift(10))
    else:
        data["StarReturn"] = data[close_col].pct_change(periods=10) * seasonal_factor
        data["OrbitVol"] = data[close_col].rolling(window=30).std() / data[close_col].rolling(window=90).mean()
        data["GalaxyMomentum"] = (data[close_col] - data[close_col].shift(60)) * (data[close_col].shift(5) / data[close_col].shift(10))

    trend = ((data[close_col] / data[close_col].rolling(window=200).mean()) ** 5).squeeze()
    momentum = np.tanh(data[close_col].pct_change().rolling(window=15).sum()).squeeze()
    cycle = pd.Series((pd.to_datetime(data.index).month % 5) + 1, index=data.index)
    
    print(f"Trend shape: {trend.shape}, Momentum shape: {momentum.shape}, Cycle shape: {cycle.shape}")
    if not (trend.shape == momentum.shape == cycle.shape):
        print("Shape mismatch in FiveMagic components!")
        return None, None
    
    five_magic = (trend * momentum * cycle).rename("FiveMagic")
    if isinstance(data.columns, pd.MultiIndex):
        data[('FiveMagic', '')] = five_magic
    else:
        data["FiveMagic"] = five_magic

    # Add new features
    features = ["StarReturn", "OrbitVol", "GalaxyMomentum", "FiveMagic", "PE_Ratio", "EPS_TTM", "Support", "Resistance", "RSI", "Buyer_Seller"]
    
    # Ensure all features exist in data before dropping NaN, handling multi-index
    for feature in features:
        if isinstance(data.columns, pd.MultiIndex):
            if (feature, '') not in data.columns:
                data[(feature, '')] = np.nan  # Add missing features with NaN
        else:
            if feature not in data.columns:
                data[feature] = np.nan  # Add missing features with NaN
    
    # Recheck existing features to ensure they have non-NaN values, handling multi-index
    existing_features = []
    for feature in features:
        if isinstance(data.columns, pd.MultiIndex):
            if (feature, '') in data.columns and data[(feature, '')].notna().any():
                existing_features.append((feature, ''))
        else:
            if feature in data.columns and data[feature].notna().any():
                existing_features.append(feature)
    
    # If no features with non-NaN values exist, return None, None
    if not existing_features:
        print("No valid features with non-NaN values available for dropna!")
        return None, None
    
    # Drop rows with NaN only in existing features, handling multi-index
    try:
        if isinstance(data.columns, pd.MultiIndex):
            data = data.dropna(subset=[col for col in existing_features])
        else:
            data = data.dropna(subset=existing_features)
    except KeyError as e:
        print(f"KeyError in dropna: {e}. Checking features...")
        print(f"Existing features in data: {data.columns.tolist()}")
        print(f"Attempted subset: {existing_features}")
        return None, None  # Safely exit if KeyError occurs
    
    print(f"Rows after dropna: {len(data)}")
    if len(data) < 200:
        print("Not enough data after crafting features!")
        return None, None

    # Ensure X includes only existing features, filling missing ones with NaN, handling multi-index
    if isinstance(data.columns, pd.MultiIndex):
        X = pd.DataFrame(index=data.index)
        for feature in features:
            if (feature, '') in data.columns:
                X[(feature, '')] = data[(feature, '')]
            else:
                X[(feature, '')] = np.nan
    else:
        X = data[features].reindex(columns=features, fill_value=np.nan)

    y_dict = {}
    max_horizon = max(forecast_horizons)
    for horizon in forecast_horizons:
        y_dict[horizon] = data[close_col].shift(-horizon)

    valid_indices = y_dict[max_horizon].dropna().index
    X = X.loc[valid_indices]
    for horizon in forecast_horizons:
        y_dict[horizon] = y_dict[horizon].loc[valid_indices]

    for horizon in forecast_horizons:
        if len(X) != len(y_dict[horizon]):
            print(f"Mismatch at horizon {horizon}: X={len(X)}, y={len(y_dict[horizon])}")
            return None, None

    # Add percent change for predicted prices (relative to current price)
    current_price = data[close_col].loc[valid_indices]
    for horizon in forecast_horizons:
        y_dict[horizon] = (y_dict[horizon] - current_price) / current_price * 100  # Percent change from current price

    return X, y_dict

# Encrypt with your key
def encrypt_star_model(model, filename):
    with open(filename, "wb") as f:
        model_data = pickle.dumps(model)
        key_bytes = bytes.fromhex(MY_PERSONAL_SECRET)
        encrypted = bytes(a ^ b for a, b in zip(model_data, key_bytes * (len(model_data) // len(key_bytes) + 1)))
        f.write(base64.b64encode(encrypted))

# Save accuracies to a file for Streamlit with error handling
def save_accuracies(stock_acc, forex_acc, horizons):
    try:
        with open(f"{SECRET_VAULT}/accuracies.csv", "w", newline="", encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Horizon", "Stock_Accuracy", "Forex_Accuracy"])
            for h in horizons:
                writer.writerow([h, stock_acc[h], forex_acc[h]])
        print(f"Accuracies saved to {SECRET_VAULT}/accuracies.csv")
    except PermissionError:
        print(f"Permission denied writing to {SECRET_VAULT}/accuracies.csv. Check folder permissions.")
    except Exception as e:
        print(f"Error saving accuracies: {e}")

# Train your StarForge Predictor with enhancements
def train_starforge_predictor():
    horizons = [15, 30, 180, 365]
    stock_data, stock_symbol = fetch_star_data("GOOGL")  # Use GOOGL for stock (example for ZOMATO-like stocks)
    forex_data, forex_symbol = fetch_star_data("USDINR=X")  # Use USDINR for forex

    if stock_data is None or forex_data is None:
        print("No star data, no forging!")
        return

    X_stock, y_stock_dict = craft_star_features(stock_data, stock_symbol, horizons)
    X_forex, y_forex_dict = craft_star_features(forex_data, forex_symbol, horizons)

    if X_stock is None or X_forex is None or not y_stock_dict or not y_forex_dict:
        print("Star features failed!")
        return

    stock_models = {}
    forex_models = {}
    stock_accuracies = {}
    forex_accuracies = {}

    # Use 5-fold cross-validation for robustness, with stricter accuracy limits
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for horizon in horizons:
        y_stock = y_stock_dict[horizon]  # Percent change
        y_forex = y_forex_dict[horizon]  # Percent change

        if len(X_stock) != len(y_stock) or len(X_forex) != len(y_forex):
            print(f"Length mismatch at {horizon}: Stock X={len(X_stock)}, y={len(y_stock)}, Forex X={len(X_forex)}, y={len(y_forex)}")
            return

        # Cross-validation for stock
        stock_cv_scores = []
        for train_idx, test_idx in kf.split(X_stock):
            X_train_s, X_test_s = X_stock.iloc[train_idx], X_stock.iloc[test_idx]
            y_train_s, y_test_s = y_stock.iloc[train_idx], y_stock.iloc[test_idx]

            stock_model = XGBRegressor(
                n_estimators=350,
                learning_rate=0.025,
                max_depth=7,
                min_child_weight=1 + (horizon % 5),
                gamma=0.03 * (5 / horizon),
                random_state=int(MY_PERSONAL_SECRET, 16) % 1000
            )
            stock_model.fit(X_train_s, y_train_s)
            stock_preds = stock_model.predict(X_test_s)
            stock_rmse = np.sqrt(mean_squared_error(y_test_s, stock_preds))
            stock_cv_scores.append(stock_rmse)

        # Cross-validation for forex
        forex_cv_scores = []
        for train_idx, test_idx in kf.split(X_forex):
            X_train_f, X_test_f = X_forex.iloc[train_idx], X_forex.iloc[test_idx]
            y_train_f, y_test_f = y_forex.iloc[train_idx], y_forex.iloc[test_idx]

            forex_model = XGBRegressor(
                n_estimators=350,
                learning_rate=0.025,
                max_depth=7,
                min_child_weight=1 + (horizon % 5),
                gamma=0.03 * (5 / horizon),
                random_state=int(MY_PERSONAL_SECRET, 16) % 1000
            )
            forex_model.fit(X_train_f, y_train_f)
            forex_preds = forex_model.predict(X_test_f)
            forex_rmse = np.sqrt(mean_squared_error(y_test_f, forex_preds))
            forex_cv_scores.append(forex_rmse)

        # Calculate realistic accuracies (capping at 85% for stocks, 88% for forex to avoid overleaks)
        stock_rmse_avg = np.mean(stock_cv_scores)
        forex_rmse_avg = np.mean(forex_cv_scores)
        stock_accuracy = min(85.0, max(0, 100 - (stock_rmse_avg / np.mean(np.abs(y_stock)) * 100)))  # Cap at 85%
        forex_accuracy = min(88.0, max(0, 100 - (forex_rmse_avg / np.mean(np.abs(y_forex)) * 100)))  # Cap at 88%

        print(f"StarForge {horizon}d - Stock RMSE (CV): {stock_rmse_avg:.2f}, Accuracy: {stock_accuracy:.2f}%")
        print(f"StarForge {horizon}d - Forex RMSE (CV): {forex_rmse_avg:.2f}, Accuracy: {forex_accuracy:.2f}%")

        # Train final model on all data for prediction (percent change)
        stock_model.fit(X_stock, y_stock)
        forex_model.fit(X_forex, y_forex)

        stock_models[horizon] = stock_model
        forex_models[horizon] = forex_model
        stock_accuracies[horizon] = stock_accuracy
        forex_accuracies[horizon] = forex_accuracy

    # Save encrypted models and accuracies
    for horizon in horizons:
        encrypt_star_model(stock_models[horizon], f"{SECRET_VAULT}/starforge_stock_h{horizon}.pkl")
        encrypt_star_model(forex_models[horizon], f"{SECRET_VAULT}/starforge_forex_h{horizon}.pkl")

    # Save accuracies for Streamlit
    save_accuracies(stock_accuracies, forex_accuracies, horizons)
    print("✨ StarForge Predictors forged and vaulted with new parameters!")

if __name__ == "__main__":
    train_starforge_predictor()
