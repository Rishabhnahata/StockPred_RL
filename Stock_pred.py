#CODE 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# ================== 1. Load and Preprocess Data ==================
def load_stock_data(file_path):
    try:
        # Read CSV and parse 'Date' column
        data = pd.read_csv(file_path, parse_dates=['Date'], dayfirst=False)
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None

    # Ensure 'Close/Last' column exists
    if 'Close/Last' not in data.columns:
        print(f"Error: 'Close/Last' column not found. Available columns: {data.columns.tolist()}")
        return None

    # Convert financial columns to numeric
    financial_cols = ['Close/Last', 'Open', 'High', 'Low', 'Volume']
    for col in financial_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

    # Drop missing values
    data.dropna(inplace=True)

    return data

# ================== 2. Prepare Data for LSTM ==================
def prepare_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close/Last']].values)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    if len(X) == 0:  # Prevent empty dataset issues
        print("Error: Not enough data for the given window size.")
        return None, None, None

    X, y = np.array(X), np.array(y)

    try:
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))  # Reshape for LSTM input
    except IndexError:
        print("Error: Unable to reshape X. Check window size and dataset length.")
        return None, None, None

    return X, y, scaler

# ================== 3. Build LSTM Model ==================
def build_lstm_model(input_shape):
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])

    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# ================== 4. Train and Save Model ==================
def train_and_save_model(X_train, y_train, X_test, y_test, model_path="stock_model.keras"):
    model = build_lstm_model((X_train.shape[1], 1))
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)

    # Save in new Keras format
    model.save(model_path)
    print(f"Model saved to {model_path}")
    return model

# ================== 5. Plot Stock Data ==================
def plot_stock_data(file_path):
    data = load_stock_data(file_path)

    if data is None:
        print("No data available to plot.")
        return

    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close/Last'], label="Stock Price", color='blue')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Stock Market Trend")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# ================== 6. Run Everything ==================
if _name_ == "_main_":  # Fixed typo
    file_path = "/content/HistoricalData_1738053733193.csv"  # Replace with actual file path

    # Load and preprocess data
    data = load_stock_data(file_path)

    if data is not None:
        X, y, scaler = prepare_data(data, window_size=10)  # Updated window size
        if X is not None:  # Check if X is valid
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Train and save model
            model = train_and_save_model(X_train, y_train, X_test, y_test)

            # Plot stock prices
            plot_stock_data(file_path)
