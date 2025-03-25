#CODE 

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ================== 1. Set Random Seeds to Remove Randomness ==================
def set_random_seed(seed_value=42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    tf.random.set_seed(seed_value)

set_random_seed()

# ================== 2. Load and Preprocess Data ==================
def load_stock_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['Date'])
    
    if data['Close/Last'].dtype == object:
        data['Close/Last'] = pd.to_numeric(data['Close/Last'].str.replace('$', ''), errors='coerce')

    data.dropna(inplace=True)
    return data

# ================== 3. Prepare Data for LSTM ==================
def prepare_data(data, window_size=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close/Last']].values)

    X, y = [], []
    for i in range(window_size, len(scaled_data)):
        X.append(scaled_data[i-window_size:i, 0])
        y.append(scaled_data[i, 0])

    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y, scaler

# ================== 4. Build LSTM Model (Fixed Initialization) ==================
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

# ================== 5. Train LSTM Model (Fixed Training) ==================
def train_and_save_model(X_train, y_train, X_test, y_test, model_path="stock_model.keras"):
    model = build_lstm_model((X_train.shape[1], 1))

    # Train the model with fixed settings
    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=1)
    
    model.save(model_path)
    return model

# ================== 6. Predict Future Stock Prices (Fixed TD(0)) ==================
def td0_update(predictions, actual_prices, alpha=0.1, gamma=0.99):
    updated_predictions = predictions.copy()  # Prevent modifying original list
    for t in range(len(updated_predictions) - 1):
        updated_predictions[t] += alpha * (actual_prices[t] + gamma * updated_predictions[t+1] - updated_predictions[t])
    return updated_predictions

def predict_future_prices(model, scaler, last_window, future_years, data, td0_correction=True):
    future_days = future_years * 252  
    future_predictions = []
    input_data = last_window.copy()

    for _ in range(future_days):
        pred_scaled = model.predict(input_data.reshape(1, -1, 1), verbose=0)  # No randomness
        pred_actual = scaler.inverse_transform(pred_scaled.reshape(-1, 1))[0][0]
        future_predictions.append(pred_actual)
        input_data = np.append(input_data[1:], pred_scaled)

    if td0_correction:
        actual_prices = data['Close/Last'].values[-future_days:]  
        future_predictions = td0_update(future_predictions, actual_prices)

    last_date = data['Date'].max()
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]
    return future_dates, future_predictions

# ================== 7. Plot Predictions ==================
def plot_predictions(data, future_dates, future_predictions):
    plt.figure(figsize=(12, 6))
    plt.plot(data['Date'], data['Close/Last'], label="Historical Prices", color='blue')
    plt.plot(future_dates, future_predictions, label="Predicted Prices (TD(0) Adjusted)", color='red', linestyle='dashed')
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title("Stock Market Prediction using LSTM + TD(0)")
    plt.legend()
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.show()

# ================== 8. Run Everything ==================
if _name_ == "_main_":
    file_path = "/content/Adobe.csv"
    future_years = int(input("Enter number of years to predict: "))

    data = load_stock_data(file_path)

    if data is not None:
        X, y, scaler = prepare_data(data, window_size=60)

        # *Fix Randomness in Train-Test Split*
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Model
        model = train_and_save_model(X_train, y_train, X_test, y_test)

        # Predict Future Prices
        last_window = X[-1]
        future_dates, future_predictions = predict_future_prices(model, scaler, last_window, future_years, data)

        # Plot Predictions
        plot_predictions(data, future_dates, future_predictions)

# ================== 9. Evaluate Model (Consistent Results) ==================
def evaluate_model(model, X_test, y_test, scaler):
    y_pred_scaled = model.predict(X_test, verbose=0)
    
    # Inverse transform predictions and actual values
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Compute Metrics
    mse = mean_squared_error(y_test_actual, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test_actual, y_pred)
    r2 = r2_score(y_test_actual, y_pred)

    # Print Results
    print("\n===== Model Evaluation Metrics =====")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")

# Run Evaluation
evaluate_model(model, X_test, y_test, scaler)
