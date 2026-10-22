# %% [Cell 1: Import libraries]
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense
from tensorflow.keras.callbacks import EarlyStopping


# %% [Cell 2: Download daily weather data from Open-Meteo]
url = "https://archive-api.open-meteo.com/v1/archive"

params = {
    "latitude": 28.6139,
    "longitude": 77.2090,
    "start_date": "1940-01-01",
    "end_date": "2025-12-31",
    "daily": [
        "temperature_2m_mean",
        "relative_humidity_2m_mean",
        "wind_speed_10m_mean",
        "pressure_msl_mean",
    ],
    "timezone": "Asia/Kolkata",
}

response = requests.get(url, params=params, timeout=120)
response.raise_for_status()
weather_json = response.json()

df = pd.DataFrame(weather_json["daily"])
df = df.rename(
    columns={
        "time": "date",
        "temperature_2m_mean": "meantemp",
        "relative_humidity_2m_mean": "humidity",
        "wind_speed_10m_mean": "wind_speed",
        "pressure_msl_mean": "meanpressure",
    }
)

df["date"] = pd.to_datetime(df["date"])
df.to_csv("Delhi_Daily_Weather.csv", index=False)


# %% [Cell 3: Clean and inspect the dataset]
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.drop_duplicates(subset="date", keep="first")

feature_cols = ["meantemp", "humidity", "wind_speed", "meanpressure"]

for col in feature_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.set_index("date")
df[feature_cols] = df[feature_cols].interpolate(method="time")
df = df.dropna(subset=feature_cols)
df = df.reset_index()

# Save the cleaned dataset
df.to_csv("Delhi_Daily_Weather_Cleaned.csv", index=False)


# %% [Cell 4: Split chronologically into train, validation, and test sets]
n = len(df)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_df = df.iloc[:train_end].copy()
val_df = df.iloc[train_end:val_end].copy()
test_df = df.iloc[val_end:].copy()


# %% [Cell 5: Normalize features using the training data only]
scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols])

train_scaled = scaler.transform(train_df[feature_cols])
val_scaled = scaler.transform(val_df[feature_cols])
test_scaled = scaler.transform(test_df[feature_cols])


# %% [Cell 6: Create sequences using the previous 30 days]
LOOKBACK = 30
TEMP_INDEX = feature_cols.index("meantemp")


def create_sequences(data, lookback=LOOKBACK):
    X, y = [], []
    for i in range(lookback, len(data)):
        X.append(data[i - lookback:i])
        y.append(data[i, TEMP_INDEX])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


X_train, y_train = create_sequences(train_scaled, LOOKBACK)

# Use the final 30 training rows as context for validation sequences.
val_input = np.concatenate([train_scaled[-LOOKBACK:], val_scaled], axis=0)
X_val, y_val = create_sequences(val_input, LOOKBACK)

# Use the final 30 validation rows as context for test sequences.
test_input = np.concatenate([val_scaled[-LOOKBACK:], test_scaled], axis=0)
X_test, y_test = create_sequences(test_input, LOOKBACK)


# %% [Cell 7: Build and compile the LSTM model]
tf.keras.utils.set_random_seed(42)

model = Sequential([
    Input(shape=(LOOKBACK, len(feature_cols))),
    LSTM(64),
    Dropout(0.2),
    Dense(32, activation="relu"),
    Dense(1),
])

model.compile(
    optimizer="adam",
    loss="mse",
    metrics=["mae"],
)


# %% [Cell 8: Train the model]
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=5,
    restore_best_weights=True,
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=[early_stop],
    shuffle=False,
    verbose=1,
)


# %% [Cell 9: Evaluate the model on the test dataset]
test_loss, test_mae_scaled = model.evaluate(X_test, y_test, verbose=0)
y_pred_scaled = model.predict(X_test, verbose=0).reshape(-1)


def inverse_temperature(values):
    values = np.asarray(values).reshape(-1)
    dummy = np.zeros((len(values), len(feature_cols)))
    dummy[:, TEMP_INDEX] = values
    return scaler.inverse_transform(dummy)[:, TEMP_INDEX]


y_test_actual = inverse_temperature(y_test)
y_pred_actual = inverse_temperature(y_pred_scaled)

test_mae = mean_absolute_error(y_test_actual, y_pred_actual)
test_rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred_actual))


# %% [Cell 10: Plot training and validation loss]
plt.figure(figsize=(8, 5))
plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("LSTM Training and Validation Loss")
plt.legend()
plt.tight_layout()
plt.show()


# %% [Cell 11: Plot actual vs predicted temperature]
test_dates = test_df["date"].iloc[LOOKBACK:].reset_index(drop=True)

plt.figure(figsize=(12, 5))
plt.plot(test_dates, y_test_actual, label="Actual Temperature")
plt.plot(test_dates, y_pred_actual, label="Predicted Temperature")
plt.xlabel("Date")
plt.ylabel("Mean Temperature (°C)")
plt.title("Actual vs Predicted Daily Mean Temperature")
plt.legend()
plt.tight_layout()
plt.show()


# %% [Cell 12: Forecast the next day's mean temperature]
last_30_days = df[feature_cols].tail(LOOKBACK)
last_30_scaled = scaler.transform(last_30_days)

next_day_input = last_30_scaled.reshape(
    1, LOOKBACK, len(feature_cols)
)

next_temp_scaled = model.predict(next_day_input, verbose=0)[0, 0]
next_temp = inverse_temperature([next_temp_scaled])[0]

last_observed_date = df["date"].max()
forecast_date = last_observed_date + pd.Timedelta(days=1)


# %% [Cell 13: Save the trained model and forecast result]
model.save("lstm_temperature_model.keras")

forecast_result = pd.DataFrame({
    "last_observed_date": [last_observed_date],
    "forecast_date": [forecast_date],
    "predicted_mean_temperature_C": [next_temp],
})
forecast_result.to_csv("next_day_temperature_forecast.csv", index=False)
