import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ---- DATA COLLECTION ----
data = {
    "Date": [
        "4/10/2024",
        "4/11/2024",
        "4/12/2024",
        "4/15/2024",
        "4/16/2024",
        "4/17/2024",
        "4/18/2024",
        "4/19/2024",
        "4/22/2024",
        "4/23/2024",
        "4/24/2024",
        "4/25/2024",
        "4/26/2024",
        "4/29/2024",
        "4/30/2024",
    ],
    "Open": [
        6.56,
        6.57,
        6.59,
        6.58,
        6.66,
        6.62,
        6.77,
        6.84,
        6.85,
        6.80,
        6.79,
        6.80,
        6.88,
        6.74,
        6.69,
    ],
    "High": [
        6.62,
        6.62,
        6.66,
        6.68,
        6.70,
        6.77,
        6.94,
        6.91,
        6.94,
        6.88,
        6.84,
        6.88,
        6.90,
        6.80,
        6.79,
    ],
    "Low": [
        6.54,
        6.48,
        6.56,
        6.55,
        6.60,
        6.57,
        6.76,
        6.80,
        6.75,
        6.75,
        6.74,
        6.77,
        6.69,
        6.69,
        6.68,
    ],
    "Close": [
        6.58,
        6.60,
        6.58,
        6.66,
        6.64,
        6.77,
        6.83,
        6.85,
        6.81,
        6.79,
        6.81,
        6.88,
        6.75,
        6.72,
        6.71,
    ],
}

df = pd.DataFrame(data)
df["Date"] = pd.to_datetime(df["Date"])

# ---- DATA PROCESSING ----
df = df.drop_duplicates()
df = df.fillna(df.mean())

# ---- SIMULATING VOLUME DATA AS IT IS NOT PROVIDED ----
df["Volume"] = [2000] * len(df)

# ---- NORMALIZING THE FEATURES ----
features = ["Open", "High", "Low", "Volume"]
scaler = StandardScaler()
df[features] = scaler.fit_transform(df[features])

# ---- DATA PARTITION ----
X = df[["Open", "High", "Low", "Volume"]]
y = df["Close"]

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, train_size=0.7, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, train_size=0.5, random_state=42
)

# ---- MODEL CONSTRUCTION TRAINING ----
model = LinearRegression()
model.fit(X_train, y_train)

# ---- EXPERIMENTAL RESULTS AND ANALYSIS ----
val_predictions = model.predict(X_val)
val_mse = mean_squared_error(y_val, val_predictions)
print(f"Validation MSE: {val_mse}")

test_predictions = model.predict(X_test)
test_mse = mean_squared_error(y_test, test_predictions)
print(f"Test MSE: {test_mse}")

# ---- PLOTTING THE RESULTS ----
plt.figure(figsize=(10, 6))
plt.plot(df["Date"].iloc[X_test.index], y_test, label="Actual Close Price")
plt.plot(
    df["Date"].iloc[X_test.index],
    test_predictions,
    label="Predicted Close Price",
    linestyle="--",
)
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.title("Actual vs Predicted Close Prices")
plt.legend()
plt.show()
