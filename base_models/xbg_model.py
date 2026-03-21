import yfinance as yf
import numpy as np
from xgboost import XGBRegressor as xbgr
from sklearn.model_selection import train_test_split as ttsplit
from sklearn.metrics import root_mean_squared_error as rmse
import matplotlib.pyplot as plt
import pandas as pd

df = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df.tail()

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'SMA_5', 'SMA_10', 'Lag_1']
X = df[features]
y = df['Target']

X_train, X_test, y_train, y_test = ttsplit(X, y, test_size=0.2, shuffle=False, random_state=42)

model = xbgr(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

preds = model.predict(X_test)
rmse = rmse(y_test, preds)
print(f"RMSE: {rmse:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual")
plt.plot(preds, label="Predicted")
plt.title("AAPL Stock Price Prediction")
plt.xlabel("Days")
plt.ylabel("Price")
plt.legend()
plt.grid()
plt.show()

accuracy = ( (preds[1:] > preds[:-1]) == (y_test.values[1:] > y_test.values[:-1]) ).mean()
print(f"Directional Accuracy: {accuracy:.2%}")