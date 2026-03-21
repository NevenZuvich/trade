import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from joblib import dump
import sys
sys.path.insert(1, '.')
from metric_helper import augment_df
from data_helpers import load_data


def train_linearmodel(df, target='Target', features=[], train_size=0.8):

    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]
    
    X_train = train_df[features]
    y_train = train_df[target]

    X_test = test_df[features]
    y_test = test_df[target]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        'rmse' : root_mean_squared_error(y_test, y_pred)
    }

    print(metrics)

    dump(model, "trained_linear_model.joblib")
    return model
    

df = load_data(ticker='SPY', start='2015-01-01', end='2025-01-01')
df = augment_df(df)
train_linearmodel(df, target='Target', features= ['Return', 'SMA_5', 'SMA_10', 'Lag'], train_size=0.85)