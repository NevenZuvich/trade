import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import sys
sys.path.insert(1, '.')
from preprocessing_helpers import prep_df
from data_cacheload_helpers import load_data



def linear_model(df, target='Target', features=[], train_size=0.8):

    # train test split
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Assign X, y
    X_train = train_df[features]
    X_test = test_df[features]
    y_train = train_df[target]
    y_test = test_df[target]

    # set up model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # compute metrics
    metrics = {
        'MAE' : mean_absolute_error(y_test, y_pred),
        'RMSE' : root_mean_squared_error(y_test, y_pred)
    }

    print(metrics)
    return model
    


df = load_data(ticker='SPY', start='2015-01-01', end='2025-01-01')
df = prep_df(df)
linear_model(df, target='Target', features= ['Return', 'SMA_5', 'SMA_10', 'Lag'], train_size=0.85)