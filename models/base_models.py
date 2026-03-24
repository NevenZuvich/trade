import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, ElasticNet, Lasso
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import sys
sys.path.insert(1, '.')
from data_helpers import prep_df, load_data, download_data



def linear_model(df, target='Target', features=[], train_size=0.8):

    df = df.copy()

    # train test split
    split_idx = int(len(df) * train_size)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    # Fit the scaler only on training data and transform both train and test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train_df[features])
    X_test = scaler.transform(test_df[features])

    y_train = train_df[target].values
    y_test = test_df[target].values

    # set up model
    model = ElasticNet()
    model.fit(X_train, y_train)

    # make predictions
    y_pred = model.predict(X_test)

    # compute metrics
    metrics = {
        'MAE' : mean_absolute_error(y_test, y_pred),
        'RMSE' : root_mean_squared_error(y_test, y_pred)
    }

    print(metrics)
    print('Movement Acc: ', ((np.sign(y_pred) == np.sign(y_test)).mean()*100).round(2),'%')

    return model
    


download_data(tickers=['AAPL'], start='2015-01-01', end='2025-01-01', interval='1d', cache_dir="temp_cache", overwrite=False)

df = load_data(ticker='AAPL', start='2015-01-01', end='2025-01-01')
df = prep_df(df)
linear_model(df, 
             target='Target', 
             features= ['Ret_1', 'Ret_5', 'Ret_10', 'Ret_20', 'Ret_60', 'Vol_5', 
                        'zscore_5', 'SMA_60', 'SMA_200', 'Trend_60'], 
             train_size=0.80)