import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import sys
sys.path.insert(1, '.')
from data_helpers import prep_df, load_data, download_data



def linear_model(df, target='Target', features=[], train_size=0.8):

    X = df[features]
    y = df[target]

    # train test split
    split_idx = int(len(df) * train_size)

    # Assign X, y
    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

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
    print('Movement Acc: ', ((np.sign(y_pred) == np.sign(y_test)).mean()*100).round(2),'%')
    
    df['Predictions'] = np.nan
    df.loc[df.index[split_idx:], 'Prediction'] = y_pred
    df['Position'] = np.sign(df['Prediction'])
    df['Strategy_Return'] = df['Position'] * df['Target']
    df['Cumul_Return'] = (1 + df['Strategy_Return']).cumprod()
    df['Buy_Hold'] = (1 + df['Target']).cumprod()
    df = df.dropna(subset=['Prediction', 'Target'])

    print('Strategy return: ', df['Cumul_Return'].iloc[-1].round(2))
    print('Buy/Hold: ', df['Buy_Hold'].iloc[-1].round(2))

    return model
    


download_data(tickers=['AAPL'], start='2015-01-01', end='2025-01-01', interval='1d', cache_dir="temp_cache", overwrite=False)

df = load_data(ticker='AAPL', start='2015-01-01', end='2025-01-01')
df = prep_df(df)
linear_model(df, 
             target='Target', 
             features= ['Return', 'Ret_5', 'Ret_10', 'Vol_5', 'zscore_5'], 
             train_size=0.80)