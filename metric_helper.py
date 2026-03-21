import pandas as pd
from data_helpers import load_data

def augment_df(df):
    df = df.dropna()

    df['Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Lag'] = df['Close'].shift(1)
    df['Target'] = df['Close'].shift(-1)
    
    return df
    