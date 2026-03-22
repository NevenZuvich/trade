import pandas as pd
from data_cacheload_helpers import load_data
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def prep_df(df):
    # ensure data is numeric and remove NaN
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    # feature engineering
    df['Return'] = df['Close'].pct_change()
    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['Lag'] = df['Close'].shift(1)
    df['Target'] = df['Close'].shift(-1)

    df = df.dropna()
    
    return df

def pca_transformer(df):
    pca_obj = PCA() # no arguments
    x_pca = pca_obj.fit_transform(df)
    return x_pca

def lda_transformer(df):
    #TODO implement this
    pass

def zscore_normalizer(df):
    scaler = StandardScaler()
    norm_df = scaler.transform(df)
    return norm_df