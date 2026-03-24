import yfinance as yf
import pandas as pd
import os
import sys
from sklearn.decomposition import PCA
sys.path.insert(1, '.')

def download_data(tickers, start=None, end=None, period='1mo', interval='1d', cache_dir="temp_cache", overwrite=False):
    
    if isinstance(tickers, str): tickers = [tickers]
    os.makedirs(cache_dir, exist_ok=True)
    
    for ticker in tickers:

        ticker = ticker.upper()
        
        path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
        if os.path.exists(path):
            if not overwrite:
                print(f"Saved data for {ticker} is already cached at {path}. Cancelling download.")
                return
            print(f"Saved data for {ticker} is already cached at {path}. Overwriting...")
        else:
            print(f"Caching data for {ticker} at {path}.")

            try:
                if start and end:
                    data = yf.download(tickers=ticker, start=start, end=end, interval=interval)
                    filename = f"{ticker}_{start}_{end}.csv"
                else:
                    data = yf.download(tickers=ticker, period=period, interval=interval)
                    filename = f"{ticker}_{period}_{interval}.csv"

                if data is None or data.empty:
                    raise ValueError(f"No data found for {ticker}")
            
                path = os.path.join(cache_dir, filename)
                data.to_csv(path)
                print(f"Saved data for {ticker} to cache at {path}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
        

def load_data(ticker='SPY', start=None, end=None, cache_dir="temp_cache"):    
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(path):
        try:
            data = pd.read_csv(path, index_col=0, parse_dates=False)
            print(f"Loaded cached data for {ticker}.")
            return data
        except Exception as _:
            print(f"Failed to load cache at {path}.")
    else:
        print(f"No cached data found for {ticker} at {path}.")



def prep_df(df):
    # ensure data is numeric and remove NaN
    df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
    df = df.dropna()

    # feature engineering
    df['Ret_1'] = df['Close'].pct_change()
    df['Ret_5'] = df['Close'].pct_change(5)
    df['Ret_10'] = df['Close'].pct_change(10)
    df['Ret_20'] = df['Close'].pct_change(20)
    df['Ret_60'] = df['Close'].pct_change(60)

    df['Vol_5'] = df['Ret_1'].rolling(5).std()
    df['Vol_10'] = df['Ret_1'].rolling(10).std()

    df['SMA_5'] = df['Close'].rolling(5).mean()
    df['SMA_10'] = df['Close'].rolling(10).mean()
    df['SMA_60'] = df['Close'].rolling(60).mean()
    df['SMA_200'] = df['Close'].rolling(200).mean()

    df['SMA_Cross'] = df['SMA_60'] - df['SMA_200']
    df['SMA_Cross_Norm'] = df['SMA_60'] / df['SMA_200'] - 1

    df['Trend_60'] = df['Close'] / df['SMA_60'] 
    df['Trend_200'] = df['Close'] / df['SMA_200'] 

    df['Lag'] = df['Close'].shift(1)
    df['zscore_5'] = (df['Close'] - df['SMA_5']) / df['Ret_1'].rolling(5).std()

    # prediction target
    df['Target'] = df['Close'].pct_change(10).shift(-10)

    df = df.dropna()
    
    return df


def pca_transformer(df):
    pca_obj = PCA() # no arguments
    x_pca = pca_obj.fit_transform(df)
    return x_pca



def zscore_normalizer(df):
    scaler = StandardScaler()
    norm_df = scaler.transform(df)
    return norm_df