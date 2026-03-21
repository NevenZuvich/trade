import yfinance as yf
import pandas as pd
import os
import sys
sys.path.insert(1, '.')

def download_data(tickers, start=None, end=None, period='1mo', interval='1d', cache_dir="temp_cache", overwrite=False):
    if isinstance(tickers, str): tickers = [tickers]
    
    for ticker in tickers:
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
        if os.path.exists(path) and not overwrite:
            print(f"Saved data for {ticker} is already cached at {path}.")
        else:
            print(f"Caching data for {ticker} at {path}.")
            try:
                if start and end:
                    data = yf.download(tickers=ticker, start=start, end=end, interval=interval)
                else:
                    data = yf.download(tickers=ticker, period=period, interval=interval)

                if data.empty: # dont know why this has error
                    raise ValueError(f"No data found for {ticker}")
            
                data.to_csv(path)
                print(f"Saved data for {ticker} to cache at {path}")
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")



def load_data(ticker='SPY', start=None, end=None, cache_dir="temp_cache"):    
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{ticker}_{start}_{end}.csv")
    if os.path.exists(path):
        try:
            data = pd.read_csv(path, index_col=0, parse_dates=True)
            print(f"Loaded cached data for {ticker}.")
            return data
        except Exception as _:
            print(f"Failed to load cache at {path}.")
    else:
        print(f"No cached data found for {ticker} at {path}.")

# download_data(tickers=['SPY', 'AAPL'], start='2015-01-01', end='2025-01-01', interval='1d', overwrite=True)
