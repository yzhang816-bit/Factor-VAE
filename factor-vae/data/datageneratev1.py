import ccxt
import pandas as pd
import polars as pl
import os  # Import os to handle directory paths

# Initialize CCXT Binance API
exchange = ccxt.binance()

# Define a list of symbols to fetch data for
symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'XRP/USDT', 'ADA/USDT']  # Add more tickers as needed

timeframe = '1d'  # Daily historical data
limit = 500  # Number of data points to fetch

dfs = []

for symbol in symbols:
    try:
        ohlcv_data = exchange.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv_data, columns=['<DATE>', '<OPEN>', '<HIGH>', '<LOW>', '<CLOSE>', '<VOL>'])
        df['<DATE>'] = pd.to_datetime(df['<DATE>'], unit='ms')
        df['<TICKER>'] = symbol
        dfs.append(df)
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")

# Concatenate all DataFrames
df_all = pd.concat(dfs, ignore_index=True)

# Convert the Pandas DataFrame into a Polars DataFrame
df_pl = pl.from_pandas(df_all)

# Specify the directory where you want to save the CSV
DATA_DIR = './data'  # Update this to your desired directory

# Create the directory if it doesn't exist
os.makedirs(DATA_DIR, exist_ok=True)

# Save the data to CSV in the specified directory
locale = 'hk'  # You can replace this with any desired file name
csv_path = os.path.join(DATA_DIR, f"{locale}.csv")
df_pl.write_csv(csv_path)

print(f"Data saved to {csv_path}")
