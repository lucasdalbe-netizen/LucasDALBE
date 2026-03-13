import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os 

# Average True Range
# On average on the period, it is the range of change of price
# in percentage of the current price
# It is focus on the vol

def ATR(ticker, period, minimum):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_ticker = pd.read_csv(os.path.join(script_dir, f'../data/{ticker}_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
    df_ticker.set_index('Date', inplace=True)

    # Compute TR
    tr_1 = df_ticker['High'] - df_ticker['Low']
    tr_2 = (df_ticker['High'] - df_ticker['Close'].shift(1)).abs()
    tr_3 = (df_ticker['Low'] - df_ticker['Close'].shift(1)).abs()

    df_ticker['TR'] = np.maximum(tr_1, np.maximum(tr_2, tr_3))
    

    # Basic ATR
    #df_ticker['ATR'] = df_ticker['TR'].rolling(period).mean()


    # Wilder ATR
    df_ticker['ATR'] = np.nan

    # First ATR at the period (input)
    df_ticker.loc[df_ticker.index[period], 'ATR'] = df_ticker['TR'].iloc[1:period+1].mean()
    
    # ATR
    for i in range(1+period, len(df_ticker)) :
        previous_ATR = df_ticker.iloc[i-1]['ATR']
        current_TR = df_ticker.iloc[i]['TR']
        df_ticker.iloc[i, df_ticker.columns.get_loc('ATR')] = (previous_ATR * (period -1) + current_TR) /period

    # We use relative ATR in %, not in $
    df_ticker['ATR_pct'] = df_ticker['ATR'] / df_ticker['Close'] * 100
    average_ATR = df_ticker['ATR_pct'].mean()
    ATR_sup = (df_ticker['ATR_pct'] > minimum).sum()

    print("\n==========")
    print("Ticker = ", ticker)
    print("Average ATR of",ticker, "is = ", average_ATR)
    print("Number of days with ATR >", minimum, " =", ATR_sup, " on ", len(df_ticker), " days")

    return df_ticker

tickers = ["GOOGL", "TSLA", "AVGO", "ORCL", "ADBE"]

for ticker in tickers :
    ATR(ticker, 14, 3)
