import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt

def Moving_Averages():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_appl = pd.read_csv(os.path.join(script_dir, '../data/AAPL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df_appl['Date'] = pd.to_datetime(df_appl['Date'])
    df_appl.set_index('Date', inplace=True)

    df_appl['MA_50'] = df_appl['Close'].rolling(50).mean()
    df_appl['MA_200'] = df_appl['Close'].rolling(200).mean()

    df_appl['Strat'] = np.where(df_appl['MA_50'] > df_appl['MA_200'], 1, 0)

    df_appl['returns'] = df_appl['Close'].pct_change()

    df_appl['Strat_equity_returns'] = df_appl['Strat'].shift(1) * df_appl['returns']

    df_appl['Strat_buy_hold'] = (1 + df_appl['returns']).cumprod()

    df_appl['equity_curve'] = (1 + df_appl['Strat_equity_returns']).cumprod()

    Strat_returns = (1 + df_appl['Strat_equity_returns']).cumprod().iloc[-1] - 1

    plt.plot(df_appl['Close'], label = 'Spot Price')
    plt.plot(df_appl['MA_50'], label = 'MA 50 days')
    plt.plot(df_appl['MA_200'], label = 'MA 200 days')
    plt.title('Moving Averages 50 and 200')
    plt.legend()
    plt.show()

    plt.plot(df_appl['Strat_buy_hold'], label = 'Buy and Hold')
    plt.plot(df_appl['equity_curve'], label = 'Equity Strategy')
    plt.title('Strategy vs buy and hold')
    plt.legend()
    plt.show()



    print(Strat_returns)

    return df_appl

    
Moving_Averages()
