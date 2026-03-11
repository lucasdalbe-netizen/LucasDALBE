import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def Ichimoku(ticker) :
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_ticker = pd.read_csv(os.path.join(script_dir, f'../data/{ticker}_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Price', 'High', 'Low', 'Open', 'Volume'])
    df_ticker['Date'] = pd.to_datetime(df_ticker['Date'])
    df_ticker.set_index('Date', inplace=True)

    #Tenkan is the mean between the higher high and lower low on 9 days
    df_ticker['Lower_9'] = df_ticker['Low'].rolling(9).min()
    df_ticker['Higher_9'] = df_ticker['High'].rolling(9).max()
    df_ticker['Tenkan'] = (df_ticker['Higher_9'] + df_ticker['Lower_9']) / 2

    #Kijun-Sen is the mean between the higher high and lower low on 26 days
    df_ticker['Lower_26'] = df_ticker['Low'].rolling(26).min()
    df_ticker['Higher_26'] = df_ticker['High'].rolling(26).max()
    df_ticker['Kijun_Sen'] = (df_ticker['Higher_26'] + df_ticker['Lower_26']) / 2

    #Chikou Span is the n+26 day close price
    df_ticker['Chikou_Span'] = df_ticker['Price'].shift(-26)

    #Senko Span A is the n-26 mean of Tenkan and Kijun-Sen
    df_ticker['SSA'] = ((df_ticker['Tenkan'] + df_ticker['Kijun_Sen'])/2).shift(26)
    df_ticker['SSA_raw'] = ((df_ticker['Tenkan'] + df_ticker['Kijun_Sen'])/2)


    #Senko Span B is the n-52 mean of Higher high and Lower low
    df_ticker['Lower_52'] = df_ticker['Low'].rolling(52).min()
    df_ticker['Higher_52'] = df_ticker['High'].rolling(52).max()
    df_ticker['SSB'] = ((df_ticker['Higher_52'] + df_ticker['Lower_52'])/2).shift(26)
    df_ticker['SSB_raw'] = ((df_ticker['Higher_52'] + df_ticker['Lower_52'])/2)

    #Kumo is the values in between SSA and SSB 
    #The thicker the harder it is to go through
    #Kumo is a support 
    df_ticker['Kumo_low'] = df_ticker[['SSA_raw','SSB_raw']].min(axis = 1)
    df_ticker['Kumo_high'] = df_ticker[['SSA_raw','SSB_raw']].max(axis = 1)

    #Interpret those results as signals
    #There are multiple, compute each separately then cross them
    
    #1; Current price > Past 26 days price
    df_ticker['Chikou_Signal'] = 0
    df_ticker.loc[
        (df_ticker['Price'] > df_ticker['Price'].shift(26)),
        'Chikou_Signal'
    ] = 1

    #2; Price relatively to Kumo Cloud
    df_ticker['Kumo_BreakOut_Signal'] = 0
    df_ticker.loc[
        (df_ticker['Price'] > (df_ticker['Kumo_high'])),
        'Kumo_BreakOut_Signal'
    ] = 1
    df_ticker.loc[
        (df_ticker['Price'] < (df_ticker['Kumo_low'])),
        'Kumo_BreakOut_Signal'
    ] = -1

    #3; Tenkan / Kijun Cross (kinda like Moving Averages)
    df_ticker['Cross_TK_Signal'] = 0
    df_ticker.loc[
        (df_ticker['Tenkan'] > df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) < df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] > df_ticker['Kumo_high']),
        'Cross_TK_Signal'
    ] = 3
    df_ticker.loc[
        (df_ticker['Tenkan'] > df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) < df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] < df_ticker['Kumo_high']) & 
        (df_ticker['Tenkan'] > df_ticker['Kumo_low']),
        'Cross_TK_Signal'
    ] = 2
    df_ticker.loc[
        (df_ticker['Tenkan'] > df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) < df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] < df_ticker['Kumo_low']),
        'Cross_TK_Signal'
    ] = 1
    df_ticker.loc[
        (df_ticker['Tenkan'] < df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) > df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] > df_ticker['Kumo_high']),
        'Cross_TK_Signal'
    ] = -1
    df_ticker.loc[
        (df_ticker['Tenkan'] < df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) > df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] < df_ticker['Kumo_high']) & 
        (df_ticker['Tenkan'] > df_ticker['Kumo_low']),
        'Cross_TK_Signal'
    ] = -2
    df_ticker.loc[
        (df_ticker['Tenkan'] < df_ticker['Kijun_Sen']) &
        (df_ticker['Tenkan'].shift(1) > df_ticker['Kijun_Sen'].shift(1)) &
        (df_ticker['Tenkan'] < df_ticker['Kumo_low']),
        'Cross_TK_Signal'
    ] = -3

    #4; Kumo orientation
    df_ticker['Kumo_Orientation_Signal'] = 0
    df_ticker.loc[
        (df_ticker['SSA_raw'] > df_ticker['SSB_raw']),
        'Kumo_Orientation_Signal'
    ] = 1
    df_ticker.loc[
        (df_ticker['SSA_raw'] < df_ticker['SSB_raw']),
        'Kumo_Orientation_Signal'
    ] = -1

    
    #Create Entries
    df_ticker['Overall_Signal'] = 0
    df_ticker.loc[
        (df_ticker['Chikou_Signal'] > 0) &
        (df_ticker['Kumo_BreakOut_Signal'] > 0) &
        #(df_ticker['Cross_TK_Signal'] > 0) &
        # Cross_TK_Signal is to delete for more values (about 1/3 of the overall data)
        (df_ticker['Kumo_Orientation_Signal'] > 0),
        'Overall_Signal'
    ] = 1

    #Create exits  (here we just leave when price goes down)
    df_ticker['Exit'] = 0
    df_ticker.loc[
        (df_ticker['Price'] < df_ticker['Price'].shift(1)),
        'Exit'
    ] = 1



    # Compute the returns of this signal strategy 
    trades = []
    entry_index = None
    position = False
    for i in range(1, len(df_ticker)) :
    #Entry if Overall Signal is 1 and we are not in position
        if (not position) and (df_ticker.iloc[i]['Overall_Signal'] == 1):
            entry_index = i 
            position = True
    #Exit when price day n < price day n-1
        elif position and (df_ticker.iloc[i]['Price'] < df_ticker.iloc[i-1]['Price']) :
    #Exit when price < Kijun
        #elif position and (df_ticker.iloc[i]['Price'] < df_ticker.iloc[i]['Kijun_Sen']) :
    #Exit when price < Cloud
        #elif position and (df_ticker.iloc[i]['Price'] < df_ticker.iloc[i]['Kumo_low']) :
    #Exit when Tenkan < Kijun
        #elif position and (df_ticker.iloc[i]['Tenkan'] < df_ticker.iloc[i]['Kijun_Sen']) :
            exit_index = i
            trades.append((entry_index, exit_index))
            position = False
            entry_index =None

    Total_pnl = 1
    pnl = 0
    for entry_index, exit_index in trades :
        pnl = (df_ticker.iloc[exit_index]['Price']/df_ticker.iloc[entry_index]['Price']) - 1
        Total_pnl *= (1+pnl)

    #Buy & Hold Strategy return as a comparison 
    Buy_and_Hold = (df_ticker.iloc[len(df_ticker) - 1]['Price'] / df_ticker.iloc[0]['Price'])
    Trades = len(trades)

    print("Number of trades made = ", Trades)
    print("Returns Ichimoku = ", Total_pnl - 1)
    print("Returns Buy & Hold = ", Buy_and_Hold -1)


    df_plot = df_ticker.iloc[1180:]

    
    plt.plot(df_plot['Price'], label = "Spot Price")
    plt.plot(df_plot['Tenkan'], '--', label = "Tenkan")
    plt.plot(df_plot['Kijun_Sen'], '--', label = "Kijun-Sen")
    plt.plot(df_plot['Chikou_Span'], '--', label = "Chikou Span")
    plt.plot(df_plot['SSA'], '--', label = "SSA")
    plt.plot(df_plot['SSB'], '--', label = "SSB")
    plt.title(f"Ichimoku - {ticker}")
    plt.fill_between(df_plot.index,df_plot['Kumo_low'],df_plot['Kumo_high'],alpha=0.3,label="Kumo")
    plt.legend()
    plt.show()
    plt.savefig('Ichimoku.png')
    return

tickers = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    "JPM", "GS", "BAC",
    "XOM", "CVX",
    "^GSPC", "^VIX"
]

for ticker in tickers :
    print("\n==========")
    print("Ticker :", ticker)
    Ichimoku(ticker)
