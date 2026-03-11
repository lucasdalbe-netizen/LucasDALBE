import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

def Ichimoku() :
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_appl = pd.read_csv(os.path.join(script_dir, '../data/AAPL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Price', 'High', 'Low', 'Open', 'Volume'])
    df_appl['Date'] = pd.to_datetime(df_appl['Date'])
    df_appl.set_index('Date', inplace=True)

    #Tenkan is the mean between the higher high and lower low on 9 days
    df_appl['Lower_9'] = df_appl['Low'].rolling(9).min()
    df_appl['Higher_9'] = df_appl['High'].rolling(9).max()
    df_appl['Tenkan'] = (df_appl['Higher_9'] + df_appl['Lower_9']) / 2

    #Kijun-Sen is the mean between the higher high and lower low on 26 days
    df_appl['Lower_26'] = df_appl['Low'].rolling(26).min()
    df_appl['Higher_26'] = df_appl['High'].rolling(26).max()
    df_appl['Kijun_Sen'] = (df_appl['Higher_26'] + df_appl['Lower_26']) / 2

    #Chikou Span is the n+26 day close price
    df_appl['Chikou_Span'] = df_appl['Price'].shift(-26)

    #Senko Span A is the n-26 mean of Tenkan and Kijun-Sen
    df_appl['Senko_Span_A'] = ((df_appl['Tenkan'] + df_appl['Kijun_Sen'])/2).shift(26)

    #Senko Span B is the n-52 mean of Higher high and Lower low
    df_appl['Lower_52'] = df_appl['Low'].rolling(52).min()
    df_appl['Higher_52'] = df_appl['High'].rolling(52).max()
    df_appl['Senko_Span_B'] = ((df_appl['Higher_52'] + df_appl['Lower_52'])/2).shift(26)

    #Kumo is the values in between SSA and SSB 
    #The thicker the harder it is to go through
    #Kumo is a support 
    df_appl['Kumo_low'] = df_appl[['Senko_Span_A','Senko_Span_B']].min(axis = 1)
    df_appl['Kumo_high'] = df_appl[['Senko_Span_A','Senko_Span_B']].max(axis = 1)

    #Interpret those results as signals
    #There are multiple, compute each separately then cross them
    
    #1; Current price > Past 26 days price
    df_appl['Chikou_Signal'] = 0
    df_appl.loc[
        (df_appl['Price'] > df_appl['Price'].shift(26)),
        'Chikou_Signal'
    ] = 1

    #2; Price relatively to Kumo Cloud
    df_appl['Kumo_BreakOut_Signal'] = 0
    df_appl.loc[
        (df_appl['Price'] > (df_appl['Kumo_high'])),
        'Kumo_BreakOut_Signal'
    ] = 1
    df_appl.loc[
        (df_appl['Price'] < (df_appl['Kumo_low'])),
        'Kumo_BreakOut_Signal'
    ] = -1

    #3; Tenkan / Kijun Cross (kinda like Moving Averages)
    df_appl['Cross_TK_Signal'] = 0
    df_appl.loc[
        (df_appl['Tenkan'] > df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) < df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] > df_appl['Kumo_high']),
        'Cross_TK_Signal'
    ] = 3
    df_appl.loc[
        (df_appl['Tenkan'] > df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) < df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] < df_appl['Kumo_high']) & 
        (df_appl['Tenkan'] > df_appl['Kumo_low']),
        'Cross_TK_Signal'
    ] = 2
    df_appl.loc[
        (df_appl['Tenkan'] > df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) < df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] < df_appl['Kumo_low']),
        'Cross_TK_Signal'
    ] = 1
    df_appl.loc[
        (df_appl['Tenkan'] < df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) > df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] > df_appl['Kumo_high']),
        'Cross_TK_Signal'
    ] = -1
    df_appl.loc[
        (df_appl['Tenkan'] < df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) > df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] < df_appl['Kumo_high']) & 
        (df_appl['Tenkan'] > df_appl['Kumo_low']),
        'Cross_TK_Signal'
    ] = -2
    df_appl.loc[
        (df_appl['Tenkan'] < df_appl['Kijun_Sen']) &
        (df_appl['Tenkan'].shift(1) > df_appl['Kijun_Sen'].shift(1)) &
        (df_appl['Tenkan'] < df_appl['Kumo_low']),
        'Cross_TK_Signal'
    ] = -3

    #4; Kumo orientation
    df_appl['Kumo_Orientation_Signal'] = 0
    df_appl.loc[
        (df_appl['Senko_Span_A'] > df_appl['Senko_Span_B']),
        'Kumo_Orientation_Signal'
    ] = 1
    df_appl.loc[
        (df_appl['Senko_Span_A'] < df_appl['Senko_Span_B']),
        'Kumo_Orientation_Signal'
    ] = -1

    #Create exits 
    


    df_appl['Overall_Signal'] = 0
    df_appl.loc[
        (df_appl['Chikou_Signal'] > 0) &
        (df_appl['Kumo_BreakOut_Signal'] > 0) &
        #(df_appl['Cross_TK_Signal'] > 0) &
        # Cross_TK_Signal is to delete for more values (about 1/3 of the overall data)
        (df_appl['Kumo_Orientation_Signal'] > 0),
        'Overall_Signal'
    ] = 1



    # Compute the returns of this signal strategy 
    id_open = []
    i = 0
    while i < len(df_appl) - 10 :
        if df_appl.iloc[i]['Overall_Signal'] == 1:
            id_open.append(i)
            i += 10
        else :
            i += 1

    #Here search for index that respect exit condition 

    Total_pnl = 1
    pnl = 0
    for i in range(len(id_open)) :
        a = id_open[i]
        b = a +10
        pnl = (df_appl.iloc[b]['Price']/df_appl.iloc[a]['Price']) - 1
        Total_pnl *= (1+pnl)

    Buy_and_Hold = (df_appl.iloc[len(df_appl) - 1]['Price'] / df_appl.iloc[0]['Price'])
    Trades = len(id_open)


    print(Trades, " Trades have been made using Ichimoku")
    print("Returns = ", Total_pnl - 1)
    print("Returns Buy & Hold = ", Buy_and_Hold -1)

    df_plot = df_appl.iloc[1180:]
    
    plt.plot(df_plot['Price'], label = "Spot Price")
    plt.plot(df_plot['Tenkan'], '--', label = "Tenkan")
    plt.plot(df_plot['Kijun_Sen'], '--', label = "Kijun-Sen")
    plt.plot(df_plot['Chikou_Span'], '--', label = "Chikou Span")
    plt.plot(df_plot['Senko_Span_A'], '--', label = "SSA")
    plt.plot(df_plot['Senko_Span_B'], '--', label = "SSB")
    plt.fill_between(df_plot.index,df_plot['Kumo_low'],df_plot['Kumo_high'],alpha=0.3,label="Kumo")
    plt.legend()
    plt.show()
    plt.savefig('Ichimoku.png')
    return

Ichimoku()
