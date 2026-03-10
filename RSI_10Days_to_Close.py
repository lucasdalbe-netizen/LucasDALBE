import numpy as np
import os 
import pandas as pd
import matplotlib.pyplot as plt

def RSI():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_appl = pd.read_csv(os.path.join(script_dir, '../data/AAPL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Price', 'High', 'Low', 'Open', 'Volume'])
    df_appl['Date'] = pd.to_datetime(df_appl['Date'])
    df_appl.set_index('Date', inplace=True)

    #Daily Performance
    df_appl['Delta'] = df_appl['Price'].pct_change()

    #If the stock has a daily return > 0 its performance goes in 'Increase"
    df_appl['Increase'] = 0.0
    df_appl.loc[df_appl['Delta'] > 0, 'Increase'] = df_appl['Delta']

    #If the stock has a daily return < 0 its performance goes in 'Decrease"
    df_appl['Decrease'] = 0.0
    df_appl.loc[df_appl['Delta'] < 0, 'Decrease'] = - df_appl['Delta']
    
    # We calculate the RSI on a 14 days of data basis
    df_appl['RSI_up'] = df_appl['Increase'].rolling(14).mean()
    df_appl['RSI_down'] = df_appl['Decrease'].rolling(14).mean()

    df_appl['Ratio'] = df_appl['RSI_up']/df_appl['RSI_down']

    df_appl['RSI'] = (100- (100/(1+ df_appl['Ratio']))) 

    # Position = 1 to enter short, Position = -1 to enter long
    df_appl['Position'] = 0
    df_appl.loc[
        (df_appl['RSI'] > 70) & (df_appl['RSI'].shift(1) <= 70),
        'Position'
    ] = 1
    df_appl.loc[
        (df_appl['RSI'] < 30) & (df_appl['RSI'].shift(1) >= 30),
        'Position'
    ] = -1

    #Compute the pnl for each position that we close 10 days after
    Total_pnl = 1
    pnl = 0
    for i in range(len(df_appl) - 10) : 
        P = df_appl.iloc[i]['Position']
        if P == 1 :
            pnl = (df_appl.iloc[i]['Price'] / df_appl.iloc[i + 10]['Price']) - 1
        elif P == -1 :
            pnl = (df_appl.iloc[i + 10]['Price'] / df_appl.iloc[i]['Price']) - 1
        else :
            pnl = 0
        Total_pnl = Total_pnl * (1+pnl)
    
    
    plt.plot(df_appl['Price'], label = 'Spot Price')
    plt.title("Apple Price")
    plt.legend()
    plt.show()

    plt.plot(df_appl['RSI'], label = 'RSI')
    plt.title("RSI Apple")
    plt.axhline(70)
    plt.axhline(30)
    plt.legend()
    plt.savefig('RSI.png')
    plt.show()

    Trades = (df_appl['Position'] != 0).sum()

    print(Trades, " Trades have been made using RSI")
    print("Returns = ", Total_pnl - 1)
    return Total_pnl

RSI()
