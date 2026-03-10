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
    
    id_open_short = []
    id_open_long = []
    id_close_short = []
    id_close_long = []

    #Stock indexs of the rows where we enter a position
    for i in range (1,len(df_appl)) :
        if df_appl.iloc[i]['Position'] == 1 :
            id_open_short.append(i)
        elif df_appl.iloc[i]['Position'] == -1 :
            id_open_long.append(i)

    #Search for the closest day for each short position where RSI < 50 to close it
    for i in id_open_short :
        for j in range(i+1, len(df_appl)) :
            if df_appl.iloc[j]['RSI'] < 50 :
                id_close_short.append(j)
                break
    
    #Search for the closest day for each long position where RSI > 50 to close it
    for i in id_open_long :
        for j in range(i+1, len(df_appl)) :
            if df_appl.iloc[j]['RSI'] > 50 :
                id_close_long.append(j)
                break
        
    #Compute pnl for each strategy
    Total_pnl_long, Total_pnl_short = 1, 1
    pnl = 0
    for i in range(len(id_open_long)) : 
        a = id_open_long[i]
        b = id_close_long[i]
        pnl = (df_appl.iloc[b]['Price'] /df_appl.iloc[a]['Price']) - 1
        Total_pnl_long = Total_pnl_long * (1+pnl)

    for i in range(len(id_open_short)) :
        a = id_open_short[i]
        b = id_close_short[i]
        pnl = ((df_appl.iloc[a]['Price']/df_appl.iloc[b]['Price']) - 1)
        Total_pnl_short = Total_pnl_short * (1+pnl)

    #Buy & Hold comparison
    Buy_and_Hold = (df_appl.iloc[len(df_appl) - 1]['Price'] - df_appl.iloc[0]['Price']) / df_appl.iloc[0]['Price']
    
    #Plot
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
    print(len(id_open_long), " Long trades")
    print(len(id_open_short), " Short trades")
    print("Returns long = ", Total_pnl_long - 1)
    print("Returns short = ", Total_pnl_short - 1)
    print("Returns Buy & Hold = ", Buy_and_Hold - 1)
    return Total_pnl_long, Total_pnl_short

RSI()
