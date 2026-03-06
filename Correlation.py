import numpy as np
import pandas as pd
import datetime as dt
import random as rd
import os
import matplotlib.pyplot as plt

def correlation() :
    #data collection of 2 firms 
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df_appl = pd.read_csv(os.path.join(script_dir, '../data/AAPL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df_googl = pd.read_csv(os.path.join(script_dir, '../data/GOOGL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df_appl['Date'] = pd.to_datetime(df_appl['Date'])
    df_googl['Date'] = pd.to_datetime(df_googl['Date'])
    df_appl.set_index('Date', inplace=True)
    df_googl.set_index('Date', inplace=True)
    prices_appl = df_appl['Close']
    prices_googl = df_googl['Close']

    #Dayli returns of the assets
    return_appl = np.log(prices_appl/prices_appl.shift(1)).dropna()
    return_googl = np.log(prices_googl/prices_googl.shift(1)).dropna()
  
    #Use of the corr function
    correlation = return_appl.corr(return_googl)

    print("The correlation between those two assets is : ",correlation)
    return correlation

correlation()
