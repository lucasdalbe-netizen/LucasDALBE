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

    #Chiko Span is the n+26 day close price
    df_appl['Chiko_Span'] = df_appl['Price'].shift(-26)

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


    df_plot = df_appl.iloc[80:150]
    
    plt.plot(df_plot['Price'], label = "Spot Price")
    plt.plot(df_plot['Tenkan'], '--', label = "Tenkan")
    plt.plot(df_plot['Kijun_Sen'], '--', label = "Kijun-Sen")
    plt.plot(df_plot['Chiko_Span'], '--', label = "Chiko Span")
    plt.plot(df_plot['Senko_Span_A'], '--', label = "SSA")
    plt.plot(df_plot['Senko_Span_B'], '--', label = "SSB")
    plt.fill_between(df_plot.index,df_plot['Kumo_low'],df_plot['Kumo_high'],alpha=0.3,label="Kumo")
    plt.legend()
    plt.show()
    return

Ichimoku()
