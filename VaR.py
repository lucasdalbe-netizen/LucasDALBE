import numpy as np
import pandas as pd
import datetime as dt
import random as rd
import os
import matplotlib.pyplot as plt

def vol_and_expected_return_asset_historical():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    df = pd.read_csv(os.path.join(script_dir, '../data/AAPL_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Close', 'High', 'Low', 'Open', 'Volume'])
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    prices = df['Close']
    returns_historical = np.log(prices/prices.shift(1)).dropna()
    mu_daily = returns_historical.mean()
    vol_daily = returns_historical.std()
    mu_year = mu_daily * 252
    vol_year = vol_daily * np.sqrt(252)
    return mu_daily, vol_daily, mu_year, vol_year, returns_historical

mu_daily, vol_daily, mu_year, vol_year, returns_historical = vol_and_expected_return_asset_historical()

def VaR_simulation(simulation, confidence):

    #Simulation MC for the VaR simulated
    daily_returns = mu_daily + vol_daily * np.random.normal(0,1,simulation)
    alpha = 1 - confidence
    
    #Historical VaR
    VaR_day_Historical = np.quantile(returns_historical, alpha).round(4)

    VaR_day_MC = np.quantile(daily_returns, alpha).round(4)
    Var_10_day_MC = (VaR_day_MC * np.sqrt(10)).round(4)
    VaR_year_MC = (VaR_day_MC * np.sqrt(252)).round(4)

    plt.hist(daily_returns, bins = 100, label = "Monte Carlo")
    plt.axvline(VaR_day_MC, color = 'red')
    plt.show()
    
    print(
        'Historical VaR :', - VaR_day_Historical, '\n'
        'Daily simulated VaR :', - VaR_day_MC, '\n'
        '10 Days simulated VaR :',- Var_10_day_MC, '\n'
        '1 Year simulated VaR :', - VaR_year_MC
        )

    return daily_returns, VaR_day_MC, VaR_day_Historical

def Expected_Shortfall(daily_returns, VaR_day_MC, VaR_day_Historical):
    #On average when we are sure to be under the VaR, what is the return
    ES_MC = - daily_returns[daily_returns < VaR_day_MC].mean().round(4)
    ES_Historical = - returns_historical[returns_historical < VaR_day_Historical].mean().round(4)
    print('The expected shortfall simulated is :', ES_MC, '\n'
          'The expected shortfall historical is :', ES_Historical
          )
    return ES_MC, ES_Historical

def Stress_Test_Historical() :
    #Historical worst day
    worst_day = returns_historical.min().round(4)
    print("Historical worst daily return :", worst_day)
    return worst_day

def Stress_Test_n_worst_days(n):
    #loss on the n worst day historical returns cumulate to simulate a n streak of bad days
    worst_days = returns_historical.nsmallest(5)
    cumulative_worst_return_log = worst_days.sum()
    cumulative_worst_return = np.exp(cumulative_worst_return_log) - 1
    loss = - cumulative_worst_return.round(4)
    print("Worst", n, " days cumulative loss is :", loss)
    return loss

def Stress_Test_Vol(vol_variation, simulation, confidence) :
    #Shock on the assets vol
    var_stress = vol_daily * (1+vol_variation)
    new_daily_returns = mu_daily + var_stress * np.random.normal(0,1,simulation)
    alpha = 1 - confidence

    new_VaR_daily = np.quantile(new_daily_returns, alpha).round(4)

    worst_scenario_stress = new_daily_returns.min().round(4)

    new_ES = - new_daily_returns[new_daily_returns<new_VaR_daily].mean().round(4)

    print("If a daily vol shock of ", vol_variation, " occurs, the VaR of this day is : ", new_VaR_daily,'\n'
          "If a daily vol shock of ", vol_variation, " occurs, the worst scenario return is : ", worst_scenario_stress, '\n'
          "The Expected Shortfall of this day relative to this vol variation is :", new_ES)
    return

def Stress_Test_Return(return_variation, simulation, confidence) : 
    #Shock on the asset's returns
    return_stress = mu_daily * (1+return_variation)
    new_daily_returns = return_stress + VaR_day_Historical*np.random.normal(0,1,simulation)
    alpha = 1 - confidence

    new_VaR_daily = np.quantile(new_daily_returns, alpha).round(4)

    worst_scenario_stress = new_daily_returns.min().round(4)

    new_ES = - new_daily_returns[new_daily_returns<new_VaR_daily].mean().round(4)

    print("If a daily return shock of ", return_variation, " occurs, the VaR of this day is : ", new_VaR_daily,'\n'
          "If a daily return shock of ", return_variation, " occurs, the worst scenario return is : ", worst_scenario_stress, '\n'
          "The Expected Shortfall of this day relative to this vol variation is :", new_ES)
    return

def Stress_Test_Jump(jump_size, jump_prob,simulation, confidence):
    #Shock that impact the return of the asset not proportionally to its vol or average return
    jumps = np.random.binomial(1,jump_prob, simulation) * jump_size
    new_daily_returns = mu_daily + vol_daily * np.random.normal(0,1,simulation) + jumps
    alpha = 1 - confidence
    VaR_jump = - np.quantile(new_daily_returns, alpha)
    print("Jump stress VaR :", VaR_jump)
    return

daily_returns, VaR_day_MC, VaR_day_Historical = VaR_simulation(100000, 0.95)

Expected_Shortfall(daily_returns, VaR_day_MC, VaR_day_Historical)
Stress_Test_Historical()
Stress_Test_n_worst_days(10)
Stress_Test_Vol(0.2, 100000, 0.95)
Stress_Test_Return(0.15, 1000000, 0.95)
Stress_Test_Jump(-0.1, 0.05,100000,0.95)
