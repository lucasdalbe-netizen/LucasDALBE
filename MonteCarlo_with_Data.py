import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import os


def get_data(tickers):
    data = {}
    script_dir = os.path.dirname(os.path.abspath(__file__))
    for ticker in tickers :
        df = pd.read_csv(os.path.join(script_dir, f'../data/{ticker}_prices_5y_1d.csv'), skiprows=3, names=['Date', 'Price', 'High', 'Low', 'Open', 'Volume'])
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        # Use daily returns instead of prices
        df['Return'] = df['Price'].pct_change()
        data[ticker] = df
    return data
    
def generate_weights (nb_simulation, data):
    # Generate random weights assignments for each of the assets
    nb_assets = len(data)
    weights = np.random.random((nb_simulation, nb_assets))
    weights = weights / weights.sum(axis=1, keepdims=True)
    # Keep sum(weights = 1)
    return weights

def build_returns_df(data):
    # Build arrays for each ticker with their daily returns
    returns_df = pd.DataFrame()
    for ticker, df in data.items() :
        returns_df[ticker] = df['Return']
    returns_df = returns_df.dropna()
    return returns_df

def portfolio_returns(weights, data):
    returns_df = build_returns_df(data)

    mean_returns = returns_df.mean() * 252  
    cov_matrix = returns_df.cov() * 252

    portfolio_returns = weights @ mean_returns
    portfolio_volatility = []

    for w in weights :
        vol = np.sqrt(w.T @ cov_matrix.values @ w)
        portfolio_volatility.append(vol)

    portfolio_volatility = np.array(portfolio_volatility)

    # Risk free supposed = 0
    sharpe_ratio = portfolio_returns / portfolio_volatility

    results = pd.DataFrame({
        'Return' : portfolio_returns,
        'Volatility' : portfolio_volatility,
        'Sharpe Ratio' : sharpe_ratio
    })

    best_SR = results['Sharpe Ratio'].idxmax()
    best_weights = weights[best_SR]
    min_vol = results['Volatility'].idxmin()
    min_vol_value = results.loc[min_vol, 'Volatility']
    min_vol_weights = weights[min_vol]


    for ticker in data :
        print(ticker, " Annual return is ",mean_returns[ticker])

    print("\nBest Sharpe portfolio:")
    print(results.loc[results['Sharpe Ratio'].idxmax()])
    print("\nBest Sharpe Weights:")
    print(best_weights)
    print("\nMin volatility portfolio:")
    print(min_vol_value)
    print("\nMin volatility portfolio weights:")
    print(min_vol_weights)
    print("\n Assets individual returns")


    return results


def plot_results(results):
    plt.figure(figsize=(10,8))
    scatter = plt.scatter(results['Volatility'], results['Return'], c=results['Sharpe Ratio'])
    plt.colorbar(scatter, label='Sharpe Ratio')

    plt.xlabel('Volatility')
    plt.ylabel('Return')
    plt.title('Portfolio Optimization Simulation')

    best_SR = results.loc[results['Sharpe Ratio'].idxmax()]
    plt.scatter(best_SR['Volatility'], best_SR['Return'], color='red', label='Best SR')
    min_vol = results.loc[results['Volatility'].idxmin()]
    plt.scatter(min_vol['Volatility'], min_vol['Return'], color='blue', label='Min Vol')

    plt.legend()
    plt.show()


tickers = ["AVGO", "ADBE", "AMZN", "ASML", "BAC",
            "BLK", "C", "CRM", "CVX", "GOOGL", "GS", "JPM", "META",
              "MS", "MSFT", "MU", "NVDA", "ORCL", "TSLA", "TSM", "WFC", "XOM"]

data = get_data(tickers)
weights = generate_weights(100000, data)
results = portfolio_returns(weights, data)
plot_results(results)
