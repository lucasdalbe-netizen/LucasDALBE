import os
import numpy as np
import pandas as pd
import plotly.express as px

sim_runs = 100000 #The more the better
initial_investment = 100000
risk_free_rate = 0.02 
n_assets = 5 #available assets we can create the portfolio with

# Generate random assets returns
expected_returns = np.random.uniform(0.05, 0.15, n_assets)

# Generate random cov matrix between the assets
A = np.random.uniform(0.001, 0.05, (n_assets, n_assets))
covariance_matrix = A @ A.T

#Functions used
def generate_portfolio_weights(sim_runs, n_assets):
    w = np.random.random((sim_runs, n_assets))
    w /= w.sum(axis=1, keepdims=True)  #Sum of weights = 1
    return w

def expected_portfolio_return(weights, expected_returns):
    return float(np.dot(weights, expected_returns))

def portfolio_volatility(weights, covariance_matrix):
    return float(np.sqrt(weights.T @ covariance_matrix @ weights))

def sharpe_ratio(port_return, risk_free_rate, port_vol):
    return (port_return - risk_free_rate) / port_vol if port_vol != 0 else np.nan

def roi(final_value, initial_investment):
    return (final_value - initial_investment) / initial_investment

def final_portfolio_value(initial_investment, expected_return, volatility):
    return initial_investment * (1 + expected_return) * (1 - volatility)

#Simulation
weights_run = generate_portfolio_weights(sim_runs, n_assets)

expected_portfolio_returns = np.zeros(sim_runs)
volatility = np.zeros(sim_runs)
sharpe_ratios = np.zeros(sim_runs)
final_value = np.zeros(sim_runs)
roi_arr = np.zeros(sim_runs)

for i in range(sim_runs):
    w = weights_run[i]
    expected_portfolio_returns[i] = expected_portfolio_return(w, expected_returns)
    volatility[i] = portfolio_volatility(w, covariance_matrix)
    sharpe_ratios[i] = sharpe_ratio(expected_portfolio_returns[i], risk_free_rate, volatility[i])
    final_value[i] = final_portfolio_value(initial_investment, expected_portfolio_returns[i], volatility[i])
    roi_arr[i] = roi(final_value[i], initial_investment)

results_df = pd.DataFrame({
    "expected_return": expected_portfolio_returns,
    "volatility": volatility,
    "sharpe_ratio": sharpe_ratios,
    "final_value": final_value,
    "roi": roi_arr,
    "weights": [np.round(w, 4).tolist() for w in weights_run],
})

#Efficient Markowitz Frontier, Best portfolio to create for each bin
n_bins = 30
results_df["vol_bin"] = pd.cut(results_df["volatility"], bins=n_bins)
frontier = (
    results_df.groupby("vol_bin", observed = True)
    .apply (lambda g: g.loc[g["expected_return"].idxmax()])
    .reset_index(drop=True)
    .sort_values("volatility")
)

fig = px.scatter(
    results_df,
    x="volatility",
    y="expected_return",
    color="sharpe_ratio",
    hover_data=["weights"],
    title="Simulated Portfolio Performance"
)

fig.add_scatter(x=frontier["volatility"], y=frontier["expected_return"], mode="lines", name="Efficient Frontier")
#Frontier de Markowitz

best = results_df.loc[results_df['sharpe_ratio'].idxmax()]
fig.add_scatter(x=[best["volatility"]], y=[best["expected_return"]], mode="markers", name="Best Sharpe Ratio Portfolio", marker=dict(color="red", size=18, symbol="star"))
#Best Sharpe ratio portfolio

fig.update_layout(xaxis_title="Volatility", yaxis_title="Expected Return")
script_dir = os.path.dirname(os.path.abspath(__file__))
filepath = os.path.join(script_dir, "MonteCarlo.png")
fig.write_image(filepath, width=1200, height=800)
print(f"Image sauvegardée à: {filepath}")
