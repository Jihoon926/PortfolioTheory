import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize
import math


def get_data(file_path=None, tickers=None, start_date=None, end_date=None):
    if file_path is not None:
        df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        df = yf.download(tickers, start=start_date, end=end_date)['Close']
    return df


def get_log_returns_covs(data:pd.DataFrame):
    returns = data.pct_change().dropna()
    # 평균 수익률과 공분산
    mu = returns.mean() 
    cov = returns.cov()
    return mu, cov





def optimize_portfolio(expected_return, covariance, target, weights=None):
    n = len(expected_return)
    if weights is None:
        weights = np.ones(n) / n
    bounds = [(0.0, 1.0)] * n
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'eq', 'fun': lambda w: np.dot(w, expected_return) - target}
        ]
    result = minimize(lambda w: np.sqrt(np.dot(w.T, np.dot(covariance, w))), weights, method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        return result.x, result.fun
    else: 
        return None, None

def get_efficient_portfolios(er, c, risk_free=None, target_returns:list=None, plot:bool=True):
    if target_returns is None:
        target_returns = np.linspace(er.min, er.max(), 100)
    stdev_list = []
    mean_return_list = []
    weight_list = []
    sharpe_ratio_list = []
    for i in target_returns:
        opt_result = optimize_portfolio(er, c, i)
        if opt_result[0] is None:
            continue
        weight = opt_result[0]
        stdev = np.sqrt(np.dot(weight.T, np.dot(c, weight)))
        mean_return = np.dot(weight, er)
        sharpe_ratio = (mean_return - risk_free) / stdev  # Sharpe ratio
        stdev_list.append(stdev)  
        mean_return_list.append(mean_return)
        weight_list.append(weight)
        sharpe_ratio_list.append(sharpe_ratio)

    if plot:
        max_sharpe_index = np.argmax(sharpe_ratio_list)
        print(f"Maximum Sharpe Ratio: {sharpe_ratio_list[max_sharpe_index]}")
        print(f"Optimal Weights: {weight_list[max_sharpe_index]}")
        print(f"Expected Return: {mean_return_list[max_sharpe_index]}")
        print(f"Standard Deviation: {stdev_list[max_sharpe_index]}")
        print("index", max_sharpe_index)
        cal_x = [0, stdev_list[max_sharpe_index]]
        cal_y = [risk_free, mean_return_list[max_sharpe_index]]
        plt.figure(figsize=(10, 6))
        plt.plot(cal_x, cal_y, 'k--', label="Capital Allocation Line (CAL)")
        plt.plot(stdev_list[:max_sharpe_index * 3], mean_return_list[:max_sharpe_index * 3], lw=1, label="Efficient Frontier")
        plt.scatter(stdev_list[max_sharpe_index], mean_return_list[max_sharpe_index], color='blue', marker='X', s=180, label='Max Sharpe (Grid)')

        plt.xlabel("Standard Deviation")
        plt.ylabel("Expected Return")
        plt.title("Efficient Frontier (MVO)")
        # plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
        # plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("efficient_frontier.png")
        plt.show()

    return mean_return_list, stdev_list, sharpe_ratio_list, weight_list

    
if __name__ == "__main__":
    data = get_data(file_path="./data/asset_universe.csv")
    risk_free = data["3M_TBill_Rate"].mean() / 100  # Convert to decimal
    data = data.drop(columns=["3M_TBill_Rate"])  # Remove risk-free rate from asset data
    monthly_returns = data.pct_change().dropna()
    mean_monthly_returns = (monthly_returns.mean() * 100).round(3)

    # Calculate annualized return
    mean_annual_returns = ((1 + monthly_returns.mean()) ** 12 - 1) * 100
    mean_annual_returns = mean_annual_returns.round(3)

    summary_returns = pd.DataFrame({
        "Mean Monthly Return (%)": mean_monthly_returns,
        "Annualized Return (%)": mean_annual_returns
    }).sort_values(by="Annualized Return (%)", ascending=False)
    summary_returns
    expected_returns = monthly_returns.mean() * 12
    covariance = monthly_returns.cov() * 12  # Annualize covariance
    mean_return_list, stdev_list, sharpe_ratio_list, weight_list = get_efficient_portfolios(expected_returns, covariance, risk_free=risk_free)
