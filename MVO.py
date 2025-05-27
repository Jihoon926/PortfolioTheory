import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
from scipy.optimize import minimize
import seaborn as sns
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
        target_returns = np.linspace(er.min(), er.max(), 1000)
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
        delta_x = stdev_list[max_sharpe_index]
        delta_y = mean_return_list[max_sharpe_index] - risk_free

        scale = 2

        # 연장된 점
        new_x = delta_x * scale
        new_y = risk_free + delta_y * scale

        # 연장된 선 좌표
        cal_x_extended = [0, new_x]
        cal_y_extended = [risk_free, new_y]
        plt.figure(figsize=(10, 6))
        plt.plot(cal_x_extended, cal_y_extended, 'k--', label="Capital Allocation Line (CAL)")
        plt.plot(stdev_list[:max_sharpe_index*3], mean_return_list[:max_sharpe_index*3], lw=1, label="Efficient Frontier")
        plt.scatter(stdev_list[max_sharpe_index], mean_return_list[max_sharpe_index], color='blue', marker='X', s=180, label='Max Sharpe')

        plt.xlabel("Standard Deviation(Annualized)")
        plt.ylabel("Expected Return(Annualized)")
        plt.title("Efficient Frontier (MVO)")
        # plt.gca().xaxis.set_major_formatter(PercentFormatter(1.0))
        # plt.gca().yaxis.set_major_formatter(PercentFormatter(1.0))
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig("./results/efficient_frontier.png")
        plt.show()

    return mean_return_list, stdev_list, sharpe_ratio_list, weight_list


def get_covariance_plot(returns: pd.DataFrame, save_path=None):
    corr = returns.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Covariance Matrix")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def risk_parity_objective(weights, cov):
    portfolio_variance = np.dot(weights.T, np.dot(cov, weights))
    risk_contributions = np.multiply(weights, np.dot(cov, weights)) / portfolio_variance
    target = risk_contributions.mean()
    risk_parity = np.sum((risk_contributions - target) ** 2)
    return risk_parity

def get_risk_parity_portfolio(er, cov, risk_free):
    n = len(er)
    weights = np.ones(n) / n  # 초기 가중치
    bounds = [(0.0, 1.0)] * n
    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
    result = minimize(risk_parity_objective, weights, args=(cov,), method='SLSQP', bounds=bounds, constraints=constraints)
    if result.success:
        weights = result.x
        portfolio_return = np.dot(weights, er)
        portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov, weights)))
        sharpe_ratio = (portfolio_return - risk_free) / portfolio_std
        return weights, portfolio_return, portfolio_std, sharpe_ratio
    
if __name__ == "__main__":
    data = get_data(file_path="./data/asset_universe.csv")
    pd.set_option('display.max_columns', None)
    print(data.tail())
    risk_free = data["^IRX"][-1] / 100  # Convert to decimal
    data = data.drop(columns=["^IRX"])  # Remove risk-free rate from asset data
    monthly_returns = data.pct_change().dropna()
    get_covariance_plot(monthly_returns, save_path="./results/correlation_matrix.png")
    mean_monthly_returns = (monthly_returns.mean() * 100).round(3)
    
    # Calculate annualized return
    mean_annual_returns = ((1 + monthly_returns.mean()) ** 12 - 1) * 100
    mean_annual_returns = mean_annual_returns.round(3)

    summary_returns = pd.DataFrame({
        "Mean Monthly Return (%)": mean_monthly_returns,
        "Annualized Return (%)": mean_annual_returns
    }).sort_values(by="Annualized Return (%)", ascending=False)
    print(summary_returns)
    expected_returns = monthly_returns.mean() * 12
    covariance = monthly_returns.cov() * 12  # Annualize covariance
    mean_return_list, stdev_list, sharpe_ratio_list, weight_list = get_efficient_portfolios(expected_returns, covariance, risk_free=risk_free)
    column_names = data.columns.tolist()
    max_sharpe_index = np.argmax(sharpe_ratio_list)
    plt.figure(figsize=(16, 10))
    plt.bar(column_names, weight_list[max_sharpe_index], color='skyblue')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights (Max Sharpe)')
    plt.ylim(0, 1)  # 비중은 일반적으로 0~1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./results/portfolio_weights_max_sharpe.png")
    plt.show()
    # A=2라고 가정
    print("risk_free rate:", risk_free)
    for risk_aversion in range(2, 10):
        y = (mean_return_list[max_sharpe_index] - risk_free)/ (risk_aversion * (stdev_list[max_sharpe_index]**2))
        print("Risk portfolio weight:", y)
    
    weights_rp, portfolio_return_rp, portfolio_std_rp, sharpe_ratio_rp = get_risk_parity_portfolio(expected_returns, covariance, risk_free)
    print("Risk Parity Portfolio Weights:", weights_rp)
    print("Risk Parity Portfolio Return:", portfolio_return_rp)
    print("Risk Parity Portfolio Standard Deviation:", portfolio_std_rp)
    print("Risk Parity Portfolio Sharpe Ratio:", sharpe_ratio_rp)

    tickers = column_names
    weights = weight_list[max_sharpe_index]
    start_date = '2025-01-01'
    end_date = '2025-05-27'
    benchmark = 'SPY'

    # 데이터 다운로드
    data = yf.download(tickers + [benchmark], start=start_date, end=end_date)['Close']

    # 일별 수익률
    daily_returns = data.pct_change().dropna()

    # 포트폴리오 일별 수익률 계산
    rp_daily_returns = daily_returns[tickers] @ weights_rp
    rp_daily_returns.name = 'Risk Parity Portfolio Return'
    portfolio_daily_returns = daily_returns[tickers] @ weights
    portfolio_daily_returns.name = 'MVO Portfolio Return'

    # S&P500 일별 수익률
    benchmark_daily_returns = daily_returns[benchmark]
    benchmark_daily_returns.name = 'S&P 500 Return'

    # 누적 수익률 계산
    rp_cum_return = rp_daily_returns.cumsum()
    portfolio_cum_return = portfolio_daily_returns.cumsum()
    benchmark_cum_return = benchmark_daily_returns.cumsum()

    # 시각화
    plt.figure(figsize=(20, 12))
    rp_cum_return.plot(label='Risk Parity Portfolio Cumulative Return')
    portfolio_cum_return.plot(label='MVO Portfolio Cumulative Return')
    benchmark_cum_return.plot(label='S&P 500 Cumulative Return')
    plt.title('Cumulative Returns: MVO Portfolio vs S&P 500')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./results/cumulative_returns_mvo_vs_risk_parity_vs_sp500.png")
    plt.show()

    plt.figure(figsize=(16, 10))
    plt.bar(column_names, weights_rp, color='skyblue')
    plt.ylabel('Weight')
    plt.title('Portfolio Weights (Risk Parity)')
    plt.ylim(0, 1)  # 비중은 일반적으로 0~1
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig("./results/portfolio_weights_risk_parity.png")
    plt.show()
