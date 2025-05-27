import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def max_drawdown(cumulative_returns):
    rolling_max = np.maximum.accumulate(cumulative_returns)
    drawdowns = (cumulative_returns - rolling_max) / rolling_max
    return drawdowns.min()  # 최대 낙폭 (음수)

def mdd_objective(weights, log_returns):
    portfolio_returns = log_returns @ weights
    cumulative = (1 + portfolio_returns).cumprod()
    return -max_drawdown(cumulative)  # 최소화 위해 음수화

# 제약 조건
n = len(tickers)
bounds = [(0, 1)] * n
constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}

# 최적화 실행
result_mdd = minimize(mdd_objective, np.ones(n)/n,
                      args=(log_returns,), bounds=bounds, constraints=constraints)

# 결과
weights_mdd = result_mdd.x
portfolio_returns_mdd = log_returns @ weights_mdd
cumulative_mdd = (1 + portfolio_returns_mdd).cumprod()

print("\n📉 MDD 최소화 포트폴리오 비중:")
for t, w in sorted(zip(tickers, weights_mdd), key=lambda x: -x[1]):
    if w > 0.01:
        print(f"{t}: {w:.2%}")

print(f"\n✅ 포트폴리오 최대 낙폭 (MDD): {abs(max_drawdown(cumulative_mdd)):.2%}")

# 누적 수익률 시각화
plt.figure(figsize=(10, 6))
plt.plot(cumulative_mdd, label="MDD-Minimized Portfolio")
plt.title("Cumulative Returns of MDD-Minimized Portfolio")
plt.xlabel("Date")
plt.ylabel("Cumulative Return")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("mdd_minimized_portfolio.png")
plt.show()