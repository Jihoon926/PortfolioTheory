import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import pandas_datareader.data as web

tickers = ["GLD", "SLV", "GSG", "BIL", "SHY", "TLT", "EWY", "EWJ", "FXI", "FEZ", "TSLA", "NVDA", "PLTR", "IONQ", "AAPL", "MSFT", "GOOG", "META", "AMZN", "INTC", "MU", 
           "LLY", "COIN", "TEM", "LAES"]

def read_yf_data(tickers, start_date, end_date, save_path=None):
    """
    Read data from Yahoo Finance for the given tickers and date range.
    """
    df = pd.DataFrame()
    for ticker in tickers:
        hist = yf.download(ticker, start=start_date, end=end_date)["Close"]
        df[ticker] = hist
        print(hist.head())
    # 3-month T-Bill interest rate (DTB3) from FRED
    t_bill = web.DataReader("DTB3", "fred", start=start_date, end=end_date)
    t_bill = t_bill.rename(columns={"DTB3": "3M_TBill_Rate"})

    # Merge data
    df = df.join(t_bill, how="outer")
    monthly_prices = df.resample("M").last()
    if save_path:
        monthly_prices.to_csv(save_path)
    return monthly_prices

def read_csv_data(file_path):
    """
    Read data from a CSV file.
    """
    df = pd.read_csv(file_path, index_col=0, parse_dates=True)
    return df

def show_correlation_matrix(df, save_path=None):
    """
    Show the correlation matrix of the given DataFrame.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
    plt.title("Correlation Matrix")
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

if __name__ == "__main__":
    print("yfinance version:", yf.__version__)
    df = read_yf_data(tickers, "2015-05-27", "2025-05-27", "./data/asset_universe.csv")
    # df = read_csv_data("./data/asset_universe.csv")
    # pd.set_option('display.max_columns', None)
    # print(df.head())
    # pd.set_option('display.max_rows', None)
    # print(df["^XDA"])