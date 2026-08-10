from datetime import datetime
from matplotlib.ticker import FuncFormatter
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def percentage_formatter(x, pos):
    return "%.1f%%" % (x * 100)


def get_historical_prices(tickers, start, end):
    prices = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False
    )["Close"]
    # yfinance sorts the tickers; restore the caller's order so that the
    # columns line up with the labels used when annotating the plot.
    return prices[list(tickers)]


def get_parabola(a, b, c):
    return lambda r: (a - 2 * b * r + c * r * r) / (a * c - b * b)


if __name__ == '__main__':

    days = 1
    tickers = ["IBM", "GOOG", "AAPL", "TGT", "GS", "MS", "AMZN",
               "MSFT", "WMT", "NKE", "UNH", "PG", "DB", "C", "META", "NVDA"]
    start = datetime(2017, 9, 17)
    end = datetime(2020, 9, 17)
    prices = get_historical_prices(tickers, start, end)
    print(prices)
    percent_change = prices.pct_change(periods=days)
    factor = 252. / days
    mean = percent_change.mean() * factor
    cov = percent_change.cov() * factor
    stdev = np.sqrt(np.diagonal(cov))
    # print(mean)
    # print(cov)
    # print(stdev)
    ones = np.ones(len(tickers))
    inv_cov = np.linalg.inv(cov)
    x = np.dot(mean, inv_cov)
    a = np.dot(x, mean)
    b = np.sum(x)
    c = np.sum(inv_cov)

    r0 = b / c
    sigma2_0 = 1 / c

    r1 = a / b
    sigma2_1 = a / (b * b)

    x_max = max(np.sqrt(sigma2_1), max(stdev))
    y_max = max(r1, max(mean))

    mean_pts = np.arange(-0.5, y_max + 0.05, 0.001)
    parabola = get_parabola(a, b, c)
    stdev_pts = np.sqrt(parabola(mean_pts))

    _, ax = plt.subplots(figsize=(11, 7), layout="constrained")
    ax.set_xlabel(
        "Standard Deviation of Returns (Annualized)",
        fontsize=16
    )
    ax.set_ylabel("Mean Returns (Annualized)", fontsize=16)
    ax.set_title(
        "Historical Returns Mean versus Standard Deviation",
        fontsize=20
    )
    ax.tick_params(labelsize=11)
    formatter = FuncFormatter(percentage_formatter)
    ax.xaxis.set_major_formatter(formatter)
    ax.yaxis.set_major_formatter(formatter)
    ax.grid()
    plt.xlim(left=0.15, right=x_max + 0.02)
    plt.ylim(bottom=-0.15, top=y_max + 0.02)
    plt.scatter(stdev_pts, mean_pts)
    plt.scatter(stdev, mean)
    plt.scatter(np.sqrt(sigma2_0), r0, marker='x', c='black', s=100)
    plt.annotate("GMVP", xy=(np.sqrt(sigma2_0), r0), fontsize=15)
    plt.scatter(np.sqrt(sigma2_1), r1, marker='x', c='black', s=100)
    plt.annotate("SEP", xy=(np.sqrt(sigma2_1), r1), fontsize=15)
    for t, x, y in zip(tickers, stdev, mean):
        plt.annotate(t, xy=(x, y))
    plt.show()
