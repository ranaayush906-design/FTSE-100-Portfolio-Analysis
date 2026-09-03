import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from pathlib import Path


# ==========================================
# 1. Project Setup
# ==========================================

INITIAL_INVESTMENT = 10_000
TRADING_DAYS = 252

TICKERS = ["AZN.L", "HSBA.L", "SHEL.L", "ULVR.L"]

WEIGHTS = {
    "AZN.L": 0.25,
    "HSBA.L": 0.25,
    "SHEL.L": 0.25,
    "ULVR.L": 0.25
}

START_DATE = "2021-01-01"
END_DATE = "2026-01-01"

BASE_DIR = Path(__file__).resolve().parent
CHARTS_DIR = BASE_DIR / "charts"
CHARTS_DIR.mkdir(parents=True, exist_ok=True)

print("FTSE 100 Portfolio Analysis")
print("=" * 40)


# ==========================================
# 2. Download Stock Data
# ==========================================

data = yf.download(
    TICKERS,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=True
)

if data.empty:
    raise ValueError("No stock data was downloaded.")

prices = data["Close"]

# Make sure all required tickers exist
missing_tickers = [ticker for ticker in TICKERS if ticker not in prices.columns]

if missing_tickers:
    raise ValueError(f"Missing stock data for: {missing_tickers}")

prices = prices[TICKERS]


# ==========================================
# 3. Calculate Daily Returns
# ==========================================

returns = prices.pct_change().dropna()


# ==========================================
# 4. Individual Stock Performance
# ==========================================

average_daily_returns = returns.mean()
daily_volatility = returns.std()

annualised_returns = average_daily_returns * TRADING_DAYS
annualised_volatility = daily_volatility * (TRADING_DAYS ** 0.5)

print("\nAnnualised Returns (%):")
print((annualised_returns * 100).round(2))

print("\nAnnualised Volatility (%):")
print((annualised_volatility * 100).round(2))


# ==========================================
# 5. Portfolio Construction
# ==========================================

weights = pd.Series(WEIGHTS)

total_weight = weights.sum()

if abs(total_weight - 1) > 0.0001:
    raise ValueError("Portfolio weights must add up to 1.")

portfolio_returns = (returns * weights).sum(axis=1)

portfolio_average_daily_return = portfolio_returns.mean()
portfolio_daily_volatility = portfolio_returns.std()

portfolio_annual_return = (
    portfolio_average_daily_return * TRADING_DAYS
)

portfolio_annual_volatility = (
    portfolio_daily_volatility * (TRADING_DAYS ** 0.5)
)

print("\nTotal Portfolio Weight:")
print(round(total_weight, 2))

print("\nPortfolio Annualised Return:")
print(f"{portfolio_annual_return * 100:.2f} %")

print("\nPortfolio Annualised Volatility:")
print(f"{portfolio_annual_volatility * 100:.2f} %")


# ==========================================
# 6. Correlation Analysis
# ==========================================

correlation_matrix = returns.corr()

print("\nCorrelation Matrix:")
print(correlation_matrix.round(2))


# ==========================================
# 7. Portfolio Growth
# ==========================================

portfolio_growth = (1 + portfolio_returns).cumprod()

portfolio_value = (
    portfolio_growth * INITIAL_INVESTMENT
)

final_portfolio_value = portfolio_value.iloc[-1]

total_portfolio_return = (
    final_portfolio_value / INITIAL_INVESTMENT - 1
)

# CAGR based on actual investment period
number_of_years = (
    portfolio_value.index[-1] - portfolio_value.index[0]
).days / 365.25

portfolio_cagr = (
    final_portfolio_value / INITIAL_INVESTMENT
) ** (1 / number_of_years) - 1


# ==========================================
# 8. Sharpe Ratio
# ==========================================

risk_free_rate = 0

portfolio_sharpe_ratio = (
    (portfolio_annual_return - risk_free_rate)
    / portfolio_annual_volatility
)


# ==========================================
# 9. Maximum Drawdown
# ==========================================

running_max = portfolio_value.cummax()

drawdown = (
    portfolio_value - running_max
) / running_max

maximum_drawdown = drawdown.min()


# ==========================================
# 10. Stock Performance Chart
# ==========================================

plt.figure(figsize=(12, 6))

for ticker in TICKERS:
    stock_growth = (
        prices[ticker] / prices[ticker].iloc[0]
    )

    plt.plot(
        stock_growth.index,
        stock_growth,
        label=ticker
    )

plt.title("FTSE 100 Stocks - Relative Performance")
plt.xlabel("Date")
plt.ylabel("Growth of £1")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "stock_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 11. Portfolio Performance Chart
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    portfolio_value.index,
    portfolio_value,
    label="Equal-Weight Portfolio"
)

plt.axhline(
    INITIAL_INVESTMENT,
    linestyle="--",
    label="Initial Investment"
)

plt.title("Portfolio Performance")
plt.xlabel("Date")
plt.ylabel("Portfolio Value (£)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "portfolio_performance.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 12. Portfolio Drawdown Chart
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    drawdown.index,
    drawdown * 100
)

plt.title("Portfolio Maximum Drawdown")
plt.xlabel("Date")
plt.ylabel("Drawdown (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "portfolio_drawdown.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 13. Correlation Matrix Heatmap
# ==========================================

plt.figure(figsize=(8, 6))

plt.imshow(
    correlation_matrix,
    interpolation="nearest"
)

plt.colorbar(label="Correlation")

plt.xticks(
    range(len(TICKERS)),
    TICKERS
)

plt.yticks(
    range(len(TICKERS)),
    TICKERS
)

plt.title("Stock Return Correlation Matrix")

for i in range(len(TICKERS)):
    for j in range(len(TICKERS)):
        plt.text(
            j,
            i,
            f"{correlation_matrix.iloc[i, j]:.2f}",
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 14. Download FTSE 100 Benchmark
# ==========================================

benchmark = yf.download(
    "^FTSE",
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=True
)

if benchmark.empty:
    raise ValueError("FTSE 100 benchmark data could not be downloaded.")

benchmark_prices = benchmark["Close"]

# Handle possible DataFrame format
if isinstance(benchmark_prices, pd.DataFrame):
    benchmark_prices = benchmark_prices.iloc[:, 0]

benchmark_prices = benchmark_prices.dropna()

benchmark_returns = benchmark_prices.pct_change().dropna()

benchmark_growth = (
    1 + benchmark_returns
).cumprod()

benchmark_value = (
    benchmark_growth * INITIAL_INVESTMENT
)


# ==========================================
# 15. Align Portfolio and Benchmark
# ==========================================

comparison = pd.concat(
    [
        portfolio_value.rename("Portfolio"),
        benchmark_value.rename("FTSE 100")
    ],
    axis=1
).dropna()

# Rebase both investments to £10,000
comparison = (
    comparison / comparison.iloc[0]
) * INITIAL_INVESTMENT


# ==========================================
# 16. Benchmark Comparison
# ==========================================

final_benchmark_value = comparison["FTSE 100"].iloc[-1]

portfolio_benchmark_return = (
    comparison["Portfolio"].iloc[-1]
    / INITIAL_INVESTMENT
    - 1
)

ftse_total_return = (
    final_benchmark_value
    / INITIAL_INVESTMENT
    - 1
)

benchmark_outperformance = (
    portfolio_benchmark_return
    - ftse_total_return
)


# ==========================================
# 17. Portfolio vs FTSE 100 Chart
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    comparison.index,
    comparison["Portfolio"],
    label="Portfolio"
)

plt.plot(
    comparison.index,
    comparison["FTSE 100"],
    label="FTSE 100"
)

plt.title("Portfolio vs FTSE 100")
plt.xlabel("Date")
plt.ylabel("Investment Value (£)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig(
    CHARTS_DIR / "portfolio_vs_ftse100.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()


# ==========================================
# 18. Stock Summary Table
# ==========================================

summary = pd.DataFrame({
    "Annual Return (%)": annualised_returns * 100,
    "Annual Volatility (%)": annualised_volatility * 100
})

summary = summary.round(2)

print("\nStock Performance Summary:")
print(summary)


# ==========================================
# 19. Final Results
# ==========================================

print("\n")
print("=" * 50)
print("FINAL PORTFOLIO RESULTS")
print("=" * 50)

print(
    f"\nInitial Investment: "
    f"£{INITIAL_INVESTMENT:,.2f}"
)

print(
    f"Final Portfolio Value: "
    f"£{final_portfolio_value:,.2f}"
)

print(
    f"Total Portfolio Return: "
    f"{total_portfolio_return * 100:.2f}%"
)

print(
    f"Portfolio CAGR: "
    f"{portfolio_cagr * 100:.2f}%"
)

print(
    f"Annualised Return: "
    f"{portfolio_annual_return * 100:.2f}%"
)

print(
    f"Annualised Volatility: "
    f"{portfolio_annual_volatility * 100:.2f}%"
)

print(
    f"Sharpe Ratio: "
    f"{portfolio_sharpe_ratio:.2f}"
)

print(
    f"Maximum Drawdown: "
    f"{maximum_drawdown * 100:.2f}%"
)

print(
    f"\nFTSE 100 Final Value: "
    f"£{final_benchmark_value:,.2f}"
)

print(
    f"FTSE 100 Total Return: "
    f"{ftse_total_return * 100:.2f}%"
)

print(
    f"Portfolio vs FTSE 100: "
    f"{benchmark_outperformance * 100:.2f} percentage points"
)

print("\nCharts saved to:")
print(CHARTS_DIR)

print("\nAnalysis complete.")