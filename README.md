# FTSE 100 Portfolio Analysis

## Overview

This project analyses the historical performance and risk of an equal-weight portfolio of four FTSE 100 companies.

The analysis uses Python, Pandas, Matplotlib and Yahoo Finance data to calculate investment returns, volatility, correlation, Sharpe ratio and maximum drawdown.

The portfolio is also compared against the FTSE 100 index.

## Companies Analysed

- AstraZeneca (AZN.L)
- HSBC (HSBA.L)
- Shell (SHEL.L)
- Unilever (ULVR.L)

Each stock has a 25% portfolio weight.

## Analysis

The project calculates:

- Annualised returns
- Annualised volatility
- Portfolio performance
- Portfolio CAGR
- Sharpe ratio
- Maximum drawdown
- Stock return correlations
- Portfolio vs FTSE 100 performance

An initial investment of £10,000 is used to demonstrate portfolio performance.

## Tools Used

- Python
- Pandas
- Matplotlib
- yfinance

## Key Results

| Metric | Result |
|---|---:|
| Initial Investment | £10,000 |
| Final Portfolio Value | £20,353.03 |
| Total Portfolio Return | 103.53% |
| Portfolio CAGR | 15.32% |
| Annualised Return | 15.23% |
| Annualised Volatility | 14.24% |
| Sharpe Ratio | 1.07 |
| Maximum Drawdown | -14.55% |
| FTSE 100 Final Value | £15,019.59 |

## Project Structure

```text
FTSE-100-Portfolio-Analysis/
├── ftse100_portfolio_analysis.py
├── requirements.txt
├── README.md
└── charts/
    ├── stock_performance.png
    ├── portfolio_performance.png
    ├── portfolio_drawdown.png
    ├── correlation_matrix.png
    └── portfolio_vs_ftse100.png

Methodology
Historical adjusted price data is downloaded using Yahoo Finance.
Daily returns are calculated using percentage changes in adjusted prices.
Portfolio returns are calculated using equal 25% weights across the four stocks.
Annualised volatility is calculated using the standard deviation of daily returns multiplied by the square root of 252 trading days.
The Sharpe ratio is calculated using a 0% risk-free rate.
Portfolio performance is compared with the FTSE 100 index using the same £10,000 starting investment.
Disclaimer
This project is for educational and portfolio demonstration purposes only and does not constitute investment advice.