"""
demo_backtest_yfinance.py

Standalone, free-data version of the cross-sectional momentum strategy
in momentum_bot.py. Anyone can clone and run this file with no
subscription, credentials, or account of any kind.

WHY THIS FILE EXISTS
---------------------

The full backtest (see the project write-up) uses point-in-time S&P 500
membership from CRSP/WRDS to avoid survivorship bias, and cannot be
redistributed here because that data is licensed and not mine to share.
This file exists so a reviewer without a WRDS subscription can still run
the actual strategy logic end to end and see real output.

WHAT THIS FILE DOES NOT DO
---------------------------

    - It does NOT control for survivorship bias. The ticker universe is
      TODAY's S&P 500 list applied backward across the whole backtest
      window, which silently excludes every historical constituent that
      was later removed (bankruptcies, acquisitions, delistings, index
      demotions). This inflates returns versus a true point-in-time
      backtest, and the effect gets worse the further back you go.
    - It does NOT reproduce the headline numbers from the WRDS-based
      notebook. Do not compare the two CAGR figures and draw conclusions
      from the gap - the gap is partly the survivorship-bias artifact
      described above, not evidence about the strategy itself.
    - It uses yfinance's auto-adjusted close (splits + dividends), so
      unlike the raw-price path in momentum_bot.py this file does not
      have the dividend-adjustment gap noted in the project write-up.

STRATEGY (same rules as momentum_bot.py)
------------------------------------------

    1. Universe: current S&P 500 constituents (GitHub-maintained CSV).
    2. Signal: 12-month return, skipping the most recent month
       (momentum = price[t-1m] / price[t-12m] - 1).
    3. Monthly rebalance, equal-dollar weight across the top TOP_PCT
       of the ranked universe.
    4. Commission model: IBKR tiered rates, same constants as
       momentum_bot.py, applied as a simple round-trip drag estimate.
       These are uncalibrated against real fills - see project notes.

REQUIREMENTS
------------

    pip install yfinance pandas numpy matplotlib requests

Example:

    python demo_backtest_yfinance.py
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import matplotlib.pyplot as plt


# =============================================================================
# CONFIG
# =============================================================================

TOP_PCT = 0.02  # matches momentum_bot.py's TOP_PCT (~top 10 of ~500 names)

LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1

YEARS_OF_HISTORY = 15
STARTING_CASH = 10_000.0

CACHE_DIR = Path("yfinance_cache")
PRICE_CACHE_FILE = CACHE_DIR / "adj_close_prices.csv"

GITHUB_SNP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/master/data/constituents.csv"
)

# Same constants as momentum_bot.py - not re-derived here, not re-validated.
IBKR_TIERED_COMMISSION_PER_SHARE = 0.0035
IBKR_TIERED_MIN_COMMISSION = 0.35
IBKR_PASS_THROUGH_MULTIPLIER = 1.40
ANNUAL_RISK_FREE_RATE = 0.04


# =============================================================================
# UNIVERSE
# =============================================================================

def get_snp500_tickers():
    df = pd.read_csv(GITHUB_SNP500_URL)
    tickers = (
        df["Symbol"]
        .astype(str)
        .str.strip()
        .str.upper()
        .str.replace(".", "-", regex=False)  # yfinance wants BRK-B, not BRK.B
        .tolist()
    )
    return sorted(set(tickers))


# =============================================================================
# DATA
# =============================================================================

def download_prices(tickers, years):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if PRICE_CACHE_FILE.exists():
        print(f"Using cached prices: {PRICE_CACHE_FILE}")
        return pd.read_csv(PRICE_CACHE_FILE, index_col=0, parse_dates=True)

    print(f"Downloading {len(tickers)} tickers + SPY from yfinance...")
    start = time.time()

    data = yf.download(
        tickers + ["SPY"],
        period=f"{years}y",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes = {}
    for t in tickers + ["SPY"]:
        try:
            closes[t] = data[t]["Close"]
        except (KeyError, TypeError):
            continue

    prices = pd.DataFrame(closes).sort_index()
    prices.to_csv(PRICE_CACHE_FILE)

    print(f"Downloaded in {time.time() - start:.1f}s. "
          f"Got usable data for {len(prices.columns)} of {len(tickers) + 1} symbols.")

    return prices


# =============================================================================
# SIGNAL
# =============================================================================

def monthly_momentum_signal(daily_prices):
    monthly = daily_prices.resample("ME").last()

    momentum = (
        monthly.shift(SKIP_MONTHS)
        / monthly.shift(LOOKBACK_MONTHS)
        - 1
    )

    return monthly, momentum


# =============================================================================
# COMMISSION ESTIMATE
# =============================================================================

def estimate_round_trip_commission(dollar_amount, price):
    shares = dollar_amount / price
    per_side = max(
        shares * IBKR_TIERED_COMMISSION_PER_SHARE,
        IBKR_TIERED_MIN_COMMISSION,
    )
    return 2 * per_side * IBKR_PASS_THROUGH_MULTIPLIER  # buy + sell


# =============================================================================
# BACKTEST
# =============================================================================

def backtest(monthly_prices, momentum, top_pct, starting_cash):
    dates = momentum.index[LOOKBACK_MONTHS:]
    cash = starting_cash
    holdings = {}  # ticker -> shares
    curve = []

    for date in dates:
        row = momentum.loc[date].dropna()
        if row.empty:
            curve.append({"Date": date, "Account Value": cash})
            continue

        n_select = max(1, int(len(row) * top_pct))
        picks = row.sort_values(ascending=False).head(n_select).index.tolist()

        # liquidate anything not in the new picks
        for ticker in list(holdings.keys()):
            if ticker not in picks:
                price = monthly_prices.loc[date, ticker]
                if pd.notna(price):
                    proceeds = holdings[ticker] * price
                    proceeds -= estimate_round_trip_commission(proceeds, price) / 2
                    cash += proceeds
                del holdings[ticker]

        # equal-dollar buy into new picks not already held
        new_picks = [t for t in picks if t not in holdings]
        if new_picks and cash > 0:
            dollars_per_pick = cash / len(new_picks)
            for ticker in new_picks:
                price = monthly_prices.loc[date, ticker]
                if pd.isna(price) or price <= 0:
                    continue
                commission = estimate_round_trip_commission(dollars_per_pick, price) / 2
                spendable = dollars_per_pick - commission
                if spendable <= 0:
                    continue
                shares = spendable / price
                holdings[ticker] = shares
                cash -= dollars_per_pick

        portfolio_value = cash + sum(
            holdings.get(t, 0) * monthly_prices.loc[date, t]
            for t in holdings
            if pd.notna(monthly_prices.loc[date, t])
        )
        curve.append({"Date": date, "Account Value": portfolio_value})

    return pd.DataFrame(curve)


def spy_buy_and_hold(monthly_prices, starting_cash):
    spy = monthly_prices["SPY"].dropna()
    shares = starting_cash / spy.iloc[0]
    return pd.DataFrame({
        "Date": spy.index,
        "Account Value": shares * spy.values,
    })


# =============================================================================
# METRICS
# =============================================================================

def performance_metrics(account_df, starting_value, risk_free_rate):
    values = account_df["Account Value"].values
    dates = account_df["Date"]

    years = (dates.iloc[-1] - dates.iloc[0]).days / 365.25
    ending_value = values[-1]
    cagr = (ending_value / starting_value) ** (1 / years) - 1 if years > 0 else np.nan

    running_max = np.maximum.accumulate(values)
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    monthly_returns = pd.Series(values).pct_change().dropna()
    vol = monthly_returns.std() * np.sqrt(12)
    mean_excess = monthly_returns.mean() * 12 - risk_free_rate
    sharpe = mean_excess / vol if vol > 0 else np.nan

    return {
        "Ending Value": ending_value,
        "CAGR %": cagr * 100,
        "Max Drawdown %": max_dd * 100,
        "Annualized Volatility %": vol * 100,
        "Sharpe Ratio": sharpe,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 72)
    print("MOMENTUM STRATEGY DEMO - yfinance, no WRDS/CRSP required")
    print("SURVIVORSHIP BIAS IS NOT CONTROLLED FOR. See module docstring.")
    print("=" * 72)

    tickers = get_snp500_tickers()
    daily_prices = download_prices(tickers, YEARS_OF_HISTORY)
    monthly_prices, momentum = monthly_momentum_signal(daily_prices)

    strategy_curve = backtest(monthly_prices, momentum, TOP_PCT, STARTING_CASH)
    spy_curve = spy_buy_and_hold(monthly_prices, STARTING_CASH)

    strategy_metrics = performance_metrics(strategy_curve, STARTING_CASH, ANNUAL_RISK_FREE_RATE)
    spy_metrics = performance_metrics(spy_curve, STARTING_CASH, ANNUAL_RISK_FREE_RATE)

    results = pd.DataFrame([
        {"Strategy": "Momentum (survivorship-biased demo)", **strategy_metrics},
        {"Strategy": "SPY Buy & Hold", **spy_metrics},
    ])
    results["Excess CAGR vs SPY (pp)"] = (
        results["CAGR %"] - spy_metrics["CAGR %"]
    ).round(2)

    print()
    print(results.to_string(index=False))
    print()
    print("Reminder: this comparison is inflated by survivorship bias.")
    print("Do not cite this CAGR as the strategy's real edge.")

    plt.figure(figsize=(10, 6))
    plt.plot(strategy_curve["Date"], strategy_curve["Account Value"], label="Momentum (demo, survivorship-biased)")
    plt.plot(spy_curve["Date"], spy_curve["Account Value"], label="SPY Buy & Hold")
    plt.legend()
    plt.title(f"Demo backtest (top {TOP_PCT:.0%}, {YEARS_OF_HISTORY}y, yfinance data)")
    plt.ylabel("Account Value ($)")
    plt.xlabel("Date")
    plt.tight_layout()
    plt.savefig("demo_backtest_result.png", dpi=150)
    print("\nChart saved to demo_backtest_result.png")


if __name__ == "__main__":
    main()
