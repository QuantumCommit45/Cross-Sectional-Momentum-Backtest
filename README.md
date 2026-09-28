# Cross-Sectional Momentum Trading Bot

A systematic trading strategy for S&P 500 equities, backtested and
paper-traded live through Interactive Brokers (IBKR).

**Strategy:** rank S&P 500 constituents by 12-minus-1 month trailing
return, hold the top ~2% (≈10 names), equal-weighted, rebalanced monthly.

> **Not investment advice.** This is a personal / portfolio project.
> Nothing here is a recommendation to buy or sell any security. Past
> backtest performance is not indicative of future results. See
> **Known Limitations** below before drawing any conclusion from the
> numbers this code produces.

## Repo contents

| File | What it is |
|---|---|
| `momentum_bot.py` | Live/paper execution bot. Connects to a local IBKR TWS/Gateway instance, ranks the current universe from local historical CSVs, and either submits whole-share orders automatically or generates a manual order sheet for fractional buys. |
| `demo_backtest_yfinance.py` | Standalone backtest runnable by anyone with no paid data subscription. Uses `yfinance`. **Does not control for survivorship bias** — see the file's own docstring and the limitations section below. |
| `momentum_backtesting.ipynb` | The full point-in-time backtest, using CRSP/WRDS data to resolve survivorship bias properly. **Not runnable as-is** — it depends on a WRDS institutional subscription and local CSVs pulled from `crsp.stocknames` and per-PERMNO price history, neither of which are included here because that data is licensed and cannot be redistributed. Outputs are cleared from this file (git-friendly, no local-machine leakage) — for the actual rendered results, see `momentum_backtesting.html` below. |
| `momentum_backtesting.html` | Static export of the same notebook **with output intact**: every printed table, all 5 plot images (including the SNDK isolation test and the parameter heatmaps). This is the file to open if you want to see real results without running anything — the `.ipynb` above is source, this is the evidence. |

Two earlier notebooks (`momentum_backtest_core.ipynb`,
`momentum_validation_extras.ipynb`) covered overlapping ground during
development and are left out of this repo for brevity — everything
load-bearing from them is folded into `momentum_backtesting.ipynb`
above.

## Running the demo

```bash
pip install -r requirements.txt
python demo_backtest_yfinance.py
```

Downloads ~500 tickers of daily price history via `yfinance` on first
run (cached locally after that), computes the momentum signal, runs
the monthly-rebalance backtest, and prints a performance table plus a
saved equity-curve chart.

## Known limitations

- **Survivorship bias.** The `yfinance` demo applies today's S&P 500
  membership backward across the whole backtest window, which excludes
  historical constituents that were later delisted, acquired, or
  removed from the index (Enron, Lehman, etc.). This inflates the
  demo's returns relative to a true point-in-time backtest. The
  WRDS-based analysis resolves this; the public demo does not.
- **Single-stock isolation test.** An earlier concern was that most of
  the strategy's recent gains were concentrated in one stock (SNDK,
  during the 2024–2026 AI/NAND rally). Testing this directly — removing
  SNDK from the eligible universe entirely — showed the opposite:
  CAGR was *higher* without it. SNDK cost the strategy return over
  this window rather than driving it.
- **Dividend adjustment.** `momentum_bot.py`'s live-data path uses
  split-adjusted (not dividend-adjusted) prices, which understates
  trailing returns for high-dividend names. The `yfinance` demo uses
  auto-adjusted (split + dividend) prices, so the two are not directly
  comparable.
- **Commission multiplier.** Both files pad IBKR's tiered commission
  estimate by a 1.4x factor for exchange/regulatory pass-throughs.
  This factor has not been validated against real fills and should be
  treated as a rough estimate, not a calibrated cost model.
- **Fractional shares.** IBKR's API rejects fractional share orders.
  `momentum_bot.py` supports both a whole-share automatic mode and a
  manual mode that generates a dollar-amount order sheet for entry via
  the TWS desktop client.

## Requirements

- Python ≥ 3.10
- `momentum_bot.py`: `ib_async`, `pandas`, `numpy`, `requests`, `lxml`,
  a running IBKR TWS or Gateway instance
- `demo_backtest_yfinance.py`: `yfinance`, `pandas`, `numpy`,
  `matplotlib`

## License

MIT — see `LICENSE`.
