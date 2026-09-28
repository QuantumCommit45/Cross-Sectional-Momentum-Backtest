"""
momentum_bot.py

Momentum trading bot for IBKR using ib_async.

EXECUTION MODES
---------------

AUTO_WHOLE:
    - Automatically submits BUY and SELL orders through IBKR.
    - BUY orders use WHOLE shares only.
    - SELL orders sell the entire current long position.
    - No fractional shares are sent through the API.

MANUAL_FRACTIONAL:
    - Does NOT automatically execute orders.
    - Generates a manual CSV + console order sheet for TWS.
    - BUYs are specified as DOLLAR / CASH QUANTITY amounts.
    - SELLs are specified as the full current position.

STRATEGY
--------

    1. Refresh current S&P 500 constituents.
    2. Normalize symbols.
    3. Rank the current universe using local historical CSVs.
    4. Momentum = close[-21] / close[-252] - 1
    5. Select top TOP_PCT.
    6. Sell anything outside the top selection.
    7. Sell anything removed from the S&P 500.
    8. Buy only NEW top-pct names.
    9. Equal-dollar allocation among new BUYs.
   10. Always reserve CASH_SAFETY_BUFFER.

IMPORTANT STRATEGY NOTE
-----------------------

This is an ENTRY/EXIT strategy, not a full equal-weight rebalance.

If a stock is already held and remains in the top selection:

    - it is NOT bought again
    - it is NOT trimmed
    - its weight is allowed to drift

Only newly entering stocks receive BUY orders.

REQUIREMENTS
------------

Python >= 3.10
ib_async
pandas
numpy
requests
lxml

Example:

    pip install ib_async pandas numpy requests lxml
"""

from ib_async import IB, Stock, MarketOrder

import asyncio
import math
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# =============================================================================
# CONFIG
# =============================================================================

EXECUTION_MODE = "AUTO_WHOLE"
# Valid:
#   "AUTO_WHOLE"
#   "MANUAL_FRACTIONAL"

DRY_RUN = True
# AUTO_WHOLE only.
#
# True  = print intended orders, submit nothing.
# False = submit real IBKR orders.

TOP_PCT = 0.02

IB_HOST = "127.0.0.1"
IB_PORT = 7497
IB_CLIENT_ID = 88

DATA_STALENESS_WARN_DAYS = 7

# Set True to exclude stale CSVs from ranking.
REJECT_STALE_DATA = False

CASH_SAFETY_BUFFER = 30.00

# Sanity guardrail: warn (do not abort) if any single NEW position would
# exceed this fraction of investable cash. Set to 1.0 to disable.
MAX_POSITION_PCT = 0.35

TRADE_TIMEOUT_SECONDS = 60
ACCOUNT_REFRESH_SECONDS = 2

HISTORICAL_DATA_DIR = Path("yfinancehistoricaldata")

SNP500_PATH = Path("snp500.csv")
SNP500_PREV_PATH = Path("snp500_previous.csv")

ORDER_SHEET_PREFIX = "order_sheet_"

GITHUB_SNP500_URL = (
    "https://raw.githubusercontent.com/datasets/"
    "s-and-p-500-companies/master/data/constituents.csv"
)

WIKI_SNP500_URL = (
    "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
)

HTTP_TIMEOUT_SECONDS = 15


# =============================================================================
# COMMISSION ESTIMATION
# =============================================================================

IBKR_TIERED_COMMISSION_PER_SHARE = 0.0035
IBKR_TIERED_MIN_COMMISSION = 0.35
IBKR_TIERED_MAX_COMMISSION_PCT = 0.01

IBKR_PASS_THROUGH_MULTIPLIER = 1.40


# =============================================================================
# IB CONNECTION
# =============================================================================

ib = IB()


# =============================================================================
# SYMBOL NORMALIZATION
# =============================================================================

def canonical_symbol(symbol):
    """
    Normalize symbols into one internal representation.

    Examples:

        BRK.B
        BF.B
        AAPL

    IBKR may return:

        BRK B
        BF B
    """

    if symbol is None:
        return ""

    symbol = str(symbol).strip().upper()

    if not symbol:
        return ""

    symbol = symbol.replace("/", ".")
    symbol = symbol.replace("-", ".")
    symbol = re.sub(r"\s+", ".", symbol)

    return symbol


def ibkr_symbol(symbol):
    """
    Convert internal canonical symbol to IBKR-style symbol.
    """

    symbol = canonical_symbol(symbol)

    if "." in symbol:
        parts = symbol.split(".")

        if len(parts) == 2:
            return f"{parts[0]} {parts[1]}"

    return symbol


def historical_file_candidates(ticker):
    """
    Support common historical filename conventions.

    Examples:

        BRK.B.csv
        BRK-B.csv
        BRK_B.csv
        BRK B.csv
    """

    canonical = canonical_symbol(ticker)

    variants = [
        canonical,
        canonical.replace(".", "-"),
        canonical.replace(".", "_"),
        canonical.replace(".", " "),
    ]

    paths = []

    for variant in variants:

        path = (
            HISTORICAL_DATA_DIR
            / f"{variant}_adjusted.csv"
        )

        if path not in paths:
            paths.append(path)

    return paths


# =============================================================================
# CONNECTION
# =============================================================================

async def connect_ib():
    """
    Connect to TWS / IB Gateway using the async API.
    """

    if ib.isConnected():
        return

    print(
        f"Connecting to IBKR at "
        f"{IB_HOST}:{IB_PORT} "
        f"with clientId={IB_CLIENT_ID}..."
    )

    await ib.connectAsync(
        IB_HOST,
        IB_PORT,
        clientId=IB_CLIENT_ID,
        timeout=15,
    )

    if not ib.isConnected():
        raise RuntimeError(
            "IBKR connection failed."
        )

    print("IBKR connected.")

    # 3 = delayed market data.
    ib.reqMarketDataType(3)

    await asyncio.sleep(2)


# =============================================================================
# S&P 500
# =============================================================================

def load_local_snp500():
    """
    Load existing local S&P 500 file.
    """

    if not SNP500_PATH.exists():
        return None

    try:

        df = pd.read_csv(SNP500_PATH)

        if "Symbol" not in df.columns:
            raise ValueError(
                f"{SNP500_PATH} has no "
                f"'Symbol' column."
            )

        df = df.copy()

        df["Symbol"] = (
            df["Symbol"]
            .map(canonical_symbol)
        )

        df = df[df["Symbol"] != ""]

        df = (
            df
            .drop_duplicates(subset=["Symbol"])
            .reset_index(drop=True)
        )

        return df

    except Exception as exc:

        print(
            f"WARNING: Could not read "
            f"{SNP500_PATH}: {exc}"
        )

        return None


def refresh_snp500():
    """
    Refresh current S&P 500 constituents.

    Priority:

        1. GitHub
        2. Wikipedia
        3. Existing local file
    """

    old_df = load_local_snp500()

    if old_df is not None:
        old_symbols = set(old_df["Symbol"])
    else:
        old_symbols = set()

    new_df = None
    source_used = None

    # -------------------------------------------------------------------------
    # GitHub
    # -------------------------------------------------------------------------

    try:

        candidate = pd.read_csv(GITHUB_SNP500_URL)

        if "Symbol" not in candidate.columns:
            raise ValueError(
                "GitHub source does not contain "
                "a Symbol column."
            )

        candidate = candidate.copy()

        candidate["Symbol"] = (
            candidate["Symbol"]
            .map(canonical_symbol)
        )

        candidate = candidate[
            candidate["Symbol"] != ""
        ]

        candidate = (
            candidate
            .drop_duplicates(subset=["Symbol"])
            .reset_index(drop=True)
        )

        if len(candidate) < 400:
            raise ValueError(
                f"Unexpectedly small S&P 500 "
                f"list: {len(candidate)}"
            )

        new_df = candidate
        source_used = "GitHub"

    except Exception as exc:

        print(
            f"GitHub S&P 500 source failed: "
            f"{exc}"
        )

        print("Trying Wikipedia fallback...")

    # -------------------------------------------------------------------------
    # Wikipedia
    # -------------------------------------------------------------------------

    if new_df is None:

        try:

            headers = {
                "User-Agent": (
                    "Mozilla/5.0 "
                    "(Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120.0.0.0 Safari/537.36"
                )
            }

            response = requests.get(
                WIKI_SNP500_URL,
                headers=headers,
                timeout=HTTP_TIMEOUT_SECONDS,
            )

            response.raise_for_status()

            tables = pd.read_html(response.text)

            candidate = None

            for table in tables:

                if "Symbol" in table.columns:
                    candidate = table
                    break

            if candidate is None:
                raise ValueError(
                    "Could not find Wikipedia "
                    "table containing Symbol."
                )

            candidate = candidate.copy()

            candidate["Symbol"] = (
                candidate["Symbol"]
                .map(canonical_symbol)
            )

            candidate = candidate[
                candidate["Symbol"] != ""
            ]

            candidate = (
                candidate
                .drop_duplicates(subset=["Symbol"])
                .reset_index(drop=True)
            )

            if len(candidate) < 400:
                raise ValueError(
                    f"Unexpectedly small Wikipedia "
                    f"list: {len(candidate)}"
                )

            new_df = candidate
            source_used = "Wikipedia"

        except Exception as exc:

            print(
                f"WARNING: Wikipedia fallback "
                f"failed: {exc}"
            )

            if old_df is not None:

                print(
                    f"Using existing local "
                    f"{SNP500_PATH}."
                )

                return old_df

            raise RuntimeError(
                "No S&P 500 source available "
                "and no local fallback exists."
            )

    # -------------------------------------------------------------------------
    # Diff
    # -------------------------------------------------------------------------

    new_symbols = set(new_df["Symbol"])

    added = new_symbols - old_symbols
    removed = old_symbols - new_symbols

    print()
    print(
        f"S&P 500 list refreshed from "
        f"{source_used} "
        f"({len(new_symbols)} symbols)"
    )

    if added:
        print(
            "S&P 500 additions since last run: "
            f"{sorted(added)}"
        )

    if removed:
        print(
            "S&P 500 removals since last run: "
            f"{sorted(removed)}"
        )

    if not added and not removed and old_symbols:
        print(
            "No S&P 500 constituent changes "
            "detected since last run."
        )

    if old_df is not None:

        old_df.to_csv(
            SNP500_PREV_PATH,
            index=False,
        )

    new_df.to_csv(SNP500_PATH, index=False)

    return new_df


# =============================================================================
# ACCOUNT HELPERS
# =============================================================================

def parse_account_value(values, tags, currencies=("USD",)):
    """
    Search account values for a matching tag/currency.
    """

    if isinstance(tags, str):
        tags = {tags}
    else:
        tags = set(tags)

    if isinstance(currencies, str):
        currencies = {currencies}
    else:
        currencies = set(currencies)

    for item in values:

        tag = getattr(item, "tag", "")
        currency = getattr(item, "currency", "")

        if tag in tags and currency in currencies:

            try:
                return float(item.value)
            except (TypeError, ValueError):
                pass

    return None


async def find_usd_cash():
    """
    Read USD cash balance.

    Preference:

        $LEDGER-CashBalance / USD

    Fallback:

        TotalCashValue / USD
    """

    values = ib.accountValues()

    result = parse_account_value(
        values,
        {
            "$LEDGER-CashBalance",
            "CashBalance",
        },
    )

    if result is not None:
        return result

    result = parse_account_value(values, "TotalCashValue")

    if result is not None:
        return result

    print(
        "WARNING: Could not find USD cash "
        "in IBKR account values."
    )

    return None


async def find_net_liquidation_usd():
    """
    Read USD net liquidation if available.
    """

    values = ib.accountValues()

    result = parse_account_value(
        values,
        {
            "$LEDGER-NetLiquidationByCurrency",
            "NetLiquidation",
        },
    )

    return result


async def get_current_positions():
    """
    Return all non-zero USD stock positions.

    Result:

        {
            "AAPL": 10.0,
            "BRK.B": 5.0
        }
    """

    positions = await ib.reqPositionsAsync()

    result = {}

    for pos in positions:

        contract = getattr(pos, "contract", None)

        if contract is None:
            continue

        if getattr(contract, "secType", None) != "STK":
            continue

        if getattr(contract, "currency", None) != "USD":
            continue

        ticker = canonical_symbol(
            getattr(contract, "symbol", "")
        )

        if not ticker:
            continue

        try:
            quantity = float(pos.position)
        except (TypeError, ValueError):
            continue

        if quantity == 0:
            continue

        result[ticker] = quantity

    return result


# =============================================================================
# HISTORICAL DATA
# =============================================================================

def load_historical_csv(ticker):
    """
    Load historical CSV for ticker.
    """

    selected_path = None

    for path in historical_file_candidates(ticker):

        if path.exists():
            selected_path = path
            break

    if selected_path is None:
        return None

    try:
        df = pd.read_csv(selected_path)

    except Exception as exc:

        print(
            f"WARNING: Could not read "
            f"{selected_path}: {exc}"
        )

        return None

    required = {"date", "close"}

    missing = required - set(df.columns)

    if missing:

        print(
            f"WARNING: {selected_path} "
            f"missing columns: "
            f"{sorted(missing)}"
        )

        return None

    df = df[["date", "close"]].copy()

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
        utc=True,
    )

    df["close"] = pd.to_numeric(
        df["close"],
        errors="coerce",
    )

    df = df.dropna(subset=["date", "close"])

    df = df[df["close"] > 0]

    df = (
        df
        .sort_values("date")
        .drop_duplicates(
            subset=["date"],
            keep="last",
        )
        .reset_index(drop=True)
    )

    return df


def compute_momentum_from_csv(ticker):
    """
    Calculate:

        close[-21] / close[-252] - 1
    """

    df = load_historical_csv(ticker)

    if df is None:
        return None

    if len(df) < 252:
        return None

    price_21 = float(df["close"].iloc[-21])
    price_252 = float(df["close"].iloc[-252])
    latest_price = float(df["close"].iloc[-1])

    if not all(
        np.isfinite(x)
        for x in (price_21, price_252, latest_price)
    ):
        return None

    if price_21 <= 0 or price_252 <= 0 or latest_price <= 0:
        return None

    momentum = (price_21 / price_252) - 1.0

    latest_date = df["date"].iloc[-1]

    now_utc = pd.Timestamp.now(tz="UTC")

    days_stale = max(
        0,
        int(
            (now_utc - latest_date).total_seconds()
            // 86400
        ),
    )

    return {
        "momentum": float(momentum),
        "price": float(latest_price),
        "latest_date": latest_date,
        "days_stale": days_stale,
    }


def rank_universe_from_csv(tickerlist):
    """
    Rank universe using local CSV data.
    """

    momentum_data = {}

    missing = []
    stale_tickers = []

    for raw_ticker in tickerlist:

        ticker = canonical_symbol(raw_ticker)

        if not ticker:
            continue

        result = compute_momentum_from_csv(ticker)

        if result is None:
            missing.append(ticker)
            continue

        if (
            result["days_stale"]
            > DATA_STALENESS_WARN_DAYS
        ):

            stale_tickers.append(
                (ticker, result["days_stale"])
            )

            if REJECT_STALE_DATA:
                continue

        momentum_data[ticker] = result

    print(
        f"Loaded "
        f"{len(momentum_data)} / "
        f"{len(tickerlist)} "
        f"tickers from local CSVs"
    )

    if missing:

        print(
            f"  Missing/invalid/"
            f"insufficient data for "
            f"{len(missing)} tickers, "
            f"e.g. {missing[:10]}"
        )

    if stale_tickers:

        print(
            f"  WARNING: "
            f"{len(stale_tickers)} tickers "
            f"have data older than "
            f"{DATA_STALENESS_WARN_DAYS} "
            f"days, e.g. "
            f"{stale_tickers[:5]}"
        )

        if REJECT_STALE_DATA:

            print(
                "  Stale tickers were "
                "excluded from ranking."
            )

        else:

            print(
                "  Stale tickers remain "
                "in ranking because "
                "REJECT_STALE_DATA=False."
            )

    return momentum_data


# =============================================================================
# COMMISSION
# =============================================================================

def estimate_ibkr_tiered_buy_commission(
    dollar_amount,
    share_price,
):
    """
    Estimate commission.

    Planning estimate only.
    """

    if dollar_amount <= 0 or share_price <= 0:
        return 0.0

    estimated_shares = dollar_amount / share_price

    base_commission = max(
        estimated_shares
        * IBKR_TIERED_COMMISSION_PER_SHARE,
        IBKR_TIERED_MIN_COMMISSION,
    )

    base_commission = min(
        base_commission,
        dollar_amount
        * IBKR_TIERED_MAX_COMMISSION_PCT,
    )

    return float(
        base_commission
        * IBKR_PASS_THROUGH_MULTIPLIER
    )


# =============================================================================
# EQUAL-DOLLAR ALLOCATION
# =============================================================================

def solve_equal_buy_budget(cash, prices, max_iterations=100):
    """
    Solve equal-dollar BUY allocation.

    Returns (allocation_per_ticker, total_commission).
    """

    if cash <= 0 or not prices:
        return 0.0, 0.0

    clean_prices = []

    for price in prices:

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if np.isfinite(price) and price > 0:
            clean_prices.append(price)

    if not clean_prices:
        return 0.0, 0.0

    n = len(clean_prices)

    allocation = cash / n

    for _ in range(max_iterations):

        total_commission = sum(
            estimate_ibkr_tiered_buy_commission(
                allocation,
                price,
            )
            for price in clean_prices
        )

        new_allocation = max(
            0.0,
            (cash - total_commission) / n,
        )

        if abs(new_allocation - allocation) < 1e-10:
            allocation = new_allocation
            break

        allocation = new_allocation

    total_commission = sum(
        estimate_ibkr_tiered_buy_commission(
            allocation,
            price,
        )
        for price in clean_prices
    )

    return float(allocation), float(total_commission)


# =============================================================================
# WHOLE-SHARE ALLOCATION (FIXED)
# =============================================================================

def build_whole_share_buys(cash, prices_by_ticker):
    """
    Build whole-share BUY quantities.

    Fixes vs. previous version:

        1. Expensive names are no longer silently dropped. If the
           equal-dollar budget can't buy 1 share of an expensive name,
           the allocator will still try to buy 1 share if total cash
           allows. Only if even 1 share is unaffordable is the name
           dropped -- and then explicitly reported.

        2. Leftover-cash sweep is now round-robin across all buys,
           capped at one extra share per ticker per pass, so a single
           cheap ticker can never absorb the whole remainder.

        3. Adds a per-position weight warning.

    Returns:
        {
            ticker: {
                "shares": int,
                "dollar_amount": float,
                "estimated_commission": float,
            }
        }
    """

    if cash <= 0 or not prices_by_ticker:
        return {}

    # -------------------------------------------------------------------------
    # Validate prices
    # -------------------------------------------------------------------------

    valid_prices = {}

    for ticker, price in prices_by_ticker.items():

        try:
            price = float(price)
        except (TypeError, ValueError):
            continue

        if np.isfinite(price) and price > 0:
            valid_prices[ticker] = price

    if not valid_prices:
        return {}

    # -------------------------------------------------------------------------
    # Initial equal-dollar allocation
    # -------------------------------------------------------------------------

    budget_per_ticker, _ = solve_equal_buy_budget(
        cash,
        list(valid_prices.values()),
    )

    buys = {}

    for ticker, price in valid_prices.items():

        shares = int(math.floor(budget_per_ticker / price))

        if shares <= 0:
            continue

        dollar_amount = shares * price

        commission = estimate_ibkr_tiered_buy_commission(
            dollar_amount,
            price,
        )

        buys[ticker] = {
            "shares": shares,
            "dollar_amount": dollar_amount,
            "estimated_commission": commission,
        }

    # -------------------------------------------------------------------------
    # Rescue expensive names:
    #
    # Any ticker that got 0 shares only because its price exceeded the
    # equal-dollar budget gets a chance at 1 share, drawn from the
    # remaining pool. Cheapest first, so we maximize the number of
    # names we can rescue.
    # -------------------------------------------------------------------------

    def total_cost_snapshot(buy_dict):

        return sum(
            item["dollar_amount"]
            + item["estimated_commission"]
            for item in buy_dict.values()
        )

    dropped = [
        ticker
        for ticker in valid_prices
        if ticker not in buys
    ]

    dropped.sort(key=lambda t: valid_prices[t])

    rescued = []
    unrescued = []

    for ticker in dropped:

        price = valid_prices[ticker]

        commission = estimate_ibkr_tiered_buy_commission(
            price,
            price,
        )

        cost = price + commission

        leftover = cash - total_cost_snapshot(buys)

        # Only rescue if there's a per-ticker slice's worth of cash
        # available, so we don't cannibalize other positions entirely.
        per_ticker_slice = cash / max(1, len(valid_prices))

        if (
            leftover >= cost
            and leftover >= per_ticker_slice
        ):

            buys[ticker] = {
                "shares": 1,
                "dollar_amount": price,
                "estimated_commission": commission,
            }

            rescued.append(ticker)

        else:

            unrescued.append(ticker)

    if rescued:

        print(
            f"  NOTE: bought 1 share of "
            f"{len(rescued)} expensive names "
            f"that the equal-dollar budget "
            f"could not reach: {sorted(rescued)}"
        )

    if unrescued:

        print(
            f"  WARNING: {len(unrescued)} names "
            f"were DROPPED because even 1 "
            f"share was unaffordable with "
            f"remaining cash: {sorted(unrescued)}"
        )

    # -------------------------------------------------------------------------
    # Round-robin leftover sweep
    #
    # Distribute leftover cash across all buys, one extra share per
    # ticker per pass, cheapest-affordable-first within each pass.
    # This never concentrates the remainder in a single name.
    # -------------------------------------------------------------------------

    max_passes = 100

    for _ in range(max_passes):

        leftover = cash - total_cost_snapshot(buys)

        if leftover <= 0 or not buys:
            break

        made_progress = False

        # Cheapest affordable first within each pass.
        candidates = sorted(
            buys.keys(),
            key=lambda t: valid_prices[t],
        )

        for ticker in candidates:

            price = valid_prices[ticker]
            item = buys[ticker]

            old_total = (
                item["dollar_amount"]
                + item["estimated_commission"]
            )

            new_shares = item["shares"] + 1
            new_dollar_amount = new_shares * price

            new_commission = (
                estimate_ibkr_tiered_buy_commission(
                    new_dollar_amount,
                    price,
                )
            )

            new_total = new_dollar_amount + new_commission

            incremental_cost = new_total - old_total

            if incremental_cost <= leftover:

                item["shares"] = new_shares
                item["dollar_amount"] = new_dollar_amount
                item["estimated_commission"] = new_commission

                leftover -= incremental_cost
                made_progress = True

        if not made_progress:
            break

    # -------------------------------------------------------------------------
    # Final affordability pass
    # -------------------------------------------------------------------------

    while buys and total_cost_snapshot(buys) > cash + 1e-9:

        ticker = max(
            buys,
            key=lambda t: (
                buys[t]["dollar_amount"]
                + buys[t]["estimated_commission"]
            ),
        )

        item = buys[ticker]
        item["shares"] -= 1

        if item["shares"] <= 0:
            del buys[ticker]
            continue

        price = valid_prices[ticker]

        item["dollar_amount"] = item["shares"] * price
        item["estimated_commission"] = (
            estimate_ibkr_tiered_buy_commission(
                item["dollar_amount"],
                price,
            )
        )

    if total_cost_snapshot(buys) > cash + 1e-6:
        raise RuntimeError(
            "Internal allocation error: "
            "whole-share BUY cost exceeds "
            "available cash."
        )

    # -------------------------------------------------------------------------
    # Per-position weight warning
    # -------------------------------------------------------------------------

    if buys and cash > 0:

        for ticker, item in sorted(buys.items()):

            weight = (
                item["dollar_amount"] + item["estimated_commission"]
            ) / cash

            if weight > MAX_POSITION_PCT:

                print(
                    f"  WARNING: {ticker} would "
                    f"be {weight:.1%} of investable "
                    f"cash (${item['dollar_amount']:,.2f}) "
                    f"-- above MAX_POSITION_PCT="
                    f"{MAX_POSITION_PCT:.0%}."
                )

    return buys


# =============================================================================
# ORDER PLAN
# =============================================================================

async def build_order_plan(
    current_symbols,
    removed_symbols,
    top_pct=TOP_PCT,
):
    """
    Build order plan.
    """

    if not (0 < top_pct <= 1):
        raise ValueError(
            "top_pct must be > 0 and <= 1."
        )

    normalized_symbols = sorted(
        {
            canonical_symbol(symbol)
            for symbol in current_symbols
            if canonical_symbol(symbol)
        }
    )

    momentum_data = rank_universe_from_csv(
        normalized_symbols
    )

    if not momentum_data:
        raise RuntimeError(
            "No usable historical data "
            "was available."
        )

    ranked = sorted(
        momentum_data.items(),
        key=lambda item: item[1]["momentum"],
        reverse=True,
    )

    n_holdings = max(
        1,
        int(len(ranked) * top_pct),
    )

    top_picks = {
        ticker
        for ticker, _
        in ranked[:n_holdings]
    }

    current_positions = await get_current_positions()

    current_tickers = set(current_positions.keys())

    removed_symbols = {
        canonical_symbol(symbol)
        for symbol in removed_symbols
    }

    to_sell = (
        current_tickers - top_picks
    ) | (
        current_tickers & removed_symbols
    )

    to_buy = top_picks - current_tickers

    net_liq = await find_net_liquidation_usd()

    cash = await find_usd_cash()

    if cash is None:
        raise RuntimeError(
            "Could not determine USD "
            "cash balance."
        )

    print()

    if net_liq is not None:

        print(
            f"Net liquidation (USD): "
            f"${net_liq:,.2f}"
        )

    else:

        print(
            "Net liquidation (USD): "
            "unavailable"
        )

    print(
        f"Current USD cash: "
        f"${cash:,.2f}"
    )

    print(
        f"Safety buffer: "
        f"${CASH_SAFETY_BUFFER:,.2f}"
    )

    print(
        f"Currently holding "
        f"{len(current_tickers)} positions"
    )

    print(
        f"Usable ranked universe: "
        f"{len(ranked)} names"
    )

    print(
        f"Top {top_pct:.2%}: "
        f"{len(top_picks)} names"
    )

    print(
        f"Selling {len(to_sell)}, "
        f"Buying {len(to_buy)}"
    )

    return {
        "momentum_data": momentum_data,
        "ranked": ranked,
        "top_picks": top_picks,
        "current_positions": current_positions,
        "to_sell": to_sell,
        "to_buy": to_buy,
        "removed_symbols": removed_symbols,
        "cash": float(cash),
        "net_liq": net_liq,
    }


# =============================================================================
# MANUAL FRACTIONAL
# =============================================================================

def generate_manual_fractional_order_sheet(plan):
    """
    Generate manual TWS order sheet.
    """

    momentum_data = plan["momentum_data"]
    current_positions = plan["current_positions"]
    to_sell = plan["to_sell"]
    to_buy = plan["to_buy"]
    removed_symbols = plan["removed_symbols"]
    cash = plan["cash"]

    investable_cash = max(
        0.0,
        cash - CASH_SAFETY_BUFFER,
    )

    orders = []

    # -------------------------------------------------------------------------
    # SELLS
    # -------------------------------------------------------------------------

    for ticker in sorted(to_sell):

        position = current_positions[ticker]

        if position <= 0:
            continue

        if ticker in removed_symbols:
            reason = "removed from S&P 500"
        else:
            reason = "fell out of top pct"

        orders.append(
            {
                "action": "SELL",
                "ticker": ticker,
                "shares": position,
                "amount": "ALL SHARES",
                "reason": reason,
            }
        )

    # -------------------------------------------------------------------------
    # BUYS
    # -------------------------------------------------------------------------

    valid_buy_prices = {}

    for ticker in sorted(to_buy):

        price = momentum_data[ticker]["price"]

        if price > 0 and np.isfinite(price):
            valid_buy_prices[ticker] = price

    if valid_buy_prices:

        budget_per_ticker, _ = solve_equal_buy_budget(
            investable_cash,
            list(valid_buy_prices.values()),
        )

        for ticker in sorted(valid_buy_prices):

            price = valid_buy_prices[ticker]

            dollar_amount = budget_per_ticker

            commission = estimate_ibkr_tiered_buy_commission(
                dollar_amount,
                price,
            )

            reference_shares = dollar_amount / price

            orders.append(
                {
                    "action": "BUY",
                    "ticker": ticker,
                    "dollar_amount": dollar_amount,
                    "reference_shares": reference_shares,
                    "estimated_commission": commission,
                    "reason": "new top pct pick",
                }
            )

    # -------------------------------------------------------------------------
    # CONSOLE
    # -------------------------------------------------------------------------

    print()
    print("=" * 72)
    print("MANUAL FRACTIONAL ORDER SHEET")
    print("=" * 72)

    print(f"USD cash: ${cash:,.2f}")
    print(
        f"Safety buffer: "
        f"${CASH_SAFETY_BUFFER:,.2f}"
    )
    print(
        f"Investable cash: "
        f"${investable_cash:,.2f}"
    )

    sell_orders = [
        order for order in orders
        if order["action"] == "SELL"
    ]

    buy_orders = [
        order for order in orders
        if order["action"] == "BUY"
    ]

    if sell_orders:

        print()
        print(f"SELL ({len(sell_orders)})")

        for order in sell_orders:

            print(
                f"  SELL ALL SHARES "
                f"{order['ticker']:<8} "
                f"({order['reason']})"
            )

    if buy_orders:

        print()
        print(f"BUY ({len(buy_orders)})")
        print(
            "  Enter each BUY in TWS "
            "using CASH QUANTITY."
        )

        for order in buy_orders:

            print(
                f"  BUY "
                f"${order['dollar_amount']:>10,.2f} "
                f"{order['ticker']:<8} "
                f"(reference shares "
                f"{order['reference_shares']:.6f}, "
                f"estimated commission "
                f"${order['estimated_commission']:.2f})"
            )

    rows = []

    for order in sell_orders:

        rows.append(
            {
                "action": "SELL",
                "ticker": order["ticker"],
                "amount": "ALL SHARES",
                "shares": order["shares"],
                "estimated_commission": "",
                "reason": order["reason"],
                "done": False,
            }
        )

    for order in buy_orders:

        rows.append(
            {
                "action": "BUY",
                "ticker": order["ticker"],
                "amount": round(order["dollar_amount"], 2),
                "shares": "",
                "estimated_commission": round(
                    order["estimated_commission"], 2
                ),
                "reason": order["reason"],
                "done": False,
            }
        )

    if rows:

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )

        filename = (
            f"{ORDER_SHEET_PREFIX}{timestamp}.csv"
        )

        pd.DataFrame(rows).to_csv(
            filename,
            index=False,
        )

        print()
        print(f"Saved to {filename}")

    else:

        print()
        print("No orders to process.")

    print("=" * 72)


# =============================================================================
# AUTO WHOLE BUY GENERATION
# =============================================================================

def build_auto_whole_buy_orders(plan, available_cash):
    """
    Build whole-share BUY orders.
    """

    if available_cash is None:
        return []

    to_buy = plan["to_buy"]
    momentum_data = plan["momentum_data"]

    investable_cash = max(
        0.0,
        float(available_cash) - CASH_SAFETY_BUFFER,
    )

    if not to_buy:
        return []

    prices = {}

    for ticker in to_buy:

        price = momentum_data[ticker]["price"]

        if price > 0 and np.isfinite(price):
            prices[ticker] = float(price)

    if not prices:
        return []

    buy_details = build_whole_share_buys(
        investable_cash,
        prices,
    )

    orders = []

    for ticker in sorted(buy_details):

        item = buy_details[ticker]

        orders.append(
            {
                "ticker": ticker,
                "action": "BUY",
                "shares": int(item["shares"]),
                "price": prices[ticker],
                "dollar_amount": float(
                    item["dollar_amount"]
                ),
                "estimated_commission": float(
                    item["estimated_commission"]
                ),
                "reason": "new top pct pick",
            }
        )

    return orders


# =============================================================================
# CONTRACT
# =============================================================================

async def qualify_stock_contract(ticker):
    """
    Create and qualify US stock contract.
    """

    symbol = ibkr_symbol(ticker)

    contract = Stock(symbol, "SMART", "USD")

    qualified = await ib.qualifyContractsAsync(
        contract
    )

    if not qualified:

        raise RuntimeError(
            f"Could not qualify IBKR "
            f"contract for {ticker} "
            f"(IB symbol: {symbol})."
        )

    return qualified[0]


# =============================================================================
# ORDER CONSTRUCTION
# =============================================================================

def make_market_order(action, quantity):
    """
    Create a MarketOrder with an explicit TIF.

    Fixes IBKR error 10349
    (Order TIF was set to DAY based on order preset).
    """

    order = MarketOrder(action, quantity)

    order.tif = "DAY"

    return order


# =============================================================================
# TRADE STATUS
# =============================================================================

async def wait_for_trade(
    trade,
    timeout_seconds=TRADE_TIMEOUT_SECONDS,
):
    """
    Wait until trade is done or timeout expires.
    """

    start = time.perf_counter()

    while True:

        if trade.isDone():
            return trade

        elapsed = time.perf_counter() - start

        if elapsed >= timeout_seconds:
            return trade

        await asyncio.sleep(0.25)


def trade_status(trade):
    """
    Safely retrieve trade status.
    """

    status = getattr(
        getattr(trade, "orderStatus", None),
        "status",
        None,
    )

    if status is None:
        return ""

    return str(status)


def trade_filled_quantity(trade):
    """
    Safely retrieve filled quantity.
    """

    try:
        return float(trade.orderStatus.filled)

    except (AttributeError, TypeError, ValueError):
        return 0.0


def trade_is_filled(trade, expected_quantity):
    """
    Require complete fill.
    """

    status = trade_status(trade)
    filled = trade_filled_quantity(trade)

    if status != "Filled":
        return False

    return filled >= float(expected_quantity) - 1e-9


# =============================================================================
# POSITION VALIDATION
# =============================================================================

def validate_auto_whole_positions(current_positions):
    """
    AUTO_WHOLE only supports long whole-share positions.
    """

    problems = []

    for ticker, quantity in current_positions.items():

        if quantity < 0:

            problems.append(
                f"{ticker}: short "
                f"position {quantity}"
            )

            continue

        rounded = round(quantity)

        if abs(quantity - rounded) > 1e-9:

            problems.append(
                f"{ticker}: fractional "
                f"position {quantity}"
            )

    if problems:

        print()
        print("=" * 72)
        print("AUTO_WHOLE POSITION VALIDATION FAILED")
        print("=" * 72)

        for problem in problems:
            print(f"  {problem}")

        print()
        print(
            "No automatic orders will be "
            "submitted."
        )
        print(
            "AUTO_WHOLE requires existing "
            "positions to be long whole "
            "shares."
        )

        return False

    return True


# =============================================================================
# AUTO WHOLE SELLS
# =============================================================================

async def execute_auto_whole_sells(plan):
    """
    Submit SELL orders first.

    Every SELL must fully fill before BUYs proceed.
    """

    current_positions = plan["current_positions"]
    to_sell = plan["to_sell"]

    if not to_sell:

        print()
        print("No SELL orders.")
        return True

    print()
    print("=" * 72)
    print("AUTO WHOLE-SHARE SELL ORDERS")
    print("=" * 72)

    for ticker in sorted(to_sell):

        position = current_positions.get(ticker, 0.0)

        if position <= 0:

            print(
                f"Skipping {ticker}: "
                f"non-positive position "
                f"{position}"
            )

            continue

        shares = int(round(position))

        if shares <= 0:

            print(
                f"Skipping {ticker}: "
                f"no whole shares."
            )

            continue

        contract = await qualify_stock_contract(ticker)

        order = make_market_order("SELL", shares)

        print(f"SELL {shares} {ticker}")

        if DRY_RUN:

            print(
                "  [DRY RUN] "
                "Order not submitted."
            )

            continue

        trade = ib.placeOrder(contract, order)
        trade = await wait_for_trade(trade)

        status = trade_status(trade)
        filled = trade_filled_quantity(trade)

        if not trade_is_filled(trade, shares):

            print(
                f"  ERROR: SELL "
                f"{ticker} did not "
                f"fully fill. "
                f"status={status!r}, "
                f"filled={filled}, "
                f"requested={shares}"
            )

            return False

        print(
            f"  SELL {ticker} "
            f"filled: {filled:g}"
        )

    if DRY_RUN:

        print()
        print(
            "[DRY RUN] No actual "
            "SELLs were submitted."
        )

        return True

    await asyncio.sleep(ACCOUNT_REFRESH_SECONDS)

    return True


# =============================================================================
# AUTO WHOLE BUYS
# =============================================================================

async def execute_auto_whole_buys(plan):
    """
    Refresh cash after completed sells and submit BUYs.
    """

    if DRY_RUN:

        available_cash = plan["cash"]

    else:

        available_cash = await find_usd_cash()

        if available_cash is None:

            print(
                "ERROR: Could not read "
                "updated USD cash after "
                "SELLs."
            )

            return False

    orders = build_auto_whole_buy_orders(
        plan,
        available_cash,
    )

    print()
    print("=" * 72)
    print("AUTO WHOLE-SHARE BUY ORDERS")
    print("=" * 72)

    print(
        f"USD cash used for sizing: "
        f"${available_cash:,.2f}"
    )
    print(
        f"Safety buffer: "
        f"${CASH_SAFETY_BUFFER:,.2f}"
    )

    if not orders:

        print("No BUY orders.")
        return True

    # -------------------------------------------------------------------------
    # Coverage check: did the allocator drop any names that were supposed
    # to be bought?
    # -------------------------------------------------------------------------

    planned_tickers = set(plan["to_buy"])
    built_tickers = {o["ticker"] for o in orders}

    missing_tickers = planned_tickers - built_tickers

    if missing_tickers:

        print()
        print(
            f"WARNING: {len(missing_tickers)} of "
            f"{len(planned_tickers)} planned BUY "
            f"names were dropped by the allocator "
            f"(most likely too expensive for "
            f"available cash): "
            f"{sorted(missing_tickers)}"
        )

    # -------------------------------------------------------------------------
    # Print planned orders
    # -------------------------------------------------------------------------

    investable_cash = max(
        0.0,
        available_cash - CASH_SAFETY_BUFFER,
    )

    total_dollar = 0.0
    total_commission = 0.0

    print()

    for order in orders:

        total_dollar += order["dollar_amount"]
        total_commission += order["estimated_commission"]

        weight = (
            order["dollar_amount"]
            + order["estimated_commission"]
        ) / investable_cash if investable_cash > 0 else 0.0

        print(
            f"BUY "
            f"{order['shares']} shares "
            f"{order['ticker']:<8} "
            f"(~${order['dollar_amount']:,.2f}, "
            f"est. commission "
            f"${order['estimated_commission']:.2f}, "
            f"weight {weight:.1%})"
        )

    total_estimated_cost = (
        total_dollar + total_commission
    )

    print()
    print(
        f"Estimated BUY principal: "
        f"${total_dollar:,.2f}"
    )
    print(
        f"Estimated commissions: "
        f"${total_commission:,.2f}"
    )
    print(
        f"Estimated total BUY cost: "
        f"${total_estimated_cost:,.2f}"
    )
    print(
        f"Investable BUY budget: "
        f"${investable_cash:,.2f}"
    )

    if total_estimated_cost > investable_cash + 1e-6:

        print(
            "ERROR: calculated BUY cost "
            "exceeds investable cash."
        )

        return False

    if DRY_RUN:

        print()
        print(
            "[DRY RUN] No BUY "
            "orders were submitted."
        )

        return True

    # -------------------------------------------------------------------------
    # Submit sequentially.
    # -------------------------------------------------------------------------

    for order_info in orders:

        ticker = order_info["ticker"]
        shares = int(order_info["shares"])

        if shares <= 0:
            continue

        current_cash = await find_usd_cash()

        if current_cash is None:

            print(
                "ERROR: Could not refresh "
                "USD cash before "
                f"BUY {ticker}."
            )

            return False

        contract = await qualify_stock_contract(ticker)

        order = make_market_order("BUY", shares)

        print()
        print(
            f"Submitting BUY "
            f"{shares} {ticker} "
            f"(actual cash before order: "
            f"${current_cash:,.2f})..."
        )

        trade = ib.placeOrder(contract, order)
        trade = await wait_for_trade(trade)

        status = trade_status(trade)
        filled = trade_filled_quantity(trade)

        if not trade_is_filled(trade, shares):

            print(
                f"WARNING: BUY "
                f"{ticker} did not "
                f"fully fill. "
                f"status={status!r}, "
                f"filled={filled}, "
                f"requested={shares}"
            )
            print("Stopping further BUY submission.")
            return False

        print(
            f"BUY {ticker} "
            f"filled: {filled:g}"
        )

    return True


# =============================================================================
# AUTO MODE
# =============================================================================

async def run_auto_whole(plan):
    """
    Full automatic whole-share execution.
    """

    print()
    print("=" * 72)
    print("EXECUTION MODE: AUTO_WHOLE")
    print("=" * 72)

    print(f"DRY_RUN = {DRY_RUN}")

    if DRY_RUN:
        print("No real orders will be submitted.")
    else:
        print("REAL IBKR ORDERS WILL BE SUBMITTED.")

    if not validate_auto_whole_positions(
        plan["current_positions"]
    ):
        return False

    sell_ok = await execute_auto_whole_sells(plan)

    if not sell_ok:

        print()
        print(
            "ABORTING BUY SIDE because "
            "one or more SELL orders "
            "did not fully fill."
        )

        return False

    buy_ok = await execute_auto_whole_buys(plan)

    if not buy_ok:

        print()
        print(
            "BUY execution encountered "
            "an error or partial fill."
        )

        return False

    print()
    print("=" * 72)
    print("AUTO WHOLE-SHARE EXECUTION COMPLETE")
    print("=" * 72)

    return True


# =============================================================================
# MANUAL MODE
# =============================================================================

async def run_manual_fractional(plan):
    """
    Generate manual TWS fractional order sheet.
    """

    print()
    print("=" * 72)
    print("EXECUTION MODE: MANUAL_FRACTIONAL")
    print("=" * 72)
    print("No API orders will be submitted.")

    generate_manual_fractional_order_sheet(plan)


# =============================================================================
# VALIDATION
# =============================================================================

def validate_config():
    """
    Validate configuration.
    """

    valid_modes = {
        "AUTO_WHOLE",
        "MANUAL_FRACTIONAL",
    }

    if EXECUTION_MODE not in valid_modes:
        raise ValueError(
            f"Invalid EXECUTION_MODE: "
            f"{EXECUTION_MODE!r}"
        )

    if not (
        isinstance(TOP_PCT, (int, float))
        and 0 < TOP_PCT <= 1
    ):
        raise ValueError(
            "TOP_PCT must be > 0 and <= 1."
        )

    if (
        not isinstance(CASH_SAFETY_BUFFER, (int, float))
        or CASH_SAFETY_BUFFER < 0
    ):
        raise ValueError(
            "CASH_SAFETY_BUFFER must be non-negative."
        )

    if IB_PORT <= 0:
        raise ValueError("IB_PORT must be positive.")

    if IB_CLIENT_ID < 0:
        raise ValueError(
            "IB_CLIENT_ID must be non-negative."
        )

    if (
        not isinstance(MAX_POSITION_PCT, (int, float))
        or MAX_POSITION_PCT <= 0
    ):
        raise ValueError(
            "MAX_POSITION_PCT must be positive."
        )

    HISTORICAL_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


# =============================================================================
# MAIN
# =============================================================================

async def main():
    """
    Main program.
    """

    validate_config()

    print()
    print("=" * 72)
    print("MOMENTUM BOT")
    print("=" * 72)

    print(
        "Run started: "
        f"{datetime.now(timezone.utc).isoformat()}"
    )
    print(f"Execution mode: {EXECUTION_MODE}")
    print(f"DRY_RUN: {DRY_RUN}")
    print(f"TOP_PCT: {TOP_PCT:.2%}")
    print(
        f"Safety buffer: "
        f"${CASH_SAFETY_BUFFER:,.2f}"
    )
    print(
        f"Reject stale data: "
        f"{REJECT_STALE_DATA}"
    )
    print(
        f"Max single-position weight: "
        f"{MAX_POSITION_PCT:.0%}"
    )

    await connect_ib()

    old_df = load_local_snp500()

    if old_df is not None:
        previous_symbols = set(old_df["Symbol"])
    else:
        previous_symbols = set()

    snp_df = refresh_snp500()

    current_symbols_list = [
        canonical_symbol(symbol)
        for symbol in snp_df["Symbol"].tolist()
    ]

    current_symbols = set(current_symbols_list)

    removed_symbols = (
        previous_symbols - current_symbols
    )

    plan = await build_order_plan(
        current_symbols_list,
        removed_symbols,
        top_pct=TOP_PCT,
    )

    print()
    print("=" * 72)
    print("TOP MOMENTUM SELECTION")
    print("=" * 72)

    top_count = len(plan["top_picks"])

    for rank, (ticker, data) in enumerate(
        plan["ranked"][:top_count],
        start=1,
    ):

        print(
            f"{rank:>3}. "
            f"{ticker:<8} "
            f"momentum="
            f"{data['momentum']:>9.2%} "
            f"price="
            f"${data['price']:>10,.2f} "
            f"data_age="
            f"{data['days_stale']}d"
        )

    print()
    print("=" * 72)
    print("ORDER PLAN")
    print("=" * 72)

    if plan["to_sell"]:

        print("SELL:")

        for ticker in sorted(plan["to_sell"]):

            position = plan["current_positions"].get(
                ticker, 0
            )

            if ticker in plan["removed_symbols"]:
                reason = "removed from S&P 500"
            else:
                reason = "fell out of top pct"

            print(
                f"  SELL ALL "
                f"{ticker:<8} "
                f"current={position:g} "
                f"({reason})"
            )

    else:

        print("SELL: none")

    if plan["to_buy"]:

        print("BUY:")

        for ticker in sorted(plan["to_buy"]):

            momentum = plan["momentum_data"][ticker][
                "momentum"
            ]
            price = plan["momentum_data"][ticker]["price"]

            print(
                f"  BUY NEW "
                f"{ticker:<8} "
                f"momentum="
                f"{momentum:.2%} "
                f"reference price="
                f"${price:,.2f}"
            )

    else:

        print("BUY: none")

    if EXECUTION_MODE == "AUTO_WHOLE":
        await run_auto_whole(plan)

    elif EXECUTION_MODE == "MANUAL_FRACTIONAL":
        await run_manual_fractional(plan)

    else:
        raise RuntimeError("Unexpected execution mode.")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":

    start_time = time.perf_counter()

    try:

        asyncio.run(main())

    except KeyboardInterrupt:

        print()
        print("Interrupted by user.")

    except Exception as exc:

        print()
        print("=" * 72)
        print("FATAL ERROR")
        print("=" * 72)
        print(f"{type(exc).__name__}: {exc}")
        raise

    finally:

        if ib.isConnected():
            ib.disconnect()

        elapsed = time.perf_counter() - start_time

        print()
        print(f"Time taken: {elapsed:.1f} seconds")