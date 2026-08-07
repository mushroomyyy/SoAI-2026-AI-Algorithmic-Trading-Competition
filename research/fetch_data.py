"""
Download crypto spot OHLCV bars from a CCXT exchange for local research.

Not used by the official evaluation -- the organizers feed our strategy their
own bars. This exists so we can validate honestly before submitting.

Two outputs, deliberately different:

* ``research/cache/{SYMBOL}_{TIMEFRAME}.parquet`` -- long history (years) at a
  coarse timeframe, for the fast vectorized sweeps. Cheap to fetch.
* ``data/{SYMBOL}_1m_spot.csv`` -- minute bars over a shorter window, in the
  exact schema ``backtest.py`` expects, for faithful Lumibot validation of the
  handful of configs that survive.

Why split them: a 30-day minute-bar Lumibot run was measured at 20+ minutes per
config, so sweeping on minute bars is impossible. But validating only on coarse
bars would hide execution effects. Sweep coarse, validate fine.

Integrity is checked, not assumed. Bad data that silently produces a beautiful
backtest is the most expensive failure mode in this kind of work, so every
series is validated and the report is printed rather than swallowed.

Usage:
    python research/fetch_data.py                # both datasets, default universe
    python research/fetch_data.py --hourly-only  # skip the slow minute pull
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "research" / "cache"
CSV_DIR = REPO_ROOT / "data"

EXCHANGE_ID = "binance"

# Liquid USDT spot pairs. Liquidity is the binding constraint: the official
# engine caps each child order at a fraction of the bar's real minute volume and
# does not fill the excess, so illiquid names are untradeable at size regardless
# of how good the signal looks. Stablecoins and wrapped/pegged assets are
# excluded -- they carry no directional signal.
DEFAULT_UNIVERSE = [
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "LINK",
    "DOT", "DOGE", "LTC", "ATOM", "UNI", "AAVE", "NEAR", "APT",
]

# Long, coarse history for sweeps; short, fine history for faithful validation.
HOURLY_TIMEFRAME = "1h"
HOURLY_LOOKBACK_DAYS = 3 * 365
MINUTE_TIMEFRAME = "1m"
MINUTE_LOOKBACK_DAYS = 180

MAX_RETRIES = 5
CANDLES_PER_REQUEST = 1000


@dataclass
class IntegrityReport:
    """Everything that could quietly poison a backtest, counted explicitly."""

    symbol: str
    timeframe: str
    rows: int = 0
    start: pd.Timestamp | None = None
    end: pd.Timestamp | None = None
    duplicates_dropped: int = 0
    out_of_order_fixed: int = 0
    missing_bars: int = 0
    ohlc_violations: int = 0
    nonpositive_prices: int = 0
    zero_volume_bars: int = 0
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def line(self) -> str:
        status = "OK  " if self.ok else "WARN"
        gap_pct = (self.missing_bars / self.rows * 100) if self.rows else 0.0
        return (
            f"[{status}] {self.symbol:<6} {self.timeframe:<3} "
            f"rows={self.rows:>8,}  {str(self.start)[:10]}..{str(self.end)[:10]}  "
            f"dups={self.duplicates_dropped:<4} gaps={self.missing_bars:<6} ({gap_pct:4.1f}%) "
            f"zerovol={self.zero_volume_bars:<5} ohlcbad={self.ohlc_violations}"
            + (f"  <- {'; '.join(self.problems)}" if self.problems else "")
        )


def _timeframe_delta(timeframe: str) -> timedelta:
    unit = timeframe[-1]
    value = int(timeframe[:-1])
    if unit == "m":
        return timedelta(minutes=value)
    if unit == "h":
        return timedelta(hours=value)
    if unit == "d":
        return timedelta(days=value)
    raise ValueError(f"unsupported timeframe: {timeframe}")


def _fetch_paginated(
    exchange: ccxt.Exchange, market: str, timeframe: str, since_ms: int
) -> list[list]:
    """
    Page through fetch_ohlcv until we reach the present.

    CCXT returns at most ~1000 candles per call, so long histories need many
    round trips. Transient exchange errors are retried with backoff rather than
    aborting a download that is minutes deep.
    """
    step_ms = int(_timeframe_delta(timeframe).total_seconds() * 1000)
    all_rows: list[list] = []
    cursor = since_ms
    now_ms = exchange.milliseconds()

    while cursor < now_ms:
        for attempt in range(MAX_RETRIES):
            try:
                batch = exchange.fetch_ohlcv(
                    market, timeframe=timeframe, since=cursor, limit=CANDLES_PER_REQUEST
                )
                break
            except (ccxt.NetworkError, ccxt.ExchangeError) as exc:
                if attempt == MAX_RETRIES - 1:
                    raise
                backoff = 2**attempt
                print(f"    retry {attempt + 1}/{MAX_RETRIES} after {exc.__class__.__name__}; "
                      f"sleeping {backoff}s", file=sys.stderr)
                time.sleep(backoff)

        if not batch:
            break

        all_rows.extend(batch)
        advanced = batch[-1][0] + step_ms
        if advanced <= cursor:  # exchange stopped advancing; avoid an infinite loop
            break
        cursor = advanced

        if len(batch) < CANDLES_PER_REQUEST:
            break

    return all_rows


def _to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df.set_index("timestamp").sort_index()


def validate(df: pd.DataFrame, symbol: str, timeframe: str) -> tuple[pd.DataFrame, IntegrityReport]:
    """
    Clean and audit a bar series.

    Returns the cleaned frame plus a report. Anything that would materially
    distort a backtest is escalated into ``report.problems`` so it shows up in
    the summary instead of being silently tolerated.
    """
    report = IntegrityReport(symbol=symbol, timeframe=timeframe)

    if df.empty:
        report.problems.append("empty series")
        return df, report

    before = len(df)
    df = df[~df.index.duplicated(keep="first")]
    report.duplicates_dropped = before - len(df)

    if not df.index.is_monotonic_increasing:
        report.out_of_order_fixed = 1
        df = df.sort_index()

    # OHLC internal consistency: low must be the floor, high the ceiling.
    lo = df[["open", "close", "low"]].min(axis=1)
    hi = df[["open", "close", "high"]].max(axis=1)
    violations = (df["low"] > lo) | (df["high"] < hi)
    report.ohlc_violations = int(violations.sum())
    if report.ohlc_violations:
        df = df[~violations]

    nonpositive = (df[["open", "high", "low", "close"]] <= 0).any(axis=1)
    report.nonpositive_prices = int(nonpositive.sum())
    if report.nonpositive_prices:
        df = df[~nonpositive]

    report.zero_volume_bars = int((df["volume"] <= 0).sum())

    # Gap detection against the ideal grid. Exchanges have real outages, so a
    # few gaps are expected; a large fraction means the series is unusable.
    step = _timeframe_delta(timeframe)
    expected = int((df.index[-1] - df.index[0]) / step) + 1
    report.missing_bars = max(0, expected - len(df))

    report.rows = len(df)
    report.start = df.index[0]
    report.end = df.index[-1]

    if report.rows and report.missing_bars / max(expected, 1) > 0.02:
        report.problems.append(f"{report.missing_bars / expected:.1%} of bars missing")
    if report.ohlc_violations:
        report.problems.append(f"{report.ohlc_violations} OHLC violations dropped")
    if report.rows and report.zero_volume_bars / report.rows > 0.05:
        report.problems.append(">5% zero-volume bars (illiquid)")

    return df, report


def fetch_symbol(
    exchange: ccxt.Exchange, symbol: str, timeframe: str, lookback_days: int
) -> tuple[pd.DataFrame, IntegrityReport]:
    market = f"{symbol}/USDT"
    since = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = _fetch_paginated(exchange, market, timeframe, int(since.timestamp() * 1000))
    return validate(_to_frame(rows), symbol, timeframe)


def write_lumibot_csv(df: pd.DataFrame, symbol: str) -> Path:
    """
    Write minute bars in the exact schema ``backtest.py`` reads.

    It requires columns open/high/low/close/volume/timestamp with an ISO-8601
    UTC timestamp, and expects the filename ``{SYMBOL}_1m_spot.csv``.
    """
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    out = df.reset_index()[["open", "high", "low", "close", "volume", "timestamp"]]
    path = CSV_DIR / f"{symbol}_1m_spot.csv"
    out.to_csv(path, index=False)
    return path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbols", nargs="*", default=DEFAULT_UNIVERSE)
    parser.add_argument("--hourly-only", action="store_true",
                        help="skip the slow minute-bar pull")
    parser.add_argument("--minute-days", type=int, default=MINUTE_LOOKBACK_DAYS)
    parser.add_argument("--hourly-days", type=int, default=HOURLY_LOOKBACK_DAYS)
    args = parser.parse_args()

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    exchange = getattr(ccxt, EXCHANGE_ID)({"enableRateLimit": True})

    reports: list[IntegrityReport] = []

    print(f"== {EXCHANGE_ID}: {HOURLY_TIMEFRAME} bars, {args.hourly_days}d, "
          f"{len(args.symbols)} symbols ==")
    for symbol in args.symbols:
        print(f"  {symbol} ...", flush=True)
        df, report = fetch_symbol(exchange, symbol, HOURLY_TIMEFRAME, args.hourly_days)
        reports.append(report)
        if not df.empty:
            df.to_parquet(CACHE_DIR / f"{symbol}_{HOURLY_TIMEFRAME}.parquet")

    if not args.hourly_only:
        print(f"\n== {EXCHANGE_ID}: {MINUTE_TIMEFRAME} bars, {args.minute_days}d ==")
        for symbol in args.symbols:
            print(f"  {symbol} ...", flush=True)
            df, report = fetch_symbol(exchange, symbol, MINUTE_TIMEFRAME, args.minute_days)
            reports.append(report)
            if not df.empty:
                df.to_parquet(CACHE_DIR / f"{symbol}_{MINUTE_TIMEFRAME}.parquet")
                write_lumibot_csv(df, symbol)

    print("\n== integrity report ==")
    for report in reports:
        print(report.line())

    failed = [r for r in reports if not r.ok]
    print(f"\n{len(reports) - len(failed)}/{len(reports)} series clean.")
    if failed:
        print("Series with warnings are still written; review before trusting a backtest "
              "that depends on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
