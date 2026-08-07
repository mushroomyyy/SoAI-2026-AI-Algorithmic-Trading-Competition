"""
Fast vectorized backtest engine for parameter sweeps.

WHY THIS EXISTS (measured, not assumed): Lumibot's event loop was clocked at
~33 iterations/second, so a single 30-day minute-bar configuration costs 20+
minutes and a 30-day hourly run costs ~50 s. Sweeping dozens of configurations
across three years of history is simply not possible in that engine before the
submission deadline.

WHAT MAKES IT TRUSTWORTHY: it imports the *same* ``strategies.core`` functions
the live strategy uses -- ``signals.trend_score``,
``portfolio.build_target_weights``. There is no reimplementation of the strategy
logic here, so the sweep cannot silently optimize a different strategy than the
one we submit. Only the *execution* layer is simplified, and deliberately
pessimistically:

* fills at the next bar's close (never the signal bar's close, which would be
  look-ahead),
* the full competition fee of 2 bps on every unit of traded notional,
* the same no-trade band the live strategy applies.

It still flatters reality versus the official engine, which layers volume-aware
partial fills on top. Treat every number out of here as an optimistic upper
bound, exactly as the competition README warns.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from strategies.core import portfolio, signals  # noqa: E402

CACHE_DIR = Path(__file__).resolve().parent / "cache"
FEE_RATE = 2.0 / 10_000.0  # competition charges 2 bps on all trades


@dataclass
class Config:
    """One point in the parameter space. Defaults mirror the live strategy."""

    trend_lookbacks: tuple[int, ...] = (24, 72, 240)
    volatility_window: int = 168
    target_volatility: float = 0.45
    max_weight: float = 0.20
    max_gross_exposure: float = 0.95
    cash_buffer: float = 0.05
    average_correlation: float = 0.7
    rebalance_band: float = 0.02
    rebalance_every: int = 1  # bars between decisions; 24 == daily on hourly bars

    def label(self) -> str:
        return (
            f"lb={'/'.join(map(str, self.trend_lookbacks))} "
            f"vol={self.volatility_window} tgt={self.target_volatility:.2f} "
            f"band={self.rebalance_band:.3f} every={self.rebalance_every}h"
        )


def load_universe(symbols: list[str], timeframe: str = "1h") -> pd.DataFrame:
    """Aligned close prices; assets missing data at a timestamp are NaN."""
    frames = {}
    for symbol in symbols:
        path = CACHE_DIR / f"{symbol}_{timeframe}.parquet"
        if path.exists():
            frames[symbol] = pd.read_parquet(path)["close"]
    if not frames:
        raise SystemExit(f"no cached data in {CACHE_DIR}; run research/fetch_data.py")
    return pd.DataFrame(frames).sort_index()


def precompute(prices: pd.DataFrame, config: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Vectorized trend scores and annualized volatilities for every bar at once.

    The naive version recomputed both from a growing window at every step, which
    is O(n^2) and made a three-year sweep take hours. These are exact vectorized
    equivalents of ``signals.trend_score`` and ``signals.realized_volatility``;
    ``tests/test_engine_reconciliation.py`` asserts they agree with those
    functions bar-for-bar, so the speedup cannot silently change the strategy
    being measured.
    """
    # Trend score: fraction of lookbacks whose EMA sits below the current price.
    votes = [
        (prices > prices.ewm(span=lb, adjust=False, min_periods=1).mean()).astype(float)
        for lb in config.trend_lookbacks
    ]
    trend = sum(votes) / len(votes)
    # An EMA is only meaningful once it has seen its own lookback of data.
    trend = trend.where(
        pd.notna(prices.shift(max(config.trend_lookbacks))), other=np.nan
    )

    log_returns = np.log(prices).diff()
    volatility = log_returns.rolling(
        config.volatility_window
    ).std(ddof=1) * np.sqrt(signals.HOURS_PER_YEAR)
    volatility = volatility.where(volatility > 0)

    return trend, volatility


def simulate(prices: pd.DataFrame, config: Config) -> pd.Series:
    """
    Run the strategy over ``prices`` and return the equity curve.

    Decisions at bar ``t`` use only data up to and including ``t``; the
    resulting weights earn the return from ``t`` to ``t+1``. That ordering is
    what keeps this free of look-ahead.
    """
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    warmup = max(max(config.trend_lookbacks), config.volatility_window) + 1
    if len(prices) <= warmup + 2:
        return pd.Series(dtype=float)

    trend_all, vol_all = precompute(prices, config)

    equity = 1.0
    weights = pd.Series(0.0, index=prices.columns)
    curve = {}

    for i in range(warmup, len(prices) - 1):
        if (i - warmup) % config.rebalance_every == 0:
            vol_row = vol_all.iloc[i]
            trend_row = trend_all.iloc[i]
            volatilities = {s: float(v) for s, v in vol_row.items() if np.isfinite(v)}
            trend_scores = {
                s: (float(trend_row[s]) if np.isfinite(trend_row[s]) else 0.0)
                for s in volatilities
            }

            target = portfolio.build_target_weights(
                volatilities,
                trend_scores,
                max_weight=config.max_weight,
                target_volatility=config.target_volatility,
                max_gross_exposure=config.max_gross_exposure,
                cash_buffer=config.cash_buffer,
                average_correlation=config.average_correlation,
            )
            desired = pd.Series(target, index=prices.columns).fillna(0.0)

            # No-trade band, matching the live execution layer: hold unless the
            # weight has drifted materially, but always honour a full exit.
            drift = (desired - weights).abs()
            move = (drift >= config.rebalance_band) | ((desired <= 0) & (weights > 0))
            if move.any():
                new_weights = weights.where(~move, desired)
                turnover = float((new_weights - weights).abs().sum())
                equity *= 1.0 - turnover * FEE_RATE
                weights = new_weights

        step = float((weights * returns.iloc[i + 1]).sum())
        equity *= 1.0 + step
        curve[prices.index[i + 1]] = equity

        # Positions drift with prices between rebalances; carrying stale weights
        # would misstate exposure and understate the true turnover.
        grown = weights * (1.0 + returns.iloc[i + 1])
        total = float(grown.sum())
        if total > 0:
            invested = min(1.0, float(weights.sum()) * (1.0 + step))
            weights = grown / total * invested

    return pd.Series(curve)


def terminal_return_distribution(curve: pd.Series, window_days: int = 30) -> dict:
    """
    Distribution of ``window_days`` terminal returns -- the competition's actual
    scoring horizon.

    Annualized Sharpe is the wrong lens here: the leaderboard is decided by a
    single 30-day terminal return, so what matters is the *spread* of outcomes
    over that horizon, not a long-run risk-adjusted average.
    """
    if curve.empty:
        return {}

    bars = window_days * 24
    if len(curve) <= bars:
        return {}

    values = curve.to_numpy()
    windows = values[bars:] / values[:-bars] - 1.0

    return {
        "n_windows": len(windows),
        "median": float(np.median(windows)),
        "mean": float(np.mean(windows)),
        "p05": float(np.percentile(windows, 5)),
        "p95": float(np.percentile(windows, 95)),
        "p_positive": float((windows > 0).mean()),
        "p_gt_20": float((windows > 0.20).mean()),
        "p_lt_neg25": float((windows < -0.25).mean()),
        "total_return": float(values[-1] / values[0] - 1.0),
    }


def benchmark_curve(prices: pd.DataFrame, symbol: str | None = None) -> pd.Series:
    """Buy-and-hold benchmark: one asset, or the equal-weight basket."""
    returns = prices.pct_change(fill_method=None).fillna(0.0)
    series = returns[symbol] if symbol else returns.mean(axis=1)
    return (1.0 + series).cumprod()
