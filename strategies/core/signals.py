"""
Signal computation. Pure functions over price series -- no I/O, no state.

Implements candidate 5.1 (time-series momentum with cash defense) and the
volatility estimate that candidate 5.4 sizes against.

Design note: everything here degrades gracefully on short or ragged input and
returns a neutral value rather than raising. The strategy runs unattended for 30
days with no chance to intervene, so a signal function that throws on a missing
bar is a strategy that dies on day three.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Fraction of a year, in hours, used to annualize volatility. Crypto trades
# 24/7, so unlike equities there are no market-closed hours to exclude.
HOURS_PER_YEAR = 365 * 24


def ema(series: pd.Series, span: int) -> pd.Series:
    """Exponential moving average. Returns an empty series for empty input."""
    if series is None or len(series) == 0 or span <= 0:
        return pd.Series(dtype=float)
    return series.ewm(span=span, adjust=False, min_periods=1).mean()


def realized_volatility(
    close: pd.Series, window: int, periods_per_year: float = HOURS_PER_YEAR
) -> float:
    """
    Annualized realized volatility from log returns over a trailing window.

    Returns ``nan`` when there is not enough data to form an estimate -- callers
    must treat that as "no opinion" and size the asset to zero rather than
    guessing, since a fabricated low vol becomes a huge inverse-vol weight.
    """
    if close is None or len(close) < window + 1 or window < 2:
        return float("nan")

    returns = np.log(close.astype(float)).diff().dropna()
    if len(returns) < window:
        return float("nan")

    sigma = float(returns.tail(window).std(ddof=1))
    if not np.isfinite(sigma) or sigma <= 0:
        return float("nan")

    return sigma * np.sqrt(periods_per_year)


def trend_score(close: pd.Series, lookbacks: tuple[int, ...]) -> float:
    """
    Graded trend strength in [0, 1]: the fraction of lookbacks in an uptrend.

    Why graded rather than binary: a single moving-average crossover flips the
    whole position on and off at one threshold, which maximizes whipsaw right
    where price oscillates around the average. Averaging several horizons lets
    exposure step down (1 -> 2/3 -> 1/3 -> 0) as successively slower trends roll
    over, which both smooths turnover and encodes genuine uncertainty about the
    regime instead of pretending to certainty.

    Returns 0.0 (fully defensive, i.e. hold cash) when data is insufficient --
    the safe direction to fail in a long-only book, where cash is the only
    defensive asset available.
    """
    if close is None or len(close) == 0 or not lookbacks:
        return 0.0

    usable = [lb for lb in lookbacks if lb > 0 and len(close) >= lb]
    if not usable:
        return 0.0

    latest = float(close.iloc[-1])
    if not np.isfinite(latest) or latest <= 0:
        return 0.0

    votes = []
    for lookback in usable:
        averages = ema(close, lookback)
        if averages.empty:
            continue
        level = float(averages.iloc[-1])
        if np.isfinite(level) and level > 0:
            votes.append(1.0 if latest > level else 0.0)

    if not votes:
        return 0.0

    return float(np.mean(votes))


def momentum(close: pd.Series, lookback: int) -> float:
    """
    Trailing simple return over ``lookback`` bars, for cross-sectional ranking.

    Returns ``nan`` on insufficient data so ranking code can drop the name
    rather than treating it as zero momentum, which would wrongly place it mid
    pack instead of excluding it.
    """
    if close is None or len(close) < lookback + 1 or lookback <= 0:
        return float("nan")

    start = float(close.iloc[-(lookback + 1)])
    end = float(close.iloc[-1])
    if not (np.isfinite(start) and np.isfinite(end)) or start <= 0:
        return float("nan")

    return end / start - 1.0
