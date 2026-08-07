"""
Reconciliation: the fast engine must compute the same signals as the strategy.

This is the test that makes sweep results meaningful. ``research/engine.py``
replaces the per-bar calls to ``strategies.core.signals`` with vectorized
equivalents for speed. If those diverge even slightly, every sweep optimizes a
strategy we are not going to submit -- and nothing else in the suite would
notice, because both sides would still be internally consistent.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from research.engine import Config, precompute
from strategies.core import signals


@pytest.fixture(scope="module")
def prices() -> pd.DataFrame:
    """Deterministic multi-asset price panel with distinct dynamics per asset."""
    rng = np.random.default_rng(7)
    n = 900
    index = pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC")
    data = {}
    for i, name in enumerate(["A", "B", "C"]):
        drift = (i - 1) * 0.0002  # one trending up, one flat, one down
        steps = rng.normal(drift, 0.01, n)
        data[name] = 100 * np.exp(np.cumsum(steps))
    return pd.DataFrame(data, index=index)


CONFIGS = [
    Config(),
    Config(trend_lookbacks=(48, 168, 480), volatility_window=336),
    Config(trend_lookbacks=(12,), volatility_window=48),
]


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.label())
def test_vectorized_trend_matches_strategy_function(prices, config):
    trend, _ = precompute(prices, config)

    # Check several bars spread across the series, not just the last one.
    for i in (600, 700, 800, len(prices) - 1):
        for symbol in prices.columns:
            expected = signals.trend_score(prices[symbol].iloc[: i + 1], config.trend_lookbacks)
            actual = trend[symbol].iloc[i]
            assert np.isfinite(actual), f"{symbol}@{i} produced no trend score"
            assert actual == pytest.approx(expected, abs=1e-9), (
                f"{symbol}@{i}: engine={actual} vs strategy={expected}"
            )


@pytest.mark.parametrize("config", CONFIGS, ids=lambda c: c.label())
def test_vectorized_volatility_matches_strategy_function(prices, config):
    _, volatility = precompute(prices, config)

    for i in (600, 700, 800, len(prices) - 1):
        for symbol in prices.columns:
            expected = signals.realized_volatility(
                prices[symbol].iloc[: i + 1], config.volatility_window
            )
            actual = volatility[symbol].iloc[i]
            assert np.isfinite(actual), f"{symbol}@{i} produced no volatility"
            assert actual == pytest.approx(expected, rel=1e-9), (
                f"{symbol}@{i}: engine={actual} vs strategy={expected}"
            )


def test_trend_is_nan_before_warmup(prices):
    """Un-warmed bars must not be traded on -- an EMA seeded by one point is noise."""
    config = Config()
    trend, _ = precompute(prices, config)
    assert trend.iloc[max(config.trend_lookbacks) - 2].isna().all()


@pytest.mark.parametrize("lookback", [24, 168, 336])
def test_vectorized_momentum_matches_strategy_function(prices, lookback):
    """Cross-sectional ranking is only meaningful if both sides rank identically."""
    from research.engine import precompute_momentum

    config = Config(momentum_lookback=lookback)
    momentum = precompute_momentum(prices, config)

    for i in (600, 700, len(prices) - 1):
        for symbol in prices.columns:
            expected = signals.momentum(prices[symbol].iloc[: i + 1], lookback)
            actual = momentum[symbol].iloc[i]
            assert actual == pytest.approx(expected, rel=1e-9), (
                f"{symbol}@{i}: engine={actual} vs strategy={expected}"
            )
