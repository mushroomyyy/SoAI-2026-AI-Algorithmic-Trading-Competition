"""
The live strategy must build the SAME book the research engine measured.

``strategies/core/`` is shared, and the reconciliation suite already proves the
engine's vectorized indicators match the strategy's signal functions. But the
COMPOSITION -- select top-k, size the core, scale for the sleeve, merge, cap --
is written out twice: once in ``research.engine.simulate`` and once in
``Strategy._target_book``. Two copies of the same sequence can drift, and if
they do, every sweep result describes a portfolio we do not actually trade,
while both halves stay internally consistent and every other test still passes.

This closes that gap by driving both paths from identical inputs on real market
data and asserting the resulting weights agree to floating-point tolerance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import strategies.strategy as live
from research.engine import Config, load_universe, precompute, precompute_momentum
from strategies.core import portfolio

TOLERANCE = 1e-12


def shipped_config() -> Config:
    """A research Config carrying exactly the live strategy's parameters."""
    return Config(
        trend_lookbacks=live.TREND_LOOKBACKS,
        volatility_window=live.VOLATILITY_WINDOW,
        target_volatility=live.TARGET_VOLATILITY,
        max_weight=live.MAX_WEIGHT_PER_ASSET,
        max_gross_exposure=live.MAX_GROSS_EXPOSURE,
        cash_buffer=live.CASH_BUFFER,
        average_correlation=live.AVERAGE_CORRELATION,
        momentum_lookback=live.MOMENTUM_LOOKBACK,
        top_k=live.TOP_K,
        equal_weight=live.EQUAL_WEIGHT_CORE,
        core_trend_gated=live.CORE_TREND_GATED,
        sleeve_fraction=live.SLEEVE_FRACTION,
        sleeve_k=live.SLEEVE_K,
        sleeve_trend_gated=live.SLEEVE_TREND_GATED,
    )


def engine_target_book(volatilities, trend_scores, raw_scores, config: Config) -> dict:
    """The composition exactly as ``research.engine.simulate`` performs it."""
    scores = {s: raw_scores[s] / volatilities[s] for s in volatilities if s in raw_scores}

    core_universe = volatilities
    if config.top_k > 0:
        chosen = set(portfolio.select_top_k(scores, config.top_k))
        core_universe = {s: v for s, v in volatilities.items() if s in chosen}

    sizing = {s: 1.0 for s in core_universe} if config.equal_weight else core_universe
    core = portfolio.build_target_weights(
        sizing,
        trend_scores if config.core_trend_gated else {s: 1.0 for s in core_universe},
        max_weight=config.max_weight,
        target_volatility=config.target_volatility,
        max_gross_exposure=config.max_gross_exposure,
        cash_buffer=config.cash_buffer,
        average_correlation=config.average_correlation,
    )

    if config.sleeve_fraction <= 0:
        return core

    core = {s: w * (1.0 - config.sleeve_fraction) for s, w in core.items()}
    sleeve = portfolio.concentrated_weights(
        portfolio.select_top_k(scores, config.sleeve_k),
        config.sleeve_fraction,
        trend_scores if config.sleeve_trend_gated else None,
    )
    return portfolio.combine(core, sleeve, config.max_gross_exposure)


class _Bare(live.Strategy):
    """The real Strategy with Lumibot's broker-dependent __init__ bypassed."""

    def __init__(self):
        pass


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    try:
        prices = load_universe(list(live.UNIVERSE))
    except FileNotFoundError as exc:  # pragma: no cover - only without any data
        pytest.skip(str(exc))
    if prices.empty:
        pytest.skip("market data present but empty")
    return prices


def test_live_and_engine_agree_on_real_market_data(panel):
    """Sampled across three years, on the data the decision was actually made on."""
    config = shipped_config()
    trend_all, vol_all = precompute(panel, config)
    mom_all = precompute_momentum(panel, config)

    warmup = max(max(config.trend_lookbacks), config.volatility_window,
                 config.momentum_lookback) + 1
    checkpoints = np.linspace(warmup, len(panel) - 1, 40, dtype=int)

    compared = 0
    for i in checkpoints:
        vol_row, trend_row, mom_row = vol_all.iloc[i], trend_all.iloc[i], mom_all.iloc[i]
        volatilities = {s: float(v) for s, v in vol_row.items() if np.isfinite(v)}
        if not volatilities:
            continue
        trends = {
            s: (float(trend_row[s]) if np.isfinite(trend_row[s]) else 0.0)
            for s in volatilities
        }
        momenta = {
            s: float(mom_row[s]) for s in volatilities if np.isfinite(mom_row.get(s, np.nan))
        }
        if not momenta:
            continue

        expected = engine_target_book(volatilities, trends, momenta, config)
        actual = _Bare()._target_book(volatilities, trends, momenta)

        assert set(actual) == set(expected), (
            f"bar {i} ({panel.index[i]:%Y-%m-%d}): different holdings\n"
            f"  live only:   {sorted(set(actual) - set(expected))}\n"
            f"  engine only: {sorted(set(expected) - set(actual))}"
        )
        for symbol in expected:
            assert actual[symbol] == pytest.approx(expected[symbol], abs=TOLERANCE), (
                f"bar {i} ({panel.index[i]:%Y-%m-%d}) {symbol}: "
                f"live={actual[symbol]:.10f} engine={expected[symbol]:.10f}"
            )
        compared += 1

    assert compared >= 30, f"only {compared} bars compared; the check is too weak"


def test_agreement_holds_when_some_assets_are_unusable(panel):
    """Degraded inputs must not push the two paths onto different branches."""
    config = shipped_config()
    trend_all, vol_all = precompute(panel, config)
    mom_all = precompute_momentum(panel, config)
    i = len(panel) - 1

    volatilities = {s: float(v) for s, v in vol_all.iloc[i].items() if np.isfinite(v)}
    trends = {s: float(trend_all.iloc[i][s]) for s in volatilities}
    momenta = {s: float(mom_all.iloc[i][s]) for s in volatilities}

    # Half the universe goes dark, as it would on a degraded feed.
    for symbol in list(volatilities)[: len(volatilities) // 2]:
        momenta.pop(symbol, None)

    expected = engine_target_book(volatilities, trends, momenta, config)
    actual = _Bare()._target_book(volatilities, trends, momenta)

    assert set(actual) == set(expected)
    for symbol in expected:
        assert actual[symbol] == pytest.approx(expected[symbol], abs=TOLERANCE)


def test_shipped_book_respects_its_stated_limits(panel):
    """Gross exposure, long-only, and the documented sleeve concentration."""
    config = shipped_config()
    trend_all, vol_all = precompute(panel, config)
    mom_all = precompute_momentum(panel, config)

    warmup = max(max(config.trend_lookbacks), config.volatility_window,
                 config.momentum_lookback) + 1
    for i in np.linspace(warmup, len(panel) - 1, 25, dtype=int):
        volatilities = {s: float(v) for s, v in vol_all.iloc[i].items() if np.isfinite(v)}
        trends = {s: float(trend_all.iloc[i][s]) for s in volatilities}
        momenta = {
            s: float(mom_all.iloc[i][s])
            for s in volatilities
            if np.isfinite(mom_all.iloc[i].get(s, np.nan))
        }
        if not momenta:
            continue

        book = _Bare()._target_book(volatilities, trends, momenta)
        if not book:
            continue

        assert sum(book.values()) <= live.MAX_GROSS_EXPOSURE + 1e-9
        assert all(w > 0 for w in book.values()), "long-only: no short weights"
        assert len(book) <= live.TOP_K, "must not hold more than the selected top-k"
        # Sleeve names are exempt from the core cap; nothing may exceed the
        # sleeve's own per-name share plus a full core slot.
        ceiling = live.SLEEVE_FRACTION / live.SLEEVE_K + live.MAX_WEIGHT_PER_ASSET
        assert max(book.values()) <= ceiling + 1e-9
