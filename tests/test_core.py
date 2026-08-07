"""
Unit tests for the strategy internals.

Emphasis is on degenerate input, not the happy path. The strategy runs
unattended for 30 days across ~700 iterations, so "rare" inputs -- an empty
frame, a NaN volatility, a zero price, a missing position -- are all certainties
over that horizon. Anything that raises here would end the competition run.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from strategies.core import execution, portfolio, signals


def series(values) -> pd.Series:
    return pd.Series([float(v) for v in values])


def rising(n: int = 300, start: float = 100.0, step: float = 0.5) -> pd.Series:
    return series([start + step * i for i in range(n)])


def falling(n: int = 300, start: float = 250.0, step: float = 0.5) -> pd.Series:
    return series([start - step * i for i in range(n)])


# --------------------------------------------------------------------------
# signals
# --------------------------------------------------------------------------

class TestTrendScore:
    def test_uptrend_scores_high(self):
        assert signals.trend_score(rising(), (24, 72, 240)) == pytest.approx(1.0)

    def test_downtrend_scores_zero(self):
        assert signals.trend_score(falling(), (24, 72, 240)) == pytest.approx(0.0)

    def test_score_is_graded_not_binary(self):
        """A recent bounce inside a longer downtrend should be partial, not all-or-nothing."""
        prices = series([200 - i for i in range(200)] + [i for i in range(40)])
        score = signals.trend_score(prices, (5, 50, 150))
        assert 0.0 < score < 1.0

    @pytest.mark.parametrize(
        "bad",
        [pd.Series(dtype=float), series([]), series([100.0])],
    )
    def test_insufficient_data_is_defensive(self, bad):
        """Fails to zero (hold cash), never raises -- cash is the only safe default."""
        assert signals.trend_score(bad, (24, 72)) == 0.0

    def test_handles_none_and_empty_lookbacks(self):
        assert signals.trend_score(None, (24,)) == 0.0
        assert signals.trend_score(rising(), ()) == 0.0

    def test_nonpositive_latest_price_is_defensive(self):
        prices = series([100.0] * 100 + [0.0])
        assert signals.trend_score(prices, (24,)) == 0.0


class TestRealizedVolatility:
    def test_positive_for_noisy_series(self):
        rng = np.random.default_rng(0)
        prices = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 500))))
        vol = signals.realized_volatility(prices, 168)
        assert math.isfinite(vol) and vol > 0

    def test_higher_noise_gives_higher_vol(self):
        rng = np.random.default_rng(1)
        quiet = series(100 * np.exp(np.cumsum(rng.normal(0, 0.002, 500))))
        wild = series(100 * np.exp(np.cumsum(rng.normal(0, 0.02, 500))))
        assert signals.realized_volatility(wild, 168) > signals.realized_volatility(quiet, 168)

    def test_insufficient_data_returns_nan(self):
        """NaN means 'no opinion'. A fabricated low vol becomes a huge weight."""
        assert math.isnan(signals.realized_volatility(series([100.0] * 10), 168))

    def test_flat_series_returns_nan_not_zero(self):
        """Zero vol would divide by zero downstream and produce infinite weight."""
        assert math.isnan(signals.realized_volatility(series([100.0] * 500), 168))


class TestMomentum:
    def test_computes_trailing_return(self):
        assert signals.momentum(series([100, 110, 121]), 2) == pytest.approx(0.21)

    def test_insufficient_data_returns_nan(self):
        assert math.isnan(signals.momentum(series([100.0]), 24))


# --------------------------------------------------------------------------
# portfolio
# --------------------------------------------------------------------------

class TestInverseVolatilityWeights:
    def test_lower_vol_gets_more_weight(self):
        weights = portfolio.inverse_volatility_weights({"A": 0.2, "B": 0.8}, max_weight=1.0)
        assert weights["A"] > weights["B"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_respects_per_asset_cap(self):
        weights = portfolio.inverse_volatility_weights(
            {"A": 0.01, "B": 1.0, "C": 1.0, "D": 1.0, "E": 1.0}, max_weight=0.2
        )
        assert all(w <= 0.2 + 1e-9 for w in weights.values())
        assert sum(weights.values()) == pytest.approx(1.0)

    @pytest.mark.parametrize("bad", [float("nan"), 0.0, -1.0, float("inf")])
    def test_unusable_volatility_drops_the_asset(self, bad):
        """Must never become a position -- a bad vol estimate means no estimate."""
        weights = portfolio.inverse_volatility_weights({"A": 0.5, "B": bad}, max_weight=1.0)
        assert "B" not in weights
        assert weights["A"] == pytest.approx(1.0)

    def test_all_unusable_returns_empty(self):
        assert portfolio.inverse_volatility_weights({"A": float("nan")}, max_weight=1.0) == {}


class TestTrendGate:
    def test_zero_score_moves_weight_to_cash(self):
        gated = portfolio.apply_trend_gate({"A": 0.5, "B": 0.5}, {"A": 1.0, "B": 0.0})
        assert gated["A"] == pytest.approx(0.5)
        assert gated["B"] == 0.0
        assert sum(gated.values()) < 1.0  # the shortfall IS the risk management

    def test_missing_or_invalid_score_is_defensive(self):
        gated = portfolio.apply_trend_gate({"A": 0.5, "B": 0.5}, {"A": float("nan")})
        assert gated["A"] == 0.0
        assert gated["B"] == 0.0


class TestVolatilityTargeting:
    def test_scales_down_when_vol_exceeds_target(self):
        scalar = portfolio.volatility_target_scalar(
            {"A": 1.0}, {"A": 1.0}, target_volatility=0.45
        )
        assert 0 < scalar < 1

    def test_never_levers_above_max(self):
        """Spot has no leverage available, so >1.0 would be unexecutable."""
        scalar = portfolio.volatility_target_scalar(
            {"A": 1.0}, {"A": 0.01}, target_volatility=0.45, max_leverage=1.0
        )
        assert scalar == pytest.approx(1.0)

    def test_empty_weights_gives_zero(self):
        assert portfolio.volatility_target_scalar({}, {}, target_volatility=0.45) == 0.0


class TestBuildTargetWeights:
    def _build(self, vols, trends):
        return portfolio.build_target_weights(
            vols,
            trends,
            max_weight=0.2,
            target_volatility=0.45,
            max_gross_exposure=0.95,
            cash_buffer=0.05,
        )

    def test_respects_gross_exposure_ceiling(self):
        vols = {c: 0.05 for c in "ABCDEFGH"}  # very low vol -> wants max exposure
        weights = self._build(vols, {c: 1.0 for c in "ABCDEFGH"})
        assert sum(weights.values()) <= 0.95 + 1e-9

    def test_all_downtrend_goes_fully_to_cash(self):
        weights = self._build({c: 0.5 for c in "ABC"}, {c: 0.0 for c in "ABC"})
        assert weights == {}

    def test_no_usable_data_returns_empty(self):
        assert self._build({}, {}) == {}

    def test_per_asset_cap_holds_end_to_end(self):
        weights = self._build({"A": 0.02, "B": 1.5, "C": 1.5}, {"A": 1.0, "B": 1.0, "C": 1.0})
        assert all(w <= 0.2 + 1e-9 for w in weights.values())


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------

class TestTargetQuantity:
    def test_converts_weight_to_quantity(self):
        assert execution.target_quantity(0.1, 1_000_000, 50.0) == pytest.approx(2000.0)

    @pytest.mark.parametrize(
        "weight,value,price",
        [(0.1, 1e6, 0.0), (0.1, 0.0, 50.0), (float("nan"), 1e6, 50.0), (-0.1, 1e6, 50.0)],
    )
    def test_degenerate_inputs_yield_zero(self, weight, value, price):
        assert execution.target_quantity(weight, value, price) == 0.0


class TestVolumeCap:
    def test_caps_to_participation_share(self):
        assert execution.cap_by_volume(10_000, 100_000, 0.02) == pytest.approx(2000.0)

    def test_passes_through_when_under_cap(self):
        assert execution.cap_by_volume(100, 100_000, 0.02) == pytest.approx(100.0)

    @pytest.mark.parametrize("volume", [0.0, -5.0, float("nan")])
    def test_unknown_volume_blocks_the_trade(self, volume):
        """Better to skip than to send an order that cannot fill."""
        assert execution.cap_by_volume(1000, volume, 0.02) == 0.0


class TestPlanRebalance:
    BASE = dict(participation_cap=0.5, min_trade_notional=100.0)

    def test_buys_toward_target(self):
        intents = execution.plan_rebalance(
            {"BTC": 0.5}, {}, {"BTC": 100.0}, {"BTC": 1e9}, 1_000_000, **self.BASE
        )
        assert len(intents) == 1
        assert intents[0].side == "buy"
        assert intents[0].quantity == pytest.approx(5000.0)

    def test_sells_assets_dropped_from_target(self):
        """
        Regression guard: if we only iterated over target_weights, a name whose
        trend rolled over would be held forever, silently turning the trend
        strategy into buy-and-hold exactly when the gate should protect us.
        """
        intents = execution.plan_rebalance(
            {}, {"BTC": 10.0}, {"BTC": 100.0}, {"BTC": 1e9}, 1_000_000, **self.BASE
        )
        assert [i.side for i in intents] == ["sell"]
        assert intents[0].quantity == pytest.approx(10.0)

    def test_skips_dust_trades(self):
        intents = execution.plan_rebalance(
            {"BTC": 0.5}, {"BTC": 4999.9}, {"BTC": 100.0}, {"BTC": 1e9},
            1_000_000, **self.BASE
        )
        assert intents == []

    def test_unusable_price_skips_asset_without_raising(self):
        for price in (0.0, float("nan"), -1.0):
            assert execution.plan_rebalance(
                {"BTC": 0.5}, {}, {"BTC": price}, {"BTC": 1e9}, 1_000_000, **self.BASE
            ) == []

    def test_thin_volume_throttles_order_size(self):
        intents = execution.plan_rebalance(
            {"BTC": 0.5}, {}, {"BTC": 100.0}, {"BTC": 1000.0}, 1_000_000,
            participation_cap=0.02, min_trade_notional=100.0,
        )
        assert intents[0].quantity == pytest.approx(20.0)  # 1000 * 0.02, not 5000

    def test_missing_volume_entry_skips_asset(self):
        assert execution.plan_rebalance(
            {"BTC": 0.5}, {}, {"BTC": 100.0}, {}, 1_000_000, **self.BASE
        ) == []


class TestSelectTopK:
    def test_picks_highest_scores(self):
        assert portfolio.select_top_k({"A": 0.1, "B": 0.9, "C": 0.5}, 2) == ["B", "C"]

    def test_excludes_unusable_scores(self):
        """nan must be dropped, not treated as zero -- zero ranks mid-pack."""
        assert portfolio.select_top_k({"A": float("nan"), "B": -0.5}, 2) == ["B"]

    def test_ties_break_deterministically(self):
        """An unstable sort would churn the book for no reason."""
        first = portfolio.select_top_k({"C": 0.5, "A": 0.5, "B": 0.5}, 2)
        assert first == ["A", "B"] == portfolio.select_top_k({"A": 0.5, "B": 0.5, "C": 0.5}, 2)

    def test_zero_k_selects_nothing(self):
        assert portfolio.select_top_k({"A": 1.0}, 0) == []


class TestConcentratedWeights:
    def test_equal_weights_to_sleeve_fraction(self):
        w = portfolio.concentrated_weights(["A", "B"], 0.30)
        assert w == {"A": pytest.approx(0.15), "B": pytest.approx(0.15)}

    def test_trend_gate_removes_downtrending_pick(self):
        w = portfolio.concentrated_weights(["A", "B"], 0.30, {"A": 1.0, "B": 0.0})
        assert "B" not in w and w["A"] == pytest.approx(0.15)

    def test_downside_is_bounded_by_sleeve_fraction(self):
        """The whole point: concentration buys convexity with a hard floor."""
        assert sum(portfolio.concentrated_weights(["A"], 0.25).values()) <= 0.25 + 1e-9

    def test_empty_selection_is_safe(self):
        assert portfolio.concentrated_weights([], 0.3) == {}


class TestCombine:
    def test_overlapping_names_are_summed(self):
        merged = portfolio.combine({"A": 0.3}, {"A": 0.2}, max_gross=0.95)
        assert merged["A"] == pytest.approx(0.5)

    def test_respects_gross_ceiling(self):
        merged = portfolio.combine({"A": 0.7}, {"B": 0.6}, max_gross=0.95)
        assert sum(merged.values()) == pytest.approx(0.95)
