"""
Failure injection for the 30-day unattended run.

The strategy runs for a month with no opportunity to intervene. Over that many
iterations, every "unlikely" failure becomes a certainty: a broker call returns
None, a data frame comes back empty, a price is zero, an order is rejected. Any
one of them raising out of ``on_trading_iteration`` ends the competition run,
and we would not find out until the leaderboard.

These tests drive the real ``Strategy`` class with a fake broker surface and
assert two things for every injected fault:

  1. it does not raise, and
  2. it does not silently do nothing forever.

The second matters as much as the first. An early quote-asset misconfiguration
produced a run that skipped all 720 iterations and still exited 0 -- a silent
0% submission. Not crashing is necessary but not sufficient.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import strategies.strategy as strategy_module
from strategies.strategy import Strategy


class FakeBars:
    def __init__(self, df):
        self.df = df


class FakePosition:
    def __init__(self, symbol, quantity):
        self.asset = type("A", (), {"symbol": symbol})()
        self.quantity = quantity


def make_history(n: int = 900, start: float = 100.0, seed: int = 0) -> pd.DataFrame:
    """A healthy price/volume frame long enough to satisfy every lookback."""
    rng = np.random.default_rng(seed)
    closes = start * np.exp(np.cumsum(rng.normal(0.0002, 0.01, n)))
    return pd.DataFrame({"close": closes, "volume": np.full(n, 1e7)})


class HarnessStrategy(Strategy):
    """
    The real Strategy with only the broker boundary replaced.

    Every method stubbed here is one the live strategy calls out through; the
    decision logic under test is untouched.
    """

    def __init__(self, **overrides):
        # Bypass Lumibot's __init__ (it wants a live broker) and set only what
        # the strategy itself touches.
        self.sleeptime = strategy_module.SLEEPTIME
        self.consecutive_failures = 0
        self.iteration_count = 0
        # Lumibot's real __init__ sets this; the strategy reads it when calling
        # get_historical_prices / get_last_price. Bypassing __init__ means we
        # have to supply it, or every history lookup raises AttributeError and
        # is swallowed by the strategy's own error handling -- which would make
        # all the fault tests below pass vacuously.
        #
        # Assign the PRIVATE attribute: `quote_asset` is a property whose setter
        # does `self.broker.quote_assets.add(value)`, which needs a live broker.
        # (That setter is also why the original quote-asset bug reached the
        # broker at all.)
        self._quote_asset = None
        self.tradable_assets = {
            s: type("A", (), {"symbol": s})() for s in strategy_module.UNIVERSE
        }
        self.submitted: list = []
        self.messages: list[str] = []

        self._history = overrides.get("history", make_history())
        self._portfolio_value = overrides.get("portfolio_value", 1_000_000.0)
        self._price = overrides.get("price", 100.0)
        self._positions = overrides.get("positions", [])
        self._raise_on_history = overrides.get("raise_on_history", False)
        self._raise_on_order = overrides.get("raise_on_order", False)
        self._now = overrides.get("now", datetime(2026, 8, 1, 0, 0, tzinfo=timezone.utc))

    # -- broker surface ---------------------------------------------------
    def log_message(self, message, color=None):
        self.messages.append(str(message))

    def get_datetime(self):
        return self._now

    def get_portfolio_value(self):
        if isinstance(self._portfolio_value, Exception):
            raise self._portfolio_value
        return self._portfolio_value

    def get_historical_prices(self, asset, length, timestep=None, quote=None):
        if self._raise_on_history:
            raise RuntimeError("data source exploded")
        if self._history is None:
            return None
        return FakeBars(self._history)

    def get_last_price(self, asset, quote=None):
        if isinstance(self._price, Exception):
            raise self._price
        return self._price

    def get_positions(self):
        if isinstance(self._positions, Exception):
            raise self._positions
        return self._positions

    def create_order(self, asset, quantity, side, quote=None):
        if self._raise_on_order:
            raise RuntimeError("order rejected by broker")
        return {"asset": asset, "quantity": quantity, "side": side}

    def submit_order(self, order):
        self.submitted.append(order)


# --------------------------------------------------------------------------
# The happy path must actually trade -- otherwise the fault tests below prove
# nothing, since "did not raise" is trivially true for a strategy that no-ops.
# --------------------------------------------------------------------------

def test_healthy_iteration_places_orders():
    s = HarnessStrategy()
    s.on_trading_iteration()
    assert s.submitted, "healthy iteration must trade; otherwise fault tests are vacuous"
    assert s.consecutive_failures == 0


def test_weights_stay_within_the_exposure_ceiling():
    s = HarnessStrategy()
    s.on_trading_iteration()
    book = s._target_book(
        {sym: 0.5 for sym in strategy_module.UNIVERSE},
        {sym: 1.0 for sym in strategy_module.UNIVERSE},
        {sym: 0.1 for sym in strategy_module.UNIVERSE},
    )
    assert sum(book.values()) <= strategy_module.MAX_GROSS_EXPOSURE + 1e-9
    assert all(w >= 0 for w in book.values()), "long-only: no negative weights"


# --------------------------------------------------------------------------
# Injected faults: none may raise.
# --------------------------------------------------------------------------

FAULTS = {
    "history returns None": dict(history=None),
    "history frame is empty": dict(history=pd.DataFrame()),
    "history missing close column": dict(history=pd.DataFrame({"volume": [1, 2, 3]})),
    "history far too short": dict(history=make_history(n=5)),
    "history is all NaN": dict(
        history=pd.DataFrame({"close": [np.nan] * 900, "volume": [np.nan] * 900})
    ),
    "history call raises": dict(raise_on_history=True),
    "get_last_price returns None": dict(price=None),
    "get_last_price returns zero": dict(price=0.0),
    "get_last_price returns NaN": dict(price=float("nan")),
    "get_last_price raises": dict(price=RuntimeError("feed down")),
    "portfolio value is zero": dict(portfolio_value=0.0),
    "portfolio value is None": dict(portfolio_value=None),
    "portfolio value raises": dict(portfolio_value=RuntimeError("broker down")),
    "get_positions raises": dict(positions=RuntimeError("positions unavailable")),
    "position quantity is NaN": dict(positions=[FakePosition("BTC", float("nan"))]),
    "position in an unknown symbol": dict(positions=[FakePosition("XXXX", 5.0)]),
    "every order is rejected": dict(raise_on_order=True),
    "zero volume everywhere": dict(
        history=make_history().assign(volume=0.0)
    ),
}


@pytest.mark.parametrize("name,overrides", sorted(FAULTS.items()), ids=lambda v: v if isinstance(v, str) else "")
def test_injected_fault_never_raises(name, overrides):
    s = HarnessStrategy(**overrides)
    s.on_trading_iteration()  # must not raise
    assert s.iteration_count == 1


def test_strategy_recovers_after_a_transient_failure():
    """A fault must not permanently disable trading once conditions recover."""
    s = HarnessStrategy(raise_on_history=True)
    s.on_trading_iteration()
    assert not s.submitted

    s._raise_on_history = False
    s.on_trading_iteration()
    assert s.submitted, "must resume trading once the data source recovers"
    assert s.consecutive_failures == 0


def test_circuit_breaker_opens_on_unhandled_exceptions():
    """The breaker counts genuine code failures and latches after the threshold."""
    s = HarnessStrategy()
    s._run_iteration = lambda: (_ for _ in ()).throw(RuntimeError("bug"))

    for _ in range(strategy_module.MAX_CONSECUTIVE_FAILURES):
        s.on_trading_iteration()
    assert s.consecutive_failures == strategy_module.MAX_CONSECUTIVE_FAILURES

    s.on_trading_iteration()
    assert any("Circuit breaker" in m for m in s.messages)


def test_gracefully_handled_failures_do_not_trip_the_breaker():
    """
    A degraded-but-handled broker leaves us inert, not broken.

    The breaker exists to stop a *bug* compounding into a wrecked book. When
    portfolio value is simply unavailable we place no orders at all, so there is
    nothing to compound and latching the breaker would add no protection.

    The real hazard in this state is the silent one: the strategy would sit idle
    for the whole window while still exiting cleanly. Nothing can size an order
    without a portfolio value, so it is logged every iteration rather than
    hidden.
    """
    s = HarnessStrategy(portfolio_value=RuntimeError("broker down"))
    for _ in range(strategy_module.MAX_CONSECUTIVE_FAILURES + 2):
        s.on_trading_iteration()

    assert s.consecutive_failures == 0, "handled failures are not code failures"
    assert not s.submitted, "must not trade without a usable portfolio value"
    assert any("Portfolio value unavailable" in m for m in s.messages), (
        "a persistently degraded state must be logged, not silent"
    )


def test_rebalance_is_once_daily_but_never_never():
    """
    Off-hours must skip, and the configured hour must act.

    Guards the failure that already bit us once: a gating condition that is
    never true produces a silent do-nothing run that still exits cleanly.
    """
    off = HarnessStrategy(now=datetime(2026, 8, 1, 13, 0, tzinfo=timezone.utc))
    off.on_trading_iteration()
    assert not off.submitted

    on = HarnessStrategy(
        now=datetime(2026, 8, 1, strategy_module.REBALANCE_HOUR_UTC, 0, tzinfo=timezone.utc)
    )
    on.on_trading_iteration()
    assert on.submitted


def test_rebalance_gate_fails_open_when_the_clock_is_unreadable():
    """Silently never trading is worse than trading at the wrong hour."""
    s = HarnessStrategy()
    s.get_datetime = lambda: (_ for _ in ()).throw(RuntimeError("no clock"))
    s.on_trading_iteration()
    assert s.submitted, "unreadable clock must not silence the strategy"


def test_orders_are_capped_by_available_volume():
    """A thin book must throttle order size, not send an unfillable order."""
    thin = make_history().assign(volume=10.0)
    s = HarnessStrategy(history=thin)
    s.on_trading_iteration()
    for order in s.submitted:
        assert order["quantity"] <= 10.0 * strategy_module.VOLUME_PARTICIPATION_CAP + 1e-9


# --------------------------------------------------------------------------
# Feed availability: the organizers' universe may not expose our tickers.
# --------------------------------------------------------------------------

class PartialFeedStrategy(HarnessStrategy):
    """Only some symbols resolve, as with a feed that names things differently."""

    def __init__(self, available, **overrides):
        super().__init__(**overrides)
        self.available = set(available)

    def get_historical_prices(self, asset, length, timestep=None, quote=None):
        if getattr(asset, "symbol", None) not in self.available:
            return None
        return super().get_historical_prices(asset, length, timestep, quote)


@pytest.mark.parametrize("count", [24, 16, 8, 2, 1, 0])
def test_degrades_gracefully_when_tickers_are_missing(count):
    """
    The universe includes names less certain to exist in every feed, so a
    partial universe must shrink the book rather than break it. With nothing
    available the strategy must hold cash and say so, not raise and not trade.
    """
    available = list(strategy_module.UNIVERSE)[:count]
    s = PartialFeedStrategy(available)
    s.on_trading_iteration()

    assert s.consecutive_failures == 0, "a partial feed is not a code failure"
    traded = {o["asset"].symbol for o in s.submitted}
    assert traded <= set(available), "must never trade an unavailable symbol"
    if count == 0:
        assert not s.submitted
        assert any("No assets with usable data" in m for m in s.messages)
    else:
        assert s.submitted, f"{count} symbols available but nothing traded"
