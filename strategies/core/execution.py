"""
Execution: target weights -> order quantities.

Two competition constraints drive this module.

1. **Volume-aware fills.** The official engine "caps each submitted child order
   at a fraction of the bar's real historical minute-volume" and "orders larger
   than the available liquidity will not fill". A backtest that assumes a
   million dollars of an illiquid name fills instantly produces P&L that will
   not materialize. We therefore cap our own orders at a conservative share of
   recent volume -- assumed tighter than theirs, since the exact fraction is
   unpublished.

2. **Idempotent rebalancing.** We compute the target book from scratch each
   iteration and trade the difference, rather than issuing incremental
   buys/sells. If an order failed, partially filled, or was rejected last
   iteration, the next one self-heals toward the target. Incremental logic
   drifts and cannot recover -- and over a 30-day unattended run, something will
   go wrong at least once.
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class OrderIntent:
    """A desired trade, before it becomes a broker-specific order object."""

    symbol: str
    quantity: float
    side: str  # "buy" or "sell"

    @property
    def is_actionable(self) -> bool:
        return self.quantity > 0 and math.isfinite(self.quantity)


def target_quantity(
    target_weight: float, portfolio_value: float, price: float
) -> float:
    """
    Convert a portfolio weight into a quantity of the asset.

    Sizing in *weights* rather than absolute notional is deliberate: the
    starting capital for the official run was never published, so any hard-coded
    dollar amount would be wrong. Weights are correct at any account size.
    """
    if not all(math.isfinite(x) for x in (target_weight, portfolio_value, price)):
        return 0.0
    if price <= 0 or portfolio_value <= 0 or target_weight <= 0:
        return 0.0
    return (target_weight * portfolio_value) / price


def cap_by_volume(
    quantity: float, recent_volume: float, participation_cap: float
) -> float:
    """
    Clamp an order to a fraction of recent per-bar volume.

    ``recent_volume`` should be a conservative measure of what actually trades in
    one bar -- a trailing median is safer than a mean, which a single spike
    inflates. An unknown or non-positive volume yields zero: we would rather
    skip a trade than send one that cannot fill.
    """
    if not math.isfinite(quantity) or quantity <= 0:
        return 0.0
    if not math.isfinite(recent_volume) or recent_volume <= 0:
        return 0.0
    if not math.isfinite(participation_cap) or participation_cap <= 0:
        return 0.0
    return min(quantity, recent_volume * participation_cap)


def plan_rebalance(
    target_weights: dict[str, float],
    current_quantities: dict[str, float],
    prices: dict[str, float],
    volumes: dict[str, float],
    portfolio_value: float,
    *,
    participation_cap: float,
    min_trade_notional: float,
    rebalance_band: float = 0.0,
) -> list[OrderIntent]:
    """
    Diff the target book against the actual book and return the trades to close
    the gap.

    Every asset currently held is considered, not just those in
    ``target_weights`` -- otherwise a name that drops out of the target (because
    its trend rolled over) would be held forever. That omission is an easy bug
    and an expensive one: it silently converts a trend strategy into buy-and-hold
    exactly when the trend gate is trying to protect us.

    ``rebalance_band`` is a NO-TRADE ZONE, and it is the single most important
    parameter in this module. An asset is only traded once its actual weight has
    drifted from target by more than the band, in absolute portfolio fraction.

    Why it exists, measured rather than assumed: without a band, a 30-day
    backtest turned over the book **42.6 times** -- 142 fills a day -- because
    the volatility-target scalar and the graded trend score both move a little
    every hour, so the target weights are never exactly met and the strategy
    chases them continuously. That cost 0.85% in fees outright, and far more in
    whipsaw: buying strength and selling weakness over and over in a choppy
    market. The band converts a continuous chase into occasional discrete
    corrections, which is what actually makes an hourly cadence viable.

    Trades below ``min_trade_notional`` are skipped as a second floor. At 2 bps
    a tiny rebalance costs little, but it still consumes liquidity and adds fill
    risk for no meaningful change in exposure.
    """
    intents: list[OrderIntent] = []
    symbols = set(target_weights) | set(current_quantities)

    for symbol in sorted(symbols):
        price = prices.get(symbol, float("nan"))
        if not math.isfinite(price) or price <= 0:
            continue  # no usable mark: do nothing rather than guess

        target_weight = target_weights.get(symbol, 0.0)
        held = current_quantities.get(symbol, 0.0)
        if not math.isfinite(held):
            held = 0.0

        # No-trade band, evaluated in weight space so it is independent of both
        # account size and price level.
        current_weight = (held * price / portfolio_value) if portfolio_value > 0 else 0.0
        drift = abs(target_weight - current_weight)
        exiting = target_weight <= 0.0 and held > 0.0
        # Always honour a full exit: when the trend gate turns an asset off, the
        # band must not strand the position in a downtrend.
        if drift < rebalance_band and not exiting:
            continue

        desired = target_quantity(target_weight, portfolio_value, price)
        delta = desired - held
        if abs(delta) * price < min_trade_notional:
            continue

        capped = cap_by_volume(
            abs(delta), volumes.get(symbol, float("nan")), participation_cap
        )
        if capped <= 0:
            continue

        intent = OrderIntent(
            symbol=symbol, quantity=capped, side="buy" if delta > 0 else "sell"
        )
        if intent.is_actionable:
            intents.append(intent)

    return intents
