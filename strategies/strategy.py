"""
SoAI 2026 AI Algorithmic Trading Competition -- competition entrypoint.

The official execution environment imports this class, so the file path
(``strategies/strategy.py``), the class name (``Strategy``) and the base class
(``lumibot.strategies.Strategy``) are all fixed by the rules.

APPROACH (baseline)
-------------------
Long-only, trend-gated, volatility-targeted basket of liquid crypto spot pairs,
with cash as the defensive asset.

Three constraints shape the design:

* **Long-only spot.** Crypto spot cannot be shorted, so there is no
  market-neutral construction available and every position carries full market
  beta. The only way to reduce risk is to rotate into cash. Timing exposure is
  therefore a bigger lever than asset selection, which is why the trend gate --
  not the ranking -- is the core of the strategy.
* **Terminal return is the only score.** Risk management earns no points
  directly. It matters because the strategy runs unattended for 30 days: a book
  that blows up, or code that raises, cannot recover.
* **Volume-aware fills.** The official engine will not fill orders exceeding a
  fraction of the bar's real volume, so orders are sized against recent volume
  rather than against how much we would like to trade.

ROBUSTNESS
----------
``on_trading_iteration`` never raises. Every external call is treated as able to
return ``None``, a short frame, or garbage, because over 43,000 iterations it
eventually will. The strategy recomputes its target book from live portfolio
state every iteration and trades the difference, so a failed or partial fill
self-heals on the next pass rather than leaving the book permanently skewed.
"""

from __future__ import annotations

import math

from lumibot.entities import Asset
from lumibot.strategies import Strategy as _LumibotStrategy

from strategies.core import execution, portfolio, signals

# --------------------------------------------------------------------------
# Tunable parameters. Held as plain module-level constants so the research
# bake-off can sweep them and the live strategy imports the identical values --
# there is no second copy of these numbers anywhere.
# --------------------------------------------------------------------------

# Liquid USDT spot pairs. Liquidity is the binding constraint: an order that
# exceeds available minute-volume simply does not fill on the official engine.
UNIVERSE: tuple[str, ...] = (
    "BTC", "ETH", "SOL", "BNB", "XRP", "ADA", "AVAX", "LINK",
    "DOT", "DOGE", "LTC", "ATOM", "UNI", "AAVE", "NEAR", "APT",
)

# Hourly cadence. Comfortably inside the allowed minute/hour/day range, and far
# enough from sub-minute that there is no risk of rejection at verification.
SLEEPTIME = "60M"

# Trend lookbacks in bars (hours). Three horizons -- roughly 1 day, 3 days and
# 10 days -- so exposure steps down gradually as successively slower trends roll
# over, instead of flipping the whole book at a single threshold.
TREND_LOOKBACKS: tuple[int, ...] = (24, 72, 240)

# Volatility estimation window in bars (~7 days of hourly data). Long enough to
# be stable, short enough to react to a regime change within the 30-day window.
VOLATILITY_WINDOW = 168

# Annualized portfolio volatility target. Crypto majors run far hotter than
# this, so in practice this scales the book down rather than up.
TARGET_VOLATILITY = 0.45

# Assumed average pairwise correlation, used instead of a sample covariance
# matrix. Crypto majors co-move strongly and persistently; a full covariance
# estimate over 16 assets is noisy and its errors concentrate exactly in the
# low-variance directions an optimizer would lever up.
AVERAGE_CORRELATION = 0.7

# Risk limits. max_gross < 1.0 and a cash buffer together guarantee we never
# attempt to spend money we do not have, even if a price mark is stale.
MAX_WEIGHT_PER_ASSET = 0.20
MAX_GROSS_EXPOSURE = 0.95
CASH_BUFFER = 0.05

# Share of recent per-bar volume we are willing to be. The official cap is
# unpublished, so this is deliberately conservative -- we assume ours must be
# tighter than theirs.
VOLUME_PARTICIPATION_CAP = 0.02

# Skip trades too small to change exposure meaningfully.
MIN_TRADE_NOTIONAL_FRACTION = 0.002

# No-trade band: an asset is only rebalanced once its weight has drifted this
# far (in absolute portfolio fraction) from target. Measured necessity, not a
# stylistic choice -- without it a 30-day backtest turned the book over 42.6
# times, paying 0.85% in fees and considerably more in whipsaw, and finished at
# -4.26% while the underlying basket returned +1.26%.
REBALANCE_BAND = 0.02

# Bars requested per asset per iteration. Must exceed the longest lookback with
# headroom for gaps.
HISTORY_LENGTH = max(max(TREND_LOOKBACKS), VOLATILITY_WINDOW) + 50

# Consecutive failures tolerated before the strategy stops trading and holds
# whatever it has. This is a BUG circuit breaker, not a market one: it fires on
# our own defects, never on a drawdown. Cutting risk on a market drawdown would
# cap upside for no scoring benefit, but continuing to trade through a code
# fault risks compounding a bug into a wrecked book.
MAX_CONSECUTIVE_FAILURES = 20


class Strategy(_LumibotStrategy):
    """Trend-gated, volatility-targeted long-only crypto basket."""

    # ------------------------------------------------------------------
    # Lifecycle: setup
    # ------------------------------------------------------------------
    def initialize(self):
        self.sleeptime = SLEEPTIME

        # Crypto trades continuously; without this the runner can skip
        # iterations outside equity market hours.
        self.set_market("24/7")

        # NOTE: we deliberately do NOT override ``self.quote_asset``. Lumibot
        # defaults it to Asset("USD", "forex"), and an earlier version set it to
        # a CRYPTO-typed USD instead. That made the engine try to price "USD"
        # as a tradable crypto pair, get_portfolio_value() returned nothing, and
        # every one of the 720 iterations bailed out before trading -- a silent
        # do-nothing run that still exited 0. Inheriting the default keeps us
        # aligned with whatever the official environment configures.
        self.tradable_assets = {
            symbol: Asset(symbol=symbol, asset_type=Asset.AssetType.CRYPTO)
            for symbol in UNIVERSE
        }

        # Counters only. Nothing here is required for correctness -- the target
        # book is recomputed from live portfolio state every iteration, so the
        # strategy is safe even if it is re-instantiated mid-run (we do not know
        # whether the official runner keeps one long-lived process).
        self.consecutive_failures = 0
        self.iteration_count = 0

        self.log_message(
            f"Initialized: {len(UNIVERSE)} assets, sleeptime={SLEEPTIME}, "
            f"target_vol={TARGET_VOLATILITY}, max_weight={MAX_WEIGHT_PER_ASSET}"
        )

    # ------------------------------------------------------------------
    # Lifecycle: per-step decision making
    # ------------------------------------------------------------------
    def on_trading_iteration(self):
        """
        Never raises. A crash here ends the competition run, and there is no
        opportunity to intervene during the 30-day window, so every failure is
        caught, logged, and retried on the next iteration.
        """
        try:
            self.iteration_count += 1

            if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                self.log_message(
                    f"Circuit breaker open after {self.consecutive_failures} consecutive "
                    f"failures; holding current positions and not trading.",
                    color="red",
                )
                return

            self._run_iteration()
            self.consecutive_failures = 0

        except Exception as exc:  # noqa: BLE001 - deliberately catching everything
            self.consecutive_failures += 1
            self.log_message(
                f"Iteration failed ({self.consecutive_failures}/"
                f"{MAX_CONSECUTIVE_FAILURES}): {exc!r}",
                color="red",
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _run_iteration(self) -> None:
        portfolio_value = self._safe_portfolio_value()
        if portfolio_value <= 0:
            self.log_message("Portfolio value unavailable or non-positive; skipping.")
            return

        volatilities: dict[str, float] = {}
        trend_scores: dict[str, float] = {}
        prices: dict[str, float] = {}
        volumes: dict[str, float] = {}

        for symbol, asset in self.tradable_assets.items():
            closes, volume = self._safe_history(asset)
            if closes is None:
                continue

            volatility = signals.realized_volatility(closes, VOLATILITY_WINDOW)
            if not math.isfinite(volatility):
                continue  # no usable risk estimate -> no position

            price = self._safe_price(asset, fallback=closes)
            if price is None:
                continue

            volatilities[symbol] = volatility
            trend_scores[symbol] = signals.trend_score(closes, TREND_LOOKBACKS)
            prices[symbol] = price
            volumes[symbol] = volume

        if not volatilities:
            self.log_message("No assets with usable data this iteration; skipping.")
            return

        target_weights = portfolio.build_target_weights(
            volatilities,
            trend_scores,
            max_weight=MAX_WEIGHT_PER_ASSET,
            target_volatility=TARGET_VOLATILITY,
            max_gross_exposure=MAX_GROSS_EXPOSURE,
            cash_buffer=CASH_BUFFER,
            average_correlation=AVERAGE_CORRELATION,
        )

        intents = execution.plan_rebalance(
            target_weights,
            self._current_quantities(),
            prices,
            volumes,
            portfolio_value,
            participation_cap=VOLUME_PARTICIPATION_CAP,
            min_trade_notional=portfolio_value * MIN_TRADE_NOTIONAL_FRACTION,
            rebalance_band=REBALANCE_BAND,
        )

        self._submit(intents)

        invested = sum(target_weights.values())
        self.log_message(
            f"iter={self.iteration_count} pv=${portfolio_value:,.0f} "
            f"invested={invested:.1%} cash={1 - invested:.1%} "
            f"holdings={len(target_weights)} orders={len(intents)}"
        )

    def _submit(self, intents: list[execution.OrderIntent]) -> None:
        """Submit orders one at a time so a single rejection cannot lose the batch."""
        for intent in intents:
            try:
                asset = self.tradable_assets.get(intent.symbol)
                if asset is None:
                    continue
                order = self.create_order(
                    asset, intent.quantity, intent.side, quote=self.quote_asset
                )
                if order is not None:
                    self.submit_order(order)
            except Exception as exc:  # noqa: BLE001
                self.log_message(
                    f"Order rejected {intent.side} {intent.quantity:.6f} "
                    f"{intent.symbol}: {exc!r}"
                )

    # -- defensive wrappers -------------------------------------------------
    # Each of these can fail or return something unusable in the official
    # environment. Over ~700 hourly iterations, "unlikely" becomes "expected".

    def _safe_portfolio_value(self) -> float:
        try:
            value = float(self.get_portfolio_value())
            return value if math.isfinite(value) and value > 0 else 0.0
        except Exception:  # noqa: BLE001
            return 0.0

    def _safe_history(self, asset: Asset):
        """Return (close series, recent bar volume) or (None, nan) if unusable."""
        try:
            bars = self.get_historical_prices(
                asset, HISTORY_LENGTH, timestep="hour", quote=self.quote_asset
            )
            if bars is None:
                return None, float("nan")

            frame = getattr(bars, "df", None)
            if frame is None or frame.empty or "close" not in frame:
                return None, float("nan")

            closes = frame["close"].dropna()
            if len(closes) < max(TREND_LOOKBACKS[0], 2):
                return None, float("nan")

            # Median rather than mean: a single volume spike would otherwise
            # inflate our participation budget exactly when it should not.
            volume = float("nan")
            if "volume" in frame:
                recent = frame["volume"].dropna().tail(24)
                if len(recent) > 0:
                    volume = float(recent.median())

            return closes, volume
        except Exception:  # noqa: BLE001
            return None, float("nan")

    def _safe_price(self, asset: Asset, fallback) -> float | None:
        """Last price, falling back to the most recent close if unavailable."""
        try:
            price = self.get_last_price(asset, quote=self.quote_asset)
            if price is not None:
                price = float(price)
                if math.isfinite(price) and price > 0:
                    return price
        except Exception:  # noqa: BLE001
            pass

        try:
            price = float(fallback.iloc[-1])
            return price if math.isfinite(price) and price > 0 else None
        except Exception:  # noqa: BLE001
            return None

    def _current_quantities(self) -> dict[str, float]:
        """Live holdings keyed by symbol. Empty dict on any failure."""
        held: dict[str, float] = {}
        try:
            for position in self.get_positions() or []:
                asset = getattr(position, "asset", None)
                symbol = getattr(asset, "symbol", None)
                if symbol is None or symbol not in self.tradable_assets:
                    continue
                quantity = float(getattr(position, "quantity", 0.0) or 0.0)
                if math.isfinite(quantity):
                    held[symbol] = held.get(symbol, 0.0) + quantity
        except Exception:  # noqa: BLE001
            return {}
        return held
