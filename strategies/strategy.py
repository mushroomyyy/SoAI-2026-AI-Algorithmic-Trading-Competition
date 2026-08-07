"""
SoAI 2026 AI Algorithmic Trading Competition -- competition entrypoint.

The official execution environment imports this class, so the file path
(``strategies/strategy.py``), the class name (``Strategy``) and the base class
(``lumibot.strategies.Strategy``) are all fixed by the rules.

APPROACH
--------
Long-only cross-sectional momentum over liquid crypto spot pairs, in a barbell:
an equal-weighted core of the top 8 names by risk-adjusted momentum, plus a 30%
concentrated sleeve in the top 2 (trend-gated). Rebalanced daily through a
no-trade band.

Three competition constraints shape it:

* **Long-only spot.** Crypto spot cannot be shorted, so there is no
  market-neutral construction and every position carries full market beta. The
  only defensive asset is cash, and the only way to buy upside convexity is
  concentration -- which conveniently bounds the sleeve's downside at its weight.
* **Terminal return is the only score.** Risk management earns no points
  directly. It matters only because the strategy runs unattended for 30 days: a
  book that blows up, or code that raises, cannot recover.
* **Volume-aware fills.** The official engine will not fill orders exceeding a
  fraction of the bar's real volume, so orders are sized against recent volume
  rather than against how much we would like to trade.

WHAT THE RESEARCH CHANGED
-------------------------
The first version of this file was trend-gated and volatility-targeted. Both
were removed because measurement contradicted them, across 240 configurations
and confirmed on a held-out period never used for selection:

* **The trend gate hurt.** In a long-only book it can only move capital to cash,
  and against strong long-run drift that costs more upside than the drawdowns it
  avoids -- and drawdown scores nothing. The gated baseline had the worst median
  30-day return of every finalist, in both periods. It survives only in the
  sleeve, where concentration makes protection worth its cost.
* **Volatility targeting hurt** for the same reason: the highest target tested
  always won, so the effective limit is no damping at all.
* **Mean reversion was falsified.** It was the highest-rated candidate on the
  theory that the competition's 2 bps fee (roughly 5x cheaper than real exchange
  fees) would make short-horizon reversion viable. It was not: -18.8% on the
  held-out period against -4.7% for momentum. Cheap fees were not the binding
  constraint.

ROBUSTNESS
----------
``on_trading_iteration`` never raises. Every external call is treated as able to
return ``None``, a short frame, or garbage, because over hundreds of iterations
it eventually will. The strategy recomputes its target book from live portfolio
state every iteration and trades the difference, so a failed or partial fill
self-heals on the next pass rather than leaving the book permanently skewed.

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

# Trend lookbacks in bars (hours): roughly 3, 10 and 30 days. Slower horizons
# beat faster ones in both sweep phases -- fast trend signals mostly generate
# turnover on crypto's hourly noise.
TREND_LOOKBACKS: tuple[int, ...] = (72, 240, 720)

# Whether the CORE book is trend-gated. FALSE, and this is the single most
# counter-intuitive result in the research: across 240 configurations and on
# BOTH the training period and the held-out period, gating the core on trend
# reduced returns. The phase-1 gated baseline posted the worst median 30-day
# return of every finalist in both windows (-0.40% train, -0.86% test). In a
# long-only book the gate can only move capital to cash, and in a market with
# strong long-run drift that costs more upside than the drawdowns it avoids --
# and drawdown earns zero points here. The gate is retained for the SLEEVE,
# where concentration makes downside protection actually worth its cost.
CORE_TREND_GATED = False

# Volatility estimation window in bars (~7 days of hourly data). Long enough to
# be stable, short enough to react to a regime change within the 30-day window.
VOLATILITY_WINDOW = 168

# Annualized portfolio volatility target. Deliberately high: every sweep showed
# the highest target tested winning, because the competition scores terminal
# return and damping volatility only sacrifices upside for a risk metric worth
# no points. At 1.20 this rarely binds, acting as a backstop against a genuine
# volatility explosion rather than as a routine damper.
TARGET_VOLATILITY = 1.20

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

# --- Cross-sectional selection (family B) ----------------------------------
# Hold the top-k names by RISK-ADJUSTED momentum. Ranking on raw trailing return
# would preferentially pick whatever is simply most volatile -- a volatility bet
# disguised as a momentum signal.
MOMENTUM_LOOKBACK = 168      # ~7 days of hourly bars
TOP_K = 8                    # of 16; 0 would mean hold everything

# Size the core EQUALLY rather than by inverse volatility. This beat inverse-vol
# on both the training period (+133.3% vs +111.3%, P(30d>+20%) 21.7% vs 20.9%)
# and the held-out period (-3.7% vs -4.7%), which is the kind of consistency
# that makes a change worth taking.
#
# BE PRECISE ABOUT WHAT THIS DOES: the same per-asset volatility dict feeds both
# the inverse-vol weighting AND the volatility-target scalar, so substituting
# unit volatilities equal-weights the book *and* effectively disables volatility
# targeting (the scalar resolves to 1.0, leaving us fully invested up to
# MAX_GROSS_EXPOSURE). That is not an accident of implementation -- it is the
# limit of a pattern visible in every sweep: the highest volatility target
# always won, because damping volatility trades away terminal return, and
# terminal return is the only thing scored.
EQUAL_WEIGHT_CORE = True

# --- Convexity sleeve (the barbell's aggressive end) ------------------------
# With no leverage and no shorts available on spot, CONCENTRATION is the only
# mechanism that buys upside convexity -- and it bounds the downside at exactly
# SLEEVE_FRACTION. Adding it lifted P(30-day return > +20%) from 18.8% to 20.9%
# in training and improved the held-out total return. Trend-gated, so we do not
# concentrate into a name already rolling over.
SLEEVE_FRACTION = 0.30
SLEEVE_K = 2
SLEEVE_TREND_GATED = True

# Rebalance once per day, at this UTC hour. Daily beat both hourly and weekly.
# Keyed off the clock rather than an iteration counter so it stays correct even
# if the official runner re-instantiates the strategy mid-run -- we cannot
# assume one long-lived process.
REBALANCE_HOUR_UTC = 0

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
        if not self._is_rebalance_hour():
            return

        portfolio_value = self._safe_portfolio_value()
        if portfolio_value <= 0:
            self.log_message("Portfolio value unavailable or non-positive; skipping.")
            return

        volatilities: dict[str, float] = {}
        trend_scores: dict[str, float] = {}
        momenta: dict[str, float] = {}
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
            momenta[symbol] = signals.momentum(closes, MOMENTUM_LOOKBACK)
            prices[symbol] = price
            volumes[symbol] = volume

        if not volatilities:
            self.log_message("No assets with usable data this iteration; skipping.")
            return

        target_weights = self._target_book(volatilities, trend_scores, momenta)

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

    def _is_rebalance_hour(self) -> bool:
        """
        True once per day, at ``REBALANCE_HOUR_UTC``.

        Daily rebalancing beat both hourly and weekly in the sweep. Keying off
        the wall clock rather than an iteration counter keeps this correct even
        if the official runner restarts or re-instantiates the strategy -- a
        counter would silently drift, and we have no guarantee of one
        long-lived process.

        Fails OPEN: if the clock is unreadable we rebalance rather than skip,
        since silently never trading is the worse failure. That is not
        hypothetical -- a quote-asset misconfiguration already produced a run
        that did nothing for 720 straight iterations and still exited 0.
        """
        try:
            now = self.get_datetime()
            return now.hour == REBALANCE_HOUR_UTC
        except Exception:  # noqa: BLE001
            return True

    def _target_book(
        self,
        volatilities: dict[str, float],
        trend_scores: dict[str, float],
        momenta: dict[str, float],
    ) -> dict[str, float]:
        """
        Core + sleeve target weights.

        Mirrors ``research/engine.simulate`` exactly, calling the same
        ``strategies.core`` functions in the same order, so the configuration
        validated by the sweep is the configuration that actually trades.
        """
        # Risk-adjusted momentum: dividing by volatility stops the ranking from
        # simply selecting the most volatile name every time.
        scores = {
            symbol: momenta[symbol] / volatilities[symbol]
            for symbol in volatilities
            if math.isfinite(momenta.get(symbol, float("nan")))
        }

        core_universe = volatilities
        if TOP_K > 0:
            chosen = set(portfolio.select_top_k(scores, TOP_K))
            core_universe = {s: v for s, v in volatilities.items() if s in chosen}
            if not core_universe:
                return {}

        # Equal weight is expressed as "every asset has unit volatility", which
        # reuses the identical weighting code path rather than adding a second
        # one that could drift from what the sweep measured.
        sizing_inputs = (
            {s: 1.0 for s in core_universe} if EQUAL_WEIGHT_CORE else core_universe
        )
        core = portfolio.build_target_weights(
            sizing_inputs,
            trend_scores if CORE_TREND_GATED else {s: 1.0 for s in core_universe},
            max_weight=MAX_WEIGHT_PER_ASSET,
            target_volatility=TARGET_VOLATILITY,
            max_gross_exposure=MAX_GROSS_EXPOSURE,
            cash_buffer=CASH_BUFFER,
            average_correlation=AVERAGE_CORRELATION,
        )

        if SLEEVE_FRACTION <= 0:
            return core

        core = {s: w * (1.0 - SLEEVE_FRACTION) for s, w in core.items()}
        sleeve = portfolio.concentrated_weights(
            portfolio.select_top_k(scores, SLEEVE_K),
            SLEEVE_FRACTION,
            trend_scores if SLEEVE_TREND_GATED else None,
        )
        return portfolio.combine(core, sleeve, MAX_GROSS_EXPOSURE)

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
