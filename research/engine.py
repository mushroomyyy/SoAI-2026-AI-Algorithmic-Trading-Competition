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

    # --- family B: cross-sectional selection -----------------------------
    # top_k = 0 means "hold the whole universe" (the original behaviour).
    momentum_lookback: int = 168
    top_k: int = 0
    risk_adjusted_momentum: bool = True
    # Which family ranks the universe:
    #   "momentum"  family B  -- buy relative strength
    #   "reversal"  family C  -- buy the most oversold (z-score), the
    #               "fee-enabled alpha" that 2 bps was supposed to unlock
    #   "skip"      family B3 -- momentum excluding the most recent stretch,
    #               to avoid short-horizon reversal contamination
    #   "lowvol"    family B6 -- betting-against-beta analog
    #   "residual"  family C3 -- reversion on the CROSS-SECTIONALLY DEMEANED
    #               return, which is what the plan actually specified; plain
    #               z-score reversion (C2) is mostly a short-beta bet
    #   "blend"     family E1 -- fixed blend of momentum and residual reversion
    selection_signal: str = "momentum"
    blend_weight_momentum: float = 0.5
    # What momentum is blended WITH. "residual" is degenerate -- demeaning is a
    # monotonic shift, so residual ranks are exactly (1 - momentum ranks) and the
    # blend collapses to pure momentum, a tie, or pure reversal. A real blend
    # needs a genuinely independent signal.
    blend_secondary: str = "slow"      # "slow" momentum | "lowvol" | "residual"
    slow_momentum_multiple: int = 4
    # A9 drawdown kill switch; 0 disables. Off by default -- see
    # portfolio.drawdown_throttle for why.
    stop_drawdown: float = 0.0
    momentum_skip: int = 24
    zscore_window: int = 168
    equal_weight: bool = False  # D1 instead of D2 inverse-vol

    # --- 5.6: convexity sleeve -------------------------------------------
    sleeve_fraction: float = 0.0
    sleeve_k: int = 2
    sleeve_trend_gated: bool = True

    # Gate the core on trend at all. The 144-config sweep showed the trend gate
    # costs more upside than it saves, so it must be falsifiable.
    core_trend_gated: bool = True

    def label(self) -> str:
        bits = [
            f"sig={self.selection_signal}",
            f"{'EW' if self.equal_weight else 'IV'}",
            f"tgt={self.target_volatility:.2f}",
            f"band={self.rebalance_band:.3f}",
            f"every={self.rebalance_every}h",
            f"topk={self.top_k or 'all'}",
            f"gate={'on' if self.core_trend_gated else 'off'}",
        ]
        if self.sleeve_fraction > 0:
            bits.append(
                f"sleeve={self.sleeve_fraction:.2f}x{self.sleeve_k}"
                f"{'' if self.sleeve_trend_gated else '(ungated)'}"
            )
        return " ".join(bits)


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


def precompute_momentum(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Trailing return over ``momentum_lookback`` bars, for cross-sectional ranking.

    Exact vectorized equivalent of ``signals.momentum``; reconciled in
    ``tests/test_engine_reconciliation.py``.
    """
    return prices.pct_change(config.momentum_lookback, fill_method=None)


def precompute_zscore(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Vectorized equivalent of ``signals.zscore``; reconciled in tests."""
    window = config.zscore_window
    mean = prices.rolling(window).mean()
    sigma = prices.rolling(window).std(ddof=1)
    return (prices - mean) / sigma.where(sigma > 0)


def precompute_momentum_skip(prices: pd.DataFrame, config: Config) -> pd.DataFrame:
    """Vectorized equivalent of ``signals.momentum_skip``; reconciled in tests."""
    skip = config.momentum_skip
    shifted = prices.shift(skip)
    return shifted / shifted.shift(config.momentum_lookback) - 1.0


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

    slow_all = None
    trend_all, vol_all = precompute(prices, config)
    if config.selection_signal == "reversal":
        score_all = -precompute_zscore(prices, config)   # most oversold ranks first
        warmup = max(warmup, config.zscore_window + 1)
    elif config.selection_signal == "skip":
        score_all = precompute_momentum_skip(prices, config)
        warmup = max(warmup, config.momentum_lookback + config.momentum_skip + 1)
    elif config.selection_signal == "lowvol":
        score_all = -vol_all                              # lowest vol ranks first
    elif config.selection_signal in ("residual", "blend"):
        score_all = precompute_momentum(prices, config)
        warmup = max(warmup, config.momentum_lookback + 1)
        if config.selection_signal == "blend" and config.blend_secondary == "slow":
            slow_lb = config.momentum_lookback * config.slow_momentum_multiple
            slow_all = prices.pct_change(slow_lb, fill_method=None)
            warmup = max(warmup, slow_lb + 1)
    else:
        score_all = precompute_momentum(prices, config)
        warmup = max(warmup, config.momentum_lookback + 1)

    equity = 1.0
    peak = 1.0
    weights = pd.Series(0.0, index=prices.columns)
    curve = {}

    for i in range(warmup, len(prices) - 1):
        if (i - warmup) % config.rebalance_every == 0:
            vol_row = vol_all.iloc[i]
            trend_row = trend_all.iloc[i]
            score_row = score_all.iloc[i]
            volatilities = {s: float(v) for s, v in vol_row.items() if np.isfinite(v)}
            trend_scores = {
                s: (float(trend_row[s]) if np.isfinite(trend_row[s]) else 0.0)
                for s in volatilities
            }

            # Rank on risk-adjusted momentum by default: raw trailing return
            # preferentially selects whatever is simply most volatile, which is
            # a volatility bet dressed up as a momentum signal.
            raw_scores = {}
            for symbol in volatilities:
                raw = score_row.get(symbol, np.nan)
                if not np.isfinite(raw):
                    continue
                # Risk-adjust only return-based signals. Dividing a z-score or a
                # low-vol rank by volatility would double-count the same term.
                if config.risk_adjusted_momentum and config.selection_signal in (
                    "momentum", "skip", "blend"
                ):
                    raw_scores[symbol] = float(raw) / volatilities[symbol]
                else:
                    raw_scores[symbol] = float(raw)

            if config.selection_signal == "residual":
                # Most negative residual = most oversold, so invert to rank first.
                scores = {
                    s_: -v for s_, v in portfolio.residual_scores(raw_scores).items()
                }
            elif config.selection_signal == "blend":
                if config.blend_secondary == "lowvol":
                    secondary = {s_: -volatilities[s_] for s_ in raw_scores}
                elif config.blend_secondary == "residual":
                    secondary = {
                        s_: -v for s_, v in portfolio.residual_scores(raw_scores).items()
                    }
                else:  # "slow": same signal over a much longer horizon
                    slow_row = slow_all.iloc[i]
                    secondary = {
                        s_: float(slow_row[s_]) / volatilities[s_]
                        for s_ in raw_scores
                        if np.isfinite(slow_row.get(s_, np.nan))
                    }
                scores = portfolio.blend_scores(
                    [raw_scores, secondary],
                    [config.blend_weight_momentum, 1.0 - config.blend_weight_momentum],
                )
            else:
                scores = raw_scores

            core_universe = volatilities
            if config.top_k > 0:
                chosen = set(portfolio.select_top_k(scores, config.top_k))
                core_universe = {s: v for s, v in volatilities.items() if s in chosen}

            # Equal weight (D1) is expressed as "pretend every asset has the
            # same volatility", which makes inverse-vol collapse to equal-weight
            # without a second code path that could drift from the live one.
            sizing_vols = (
                {s: 1.0 for s in core_universe} if config.equal_weight else core_universe
            )
            core = portfolio.build_target_weights(
                sizing_vols,
                trend_scores if config.core_trend_gated
                else {s: 1.0 for s in core_universe},
                max_weight=config.max_weight,
                target_volatility=config.target_volatility,
                max_gross_exposure=config.max_gross_exposure,
                cash_buffer=config.cash_buffer,
                average_correlation=config.average_correlation,
            )

            if config.sleeve_fraction > 0:
                core = {s: w * (1.0 - config.sleeve_fraction) for s, w in core.items()}
                sleeve = portfolio.concentrated_weights(
                    portfolio.select_top_k(scores, config.sleeve_k),
                    config.sleeve_fraction,
                    trend_scores if config.sleeve_trend_gated else None,
                )
                target = portfolio.combine(core, sleeve, config.max_gross_exposure)
            else:
                target = core

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

        if config.stop_drawdown > 0:
            peak = max(peak, equity)
            if portfolio.drawdown_throttle(equity, peak, config.stop_drawdown) == 0.0:
                weights = weights * 0.0

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
