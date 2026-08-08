"""
Walk-forward stability across sequential regimes.

Everything so far rested on ONE train/test split. A single split can flatter a
configuration by accident -- if the held-out period happens to suit it, you learn
almost nothing. This cuts the history into sequential blocks and scores the
shipped configuration on each, so the question becomes "does it work in most
regimes" rather than "did it work in the one slice we set aside".

No refitting happens here. The parameters are already frozen, so every block is
genuinely out-of-sample for a strategy that was selected before seeing it (with
the honest exception of the early blocks, which overlap the selection period --
those are labelled IN-SAMPLE in the output rather than quietly counted).

More parameter search is deliberately NOT what this does. At 280+ configurations
already tried, extra trials mostly buy a better-looking winner rather than a
better strategy. Stability across regimes is the thing still worth measuring.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import strategies.strategy as live  # noqa: E402
from research.engine import (  # noqa: E402
    Config,
    benchmark_curve,
    load_universe,
    simulate,
)

SELECTION_END_DAYS_AGO = 180  # everything more recent was held out


def shipped() -> Config:
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
        rebalance_band=live.REBALANCE_BAND,
        rebalance_every=24,
    )


def stats(curve) -> dict | None:
    if curve is None or curve.empty or len(curve) <= 30 * 24:
        return None
    v = curve.to_numpy()
    w = v[30 * 24:] / v[:-30 * 24] - 1.0
    return {
        "total": v[-1] / v[0] - 1.0,
        "median": float(np.median(w)),
        "p20": float((w > 0.20).mean()),
        "worst": float(w.min()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folds", type=int, default=6)
    args = parser.parse_args()

    prices = load_universe(list(live.UNIVERSE))
    config = shipped()

    # Each fold needs warmup before it can trade, so blocks carry the preceding
    # history and only the block itself is scored.
    warmup = max(max(config.trend_lookbacks), config.volatility_window,
                 config.momentum_lookback) + 24
    usable = len(prices) - warmup
    block = usable // args.folds
    holdout_start = len(prices) - SELECTION_END_DAYS_AGO * 24

    print(f"{len(prices)} hourly bars, {prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d}")
    print(f"{args.folds} sequential folds of ~{block // 24} days each\n")

    header = (f"{'fold':<6}{'period':<26}{'sample':<12}"
              f"{'strategy':>10}{'BTC':>10}{'basket':>10}{'beat BTC':>10}")
    print(header)
    print("-" * len(header))

    wins = comparable = 0
    for i in range(args.folds):
        lo = warmup + i * block
        hi = warmup + (i + 1) * block if i < args.folds - 1 else len(prices)
        panel = prices.iloc[max(0, lo - warmup):hi]

        s = stats(simulate(panel, config))
        b = stats(benchmark_curve(panel, "BTC"))
        e = stats(benchmark_curve(panel, None))
        if not (s and b):
            print(f"{i + 1:<6}{'(too short to score)':<26}")
            continue

        label = "held out" if lo >= holdout_start else "in-sample"
        beat = s["total"] > b["total"]
        wins += beat
        comparable += 1
        period = f"{prices.index[lo]:%Y-%m-%d}..{prices.index[hi - 1]:%Y-%m-%d}"
        print(f"{i + 1:<6}{period:<26}{label:<12}"
              f"{s['total']:>9.1%}{b['total']:>10.1%}{e['total']:>10.1%}"
              f"{'  yes' if beat else '   no':>10}")

    print("-" * len(header))
    print(f"\nbeat BTC buy-and-hold in {wins} of {comparable} folds")
    print("A configuration that only works in one regime would show it here.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
