"""
Out-of-sample validation for the sweep's finalists.

The sweep ranks configurations using the whole history, so its winner is partly
selected by luck: across ~240 trials, the top of a noisy ranking is expected to
be flattered. This script splits the history and re-scores the finalists on a
period never used for selection.

Reading the output honestly:

* A finalist that keeps its edge out-of-sample is worth submitting.
* A finalist that collapses out-of-sample was fitted to the training period, and
  the correct response is to fall back to a simpler configuration -- NOT to
  re-select on the test period, which would burn the only unbiased data we have.
* Benchmarks are re-scored on the same split, because "beats BTC in-sample" and
  "beats BTC out-of-sample" are very different claims.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from research.engine import (  # noqa: E402
    Config,
    benchmark_curve,
    load_universe,
    simulate,
    terminal_return_distribution,
)
from strategies.strategy import UNIVERSE  # noqa: E402

# The sweep's top finalists, plus the simpler variants they must justify
# themselves against. Naming them explicitly keeps the comparison honest: a
# complex winner has to beat the simple ones on the held-out period too.
FINALISTS: list[tuple[str, Config]] = [
    ("topk8 gate-off sleeve.30x2", Config(
        trend_lookbacks=(72, 240, 720), target_volatility=1.20, rebalance_every=24,
        rebalance_band=0.02, top_k=8, core_trend_gated=False,
        sleeve_fraction=0.30, sleeve_k=2)),
    ("topk8 gate-off sleeve.20x2", Config(
        trend_lookbacks=(72, 240, 720), target_volatility=1.20, rebalance_every=24,
        rebalance_band=0.02, top_k=8, core_trend_gated=False,
        sleeve_fraction=0.20, sleeve_k=2)),
    ("all gate-off sleeve.30x2", Config(
        trend_lookbacks=(72, 240, 720), target_volatility=1.20, rebalance_every=24,
        rebalance_band=0.02, top_k=0, core_trend_gated=False,
        sleeve_fraction=0.30, sleeve_k=2)),
    ("topk8 gate-off no sleeve", Config(
        trend_lookbacks=(72, 240, 720), target_volatility=1.20, rebalance_every=24,
        rebalance_band=0.02, top_k=8, core_trend_gated=False)),
    ("all gate-ON no sleeve (phase-1 baseline)", Config(
        trend_lookbacks=(72, 240, 720), target_volatility=0.70, rebalance_every=24,
        rebalance_band=0.02, top_k=0, core_trend_gated=True)),
]


def score(label: str, stats: dict) -> None:
    if not stats:
        print(f"  {label:<42} (insufficient data)")
        return
    print(f"  {label:<42} total={stats['total_return']:+7.1%} "
          f"med={stats['median']:+6.2%} P(>20%)={stats['p_gt_20']:5.1%} "
          f"P(<-25%)={stats['p_lt_neg25']:5.1%} n={stats['n_windows']}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-days", type=int, default=180,
                        help="held-out period at the end of the history")
    args = parser.parse_args()

    prices = load_universe(list(UNIVERSE))
    split = len(prices) - args.test_days * 24
    train, test = prices.iloc[:split], prices.iloc[split:]

    for name, panel in (("TRAIN (used for selection)", train), ("TEST (held out)", test)):
        print(f"\n== {name}: {panel.index[0]:%Y-%m-%d}..{panel.index[-1]:%Y-%m-%d} "
              f"({len(panel)} bars) ==")
        score("BTC hold", terminal_return_distribution(benchmark_curve(panel, "BTC")))
        score("EW basket", terminal_return_distribution(benchmark_curve(panel, None)))
        print("  " + "-" * 76)
        for label, config in FINALISTS:
            score(label, terminal_return_distribution(simulate(panel, config)))

    print("\nA finalist that only wins on TRAIN was selected by luck. Prefer the "
          "simplest configuration that holds up on TEST.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
