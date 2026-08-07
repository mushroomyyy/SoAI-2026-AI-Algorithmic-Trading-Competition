"""
Parameter sweep over three years of hourly data.

Ranks configurations on the distribution of 30-day terminal returns -- the
horizon the competition actually scores -- rather than on a single window or on
annualized Sharpe.

Every configuration tried is printed, so the trial count is visible when reading
the winner. A result that is the argmax of many noisy trials is not the same as
a robust one, and hiding the denominator is how people fool themselves.
"""

from __future__ import annotations

import argparse
import itertools
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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=1095, help="history to use")
    parser.add_argument("--quick", action="store_true", help="small grid")
    args = parser.parse_args()

    prices = load_universe(list(UNIVERSE))
    prices = prices.iloc[-args.days * 24:]
    print(f"universe={prices.shape[1]} bars={len(prices)} "
          f"{prices.index[0]:%Y-%m-%d}..{prices.index[-1]:%Y-%m-%d}\n")

    # Benchmarks first: any configuration that cannot beat holding BTC has no
    # reason to exist, and seeing that number before the sweep keeps the results
    # honest.
    print("== benchmarks ==")
    for name, sym in (("BTC hold", "BTC"), ("EW basket", None)):
        stats = terminal_return_distribution(benchmark_curve(prices, sym))
        print(f"  {name:<10} total={stats['total_return']:+7.1%} "
              f"median30d={stats['median']:+6.2%} p05={stats['p05']:+6.2%} "
              f"p95={stats['p95']:+6.2%} P(>0)={stats['p_positive']:.0%} "
              f"P(>20%)={stats['p_gt_20']:.0%} P(<-25%)={stats['p_lt_neg25']:.0%}")

    if args.quick:
        grid = {
            "rebalance_band": [0.02, 0.05],
            "rebalance_every": [1, 24],
        }
    else:
        grid = {
            "rebalance_band": [0.01, 0.02, 0.05, 0.10],
            "rebalance_every": [1, 6, 24, 168],
            "target_volatility": [0.30, 0.45, 0.70],
            "trend_lookbacks": [(24, 72, 240), (48, 168, 480), (72, 240, 720)],
        }

    keys = list(grid)
    combos = list(itertools.product(*(grid[k] for k in keys)))
    print(f"\n== sweep: {len(combos)} configurations ==")

    results = []
    for values in combos:
        config = Config(**dict(zip(keys, values)))
        stats = terminal_return_distribution(simulate(prices, config))
        if not stats:
            continue
        results.append((stats, config))
        print(f"  total={stats['total_return']:+7.1%} med30={stats['median']:+6.2%} "
              f"p05={stats['p05']:+6.2%} p95={stats['p95']:+6.2%} "
              f"P(>0)={stats['p_positive']:.0%} P(>20%)={stats['p_gt_20']:.0%} "
              f"P(<-25%)={stats['p_lt_neg25']:.0%}  {config.label()}", flush=True)

    if not results:
        print("no results")
        return 1

    print(f"\n== ranked by P(30d > +20%), {len(results)} trials ==")
    results.sort(key=lambda r: (-r[0]["p_gt_20"], -r[0]["median"]))
    for stats, config in results[:10]:
        print(f"  P(>20%)={stats['p_gt_20']:.1%} med={stats['median']:+6.2%} "
              f"total={stats['total_return']:+7.1%} P(<-25%)={stats['p_lt_neg25']:.1%} "
              f"| {config.label()}")

    print(f"\n== ranked by median 30d return ==")
    results.sort(key=lambda r: -r[0]["median"])
    for stats, config in results[:10]:
        print(f"  med={stats['median']:+6.2%} P(>20%)={stats['p_gt_20']:.1%} "
              f"total={stats['total_return']:+7.1%} | {config.label()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
