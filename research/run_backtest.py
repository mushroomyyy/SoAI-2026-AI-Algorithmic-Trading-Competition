"""
Run the strategy over an arbitrary window using the Lumibot Pandas engine.

``backtest.py`` at the repo root is the organizers' entrypoint and is
deliberately left simple. This wrapper adds a date window and a compact result
summary so we can score a 30-day window -- the horizon the competition actually
measures -- without editing the submitted file.

Usage:
    python research/run_backtest.py --days 30
    python research/run_backtest.py --start 2026-05-01 --end 2026-06-01
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

# Run from anywhere: this script lives one level below the repo root, and the
# harness it wraps is importable only from there.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import backtest as harness  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=30,
                        help="length of the window ending at --end (default 30)")
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--budget", type=float, default=1_000_000)
    args = parser.parse_args()

    pandas_data, starts, ends, missing = harness._load_pandas_data()
    if not pandas_data:
        raise SystemExit("no data loaded; run `python research/fetch_data.py` first")

    data_start, data_end = max(starts), min(ends)

    end = (
        datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
        if args.end else data_end.to_pydatetime()
    )
    start = (
        datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
        if args.start else end - timedelta(days=args.days)
    )
    start = max(start, data_start.to_pydatetime())
    end = min(end, data_end.to_pydatetime())

    print(f"[INFO] {len(pandas_data)} assets | data {data_start} .. {data_end}")
    print(f"[INFO] backtest window {start} .. {end} ({(end - start).days}d)")
    if missing:
        print(f"[WARN] missing CSVs: {sorted(missing)}")

    result = harness.Strategy.run_backtest(
        harness.PandasDataBacktesting,
        start,
        end,
        pandas_data=pandas_data,
        budget=args.budget,
        show_plot=False,
        show_tearsheet=False,
        save_tearsheet=False,
        show_indicators=False,
        **harness._execution_cost_kwargs(),
    )

    print("\n== result ==")
    if isinstance(result, dict):
        for key in ("total_return", "cagr", "max_drawdown", "sharpe", "volatility"):
            if key in result:
                print(f"  {key:16s} {result[key]}")
    else:
        print(f"  {result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
