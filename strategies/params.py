"""
Parameters for the LOCAL backtest harness (``backtest.py``) only.

The official execution environment does not read this file -- it imports
``strategies.strategy.Strategy`` and supplies its own data feed. The universe
the strategy actually trades is ``strategies.strategy.UNIVERSE``; this file just
tells the local harness which CSVs to load so the two line up.

Populate ``data/`` first:

    python research/fetch_data.py
"""

from strategies.strategy import UNIVERSE

# Crypto-only. Per the plan, free 1-minute US-equity history is only a few days
# deep, so an equity sleeve could not be validated honestly in the time
# available; CCXT provides years of free minute bars for crypto.
STOCK_SLEEVE_SYMBOLS: list[str] = []

# Mirrors the live universe so local backtests and the official run trade the
# same names. Single source of truth: strategies/strategy.py.
CRYPTO_SLEEVE_SYMBOLS: list[str] = list(UNIVERSE)

# Benchmark line on the Lumibot tearsheet. BTC is the honest benchmark for a
# long-only crypto book -- if the strategy cannot beat simply holding BTC, it
# has no reason to exist.
STOCK_BENCH: str = "BTC"
CRYPTO_BENCH: str = "BTC"

# Derived set used by ``backtest.py`` to classify assets as CRYPTO vs STOCK.
CRYPTO_SYMBOLS: set[str] = set(CRYPTO_SLEEVE_SYMBOLS)
