"""Strategy internals, kept as pure functions.

Nothing in this package touches Lumibot, the network, the clock, or disk. That
is deliberate: the same functions are exercised by the unit tests, by the fast
research sweeps, and by the live strategy, so there is no "research code" that
can quietly drift from "production code" -- the classic way a backtested edge
evaporates in live trading.
"""
