# SoAI 2026 AI Algorithmic Trading Competition — Submission

Long-only cross-sectional momentum over liquid crypto spot pairs, structured as
a barbell and rebalanced daily.

Entrypoint: [`strategies/strategy.py`](strategies/strategy.py) →
`Strategy(lumibot.strategies.Strategy)`.

---

## The strategy

**Universe.** 24 liquid CCXT spot USDT pairs — 16 majors plus 8 deliberately
higher-volatility names (SUI, PEPE, INJ, FET, ARB, OP, SEI, TIA). Liquidity is
the binding constraint, not an afterthought: the official engine caps each child
order at a fraction of the bar's real minute volume and does not fill the
excess, so an illiquid name is untradeable at size no matter how good its signal
looks — every addition was screened on median daily dollar volume first.

*Why higher volatility?* There is no measurable return edge to improve (see the
significance section below), so the only lever that raises the odds of finishing
**first** is dispersion, and higher-volatility constituents produce it
mechanically. Measured: 30-day dispersion 23.6% → 33.1%, windows above +50%
1 → 3, and in the faithful Lumibot engine +122.7% → +218.9% over three years at
modestly higher turnover (656× → 830×, fees 13.1% → 16.6%, all figures net).

*What that does not claim.* The aggregate return gain comes almost entirely from
one 2023 fold, when several of these names had launch runs — and they were
screened in 2026, so the list is survivor-selected. One candidate (RNDR) was
dropped precisely because its history ends mid-2024 at a ticker migration, which
is the same failure this whole universe is exposed to. Fold-level consistency
against BTC actually got *worse* (2/6 → 1/6). The dispersion gain is the part
that should transfer, because it does not depend on which specific names were
picked; the return gain should not be believed.

**Core (40%).** Equal-weighted across the top 8 names ranked by *risk-adjusted*
momentum — trailing 7-day return divided by realized volatility. Ranking on raw
return would preferentially select whatever is simply most volatile, which is a
volatility bet wearing a momentum costume. Measured, not assumed: raw momentum
returned +97.8% in training against +133.3% risk-adjusted.

**Sleeve (60%).** Split between the top 2 names by the same ranking, and
trend-gated so we do not concentrate into something already rolling over. After
gross-exposure scaling those two names carry roughly **34% of the book each**,
the other six about 4.6%. On spot there is no leverage and no short leg, so
concentration is the *only* mechanism available to buy upside convexity — and it
bounds the downside at exactly the sleeve weight, which is what makes a bet this
size deliberate rather than reckless.

**Execution.** Daily rebalance through a 2% no-trade band, with every order
capped at **4%** of recent per-bar volume. That cap was chosen for robustness to
the run's starting capital, which is unpublished — see below.

**Cadence.** `self.sleeptime = "60M"`; the strategy wakes hourly and acts once
per day. Timing is keyed to the wall clock rather than an iteration counter, so
it stays correct if the runner restarts or re-instantiates the strategy.

---

## Why it looks like this

Design decisions came from measurement, and several contradicted the priors we
started with. The full record is in `research/`.

**Method.** A vectorized engine (`research/engine.py`) swept 280+ configurations
across ~14 distinct algorithms over three years of hourly bars, scoring each on the *distribution of 30-day
terminal returns* — the horizon the competition actually measures — rather than
on Sharpe. Finalists were then re-scored on a held-out period never used for
selection (`research/validate.py`). The engine imports the same
`strategies/core/` functions the live strategy uses, and
`tests/test_engine_reconciliation.py` asserts its vectorized indicators match
those functions bar-for-bar. `tests/test_live_matches_research.py` goes further
and asserts the live strategy and the engine build the *same book* from the same
inputs, to 1e-12, across three years — so a sweep cannot optimize a portfolio
different from the one submitted.

**Three things we removed because the data said so:**

| Removed | Why |
|---|---|
| Trend gate on the core | In a long-only book it can only move capital to cash. Against strong long-run drift that costs more upside than the drawdowns it avoids — and drawdown scores zero. Worst median 30-day return of every finalist, in both periods. |
| Volatility targeting | Same logic. The highest target tested always won, so the limit is no damping at all. |
| Mean reversion | Our highest-rated prior: the competition's 2 bps fee is ~5× cheaper than real exchange fees, which should have made short-horizon reversion viable. It did not. Tested both formulations — z-score against own history (−18.8% held-out) and residual against the basket median (−16.2% held-out) — against −3.7% for momentum. |
| Drawdown kill switch | Measured rather than assumed. Even the loosest threshold (stop at 50%) cut P(30d>+20%) from 21.7% to 17.5%, and tighter stops were far worse. De-risking into a selloff caps upside for no scoring benefit. |
| Signal blending | A fixed blend of momentum with slow momentum tied on the primary metric (21.7%) but lost the median tiebreak in training and carries two extra parameters. Complexity that does not earn its place. |
| Single-name concentration | The fast engine rated it best on the held-out right tail (P(30d>+20%) of 9.3% against 1.4%). The real engine returned **+0.4%** against +122.7%, because putting most of the book in one name runs straight into the volume participation cap — and the top-ranked name changes daily, forcing a large position through a straw. A case where the optimistic engine had to be checked against the faithful one. |

**Progression**, measured in the Lumibot engine on one fixed 30-day window
(BTC buy-and-hold returned +2.30% over it):

| Version | Return | Turnover | Fees |
|---|---|---|---|
| Trend-gated, vol-targeted, no band | −4.26% | 42.6× | 0.85% |
| + no-trade band | −3.66% | 30.5× | 0.61% |
| + validated config | **+0.51%** | **5.8×** | **0.12%** |

Over three years, scored on **window-aligned** 30-day terminal returns
(24,869 windows for every series):

| | Total | Median 30d | P(>20%) | P(>50%) | P(>100%) | P(<−25%) |
|---|---|---|---|---|---|---|
| **This strategy** | **+235.4%** | −0.39% | **20.4%** | **8.4%** | **3.2%** | 5.6% |
| BTC buy-and-hold | +149.8% | **+2.13%** | 12.4% | 0.7% | 0.0% | **1.4%** |
| Equal-weight basket | +11.0% | −0.27% | 19.8% | 4.8% | 0.3% | 8.5% |

The shape is the point: a **worse median** than BTC and a **worse left tail**
(5.6% vs 1.4% of windows below −25%), bought in exchange for a right tail
several times fatter. BTC never once produced a +100% 30-day window in this
history; this strategy did in 3.2% of them. That is the trade a single-window
rank tournament rewards — and the left tail is the price, stated plainly.

In the faithful Lumibot engine, which additionally models volume-capped fills,
the same three years return **+174.3%** rather than the +235.4% above. Local
sweep figures are an optimistic upper bound throughout, exactly as the
competition brief warns.

*Earlier versions of this README quoted 21.7% / 14.4%, then 18.0% / 12.4%. The
first pair came from a comparison where the strategy curve began after its
warmup while the benchmark began at bar zero — scored over different windows.
The second was correct for a 16-name universe that has since been replaced. Both
are superseded by the table above.*

**Sleeve sizing was decided on the deep tail.** Measured on the 16-name universe,
0.30 and 0.60 tie on P(>20%) and 0.30 wins the median tiebreak, but 0.60 wins
where it counts — P(>50%) 6.4% vs 5.4%, P(>100%) 1.7% vs 1.1%, and +122.7% vs
+98.4% in the real Lumibot engine. The price is a marginally negative median and
a ~58% maximum drawdown. Accepted deliberately: drawdown scores nothing, only
terminal return is ranked.

**The volume cap was decided on robustness, not on any single scenario.** The
official run's starting capital is unpublished, and the cap interacts with it
directly — too tight and a large book cannot reach its targets, too loose and a
small book over-trades:

| Cap | at $1M | at $10M |
|---|---|---|
| 2% | **+219.9%** | +95.3% |
| 3% | +190.5% | +134.2% |
| **4% (shipped)** | +176.0% | **+173.1%** |
| 5% | +170.0% | +199.7% |

At 2% the outcome swings 125 points on a number we cannot observe. 4% is nearly
budget-independent, giving up ~44 points against the template's implied $1M to
remove that exposure. For a run that happens once and cannot be corrected, that
is the right side of the bet.

**How much of this is statistically real?** Not much, and this is the most
important caveat in the repository.

The table above uses *overlapping* 30-day windows, so `n = 24,869` is not 24,869
independent observations — three years contains only **35 non-overlapping**
30-day windows. Recomputed on those:

| | Mean | Median | Std | Best | Worst | >50% | >100% |
|---|---|---|---|---|---|---|---|
| **This strategy** | +7.3% | −2.9% | **33.1%** | **+136.8%** | −26.6% | **3** | **1** |
| BTC buy-and-hold | +3.8% | +2.3% | 15.9% | +56.1% | −27.5% | 1 | 0 |

- Mean excess over BTC: **+3.5%, t = 0.90.** Still nowhere near significance.
- The strategy beat BTC in **17 of 35** windows — still not half.
- P(>100%) rests on a **single** window. Remove that one episode and the
  headline tail claim evaporates.

**So there is no demonstrable return edge over BTC, and this repository does not
claim one.** Thirty-five observations cannot resolve a tail difference; nothing
measurable on three years of crypto could.

**What does hold is structural, not statistical.** Concentrating 60% of the book
in two names, drawn from a deliberately high-volatility universe, mechanically
produces more dispersion — arithmetic, not a fitted result — and the data agrees:
**2.08× BTC's standard deviation**, with a far better best case (+136.8% vs
+56.1%) and a comparable worst case (−26.6% vs −27.5%). A single-window rank
tournament rewards dispersion at equal mean, because finishing first requires an
outlier and finishing mid-table pays the same as finishing last.

That is the honest case for this entry: **justified by structure, not by
demonstrated alpha.**

**Walk-forward across six regimes.** A single train/test split can flatter a
configuration by accident, so the frozen parameters were also scored on six
sequential ~177-day blocks:

| Fold | Period | Strategy | BTC | Beat BTC |
|---|---|---|---|---|
| 1 | 2023-09 → 2024-03 | **+475.5%** | +141.1% | yes |
| 2 | 2024-03 → 2024-08 | −16.0% | −4.4% | no |
| 3 | 2024-08 → 2025-02 | +36.7% | +54.2% | no |
| 4 | 2025-02 → 2025-08 | −2.1% | +17.1% | no |
| 5 | 2025-08 → 2026-02 | −48.9% | −42.9% | no |
| 6 | 2026-02 → 2026-08 | −9.1% | −6.2% | no (held out) |

**One of six**, and this is the least flattering number in the repository. The
entire aggregate advantage is fold 1 — the 2023 window in which several
high-volatility constituents had launch runs. Outside it the strategy trails BTC
in every single regime, including the held-out one.

Note this got *worse* when the universe was widened (2 of 6 → 1 of 6). That was
a knowing trade: fold consistency is a **level** measure, and there is no level
edge to protect (t = 0.90). Dispersion is what a rank tournament pays for, and
dispersion improved from 1.48× to 2.08×. Fold 2 is deliberately *not* patched —
any fix chosen now would be fitted to a period already seen.

So the accurate description is **higher variance than BTC, not better than
BTC**. That is still the shape a single-window rank tournament rewards, which is
why the entry stands, but it is a weaker claim than the aggregate suggests.

**Honest limitations.** BTC buy-and-hold beats this strategy on total return
over the full history and in half the individual regimes; we beat it on the
probability of a large 30-day gain, which is what a single-window tournament
rewards. On the held-out period
*no* strategy achieved a 30-day gain above 20% — including BTC at 0.6% — so
that period confirms the configuration does not break, but cannot confirm the
upside claim. And all local results are optimistic: the official engine layers
volume-aware partial fills on top of what we model.

---

## Robustness

The strategy runs unattended for 30 days with no opportunity to intervene, so
not crashing is itself part of the edge.

- `on_trading_iteration` never raises; failures are caught, logged, and retried.
- Every broker call is wrapped and treated as able to return `None` or garbage.
- The target book is recomputed from live portfolio state each iteration and the
  difference traded, so partial or failed fills self-heal rather than leaving the
  book permanently skewed.
- Unusable inputs (NaN volatility, non-positive price, unknown volume) drop the
  asset rather than defaulting it into a position.
- The circuit breaker fires on consecutive *code* failures, never on a drawdown:
  de-risking into a selloff would cap upside for no scoring benefit, while
  trading on through a defect can compound a bug into a wrecked book.
- Rebalance timing fails *open* — if the clock is unreadable we trade rather than
  skip. Silently never trading is the worse failure, and it is not hypothetical:
  an early quote-asset misconfiguration produced a run that did nothing for 720
  consecutive iterations and still exited 0.

---

## Reproducing

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python backtest.py
```

`python research/fetch_data.py` populates `data/` and `research/cache/` with
CCXT bars (with an explicit integrity audit — duplicates, ordering, gaps, OHLC
violations, zero-volume bars) before running the research scripts.

Dependencies are pinned exactly. `numpy` is split by environment marker because
2.4.x has no Python 3.10 build, which made a flat pin fail to install there
entirely. CI verifies a clean-clone install on Python 3.10, 3.11, 3.12 and 3.13,
without a pip cache, because a warm cache hides exactly this class of failure.

## Layout

```
strategies/strategy.py   competition entrypoint
strategies/core/         signals, portfolio construction, execution (pure functions)
backtest.py              local Lumibot harness (fees set to the competition's 2 bps)
research/                data fetch, vectorized engine, sweeps, out-of-sample validation
tests/                   133 tests: submission contract, degenerate inputs,
                         engine/live equivalence, unattended-run failure injection
```
