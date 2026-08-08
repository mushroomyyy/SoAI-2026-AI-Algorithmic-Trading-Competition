# SoAI 2026 AI Algorithmic Trading Competition — Submission

Long-only cross-sectional momentum over liquid crypto spot pairs, structured as
a barbell and rebalanced daily.

Entrypoint: [`strategies/strategy.py`](strategies/strategy.py) →
`Strategy(lumibot.strategies.Strategy)`.

---

## The strategy

**Universe.** The 16 most liquid CCXT spot USDT pairs. Liquidity is the binding
constraint, not an afterthought: the official engine caps each child order at a
fraction of the bar's real minute volume and does not fill the excess, so an
illiquid name is untradeable at size no matter how good its signal looks.

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
capped at 2% of recent per-bar volume.

**Cadence.** `self.sleeptime = "60M"`; the strategy wakes hourly and acts once
per day. Timing is keyed to the wall clock rather than an iteration counter, so
it stays correct if the runner restarts or re-instantiates the strategy.

---

## Why it looks like this

Design decisions came from measurement, and several contradicted the priors we
started with. The full record is in `research/`.

**Method.** A vectorized engine (`research/engine.py`) swept 240 configurations
over three years of hourly bars, scoring each on the *distribution of 30-day
terminal returns* — the horizon the competition actually measures — rather than
on Sharpe. Finalists were then re-scored on a held-out period never used for
selection (`research/validate.py`). The engine imports the same
`strategies/core/` functions the live strategy uses, and
`tests/test_engine_reconciliation.py` asserts its vectorized indicators match
those functions bar-for-bar, so a sweep cannot optimize a strategy different
from the one submitted.

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
(24,838 windows for every series):

| | Total | Median 30d | P(>20%) | P(>50%) | P(>100%) | P(<−25%) |
|---|---|---|---|---|---|---|
| **This strategy** | **+168.3%** | −0.58% | **18.0%** | **5.3%** | **1.4%** | 1.9% |
| BTC buy-and-hold | +149.8% | **+2.12%** | 12.4% | 0.7% | 0.0% | 1.4% |
| Equal-weight basket | +29.2% | +0.57% | 16.8% | 3.8% | 0.4% | 4.9% |

The shape is the point: a **worse median** than BTC, a similar left tail, and a
right tail several times fatter. BTC never once produced a +100% 30-day window
in this history; this strategy did in 1.4% of them. That is the trade a
single-window rank tournament rewards.

*An earlier version of this README quoted 21.7% / 14.4% here. Those came from a
comparison in which the strategy curve began after its warmup while the
benchmark curve began at bar zero — the two were scored over different windows.
Corrected above; the relative advantage is essentially unchanged, the absolute
levels were overstated.*

**Sleeve sizing was the last decision, and it was made on the deep tail.** At
0.30 versus 0.60 the primary metric is tied (21.7% vs 21.6%) and 0.30 wins the
median tiebreak, but 0.60 is better where it counts: P(>50%) 6.4% vs 5.4%,
P(>100%) 1.7% vs 1.1%, a better held-out total, and **+122.7% vs +98.4%** over
the full history in the real Lumibot engine. The price is a marginally negative
median 30-day return and a ~58% maximum drawdown. Accepted deliberately, because
drawdown scores nothing here and only terminal return is ranked.

**How much of this is statistically real?** Not much, and this is the most
important caveat in the repository.

The table above uses *overlapping* 30-day windows, so `n = 24,838` is not 24,838
independent observations — three years contains only **35 non-overlapping**
30-day windows. Recomputed on those:

| | Mean | Median | Std | Best | Worst | Beat BTC |
|---|---|---|---|---|---|---|
| **This strategy** | +5.0% | 0.0% | **23.6%** | **+100.6%** | **−20.3%** | — |
| BTC buy-and-hold | +3.8% | +2.3% | 15.9% | +56.1% | −27.5% | — |

- Mean excess over BTC: **+1.1%, t = 0.40.** Nowhere near significance.
- The strategy beat BTC in **15 of 35** windows — *less* than half.
- P(>50%) is **one** window for each. P(>100%) is **one** versus zero. Remove a
  single episode and the entire "fatter right tail" claim evaporates.

**So there is no demonstrable return edge over BTC, and this repository does not
claim one.** Thirty-five observations cannot resolve a tail difference; nothing
measurable on three years of crypto could.

**What does hold is structural, not statistical.** Concentrating 60% of the book
in two names mechanically produces more dispersion — that is arithmetic, not a
fitted result — and the data agrees: **1.48× BTC's standard deviation**, with a
better best case (+100.6% vs +56.1%) and, as it happens, a better worst case
(−20.3% vs −27.5%). A single-window rank tournament rewards dispersion at equal
mean, because finishing first requires an outlier and finishing mid-table pays
the same as finishing last.

That is the honest case for this entry: **justified by structure, not by
demonstrated alpha.**

**Walk-forward across six regimes.** A single train/test split can flatter a
configuration by accident, so the frozen parameters were also scored on six
sequential ~177-day blocks:

| Fold | Period | Strategy | BTC | Beat BTC |
|---|---|---|---|---|
| 1 | 2023-09 → 2024-03 | +176.3% | +139.1% | yes |
| 2 | 2024-03 → 2024-08 | −22.4% | +0.5% | no |
| 3 | 2024-08 → 2025-02 | +65.7% | +55.3% | yes |
| 4 | 2025-02 → 2025-08 | +3.9% | +22.2% | no |
| 5 | 2025-08 → 2026-02 | −45.8% | −41.4% | no |
| 6 | 2026-02 → 2026-08 | −20.3% | −8.5% | no (held out) |

**Two of six.** The strategy wins rarely and wins big — folds 1 and 3 beat BTC
by 37 and 10 points, while the four losses are mostly modest. That is the same
convexity the tail metrics describe, seen from a different angle. The aggregate +121.5%
leans heavily on fold 1. Fold 2 is the real warning: BTC gained 47.5% while this
lost 22.4%, the classic momentum failure when a stable leader outruns a choppy
tail and rotation bleeds. Fold 2 is deliberately *not* patched — any fix chosen
now would be fitted to a period we have already seen.

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
tests/                   86 tests: submission contract, degenerate inputs, engine reconciliation
```
