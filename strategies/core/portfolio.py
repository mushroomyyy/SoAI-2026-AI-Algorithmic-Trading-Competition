"""
Portfolio construction: signals -> target weights.

Implements candidate 5.4 (inverse-volatility weighting plus volatility
targeting) gated by candidate 5.1's trend score.

The whole module is a pure function of its inputs. It never sees prices, orders
or the broker -- only per-asset signal values -- which is what makes it
straightforward to unit test and cheap to sweep.

Key constraint from the competition, and it shapes everything here: crypto spot
is LONG-ONLY. There is no short leg, so the only defensive asset is cash. Total
weight is therefore allowed to sum to less than 1.0, and that shortfall *is* the
risk-management action, not an inefficiency to be optimized away.
"""

from __future__ import annotations

import math

import numpy as np


def inverse_volatility_weights(
    volatilities: dict[str, float], max_weight: float
) -> dict[str, float]:
    """
    Risk-balanced raw weights, before any trend gating or vol targeting.

    Assets with an unusable volatility estimate (nan, non-positive, non-finite)
    are dropped rather than defaulted. A fabricated low vol would translate
    directly into an enormous position, so "no estimate" must mean "no position".

    Weights are capped per name and renormalized, so one quiet asset cannot
    dominate the book. Capping and renormalizing can push another name above the
    cap, so it iterates to a fixed point.
    """
    usable = {
        asset: vol
        for asset, vol in volatilities.items()
        if vol is not None and math.isfinite(vol) and vol > 0
    }
    if not usable:
        return {}

    # Infeasible-cap case: if every asset sitting at the cap still cannot reach
    # a fully invested book, then the cap -- not the signal -- is binding, and
    # the most diversified admissible allocation is simply everything at the
    # cap. Trying to renormalize to 1.0 here would oscillate forever, because
    # the constraint (sum == 1) and the constraint (each <= cap) are mutually
    # unsatisfiable. The shortfall is held as cash, which is a legitimate
    # outcome in a long-only book rather than an error.
    if len(usable) * max_weight <= 1.0:
        return {asset: float(max_weight) for asset in usable}

    raw = {asset: 1.0 / vol for asset, vol in usable.items()}
    total = sum(raw.values())
    if total <= 0:
        return {}

    weights = {asset: value / total for asset, value in raw.items()}

    # Iterate: capping frees weight that renormalization pushes onto others,
    # which can lift them over the cap in turn. Guaranteed to terminate because
    # each pass permanently fixes at least one asset at the cap, and the
    # feasibility check above rules out the non-convergent case.
    for _ in range(len(weights) + 1):
        over = {a: w for a, w in weights.items() if w > max_weight + 1e-12}
        if not over:
            break
        capped = {a: max_weight for a in over}
        remaining = 1.0 - sum(capped.values())
        rest = {a: w for a, w in weights.items() if a not in over}
        rest_total = sum(rest.values())
        if rest_total <= 0 or remaining <= 0:
            weights = capped
            break
        weights = {**capped, **{a: w / rest_total * remaining for a, w in rest.items()}}

    return weights


def apply_trend_gate(
    weights: dict[str, float], trend_scores: dict[str, float]
) -> dict[str, float]:
    """
    Scale each weight by its trend score, moving the remainder to cash.

    This is the long-only expression of "get out of the way": an asset in a
    confirmed downtrend gets weight zero and that capital simply is not
    deployed. Missing scores are treated as 0.0 (defensive).
    """
    gated = {}
    for asset, weight in weights.items():
        score = trend_scores.get(asset, 0.0)
        if score is None or not math.isfinite(score):
            score = 0.0
        gated[asset] = weight * max(0.0, min(1.0, float(score)))
    return gated


def volatility_target_scalar(
    weights: dict[str, float],
    volatilities: dict[str, float],
    target_volatility: float,
    max_leverage: float = 1.0,
    average_correlation: float = 0.7,
) -> float:
    """
    Scalar that pushes estimated portfolio vol toward ``target_volatility``.

    Portfolio vol is estimated with a single average-correlation assumption
    rather than a full covariance matrix. That is a deliberate simplification:
    a sample covariance matrix over ~16 assets is noisy, needs shrinkage to be
    usable, and its errors concentrate exactly in the smallest-variance
    directions the optimizer would lever up. Crypto majors are strongly and
    persistently co-moving, so one correlation number captures most of the
    diversification effect with a fraction of the estimation risk.

    Capped at ``max_leverage``, which is 1.0 by default because the competition
    universe is spot -- there is no leverage available, so scaling up beyond
    fully invested is not merely risky, it is unexecutable.
    """
    if not weights or target_volatility <= 0:
        return 0.0

    variance = 0.0
    assets = [a for a in weights if a in volatilities]
    for i, asset_i in enumerate(assets):
        vol_i = volatilities[asset_i]
        weight_i = weights[asset_i]
        if not (math.isfinite(vol_i) and vol_i > 0):
            continue
        for j, asset_j in enumerate(assets):
            vol_j = volatilities[asset_j]
            weight_j = weights[asset_j]
            if not (math.isfinite(vol_j) and vol_j > 0):
                continue
            correlation = 1.0 if i == j else average_correlation
            variance += weight_i * weight_j * vol_i * vol_j * correlation

    if variance <= 0:
        return 0.0

    portfolio_volatility = math.sqrt(variance)
    if not math.isfinite(portfolio_volatility) or portfolio_volatility <= 0:
        return 0.0

    return float(min(max_leverage, target_volatility / portfolio_volatility))


def build_target_weights(
    volatilities: dict[str, float],
    trend_scores: dict[str, float],
    *,
    max_weight: float,
    target_volatility: float,
    max_gross_exposure: float,
    cash_buffer: float,
    average_correlation: float = 0.7,
) -> dict[str, float]:
    """
    Full pipeline: inverse-vol -> trend gate -> vol target -> exposure caps.

    Returns weights as fractions of portfolio value. The sum is intentionally
    allowed to be well under 1.0; the residual is cash, which is the strategy's
    only defensive asset.
    """
    base = inverse_volatility_weights(volatilities, max_weight=max_weight)
    if not base:
        return {}

    gated = apply_trend_gate(base, trend_scores)
    if not any(w > 0 for w in gated.values()):
        return {}

    scalar = volatility_target_scalar(
        gated,
        volatilities,
        target_volatility=target_volatility,
        max_leverage=1.0,
        average_correlation=average_correlation,
    )
    scaled = {asset: weight * scalar for asset, weight in gated.items()}

    # Hard exposure ceiling, honouring the cash buffer. Both are belt-and-braces
    # on top of vol targeting, which is an estimate and can be wrong when
    # volatility regime-shifts faster than the estimation window.
    ceiling = min(max_gross_exposure, 1.0 - cash_buffer)
    gross = sum(scaled.values())
    if gross > ceiling and gross > 0:
        scaled = {asset: weight * ceiling / gross for asset, weight in scaled.items()}

    return {
        asset: float(weight)
        for asset, weight in scaled.items()
        if np.isfinite(weight) and weight > 1e-6
    }
