"""Shared risk-adjusted portfolio optimizer for production and evaluation.

No leg-count or combined-odds target is imposed beyond the supplied production
configuration. Candidates must already have passed their strategy-specific gates.
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PortfolioLimits:
    min_selections: int
    max_selections: int
    combined_odds_min: float
    combined_odds_max: float
    beam_width: int
    max_stake: float


def portfolio_utility(
    selections: list[dict], probability_key: str = "model_prob", max_stake: float = 0.05
) -> tuple[float, float, float, float]:
    combined_odds = math.prod(float(sel["odds"]) for sel in selections)
    joint_probability = math.prod(float(sel[probability_key]) for sel in selections)
    expected_value = joint_probability * combined_odds - 1.0
    net_odds = combined_odds - 1.0
    if net_odds <= 0 or expected_value < 0:
        return -math.inf, combined_odds, joint_probability, expected_value
    kelly_fraction = max(0.0, expected_value / net_odds)
    stake = min(max_stake, kelly_fraction)
    utility = (
        joint_probability * math.log1p(stake * net_odds)
        + (1.0 - joint_probability) * math.log1p(-stake)
    )
    return utility, combined_odds, joint_probability, expected_value


def optimize_portfolio(
    candidates: list[dict],
    limits: PortfolioLimits,
    probability_key: str = "model_prob",
) -> dict | None:
    """Optimize one market per fixture using the production beam search."""
    if not candidates:
        return None

    def utility_for(items):
        combined = math.prod(float(sel["odds"]) for sel in items)
        joint = math.prod(float(sel[probability_key]) for sel in items)
        ev = joint * combined - 1.0
        net = combined - 1.0
        if net <= 0 or ev < 0:
            return -math.inf, combined, joint, ev
        stake = min(limits.max_stake, max(0.0, ev / net))
        utility = joint * math.log1p(stake * net) + (1.0 - joint) * math.log1p(-stake)
        return utility, combined, joint, ev

    ranked = []
    for candidate in candidates:
        utility, *_ = utility_for([candidate])
        item = dict(candidate)
        item["_single_utility"] = utility
        ranked.append(item)

    best_by_fixture = {}
    for candidate in ranked:
        current = best_by_fixture.get(candidate["fixture_id"])
        key = (
            candidate["_single_utility"],
            candidate[probability_key],
            -candidate["odds"],
        )
        if current is None or key > current[0]:
            best_by_fixture[candidate["fixture_id"]] = (key, candidate)
    ranked = [entry[1] for entry in best_by_fixture.values()]
    ranked.sort(key=lambda c: c["_single_utility"], reverse=True)

    def state_key(state):
        selections, utility, combined, joint_probability, _ = state
        return utility, joint_probability, -len(selections), -combined

    states = [([], 0.0, 1.0, 1.0, 0.0)]
    best = None
    for candidate in ranked:
        additions = []
        for state in states:
            selections = state[0] + [candidate]
            if limits.max_selections > 0 and len(selections) > limits.max_selections:
                continue
            combined = state[2] * candidate["odds"]
            joint_probability = state[3] * candidate[probability_key]
            if not math.isfinite(combined) or not math.isfinite(joint_probability):
                continue
            if combined < limits.combined_odds_min:
                continue
            if limits.combined_odds_max > 0 and combined > limits.combined_odds_max + 1e-9:
                continue
            net_odds = combined - 1.0
            portfolio_ev = joint_probability * combined - 1.0
            if net_odds <= 0 or portfolio_ev < 0:
                continue
            stake = min(limits.max_stake, max(0.0, portfolio_ev / net_odds))
            utility = (
                joint_probability * math.log1p(stake * net_odds)
                + (1.0 - joint_probability) * math.log1p(-stake)
            )
            if not math.isfinite(utility):
                continue
            new_state = (selections, utility, combined, joint_probability, portfolio_ev)
            additions.append(new_state)
            if len(selections) >= limits.min_selections:
                key = state_key(new_state)
                if best is None or key > best[0]:
                    best = (key, new_state)

        states.extend(additions)
        states.sort(key=state_key, reverse=True)
        kept = states[:limits.beam_width]
        covered_lengths = {len(state[0]) for state in kept}
        for state in states[limits.beam_width:]:
            length = len(state[0])
            if length not in covered_lengths:
                kept.append(state)
                covered_lengths.add(length)
            if len(kept) >= limits.beam_width * 2:
                break
        states = kept

    if best is None:
        return None

    _, best_state = best
    selections, _, combined_odds, joint_probability, portfolio_ev = best_state
    return {
        "selections": selections,
        "combined_odds": round(combined_odds, 2),
        "portfolio_confidence": joint_probability,
        "portfolio_expected_value": portfolio_ev,
    }
