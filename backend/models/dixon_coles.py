"""Time-weighted Dixon-Coles football score model.

The model estimates team attack and defence ratings by weighted maximum
likelihood. It includes home advantage and the Dixon-Coles low-score
correlation adjustment. Predictions are normalized over a finite score matrix
so every mutually exclusive market sums to one.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timezone
import logging
import math
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import gammaln

logger = logging.getLogger(__name__)

MODEL_VERSION = "dixon_coles_v1"
_EPS = 1e-12


def _as_utc_datetime(value: date | datetime) -> datetime:
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime(value.year, value.month, value.day)
    return result.replace(tzinfo=result.tzinfo or timezone.utc).astimezone(timezone.utc)


def _parse_match_date(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def _low_score_tau(home_goals: np.ndarray, away_goals: np.ndarray, home_rate, away_rate, rho):
    """Dixon-Coles adjustment for 0-0, 1-0, 0-1 and 1-1."""
    tau = np.ones(np.broadcast(home_goals, away_goals).shape, dtype=float)
    tau = np.where((home_goals == 0) & (away_goals == 0), 1 - home_rate * away_rate * rho, tau)
    tau = np.where((home_goals == 0) & (away_goals == 1), 1 + home_rate * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 0), 1 + away_rate * rho, tau)
    tau = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, tau)
    return tau


@dataclass
class DixonColesModel:
    base: float
    home_advantage: float
    rho: float
    attack: dict[int, float]
    defence: dict[int, float]
    team_names: dict[int, str]
    matches_per_team: dict[int, int]
    training_matches: int
    as_of: datetime
    time_decay_days: float

    def expected_goals(self, home_team_id: int, away_team_id: int) -> tuple[float, float]:
        if home_team_id not in self.attack or away_team_id not in self.attack:
            raise KeyError("Both teams must be present in the fitted history")
        home_rate = math.exp(
            self.base + self.home_advantage
            + self.attack[home_team_id] - self.defence[away_team_id]
        )
        away_rate = math.exp(
            self.base + self.attack[away_team_id] - self.defence[home_team_id]
        )
        # Prevent an unstable fit from producing extreme tails on sparse data.
        return float(np.clip(home_rate, 0.15, 4.5)), float(np.clip(away_rate, 0.15, 4.5))

    def score_matrix(self, home_team_id: int, away_team_id: int, max_goals: int = 10) -> np.ndarray:
        home_rate, away_rate = self.expected_goals(home_team_id, away_team_id)
        goals = np.arange(max_goals + 1)
        home_pmf = np.exp(goals * math.log(home_rate) - home_rate - gammaln(goals + 1))
        away_pmf = np.exp(goals * math.log(away_rate) - away_rate - gammaln(goals + 1))
        matrix = np.outer(home_pmf, away_pmf)
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        matrix *= _low_score_tau(hg, ag, home_rate, away_rate, self.rho)
        matrix = np.clip(matrix, 0, None)
        total = float(matrix.sum())
        if not math.isfinite(total) or total <= 0:
            raise ValueError("Could not normalize Poisson score matrix")
        return matrix / total

    def probabilities(self, home_team_id: int, away_team_id: int, max_goals: int = 10) -> dict:
        matrix = self.score_matrix(home_team_id, away_team_id, max_goals=max_goals)
        home_rate, away_rate = self.expected_goals(home_team_id, away_team_id)
        home_win = float(np.tril(matrix, k=-1).sum())
        draw = float(np.trace(matrix))
        away_win = float(np.triu(matrix, k=1).sum())
        double_home = home_win + draw
        double_away = away_win + draw
        non_draw = max(home_win + away_win, _EPS)
        goals = np.arange(matrix.shape[0])
        total_goals = goals[:, None] + goals[None, :]
        return {
            "home_win": home_win,
            "draw": draw,
            "away_win": away_win,
            "btts_yes": float(1 - matrix[0, :].sum() - matrix[:, 0].sum() + matrix[0, 0]),
            "btts_no": float(matrix[0, :].sum() + matrix[:, 0].sum() - matrix[0, 0]),
            "over25": float(matrix[total_goals >= 3].sum()),
            "under25": float(matrix[total_goals <= 2].sum()),
            "over15": float(matrix[total_goals >= 2].sum()),
            "under45": float(matrix[total_goals <= 4].sum()),
            "double_chance_home": double_home,
            "double_chance_away": double_away,
            "draw_no_bet_home": home_win / non_draw,
            "draw_no_bet_away": away_win / non_draw,
            "expected_home_goals": home_rate,
            "expected_away_goals": away_rate,
        }


def fit_dixon_coles(
    matches: Iterable[dict],
    as_of: date | datetime,
    time_decay_days: float = 240.0,
    regularization: float = 0.02,
) -> DixonColesModel:
    """Fit a model from football-data.org match dictionaries.

    Each record needs homeTeam/awayTeam IDs, utcDate and a full-time score.
    Future, unfinished and malformed records are ignored.
    """
    as_of_dt = _as_utc_datetime(as_of)
    clean: list[dict] = []
    team_names: dict[int, str] = {}

    for match in matches:
        try:
            played_at = _parse_match_date(match["utcDate"])
            if played_at >= as_of_dt:
                continue
            home_id = int(match["homeTeam"]["id"])
            away_id = int(match["awayTeam"]["id"])
            full_time = match["score"]["fullTime"]
            home_goals = int(full_time["home"])
            away_goals = int(full_time["away"])
            if home_goals < 0 or away_goals < 0 or home_id == away_id:
                continue
        except (KeyError, TypeError, ValueError):
            continue
        age_days = max(0.0, (as_of_dt - played_at).total_seconds() / 86400)
        weight = 0.5 ** (age_days / max(1.0, time_decay_days))
        clean.append((home_id, away_id, home_goals, away_goals, weight))
        team_names[home_id] = match["homeTeam"].get("name", str(home_id))
        team_names[away_id] = match["awayTeam"].get("name", str(away_id))

    teams = sorted(team_names)
    if len(teams) < 4 or len(clean) < max(24, len(teams)):
        raise ValueError(f"Insufficient history: {len(clean)} usable matches for {len(teams)} teams")

    index = {team_id: i for i, team_id in enumerate(teams)}
    n = len(teams)
    home_idx = np.array([index[row[0]] for row in clean], dtype=int)
    away_idx = np.array([index[row[1]] for row in clean], dtype=int)
    home_goals = np.array([row[2] for row in clean], dtype=float)
    away_goals = np.array([row[3] for row in clean], dtype=float)
    weights = np.array([row[4] for row in clean], dtype=float)

    avg_home = max(0.2, float(np.average(home_goals, weights=weights)))
    avg_away = max(0.2, float(np.average(away_goals, weights=weights)))
    initial = np.zeros(3 + 2 * (n - 1), dtype=float)
    initial[0] = math.log(avg_away)
    initial[1] = max(-0.5, min(0.5, math.log(avg_home / avg_away)))
    initial[2] = -0.05

    def unpack(params):
        attack_values = np.r_[params[3:3 + n - 1], -np.sum(params[3:3 + n - 1])]
        start = 3 + n - 1
        defence_values = np.r_[params[start:start + n - 1], -np.sum(params[start:start + n - 1])]
        return attack_values, defence_values

    def objective(params):
        attack_values, defence_values = unpack(params)
        home_rate = np.exp(np.clip(
            params[0] + params[1] + attack_values[home_idx] - defence_values[away_idx], -4, 3
        ))
        away_rate = np.exp(np.clip(
            params[0] + attack_values[away_idx] - defence_values[home_idx], -4, 3
        ))
        tau = _low_score_tau(home_goals, away_goals, home_rate, away_rate, params[2])
        if np.any(tau <= 0) or not np.all(np.isfinite(tau)):
            return 1e12
        log_likelihood = (
            home_goals * np.log(home_rate) - home_rate - gammaln(home_goals + 1)
            + away_goals * np.log(away_rate) - away_rate - gammaln(away_goals + 1)
            + np.log(tau)
        )
        penalty = regularization * float(np.sum(attack_values ** 2) + np.sum(defence_values ** 2))
        return -float(np.sum(weights * log_likelihood)) + penalty

    bounds = [(-2.0, 2.0), (-1.0, 1.0), (-0.20, 0.20)] + [(-2.0, 2.0)] * (2 * (n - 1))
    result = minimize(objective, initial, method="L-BFGS-B", bounds=bounds, options={"maxiter": 350})
    if not result.success or not np.all(np.isfinite(result.x)):
        raise ValueError(f"Dixon-Coles fit failed: {result.message}")

    attack_values, defence_values = unpack(result.x)
    matches_per_team = {team_id: 0 for team_id in teams}
    for home_id, away_id, *_ in clean:
        matches_per_team[home_id] += 1
        matches_per_team[away_id] += 1

    return DixonColesModel(
        base=float(result.x[0]),
        home_advantage=float(result.x[1]),
        rho=float(result.x[2]),
        attack={team_id: float(attack_values[index[team_id]]) for team_id in teams},
        defence={team_id: float(defence_values[index[team_id]]) for team_id in teams},
        team_names=team_names,
        matches_per_team=matches_per_team,
        training_matches=len(clean),
        as_of=as_of_dt,
        time_decay_days=float(time_decay_days),
    )
