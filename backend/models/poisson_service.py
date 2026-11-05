"""Fetch historical matches and fit one Dixon-Coles model per competition."""

from __future__ import annotations

from datetime import date
import logging
import os

import httpx

from backend.agents.scout_agent import FOOTBALL_DATA_BASE, _fd_get
from backend.models.dixon_coles import MODEL_VERSION, fit_dixon_coles

logger = logging.getLogger(__name__)


CALENDAR_YEAR_COMPETITIONS = {"BSA", "ASL", "AAL", "JJL", "MLS"}
TOURNAMENT_COMPETITIONS = {"CL", "EL", "UCL", "DFB", "FAC", "CIT", "CDR"}


def season_start_year(target_date: date, competition_code: str = "") -> int:
    if competition_code in CALENDAR_YEAR_COMPETITIONS:
        return target_date.year
    return target_date.year if target_date.month >= 7 else target_date.year - 1


def _history_depth(matches: list[dict]) -> tuple[int, int]:
    teams = set()
    for match in matches:
        try:
            teams.add(int(match["homeTeam"]["id"]))
            teams.add(int(match["awayTeam"]["id"]))
        except (KeyError, TypeError, ValueError):
            continue
    return len(matches), len(teams)


async def build_poisson_context(matches: list[dict], target_date: date) -> dict:
    codes = sorted({m.get("competition_code") for m in matches if m.get("competition_code")})
    context = {"models": {}, "histories": {}, "model_version": MODEL_VERSION}
    if not codes:
        logger.warning("[POISSON] No competition codes supplied; no models fitted.")
        return context

    decay_days = float(os.environ.get("POISSON_TIME_DECAY_DAYS", "240"))

    async with httpx.AsyncClient(timeout=30) as client:
        for code in codes:
            try:
                current_season = season_start_year(target_date, code)
                current = await _fd_get(
                    client,
                    f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches",
                    params={"status": "FINISHED", "season": current_season},
                )
                history = list(current.get("matches") or [])
                match_count, team_count = _history_depth(history)

                # Early in a season, add the previous campaign rather than
                # fitting on a handful of noisy results.
                seasons_included = [current_season]
                needs_previous = match_count < max(60, team_count * 4) or code in TOURNAMENT_COMPETITIONS
                previous_seasons = 2 if code in {"DFB", "FAC", "CIT", "CDR"} else 1
                if needs_previous:
                    for offset in range(1, previous_seasons + 1):
                        season = current_season - offset
                        previous = await _fd_get(
                            client,
                            f"{FOOTBALL_DATA_BASE}/competitions/{code}/matches",
                            params={"status": "FINISHED", "season": season},
                        )
                        history.extend(previous.get("matches") or [])
                        seasons_included.append(season)

                # A provider can repeat records when season boundaries differ;
                # deduplicate before fitting and preserve chronological order.
                history = list({m.get("id"): m for m in history if m.get("id") is not None}.values())
                history.sort(key=lambda m: m.get("utcDate") or "")
                model = fit_dixon_coles(history, as_of=target_date, time_decay_days=decay_days)
                context["models"][code] = model
                context["histories"][code] = {
                    "training_matches": model.training_matches,
                    "teams": len(model.attack),
                    "current_season": current_season,
                    "seasons_included": seasons_included,
                    "time_decay_days": decay_days,
                }
                logger.info(
                    f"[POISSON] {code}: fitted {model.training_matches} matches, "
                    f"{len(model.attack)} teams, home_advantage={model.home_advantage:.3f}, "
                    f"rho={model.rho:.3f}."
                )
            except Exception as exc:
                logger.warning(f"[POISSON] Could not fit {code}: {exc}")

    return context
