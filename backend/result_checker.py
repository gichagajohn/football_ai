"""
RESULT CHECKER — Football Pulse AI (GitHub Actions edition)

Runs daily (after the new prediction is published) to check outcomes of
PREVIOUS tickets whose matches have since been played. Grades each
selection win/loss/void based on the actual final score, then updates
the ticket's overall outcome.

This is what feeds the Memory Agent's weekly performance report.
"""

import logging
import os
from datetime import date, timedelta

import httpx

from backend.db import supabase_client

logger = logging.getLogger(__name__)

API_FOOTBALL_KEY = os.environ.get("API_FOOTBALL_KEY", "")


def fetch_result(fixture_id: int) -> dict | None:
    """Fetch the final result for a fixture. Returns None if not finished yet."""
    try:
        resp = httpx.get(
            "https://v3.football.api-sports.io/fixtures",
            headers={"x-apisports-key": API_FOOTBALL_KEY},
            params={"id": fixture_id},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json().get("response", [])
        if not data:
            return None

        fixture = data[0]
        status = fixture["fixture"]["status"]["short"]
        if status not in ("FT", "AET", "PEN"):  # Full Time, After Extra Time, Penalties
            return None

        return {
            "home_goals": fixture["goals"]["home"],
            "away_goals": fixture["goals"]["away"],
            "status": status,
        }
    except Exception as e:
        logger.warning(f"[RESULT_CHECK] Failed to fetch result for fixture {fixture_id}: {e}")
        return None


def grade_selection(market: str, home_goals: int, away_goals: int) -> str:
    """
    Determine win/loss for a selection given the final score.
    Returns 'win', 'loss', or 'void' (if market can't be graded).
    """
    total_goals = home_goals + away_goals

    if market == "home_win":
        return "win" if home_goals > away_goals else "loss"
    if market == "away_win":
        return "win" if away_goals > home_goals else "loss"
    if market == "draw":
        return "win" if home_goals == away_goals else "loss"
    if market == "double_chance_home":
        return "win" if home_goals >= away_goals else "loss"
    if market == "double_chance_away":
        return "win" if away_goals >= home_goals else "loss"
    if market == "draw_no_bet_home":
        if home_goals == away_goals:
            return "void"  # stake returned — treat separately if needed
        return "win" if home_goals > away_goals else "loss"
    if market == "draw_no_bet_away":
        if home_goals == away_goals:
            return "void"
        return "win" if away_goals > home_goals else "loss"
    if market == "btts_yes":
        return "win" if (home_goals > 0 and away_goals > 0) else "loss"
    if market == "over25":
        return "win" if total_goals > 2.5 else "loss"

    logger.warning(f"[RESULT_CHECK] Unknown market '{market}' — marking void.")
    return "void"


def run(today: date | None = None) -> None:
    """
    Check outcomes for all published tickets dated before today whose
    selections are still 'pending', and update Supabase accordingly.
    """
    today = today or date.today()
    logger.info(f"[RESULT_CHECK] Checking pending tickets before {today}...")

    pending_tickets = supabase_client.get_pending_tickets(before_date=today)
    if not pending_tickets:
        logger.info("[RESULT_CHECK] No pending tickets to check.")
        return

    for ticket in pending_tickets:
        ticket_date = date.fromisoformat(ticket["ticket_date"])
        selections = supabase_client.get_selections_for_date(ticket_date)
        if not selections:
            continue

        all_graded = True
        any_loss = False
        any_void_only = True

        for sel in selections:
            if sel.get("outcome") != "pending":
                if sel.get("outcome") == "loss":
                    any_loss = True
                if sel.get("outcome") != "void":
                    any_void_only = False
                continue

            result = fetch_result(sel["fixture_id"])
            if result is None:
                all_graded = False
                continue

            outcome = grade_selection(sel["market"], result["home_goals"], result["away_goals"])
            supabase_client.update_selection_outcome(sel["id"], outcome)
            logger.info(
                f"[RESULT_CHECK] {sel['home_team']} vs {sel['away_team']} "
                f"({sel['market']}) -> {outcome} "
                f"(final: {result['home_goals']}-{result['away_goals']})"
            )

            if outcome == "loss":
                any_loss = True
            if outcome != "void":
                any_void_only = False

        if all_graded:
            if any_loss:
                ticket_outcome = "loss"
            elif any_void_only:
                ticket_outcome = "void"
            else:
                ticket_outcome = "win"
            supabase_client.update_ticket_outcome(ticket_date, ticket_outcome)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
