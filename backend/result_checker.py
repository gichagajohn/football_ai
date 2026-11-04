"""
RESULT CHECKER — Football Pulse AI (GitHub Actions edition)

Runs daily (after the new prediction is published) to check outcomes of
PREVIOUS tickets whose matches have since been played. Grades each
selection win/loss/void based on the actual final score, then updates
the ticket's overall outcome.

This is what feeds the Memory Agent's weekly performance report, and
stores the final score so the website's Results page can display real
scorelines instead of just win/loss.

Provider: football-data.org (v4) — the SAME provider the Scout agent
already uses. (The previous API-Football / api-sports.io provider was
returning 403, so the result checker switched to the key we already
have and pay nothing extra for.)
"""

import logging
import os
import time
from datetime import date

import httpx

from backend.db import supabase_client

logger = logging.getLogger(__name__)

FOOTBALL_DATA_BASE = "https://api.football-data.org/v4"
FOOTBALL_DATA_KEY = os.environ.get("FOOTBALL_DATA_KEY", "")

# Match statuses that mean the match is over and the score is final.
_FINAL_STATUSES = {"FINISHED", "AWARDED"}

# ── rate limiting / retries (mirrors scout_agent discipline) ──
FOOTBALL_DATA_MIN_INTERVAL_SECONDS = 6.5
FOOTBALL_DATA_MAX_RETRIES = 2
FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS = 30.0

_last_request_time = 0.0


def _throttle() -> None:
    """Respect a minimum interval between football-data.org calls."""
    global _last_request_time
    now = time.monotonic()
    wait = FOOTBALL_DATA_MIN_INTERVAL_SECONDS - (now - _last_request_time)
    if wait > 0:
        time.sleep(wait)
    _last_request_time = time.monotonic()


def _fd_get(url: str) -> httpx.Response:
    """GET with throttle + retry on 429/5xx. Raises on persistent failure."""
    global _last_request_time
    for attempt in range(FOOTBALL_DATA_MAX_RETRIES + 1):
        _throttle()
        try:
            resp = httpx.get(
                url,
                headers={"X-Auth-Token": FOOTBALL_DATA_KEY},
                timeout=20,
            )
        except httpx.HTTPError as e:
            logger.warning(f"[RESULT_CHECK] football-data.org request failed ({url}): {e}")
            if attempt >= FOOTBALL_DATA_MAX_RETRIES:
                raise
            time.sleep(FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS)
            continue

        if resp.status_code == 429:
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else FOOTBALL_DATA_DEFAULT_RETRY_WAIT_SECONDS
            logger.warning(
                f"[RESULT_CHECK] football-data.org 429 on {url} "
                f"(attempt {attempt + 1}/{FOOTBALL_DATA_MAX_RETRIES}) — retrying in {wait:.0f}s"
            )
            if attempt >= FOOTBALL_DATA_MAX_RETRIES:
                resp.raise_for_status()
            time.sleep(wait)
            continue

        resp.raise_for_status()
        return resp

    raise RuntimeError(f"Exhausted retries for {url}")


def fetch_result(fixture_id: int) -> dict | None:
    """Fetch the final result for a football-data.org match id.

    Returns None if the match isn't finished yet (so the selection stays
    pending for a later run). Returns a dict with home_goals, away_goals,
    and status on success.
    """
    if not FOOTBALL_DATA_KEY:
        logger.warning(
            "[RESULT_CHECK] FOOTBALL_DATA_KEY is not set — cannot check results. "
            "Set it in the environment (same key the Scout agent uses)."
        )
        return None

    try:
        resp = _fd_get(f"{FOOTBALL_DATA_BASE}/matches/{fixture_id}")

        # /v4/matches/{id} returns a single Match resource at the top level.
        # List endpoints use a "matches" wrapper, but this endpoint does NOT
        # wrap the object in a "match" key. Looking for that key silently
        # discarded every successful response and left all selections pending.
        data = resp.json()
        if not isinstance(data, dict) or not data:
            logger.warning(
                f"[RESULT_CHECK] Match {fixture_id} returned an unexpected empty/non-object payload."
            )
            return None

        status = data.get("status")
        if status not in _FINAL_STATUSES:
            return None

        score = data.get("score") or {}
        # football-data.org: fullTime = score at 90 minutes (the standard
        # settlement basis for 1X2 / totals markets).
        full_time = score.get("fullTime") or {}
        home_goals = full_time.get("home")
        away_goals = full_time.get("away")
        if home_goals is None or away_goals is None:
            logger.warning(f"[RESULT_CHECK] Match {fixture_id} finished but has no full-time score.")
            return None

        return {
            "home_goals": int(home_goals),
            "away_goals": int(away_goals),
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

    logger.info(f"[RESULT_CHECK] Found {len(pending_tickets)} pending ticket(s).")

    # Several old selections can reference the same fixture. Fetch each match
    # at most once per run, while still grading every database selection.
    result_cache: dict[int, dict | None] = {}
    updated_selections = 0
    completed_tickets = 0

    for ticket in pending_tickets:
        ticket_date = date.fromisoformat(ticket["ticket_date"])
        selections = supabase_client.get_selections_for_date(ticket_date)
        if not selections:
            logger.warning(f"[RESULT_CHECK] Ticket {ticket_date} has no selections; leaving it pending.")
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

            fixture_id = int(sel["fixture_id"])
            if fixture_id not in result_cache:
                result_cache[fixture_id] = fetch_result(fixture_id)
            result = result_cache[fixture_id]

            if result is None:
                all_graded = False
                continue

            outcome = grade_selection(sel["market"], result["home_goals"], result["away_goals"])

            # Store the outcome AND the final score so the website's Results
            # page can display real scorelines, not just win/loss badges.
            updated = supabase_client.update_selection_outcome(
                sel["id"], outcome,
                home_score=result["home_goals"],
                away_score=result["away_goals"],
            )
            if not updated:
                # Never mark a ticket complete when one of its database writes
                # failed. The next run can safely retry the pending selection.
                all_graded = False
                logger.error(
                    f"[RESULT_CHECK] Could not save selection {sel['id']} "
                    f"for fixture {fixture_id}; it remains pending."
                )
                continue

            updated_selections += 1
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
            if supabase_client.update_ticket_outcome(ticket_date, ticket_outcome):
                completed_tickets += 1

    logger.info(
        f"[RESULT_CHECK] Finished: {updated_selections} selection(s) updated, "
        f"{completed_tickets} ticket(s) completed, {len(result_cache)} unique fixture(s) fetched."
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run()
