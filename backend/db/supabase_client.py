"""
SUPABASE STORAGE — Football Pulse AI (GitHub Actions edition)

Stores daily tickets and tracks outcomes for the Memory Agent.
Uses Supabase's REST API (no special SDK needed — just httpx),
so this works fine inside a GitHub Actions runner.

Required environment variables:
  SUPABASE_URL       e.g. https://xxxxx.supabase.co
  SUPABASE_KEY       service_role key (has write access; keep secret!)

Schema (create this table in Supabase SQL editor first):

    create table prediction_tickets (
        id bigint generated always as identity primary key,
        ticket_date date not null unique,
        status text not null default 'pending',   -- pending | published | no_bet
        combined_odds numeric,
        selection_count smallint,
        final_confidence numeric,
        risk_level text,
        reason text,
        ticket_text text,
        outcome text default 'pending',           -- pending | win | loss | void
        created_at timestamptz default now()
    );

    create table ticket_selections (
        id bigint generated always as identity primary key,
        ticket_date date not null references prediction_tickets(ticket_date),
        fixture_id bigint not null,
        home_team text,
        away_team text,
        league text,
        market text,
        odds numeric,
        rationale text,
        outcome text default 'pending',           -- pending | win | loss | void
        home_score smallint,                       -- added for Results page (Phase 2)
        away_score smallint                         -- added for Results page (Phase 2)
    );
"""

import logging
import os
from datetime import date

import httpx

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")

HEADERS = {
    "apikey": SUPABASE_KEY,
    "Authorization": f"Bearer {SUPABASE_KEY}",
    "Content-Type": "application/json",
    "Prefer": "resolution=merge-duplicates",  # upsert on conflict
}


def _enabled() -> bool:
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("[SUPABASE] SUPABASE_URL/SUPABASE_KEY not set — skipping storage.")
        return False
    return True


def store_ticket(
    target_date: date,
    ticket_text: str,
    portfolio: dict,
    decision: dict,
) -> None:
    """Store the daily ticket and its selections in Supabase."""
    if not _enabled():
        return

    status = "published" if decision.get("decision") == "PUBLISH" else "no_bet"

    ticket_row = {
        "ticket_date": target_date.isoformat(),
        "status": status,
        "combined_odds": portfolio.get("combined_odds"),
        "selection_count": len(portfolio.get("selections", [])),
        "final_confidence": decision.get("final_confidence"),
        "risk_level": portfolio.get("risk_level"),
        "reason": decision.get("reason"),
        "ticket_text": ticket_text,
        "outcome": "pending",
    }

    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/prediction_tickets",
            headers=HEADERS,
            json=ticket_row,
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[SUPABASE] Stored ticket for {target_date} (status={status}).")
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to store ticket: {e}")
        return

    if status != "published":
        return

    selection_rows = []
    for sel in portfolio.get("selections", []):
        selection_rows.append({
            "ticket_date": target_date.isoformat(),
            "fixture_id": sel.get("fixture_id"),
            "home_team": sel.get("home_team"),
            "away_team": sel.get("away_team"),
            "league": sel.get("league"),
            "market": sel.get("market"),
            "odds": sel.get("odds"),
            "rationale": sel.get("rationale"),
            "outcome": "pending",
        })

    if not selection_rows:
        return

    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/ticket_selections",
            headers=HEADERS,
            json=selection_rows,
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[SUPABASE] Stored {len(selection_rows)} selection(s).")
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to store selections: {e}")


def get_pending_tickets(before_date: date) -> list[dict]:
    """Fetch published tickets with pending outcomes, dated before before_date."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/prediction_tickets",
            headers=HEADERS,
            params={
                "status": "eq.published",
                "outcome": "eq.pending",
                "ticket_date": f"lt.{before_date.isoformat()}",
                "select": "*",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch pending tickets: {e}")
        return []


def get_selections_for_date(ticket_date: date) -> list[dict]:
    """Fetch all selections for a given ticket date."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/ticket_selections",
            headers=HEADERS,
            params={
                "ticket_date": f"eq.{ticket_date.isoformat()}",
                "select": "*",
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch selections: {e}")
        return []


def update_selection_outcome(
    selection_id: int,
    outcome: str,
    home_score: int | None = None,
    away_score: int | None = None,
) -> None:
    """
    Update a single selection's outcome (win/loss/void) and, if known,
    the final match score. home_score/away_score are optional so this
    function still works for callers that only have the outcome.
    """
    if not _enabled():
        return
    payload = {"outcome": outcome}
    if home_score is not None:
        payload["home_score"] = home_score
    if away_score is not None:
        payload["away_score"] = away_score
    try:
        resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/ticket_selections",
            headers=HEADERS,
            params={"id": f"eq.{selection_id}"},
            json=payload,
            timeout=15,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to update selection {selection_id}: {e}")


def update_ticket_outcome(ticket_date: date, outcome: str) -> None:
    """Update a ticket's overall outcome (win = all selections won, loss = at least one lost)."""
    if not _enabled():
        return
    try:
        resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/prediction_tickets",
            headers=HEADERS,
            params={"ticket_date": f"eq.{ticket_date.isoformat()}"},
            json={"outcome": outcome},
            timeout=15,
        )
        resp.raise_for_status()
        logger.info(f"[SUPABASE] Updated ticket {ticket_date} outcome -> {outcome}")
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to update ticket outcome: {e}")


def get_recent_tickets(limit: int = 30) -> list[dict]:
    """Fetch the most recent tickets, ordered by date descending."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/prediction_tickets",
            headers=HEADERS,
            params={
                "select": "*",
                "order": "ticket_date.desc",
                "limit": str(limit),
            },
            timeout=15,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch recent tickets: {e}")
        return []
