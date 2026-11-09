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
}


def storage_configured() -> bool:
    """Return whether the required Supabase service configuration is present."""
    return bool(SUPABASE_URL and SUPABASE_KEY)


def _enabled() -> bool:
    if not storage_configured():
        logger.warning("[SUPABASE] SUPABASE_URL/SUPABASE_KEY not set — skipping storage.")
        return False
    return True


class TicketStorageError(RuntimeError):
    """Raised when the authoritative ticket snapshot could not be stored."""


def store_ticket(
    target_date: date,
    ticket_text: str,
    portfolio: dict,
    decision: dict,
    shadow_portfolio: dict | None = None,
    run_id: str | None = None,
) -> bool:
    """Atomically store one complete ticket snapshot in Supabase.

    The database RPC replaces obsolete selections and updates retained odds in
    the same transaction as the ticket summary. Existing outcomes/scores are
    never supplied for an upsert. Every successful publish creates an immutable
    prediction snapshot, and the RPC rejects selections at/after kickoff or any
    rewrite after settlement.

    Returns False only when Supabase is not configured (local/dry runs). Any
    configured storage failure raises TicketStorageError so daily_run cannot
    email a ticket that was not safely published to the website.
    """
    if not _enabled():
        return False

    status = "published" if decision.get("decision") == "PUBLISH" else "no_bet"
    selections = portfolio.get("selections", []) if status == "published" else []
    selection_rows = []
    for sel in selections:
        # Deliberately omit outcome/home_score/away_score. New rows receive the
        # database default 'pending'; retained rows keep their existing result.
        selection_rows.append({
            "fixture_id": sel.get("fixture_id"),
            "home_team": sel.get("home_team"),
            "away_team": sel.get("away_team"),
            "league": sel.get("league"),
            "kickoff_utc": sel.get("kickoff_utc"),
            "market": sel.get("market"),
            "odds": sel.get("odds"),
            "rationale": sel.get("rationale"),
        })

    payload = {
        "p_ticket_date": target_date.isoformat(),
        "p_status": status,
        "p_combined_odds": portfolio.get("combined_odds") if status == "published" else None,
        "p_final_confidence": decision.get("final_confidence"),
        "p_risk_level": portfolio.get("risk_level") if status == "published" else None,
        "p_reason": decision.get("reason"),
        "p_ticket_text": ticket_text,
        "p_selections": selection_rows,
    }

    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/rpc/store_prediction_ticket",
            headers=HEADERS,
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"[SUPABASE] Atomic ticket storage failed for {target_date}: {e}")
        raise TicketStorageError(
            f"Could not store the complete ticket for {target_date}; email suppressed."
        ) from e

    logger.info(
        f"[SUPABASE] Stored ticket for {target_date} atomically "
        f"(status={status}, selections={len(selection_rows)})."
    )
    if run_id:
        _set_ticket_run_id(target_date, run_id)

    # Keep model diagnostics separate from the public ticket/selection tables.
    # A missing optional logging table must never prevent ticket settlement or
    # suppress an otherwise safely stored ticket email.
    _store_prediction_logs(target_date, portfolio.get("candidate_evaluations") or [], run_id=run_id)
    if shadow_portfolio:
        _store_shadow_prediction_logs(target_date, shadow_portfolio, run_id=run_id)
    return True


def _set_ticket_run_id(target_date: date, run_id: str) -> bool:
    """Best-effort linkage of this publish run to the stored ticket row."""
    try:
        resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/prediction_tickets",
            headers={**HEADERS, "Prefer": "return=minimal"},
            params={"ticket_date": f"eq.{target_date.isoformat()}"},
            json={"run_id": run_id},
            timeout=15,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        logger.error(f"[SUPABASE] Could not stamp run_id on ticket {target_date}: {e}")
        return False


def _store_prediction_logs(target_date: date, evaluations: list[dict], run_id: str | None = None) -> None:
    """Persist private model diagnostics.

    Rows are inserted, never upserted, so a rerun's candidates coexist with
    earlier runs instead of silently overwriting them. Deterministic linkage
    is via ``run_id`` on both ``model_prediction_logs`` and
    ``prediction_tickets``. Readers that aggregate must deduplicate by
    (ticket_date, fixture_id, market) keeping the latest created_at.
    """
    rows = []
    for item in evaluations:
        if item.get("fixture_id") is None or not item.get("market"):
            continue
        features = dict(item.get("feature_snapshot") or {})
        if item.get("provenance_issues"):
            features["provenance"] = {"ok": False, "issues": item["provenance_issues"]}
        rows.append({
            "ticket_date": target_date.isoformat(),
            "fixture_id": item.get("fixture_id"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "league": item.get("league"),
            "kickoff_utc": item.get("kickoff_utc"),
            "market": item.get("market"),
            "strategy": "production",
            "shadow_selected": False,
            "model_version": item.get("model_version") or "unknown",
            "calibration_version": item.get("calibration_version"),
            "raw_model_prob": item.get("raw_model_prob"),
            "model_prob": item.get("model_prob"),
            "market_implied_prob": item.get("market_implied_prob"),
            "bookmaker_implied_prob": item.get("bookmaker_implied_prob"),
            "fair_odds": item.get("fair_odds"),
            "bookmaker_odds": item.get("bookmaker_odds"),
            "closing_bookmaker_odds": item.get("closing_bookmaker_odds"),
            "odds_source": item.get("odds_source"),
            "expected_value": item.get("expected_value"),
            "eligible": bool(item.get("eligible")),
            "selected": bool(item.get("selected")),
            "rejection_reason": item.get("rejection_reason"),
            "feature_snapshot": features,
            "run_id": run_id,
        })
    if not rows:
        return
    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers={**HEADERS, "Prefer": "return=minimal"},
            json=rows,
            timeout=20,
        )
        resp.raise_for_status()
        logger.info(f"[SUPABASE] Stored {len(rows)} model prediction evaluation(s).")
    except Exception as e:
        logger.error(
            f"[SUPABASE] Model logging failed (ticket storage continues): {e}. "
            "Apply backend/db/migrations/20261110_run_linkage.sql."
        )


def _store_shadow_prediction_logs(target_date: date, shadow: dict, run_id: str | None = None) -> None:
    """Persist shadow candidates privately without affecting official storage."""
    strategy = shadow.get("strategy") or "unknown_shadow"
    rows = []
    for item in shadow.get("candidate_evaluations") or []:
        if item.get("fixture_id") is None or not item.get("market"):
            continue
        features = dict(item.get("feature_snapshot") or {})
        features["shadow_evaluation"] = {
            "config": shadow.get("config") or {},
            "combined_odds": shadow.get("combined_odds"),
            "portfolio_confidence": shadow.get("portfolio_confidence"),
            "portfolio_expected_value": shadow.get("portfolio_expected_value"),
        }
        robust_probability = item.get("robust_model_prob")
        rows.append({
            "ticket_date": target_date.isoformat(),
            "fixture_id": item.get("fixture_id"),
            "home_team": item.get("home_team"),
            "away_team": item.get("away_team"),
            "league": item.get("league"),
            "kickoff_utc": item.get("kickoff_utc"),
            "market": item.get("market"),
            "strategy": strategy,
            "shadow_selected": bool(item.get("shadow_selected")),
            "model_version": f"{item.get('model_version') or 'unknown'}__shadow_{strategy}",
            "calibration_version": item.get("calibration_version"),
            "raw_model_prob": item.get("raw_model_prob"),
            "model_prob": item.get("model_prob"),
            "robust_model_prob": robust_probability,
            "market_implied_prob": item.get("market_implied_prob"),
            "bookmaker_implied_prob": item.get("bookmaker_implied_prob"),
            "fair_odds": (1.0 / float(robust_probability)) if robust_probability else None,
            "bookmaker_odds": item.get("bookmaker_odds"),
            "closing_bookmaker_odds": item.get("closing_bookmaker_odds"),
            "odds_source": item.get("odds_source"),
            "expected_value": item.get("expected_value"),
            "robust_expected_value": item.get("robust_expected_value"),
            "eligible": bool(item.get("eligible")),
            # Never train official calibration on an experimental shadow pick.
            "selected": False,
            "rejection_reason": item.get("rejection_reason"),
            "feature_snapshot": features,
            "run_id": run_id,
        })
    if not rows:
        return
    try:
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers={**HEADERS, "Prefer": "return=minimal"},
            json=rows,
            timeout=20,
        )
        resp.raise_for_status()
        logger.info(
            f"[SUPABASE] Stored {len(rows)} private shadow evaluation(s) for {strategy}."
        )
    except Exception as e:
        logger.error(
            f"[SUPABASE] Shadow logging failed (official ticket continues): {e}. "
            "Apply backend/db/migrations/20261108_portfolio_shadow.sql."
        )


def get_pending_model_logs(before_date: date, limit: int = 5000) -> list[dict]:
    """Return unsettled private diagnostics for finished-result processing."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers=HEADERS,
            params={
                "ticket_date": f"lt.{before_date.isoformat()}",
                "outcome": "eq.pending",
                "select": "id,fixture_id,market,kickoff_utc,ticket_date,strategy",
                "order": "ticket_date.asc,fixture_id.asc",
                "limit": str(limit),
            },
            timeout=20,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch pending model logs: {e}")
        return []


def update_model_log_outcome(
    log_id: int,
    outcome: str,
    home_score: int,
    away_score: int,
) -> bool:
    """Settle one diagnostic row once; a pending guard makes retries safe."""
    if not _enabled():
        return False
    try:
        resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers={**HEADERS, "Prefer": "return=representation"},
            params={"id": f"eq.{log_id}", "outcome": "eq.pending"},
            json={
                "outcome": outcome,
                "home_score": home_score,
                "away_score": away_score,
            },
            timeout=15,
        )
        resp.raise_for_status()
        return bool(resp.json())
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to settle model log {log_id}: {e}")
        return False


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
) -> bool:
    """
    Update a single selection's outcome (win/loss/void) and, if known,
    the final match score. Returns True only when Supabase accepts the write.
    """
    if not _enabled():
        return False
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
        return True
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[SUPABASE] Failed to update selection {selection_id}: {e}; "
            f"response={e.response.text}"
        )
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to update selection {selection_id}: {e}")
    return False


def update_ticket_outcome(ticket_date: date, outcome: str) -> bool:
    """Update a ticket's overall outcome and report whether the write succeeded."""
    if not _enabled():
        return False
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
        return True
    except httpx.HTTPStatusError as e:
        logger.error(
            f"[SUPABASE] Failed to update ticket outcome: {e}; "
            f"response={e.response.text}"
        )
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to update ticket outcome: {e}")
    return False


def get_strategy_prediction_logs(limit: int = 50000) -> list[dict]:
    """Fetch private production/shadow logs for the post-settlement report."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers=HEADERS,
            params={
                "select": (
                    "ticket_date,created_at,fixture_id,home_team,away_team,league,market,"
                    "strategy,model_version,selected,shadow_selected,raw_model_prob,model_prob,"
                    "robust_model_prob,market_implied_prob,bookmaker_odds,closing_bookmaker_odds,"
                    "outcome,home_score,away_score,kickoff_utc"
                ),
                "order": "ticket_date.asc,created_at.asc",
                "limit": str(limit),
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch strategy prediction logs: {e}")
        return []


def get_prediction_logs(limit: int = 10000) -> list[dict]:
    """Fetch selected private model logs for offline calibration reporting."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers=HEADERS,
            params={
                "select": "ticket_date,fixture_id,market,raw_model_prob,model_prob,bookmaker_odds,selected,created_at",
                "selected": "eq.true",
                "order": "ticket_date.desc",
                "limit": str(limit),
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch model prediction logs: {e}")
        return []


def get_graded_selections(limit: int = 10000) -> list[dict]:
    """Fetch settled outcomes used by the offline model calibration report."""
    if not _enabled():
        return []
    try:
        resp = httpx.get(
            f"{SUPABASE_URL}/rest/v1/ticket_selections",
            headers=HEADERS,
            params={
                "select": "ticket_date,fixture_id,market,outcome",
                "outcome": "in.(win,loss)",
                "order": "ticket_date.desc",
                "limit": str(limit),
            },
            timeout=30,
        )
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logger.error(f"[SUPABASE] Failed to fetch graded selections: {e}")
        return []


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


def update_closing_price(
    ticket_date: date,
    fixture_id: int,
    market: str,
    closing_odds: float,
) -> int:
    """Stamp the closing price on today's unsettled diagnostic rows.

    Only rows whose closing price is still NULL are updated, so the first
    captured snapshot wins and reruns of the capture job are no-ops.
    Returns the number of rows updated (0 when already captured or storage
    is not configured).
    """
    if not _enabled():
        return 0
    try:
        resp = httpx.patch(
            f"{SUPABASE_URL}/rest/v1/model_prediction_logs",
            headers={**HEADERS, "Prefer": "return=representation"},
            params={
                "ticket_date": f"eq.{ticket_date.isoformat()}",
                "fixture_id": f"eq.{fixture_id}",
                "market": f"eq.{market}",
                "closing_bookmaker_odds": "is.null",
            },
            json={"closing_bookmaker_odds": round(float(closing_odds), 2)},
            timeout=15,
        )
        resp.raise_for_status()
        return len(resp.json() or [])
    except Exception as e:
        logger.error(
            f"[SUPABASE] Failed to set closing price for fixture {fixture_id} "
            f"market {market}: {e}"
        )
        return 0
