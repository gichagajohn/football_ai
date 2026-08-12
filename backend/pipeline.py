"""
FOOTBALL PULSE AI — Main Pipeline Orchestrator (GitHub Actions edition)
Runs all agents in sequence, stores results in Supabase, returns the
final published ticket as a string (for emailing).
"""

import logging
import os
from datetime import date

from backend.agents.scout_agent import run as run_scout
from backend.agents.pipeline_agents import (
    run_analyst,
    run_risk_filter,
    run_portfolio,
    run_auditor,
    run_decision,
    run_publisher,
)
from backend.db import supabase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("football_pulse")


def _float_env(name: str, default: float) -> float:
    """
    Reads a float-valued env var safely. Treats both "unset" AND
    "set but empty string" (e.g. a workflow env: block with a blank
    value, or an unpopulated repo variable/secret) as "use the default"
    instead of crashing with ValueError: could not convert string to
    float: ''. Also guards against a non-numeric value being pasted in
    by mistake — logs a warning and falls back rather than crashing the
    whole pipeline over one bad env var.
    """
    raw = os.environ.get(name, "")
    raw = raw.strip() if raw else ""
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning(f"Env var {name}='{raw}' is not a valid float — using default {default}.")
        return default


# Minimum data_completeness for a match to pass the Cleaner.
# Overridable via CLEANER_THRESHOLD env var for testing (e.g. lowering it
# temporarily while also testing with a broader LEAGUE_IDS override in
# scout_agent.py). Leave unset for normal runs — defaults to 0.5.
CLEANER_THRESHOLD = _float_env("CLEANER_THRESHOLD", 0.5)


async def run_pipeline(target_date: date | None = None) -> str:
    """Full pipeline run. Returns the final published ticket as a string."""
    target_date = target_date or date.today()
    date_str = target_date.strftime("%A, %d %B %Y")
    logger.info("=" * 60)
    logger.info("  FOOTBALL PULSE AI — Pipeline Start")
    logger.info(f"  Target Date: {date_str}")
    logger.info("=" * 60)

    # ── AGENT 1: SCOUT ──────────────────────────────────────────
    logger.info("[1/9] SCOUT AGENT running...")
    intelligence = await run_scout(target_date)
    logger.info(f"[1/9] Scout collected {len(intelligence)} matches.")

    if not intelligence:
        ticket = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔵 FOOTBALL PULSE AI\n"
            f"📅 {date_str}  |  🕗 08:00 EAT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🚫 NO BET TODAY\n\n"
            f"Reason: No fixtures found in top-5 leagues + UCL for {date_str}.\n\n"
            f"Discipline over volume. We wait.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.warning(f"NO BET TODAY — Scout found no fixtures for {date_str}.")
        supabase_client.store_ticket(target_date, ticket, {}, {"decision": "NO_BET", "reason": "No fixtures found."})
        return ticket

    # ── AGENT 2: CLEANER ─────────────────────────────────────────
    logger.info("[2/9] CLEANER AGENT running...")
    for m in intelligence:
        logger.info(
            f"[2/9]   {m.get('home_team', '?')} vs {m.get('away_team', '?')} "
            f"({m.get('league', '?')}) — completeness={m.get('data_completeness', 0)}"
        )
    clean_matches = [m for m in intelligence if m.get("data_completeness", 0) >= CLEANER_THRESHOLD]
    logger.info(f"[2/9] {len(clean_matches)}/{len(intelligence)} matches passed cleaning threshold (>= {CLEANER_THRESHOLD}).")

    if not clean_matches:
        ticket = (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"🔵 FOOTBALL PULSE AI\n"
            f"📅 {date_str}  |  🕗 08:00 EAT\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
            f"🚫 NO BET TODAY\n\n"
            f"Reason: All {len(intelligence)} collected matches failed the data "
            f"quality threshold (completeness >= {CLEANER_THRESHOLD}).\n\n"
            f"Discipline over volume. We wait.\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        )
        logger.warning("No matches passed cleaning threshold.")
        supabase_client.store_ticket(target_date, ticket, {}, {"decision": "NO_BET", "reason": "All matches failed cleaning threshold."})
        return ticket

    # ── AGENT 3: ANALYST ────────────────────────────────────────
    logger.info("[3/9] ANALYST AGENT running...")
    probabilities = run_analyst(clean_matches)
    logger.info(f"[3/9] Probabilities computed for {len(probabilities)} matches.")

    # ── AGENT 4: CONTEXT (merged into analyst) ──────────────────
    logger.info("[4/9] CONTEXT AGENT adjustments applied.")

    # ── AGENT 5: RISK ────────────────────────────────────────────
    logger.info("[5/9] RISK AGENT running...")
    safe_matches = run_risk_filter(probabilities, intelligence)
    logger.info(f"[5/9] {len(safe_matches)} matches approved after risk filtering.")

    # ── AGENT 6: PORTFOLIO ───────────────────────────────────────
    logger.info("[6/9] PORTFOLIO AGENT running...")
    portfolio = run_portfolio(safe_matches)
    logger.info(f"[6/9] Portfolio: {portfolio.get('combined_odds', 'N/A')} combined odds.")

    # ── AGENT 7: AUDITOR ─────────────────────────────────────────
    logger.info("[7/9] AUDITOR AGENT running...")
    audited = run_auditor(portfolio)
    logger.info(f"[7/9] Auditor verdict: {audited.get('auditor_verdict', 'UNKNOWN')}")

    # ── AGENT 8: DECISION ────────────────────────────────────────
    logger.info("[8/9] DECISION AGENT running...")
    decision = run_decision(audited, portfolio)
    logger.info(f"[8/9] Decision: {decision.get('decision')} — {str(decision.get('reason', ''))[:80]}")

    # ── AGENT 9: PUBLISHER ───────────────────────────────────────
    logger.info("[9/9] PUBLISHER AGENT running...")
    ticket = run_publisher(portfolio, decision, audited, date_str)
    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\n{ticket}")

    supabase_client.store_ticket(target_date, ticket, portfolio, decision)

    return ticket


if __name__ == "__main__":
    import asyncio
    print(asyncio.run(run_pipeline()))
