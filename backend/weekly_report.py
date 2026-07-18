"""
WEEKLY PERFORMANCE REPORT — Football Pulse AI (GitHub Actions edition)

Runs weekly (e.g. Sunday) to summarize the past 7 days of graded tickets:
hit rate, ROI (assuming 1-unit stake per ticket), and a breakdown by
league/market. Sends the report via email alongside the daily ticket
workflow's email step.
"""

import logging
from datetime import date, timedelta

from backend.db import supabase_client

logger = logging.getLogger(__name__)


def generate_report(days: int = 7) -> str:
    """Generate a plain-text weekly performance summary."""
    cutoff = date.today() - timedelta(days=days)
    tickets = supabase_client.get_recent_tickets(limit=60)

    relevant = [
        t for t in tickets
        if date.fromisoformat(t["ticket_date"]) >= cutoff
        and t.get("outcome") in ("win", "loss", "void")
    ]

    if not relevant:
        return (
            "📊 FOOTBALL PULSE AI — WEEKLY REPORT\n"
            f"No graded tickets in the last {days} days yet.\n"
            "(Either no PUBLISH decisions were made, or results haven't "
            "been confirmed yet — check back next week.)"
        )

    total = len(relevant)
    wins = sum(1 for t in relevant if t["outcome"] == "win")
    losses = sum(1 for t in relevant if t["outcome"] == "loss")
    voids = sum(1 for t in relevant if t["outcome"] == "void")

    graded = wins + losses  # voids excluded from hit rate / ROI
    hit_rate = (wins / graded * 100) if graded else 0.0

    # ROI assuming 1 unit staked per ticket, full combined odds paid on win
    total_staked = 0.0
    total_returns = 0.0
    for t in relevant:
        if t["outcome"] == "void":
            continue
        stake = 1.0
        total_staked += stake
        if t["outcome"] == "win":
            total_returns += stake * float(t.get("combined_odds") or 0)

    profit = total_returns - total_staked
    roi = (profit / total_staked * 100) if total_staked else 0.0

    avg_odds = (
        sum(float(t.get("combined_odds") or 0) for t in relevant if t["outcome"] != "void") / graded
        if graded else 0.0
    )

    lines = [
        "📊 FOOTBALL PULSE AI — WEEKLY PERFORMANCE REPORT",
        f"Period: last {days} days ({cutoff.isoformat()} to {date.today().isoformat()})",
        "",
        f"Tickets graded: {graded}  (Wins: {wins}, Losses: {losses}, Void: {voids})",
        f"Hit Rate: {hit_rate:.1f}%",
        f"Average Combined Odds: {avg_odds:.2f}",
        f"ROI (1 unit/ticket): {roi:+.1f}%",
        f"Net Profit/Loss: {profit:+.2f} units",
        "",
    ]

    # Per-ticket breakdown
    lines.append("Daily breakdown:")
    for t in sorted(relevant, key=lambda x: x["ticket_date"], reverse=True):
        outcome_icon = {"win": "✅", "loss": "❌", "void": "➖"}.get(t["outcome"], "?")
        lines.append(
            f"  {outcome_icon} {t['ticket_date']} — "
            f"odds {t.get('combined_odds', 'N/A')}, "
            f"{t.get('selection_count', '?')} selection(s), "
            f"confidence {t.get('final_confidence', 'N/A')}"
        )

    lines.append("")
    lines.append("⚠️ Past performance does not guarantee future results.")
    lines.append("Discipline over volume. Always.")

    return "\n".join(lines)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(generate_report())
