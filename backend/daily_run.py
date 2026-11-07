"""
DAILY RUN — Football Pulse AI (GitHub Actions edition)
This is the entrypoint script run by the GitHub Actions daily workflow.
It:
  1. Checks outcomes of past published tickets (Memory Agent input)
  2. Runs the full prediction pipeline for today
  3. Emails the resulting ticket
"""
import asyncio
import logging
from datetime import date
from backend.pipeline import run_pipeline
from backend.email_sender import send_email
from backend import result_checker
from backend.db import supabase_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("daily_run")


async def main():
    today = date.today()

    # Production must never email a ticket that could not be published. Local
    # pipeline/dry-run callers can still use run_pipeline without Supabase.
    if not supabase_client.storage_configured():
        raise RuntimeError(
            "SUPABASE_URL and SUPABASE_KEY are required for the daily production run; "
            "pipeline and email suppressed."
        )

    # ── Step 1: Check outcomes of past tickets ──────────────────
    logger.info("Checking outcomes of past tickets...")
    try:
        result_checker.run(today=today)
    except Exception as e:
        logger.error(f"Result checking failed (non-fatal): {e}")

    # ── Step 2: Run the prediction pipeline ─────────────────────
    logger.info("Running prediction pipeline...")
    ticket = await run_pipeline(target_date=today)

    # ── Step 3: Email only after atomic storage has succeeded ───
    subject = f"⚽ Football Pulse AI — Daily Ticket ({today.isoformat()})"
    send_email(subject, ticket)

    logger.info("Daily run complete.")


if __name__ == "__main__":
    asyncio.run(main())
