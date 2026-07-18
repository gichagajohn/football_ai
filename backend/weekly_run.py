"""
WEEKLY RUN — Football Pulse AI (GitHub Actions edition)

Entrypoint for the weekly GitHub Actions workflow (e.g. runs Sundays).
Generates the performance report and emails it.
"""

import logging
from datetime import date

from backend.weekly_report import generate_report
from backend.email_sender import send_email

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("weekly_run")


def main():
    logger.info("Generating weekly performance report...")
    report = generate_report(days=7)
    logger.info(f"\n{report}")

    subject = f"📊 Football Pulse AI — Weekly Report ({date.today().isoformat()})"
    send_email(subject, report)

    logger.info("Weekly run complete.")


if __name__ == "__main__":
    main()
