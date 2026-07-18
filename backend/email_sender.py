"""
EMAIL SENDER — Football Pulse AI (GitHub Actions edition)

Sends the daily ticket (and optionally the weekly report) via SMTP.
Designed for Gmail with an App Password, but works with any SMTP server.

Required environment variables (set as GitHub Secrets):
  SMTP_HOST       e.g. smtp.gmail.com
  SMTP_PORT       e.g. 587
  SMTP_USERNAME   your sending email address
  SMTP_PASSWORD   app password (NOT your normal Gmail password)
  EMAIL_TO        recipient email address (can be same as SMTP_USERNAME)
"""

import logging
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)


def send_email(subject: str, body: str) -> None:
    """Send a plain-text email. Logs a warning and skips if SMTP env vars are missing."""
    smtp_host = os.environ.get("SMTP_HOST")
    smtp_port = int(os.environ.get("SMTP_PORT", "587"))
    smtp_user = os.environ.get("SMTP_USERNAME")
    smtp_pass = os.environ.get("SMTP_PASSWORD")
    email_to = os.environ.get("EMAIL_TO")

    if not all([smtp_host, smtp_user, smtp_pass, email_to]):
        logger.warning("[EMAIL] SMTP env vars not fully set — skipping email send.")
        return

    msg = MIMEMultipart()
    msg["From"] = smtp_user
    msg["To"] = email_to
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            server.starttls()
            server.login(smtp_user, smtp_pass)
            server.sendmail(smtp_user, email_to, msg.as_string())
        logger.info(f"[EMAIL] Sent: '{subject}' to {email_to}")
    except Exception as e:
        logger.error(f"[EMAIL] Failed to send email: {e}")
