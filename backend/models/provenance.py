"""Hard pre-publication provenance gate.

Every selectable candidate must prove, before it reaches the portfolio
optimiser (and before it can be published), that:

  * ``created_at`` and ``kickoff_utc`` parse as timestamps and the
    prediction was produced strictly before kickoff;
  * the price it would be published at existed at-or-before the prediction
    (a small provider clock skew is tolerated, see below) and before kickoff;
  * the decimal price is greater than 1 and both probabilities are inside
    (0, 1).

A candidate that fails any of these checks is not publishable. When no
candidate survives, the correct pipeline output is NO BET. FORCE_PUBLISH
must never override this gate.

Environment configuration:
  PROVENANCE_MAX_FUTURE_PRICE_SKEW_SECONDS  default 120 seconds (contract).
      Provider book timestamps occasionally run ahead of the worker. Small
      skews are tolerated (MODEL_INTELLIGENCE.md); tighten to 0 once the
      provider clock is fully understood. Raise via the daily workflow env
      only if the provider is demonstrably at fault — never silently.
  PROVENANCE_REQUIRE_PRICE_TIMESTAMP          default false.
      When true (set in production via the daily workflow), a missing price
      timestamp makes the candidate unverifiable and therefore rejected.

This module intentionally mirrors ``model-validation.js`` in the Pulse
repository. Keep the two implementations aligned when changing the gate.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

DEFAULT_MAX_FUTURE_PRICE_SKEW_SECONDS = 120


def _max_future_price_skew_seconds() -> int:
    raw = os.environ.get("PROVENANCE_MAX_FUTURE_PRICE_SKEW_SECONDS", "").strip()
    try:
        value = int(float(raw)) if raw else DEFAULT_MAX_FUTURE_PRICE_SKEW_SECONDS
    except ValueError:
        value = DEFAULT_MAX_FUTURE_PRICE_SKEW_SECONDS
    return max(0, value)


def require_price_timestamp() -> bool:
    return os.environ.get("PROVENANCE_REQUIRE_PRICE_TIMESTAMP", "").strip().lower() in (
        "1",
        "true",
        "yes",
        "on",
    )


def parse_time(value) -> datetime | None:
    """Parse a datetime/ISO-8601 string (including a trailing Z) to an aware UTC datetime."""
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str) and value.strip():
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text[:-1] + "+00:00" if text.endswith("Z") else text)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number != number or abs(number) == float("inf"):
        return None
    return number


def is_probability(value) -> bool:
    number = _finite_float(value)
    return number is not None and 0.0 < number < 1.0


def is_price(value) -> bool:
    number = _finite_float(value)
    return number is not None and number > 1.0


def validate_candidate(
    *,
    created_at,
    kickoff_utc,
    price_updated_at,
    bookmaker_odds,
    model_prob,
    market_implied_prob=None,
    max_future_price_skew_seconds: int | None = None,
    require_price_timestamp_flag: bool | None = None,
) -> dict:
    """Return ``{"ok": bool, "issues": [...]}``. Never raises."""
    skew = _max_future_price_skew_seconds() if max_future_price_skew_seconds is None else max(0, int(max_future_price_skew_seconds))
    require_price = require_price_timestamp() if require_price_timestamp_flag is None else bool(require_price_timestamp_flag)

    issues: list[str] = []

    created = parse_time(created_at)
    if created is None:
        issues.append("created_at_missing" if created_at is None else "created_at_invalid")

    kickoff = parse_time(kickoff_utc)
    if kickoff is None:
        issues.append("kickoff_missing" if kickoff_utc is None else "kickoff_invalid")

    if created is not None and kickoff is not None and created >= kickoff:
        issues.append("prediction_not_before_kickoff")

    price_time = parse_time(price_updated_at)
    if price_time is None:
        if require_price:
            issues.append(
                "price_updated_at_missing" if price_updated_at is None else "price_updated_at_invalid"
            )
    else:
        if created is not None:
            if price_time > created + timedelta(seconds=skew):
                issues.append("price_updated_after_prediction")
        if kickoff is not None and price_time > kickoff:
            issues.append("price_updated_after_kickoff")

    if not is_price(bookmaker_odds):
        issues.append("bookmaker_odds_invalid")
    if not is_probability(model_prob):
        issues.append("model_prob_invalid")
    if market_implied_prob is not None and not is_probability(market_implied_prob):
        issues.append("market_implied_prob_invalid")

    return {"ok": not issues, "issues": issues}
