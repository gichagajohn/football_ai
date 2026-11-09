"""Chronological backtesting through the production analyst and portfolio code.

The input is a list of football-data.org-style finished matches with genuine
pre-match ``odds_snapshot`` prices. ``closing_odds_snapshot`` is optional and is
never passed to the model or optimizer. No future result is used for a fit.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, time, timezone
import math
from typing import Iterable

from backend.agents.pipeline_agents import (
    run_analyst,
    run_conservative_shadow,
    run_portfolio,
)
from backend.models.dixon_coles import fit_dixon_coles
from backend.models.evaluation import brier_score, log_loss
from backend.models.learning import fit_guarded_calibrator
from backend.models.settlement import grade_market


def _kickoff(match: dict) -> datetime:
    value = match.get("utcDate") or match.get("kickoff_utc")
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _score(match: dict) -> tuple[int, int]:
    full_time = match["score"]["fullTime"]
    return int(full_time["home"]), int(full_time["away"])


def _competition_code(match: dict) -> str:
    return str(match.get("competition_code") or (match.get("competition") or {}).get("code") or "")


def _fixture_id(match: dict):
    value = match.get("fixture_id")
    return match.get("id") if value is None else value


def _clean_match(match: dict) -> dict:
    home = match.get("homeTeam") or {}
    away = match.get("awayTeam") or {}
    competition = match.get("competition") or {}
    odds = match.get("odds_snapshot") or match.get("odds") or {}
    code = _competition_code(match)
    return {
        **match,
        "fixture_id": _fixture_id(match),
        "home_team_id": home.get("id", match.get("home_team_id")),
        "away_team_id": away.get("id", match.get("away_team_id")),
        "home_team": home.get("name", match.get("home_team")),
        "away_team": away.get("name", match.get("away_team")),
        "competition_code": code,
        "league": match.get("league") or competition.get("name") or code,
        "kickoff_utc": _kickoff(match).isoformat(),
        "odds_snapshot": odds,
        "odds_source": match.get("odds_source") or odds.get("_source") or "historical_bookmaker",
        "odds_bookmakers": match.get("odds_bookmakers") or odds.get("_bookmakers") or [],
        "odds_last_update": match.get("odds_last_update") or odds.get("_last_update"),
    }


def _valid_price(value) -> float | None:
    try:
        price = float(value)
    except (TypeError, ValueError):
        return None
    return price if math.isfinite(price) and price > 1.0 else None


def _valid_probability(value) -> float | None:
    try:
        probability = float(value)
    except (TypeError, ValueError):
        return None
    return probability if math.isfinite(probability) and 0 < probability < 1 else None


def _closing_price(match: dict, market: str) -> float | None:
    closing = match.get("closing_odds_snapshot") or match.get("closing_odds") or {}
    return _valid_price(closing.get(market))


def _odds_band(odds: float) -> str:
    if odds < 1.50:
        return "1.01-1.49"
    if odds < 2.00:
        return "1.50-1.99"
    if odds < 3.00:
        return "2.00-2.99"
    if odds < 5.00:
        return "3.00-4.99"
    return "5.00+"


def _max_drawdown(records: list[dict]) -> float:
    equity = peak = drawdown = 0.0
    for row in records:
        equity += float(row["profit"])
        peak = max(peak, equity)
        drawdown = max(drawdown, peak - equity)
    return drawdown


def _bet_metrics(records: list[dict], segments: bool = True) -> dict:
    ordered = sorted(records, key=lambda row: (row["date"], row.get("sequence", 0)))
    count = len(ordered)
    wins = sum(row["outcome"] == "win" for row in ordered)
    losses = sum(row["outcome"] == "loss" for row in ordered)
    pushes = sum(row["outcome"] == "void" for row in ordered)
    total_return = sum(float(row["return"]) for row in ordered)
    total_profit = sum(float(row["profit"]) for row in ordered)
    prices = [float(row["odds"]) for row in ordered]
    average_odds = sum(prices) / count if count else None
    fixtures = {
        str(fixture)
        for row in ordered
        for fixture in (row.get("fixture_ids") or [row.get("fixture_id")])
        if fixture is not None
    }
    clv_rows = [row for row in ordered if _valid_price(row.get("closing_odds"))]
    clv_values = [float(row["odds"]) / float(row["closing_odds"]) - 1.0 for row in clv_rows]
    result = {
        "bets": count,
        "wins": wins,
        "losses": losses,
        "pushes": pushes,
        "hit_rate": wins / (wins + losses) if wins + losses else None,
        "total_staked": float(count),
        "total_return": total_return,
        "net_profit": total_profit,
        "roi": total_profit / count if count else None,
        "average_odds": average_odds,
        "break_even_win_rate": 1.0 / average_odds if average_odds else None,
        "max_drawdown": _max_drawdown(ordered),
        "unique_fixtures": len(fixtures),
        "clv_bets": len(clv_rows),
        "clv_coverage": len(clv_rows) / count if count else None,
        "average_clv": sum(clv_values) / len(clv_values) if clv_values else None,
        "beat_closing_rate": (
            sum(float(row["odds"]) > float(row["closing_odds"]) for row in clv_rows) / len(clv_rows)
            if clv_rows else None
        ),
        "sample_warning": "Fewer than 100 bets; estimates are unstable." if count < 100 else None,
    }
    if not segments:
        return result

    def grouped(key_function) -> dict:
        buckets: dict[str, list[dict]] = defaultdict(list)
        for row in ordered:
            buckets[str(key_function(row))].append(row)
        return {key: _bet_metrics(rows, segments=False) for key, rows in sorted(buckets.items())}

    result["by_market"] = grouped(lambda row: row.get("market") or "unknown")
    result["by_league"] = grouped(lambda row: row.get("league") or "unknown")
    result["by_odds_band"] = grouped(lambda row: _odds_band(float(row["odds"])))
    return result


def summarize_bet_records(records: list[dict]) -> dict:
    """Public betting summary; one flat unit is staked on every record."""
    return _bet_metrics(records)


def _single_records(selections: Iterable[dict], fixtures: dict, day: str, strategy: str) -> list[dict]:
    rows = []
    for selection in selections:
        fixture_id = selection.get("fixture_id")
        match = fixtures.get(str(fixture_id))
        if not match:
            continue
        try:
            home_goals, away_goals = _score(match)
            odds = _valid_price(selection.get("odds"))
        except (KeyError, TypeError, ValueError):
            continue
        if odds is None:
            continue
        outcome = grade_market(selection.get("market", ""), home_goals, away_goals)
        gross_return = odds if outcome == "win" else 1.0 if outcome == "void" else 0.0
        rows.append({
            "strategy": strategy,
            "date": day,
            "fixture_id": fixture_id,
            "fixture_ids": [fixture_id],
            "home_team": selection.get("home_team"),
            "away_team": selection.get("away_team"),
            "league": selection.get("league"),
            "market": selection.get("market"),
            "odds": odds,
            "closing_odds": _closing_price(match, selection.get("market", "")),
            "model_prob": selection.get("model_prob"),
            "robust_model_prob": selection.get("robust_model_prob"),
            "outcome": outcome,
            "return": gross_return,
            "profit": gross_return - 1.0,
            "sequence": len(rows),
        })
    return rows


def settle_accumulator(single_records: list[dict], strategy: str | None = None) -> dict | None:
    """Settle one acca, reducing rather than losing it on DNB push legs."""
    if not single_records:
        return None
    outcomes = [row["outcome"] for row in single_records]
    combined_odds = math.prod(float(row["odds"]) for row in single_records)
    if any(outcome == "loss" for outcome in outcomes):
        outcome, gross_return = "loss", 0.0
        effective_odds = None
    else:
        effective_odds = math.prod(
            float(row["odds"]) for row in single_records if row["outcome"] != "void"
        )
        # Empty product = 1: an all-push acca returns its stake.
        gross_return = effective_odds
        outcome = "void" if all(value == "void" for value in outcomes) else "win"
    closing_prices = [row.get("closing_odds") for row in single_records]
    closing_odds = (
        math.prod(float(price) for price in closing_prices)
        if all(_valid_price(price) for price in closing_prices) else None
    )
    first = single_records[0]
    leagues = {row.get("league") for row in single_records}
    markets = {row.get("market") for row in single_records}
    return {
        "strategy": strategy or first.get("strategy") or "accumulator",
        "date": first["date"],
        "fixture_id": None,
        "fixture_ids": [row.get("fixture_id") for row in single_records],
        "league": next(iter(leagues)) if len(leagues) == 1 else "multiple",
        "market": next(iter(markets)) if len(markets) == 1 else "multiple",
        "odds": combined_odds,
        "effective_odds": effective_odds,
        "closing_odds": closing_odds,
        "outcome": outcome,
        "push_legs": sum(value == "void" for value in outcomes),
        "legs": len(single_records),
        "return": gross_return,
        "profit": gross_return - 1.0,
        "sequence": first.get("sequence", 0),
    }


def _consensus_selections(analyzed: list[dict]) -> list[dict]:
    """One no-vig 1X2 favorite per fixture, using direct prices only."""
    selections = []
    for match in analyzed:
        probabilities = match.get("market_probabilities") or {}
        odds = match.get("odds_snapshot") or {}
        choices = []
        for market in ("home_win", "draw", "away_win"):
            probability = _valid_probability(probabilities.get(market))
            price = _valid_price(odds.get(market))
            if probability is not None and price is not None:
                choices.append((probability, market, price))
        if not choices:
            continue
        probability, market, price = max(choices)
        selections.append({
            "fixture_id": match.get("fixture_id"),
            "home_team": match.get("home_team"),
            "away_team": match.get("away_team"),
            "league": match.get("league"),
            "market": market,
            "model_prob": probability,
            "odds": price,
        })
    return selections


def _append_probability_rows(
    destination: dict[str, list[dict]],
    name: str,
    evaluations: list[dict],
    fixtures: dict,
    probability_key: str,
    selected_only: bool = False,
    shadow: bool = False,
) -> None:
    for evaluation in evaluations:
        selected = evaluation.get("shadow_selected") if shadow else evaluation.get("selected")
        if selected_only and not selected:
            continue
        match = fixtures.get(str(evaluation.get("fixture_id")))
        probability = _valid_probability(evaluation.get(probability_key))
        if not match or probability is None:
            continue
        try:
            home_goals, away_goals = _score(match)
        except (KeyError, TypeError, ValueError):
            continue
        outcome = grade_market(evaluation.get("market", ""), home_goals, away_goals)
        if outcome not in ("win", "loss"):
            continue
        destination[name].append({"probability": probability, "outcome": int(outcome == "win")})


def _probability_metrics(rows: list[dict]) -> dict:
    if not rows:
        return {
            "predictions": 0, "brier_score": None, "log_loss": None,
            "mean_predicted_probability": None, "observed_win_rate": None,
        }
    probabilities = [float(row["probability"]) for row in rows]
    outcomes = [int(row["outcome"]) for row in rows]
    return {
        "predictions": len(rows),
        "brier_score": brier_score(probabilities, outcomes),
        "log_loss": log_loss(probabilities, outcomes),
        "mean_predicted_probability": sum(probabilities) / len(probabilities),
        "observed_win_rate": sum(outcomes) / len(outcomes),
    }


def walk_forward_portfolio_backtest(
    matches: list[dict],
    initial_train: int | None = None,
    retrain_every: int = 1,
    time_decay_days: float = 240.0,
    calibration_min_samples: int = 200,
    include_records: bool = False,
) -> dict:
    """Replay daily portfolios using only matches before each UTC test day.

    ``retrain_every`` is measured in test days. The production analyst and
    portfolio functions are called directly. Historical LLM risk assessments
    are not reproducible, so this evaluates the deterministic analyst/portfolio
    path and explicitly does not claim to replay those LLM decisions.
    """
    ordered = sorted(matches, key=_kickoff)
    if initial_train is None:
        initial_train = max(60, len(ordered) // 2)
    if initial_train < 24 or len(ordered) <= initial_train:
        raise ValueError("Not enough chronological matches for train/test evaluation")
    retrain_every = max(1, int(retrain_every))

    days: dict[str, list[dict]] = defaultdict(list)
    for match in ordered:
        days[_kickoff(match).date().isoformat()].append(match)
    eligible_days = []
    for day, day_matches in sorted(days.items()):
        cutoff = datetime.combine(datetime.fromisoformat(day).date(), time.min, tzinfo=timezone.utc)
        history_count = sum(_kickoff(match) < cutoff for match in ordered)
        if history_count >= initial_train:
            eligible_days.append((day, day_matches))
    if not eligible_days:
        raise ValueError("No test day has the requested initial training history")

    models: dict[str, object] = {}
    calibration_examples: list[dict] = []
    calibration = {"active": False}
    betting_records: dict[str, list[dict]] = defaultdict(list)
    probability_rows: dict[str, list[dict]] = defaultdict(list)
    daily_audit = []
    model_refits = model_fit_failures = skipped = no_bet_days = 0

    for day_index, (day, day_matches) in enumerate(eligible_days):
        cutoff = datetime.combine(datetime.fromisoformat(day).date(), time.min, tzinfo=timezone.utc)
        history = [match for match in ordered if _kickoff(match) < cutoff]
        codes = {_competition_code(match) for match in day_matches if _competition_code(match)}
        should_refit = day_index % retrain_every == 0
        fit_codes = codes if should_refit else {code for code in codes if code not in models}
        for code in sorted(fit_codes):
            code_history = [match for match in history if _competition_code(match) == code]
            try:
                models[code] = fit_dixon_coles(
                    code_history, as_of=cutoff, time_decay_days=time_decay_days
                )
                model_refits += 1
            except (KeyError, TypeError, ValueError) as exc:
                models.pop(code, None)
                model_fit_failures += 1
                daily_audit.append({"date": day, "competition": code, "fit_error": str(exc)})

        calibration = fit_guarded_calibrator(
            calibration_examples, min_samples=calibration_min_samples
        )
        clean = [_clean_match(match) for match in day_matches]
        analyzed = run_analyst(clean, {"models": models, "calibration": calibration})
        skipped += len(day_matches) - len(analyzed)
        fixtures = {str(_fixture_id(match)): match for match in day_matches}

        portfolio = run_portfolio(analyzed, reference_time=cutoff)
        shadow = run_conservative_shadow(portfolio.get("candidate_evaluations") or [], reference_time=cutoff)
        consensus = _consensus_selections(analyzed)
        strategies = {
            "production": portfolio.get("selections") or [],
            "robust_conservative_v1_shadow": shadow.get("selections") or [],
            "bookmaker_consensus_1x2": consensus,
        }
        daily_row = {
            "date": day,
            "training_matches": len(history),
            "training_cutoff": cutoff.isoformat(),
            "fixtures": len(day_matches),
            "analyzed_fixtures": len(analyzed),
            "calibration_active": bool(calibration.get("active")),
            "calibration_samples": calibration.get("samples", 0),
            "production_selections": len(strategies["production"]),
            "shadow_selections": len(strategies["robust_conservative_v1_shadow"]),
        }
        daily_audit.append(daily_row)
        if not strategies["production"]:
            no_bet_days += 1

        for strategy, selections in strategies.items():
            singles = _single_records(selections, fixtures, day, strategy)
            betting_records[f"{strategy}_singles"].extend(singles)
            accumulator = settle_accumulator(singles, strategy=f"{strategy}_accumulator")
            if accumulator:
                betting_records[f"{strategy}_accumulator"].append(accumulator)

        evaluations = portfolio.get("candidate_evaluations") or []
        _append_probability_rows(probability_rows, "production_raw_all_candidates", evaluations, fixtures, "raw_model_prob")
        _append_probability_rows(probability_rows, "production_model_all_candidates", evaluations, fixtures, "model_prob")
        _append_probability_rows(probability_rows, "bookmaker_no_vig_all_candidates", evaluations, fixtures, "market_implied_prob")
        _append_probability_rows(probability_rows, "production_model_selected", evaluations, fixtures, "model_prob", selected_only=True)
        shadow_evaluations = shadow.get("candidate_evaluations") or []
        _append_probability_rows(probability_rows, "shadow_robust_all_candidates", shadow_evaluations, fixtures, "robust_model_prob", shadow=True)
        _append_probability_rows(probability_rows, "shadow_robust_selected", shadow_evaluations, fixtures, "robust_model_prob", selected_only=True, shadow=True)

        # Production calibration learns only from selected production outcomes
        # that finished on earlier test days. Shadow rows never enter this set.
        for selection in strategies["production"]:
            match = fixtures.get(str(selection.get("fixture_id")))
            if not match:
                continue
            try:
                home_goals, away_goals = _score(match)
            except (KeyError, TypeError, ValueError):
                continue
            outcome = grade_market(selection.get("market", ""), home_goals, away_goals)
            probability = _valid_probability(selection.get("raw_model_prob", selection.get("model_prob")))
            if outcome in ("win", "loss") and probability is not None:
                calibration_examples.append({
                    "ticket_date": cutoff.date(),
                    "probability": probability,
                    "outcome": int(outcome == "win"),
                })

    betting = {name: _bet_metrics(rows) for name, rows in sorted(betting_records.items())}
    # Include empty baseline keys explicitly when a short test had no prices.
    for strategy in ("production", "robust_conservative_v1_shadow", "bookmaker_consensus_1x2"):
        for suffix in ("singles", "accumulator"):
            betting.setdefault(f"{strategy}_{suffix}", _bet_metrics([]))
    probability = {name: _probability_metrics(rows) for name, rows in sorted(probability_rows.items())}
    report = {
        "evaluation": "production_portfolio_walk_forward_v1",
        "test_days": len(eligible_days),
        "test_matches": sum(len(day_matches) for _, day_matches in eligible_days),
        "initial_train": initial_train,
        "retrain_every_days": retrain_every,
        "time_decay_days": time_decay_days,
        "model_refits": model_refits,
        "model_fit_failures": model_fit_failures,
        "skipped_matches": skipped,
        "no_bet_days": no_bet_days,
        "betting": betting,
        "probability": probability,
        "daily": daily_audit,
        "methodology": {
            "stake": "One flat unit per single or daily accumulator; pushes return stake.",
            "dnb": "A draw is void. In an acca, void legs are removed and other legs continue.",
            "clv": "taken_odds / closing_odds - 1; reported only with genuine separate closing prices.",
            "code_path": "run_analyst -> run_portfolio; robust_conservative_v1 uses the shared optimizer.",
            "risk_filter": "Historical LLM risk decisions are not replayed; deterministic model/portfolio evaluation only.",
            "activation": "Shadow metrics are descriptive and never activate a production strategy.",
        },
        "note": "Historical evaluation only; no future performance is guaranteed.",
    }
    if include_records:
        report["bet_records"] = {name: rows for name, rows in sorted(betting_records.items())}
        report["probability_records"] = {name: rows for name, rows in sorted(probability_rows.items())}
    return report


def main() -> None:
    """Run a local JSON backtest without fetching or exposing any credentials."""
    import argparse
    import json
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="JSON list, or an object containing a 'matches' list.")
    parser.add_argument("--output", help="Optional report JSON path; stdout when omitted.")
    parser.add_argument("--initial-train", type=int, default=None)
    parser.add_argument("--retrain-every", type=int, default=1, help="Retrain interval in test days.")
    parser.add_argument("--include-records", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    matches = payload.get("matches") if isinstance(payload, dict) else payload
    if not isinstance(matches, list):
        raise ValueError("Input JSON must be a match list or contain a 'matches' list.")
    report = walk_forward_portfolio_backtest(
        matches,
        initial_train=args.initial_train,
        retrain_every=args.retrain_every,
        include_records=args.include_records,
    )
    rendered = json.dumps(report, indent=2, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
