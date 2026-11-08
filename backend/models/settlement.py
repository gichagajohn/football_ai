"""Shared full-time settlement rules for production and historical evaluation."""

from __future__ import annotations


def grade_market(market: str, home_goals: int, away_goals: int) -> str:
    """Return win/loss/void; DNB draws are stake-returning void pushes."""
    home_goals, away_goals = int(home_goals), int(away_goals)
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
            return "void"
        return "win" if home_goals > away_goals else "loss"
    if market == "draw_no_bet_away":
        if home_goals == away_goals:
            return "void"
        return "win" if away_goals > home_goals else "loss"
    if market == "btts_yes":
        return "win" if home_goals > 0 and away_goals > 0 else "loss"
    if market == "over25":
        return "win" if total_goals > 2.5 else "loss"
    return "void"
