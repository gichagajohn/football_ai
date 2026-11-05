from backend.agents.pipeline_agents import run_publisher


def test_long_portfolio_is_never_truncated_by_llm_output_limit():
    selections = [
        {
            "home_team": f"Home {index}", "away_team": f"Away {index}",
            "league": "League", "market": "home_win", "odds": 2.0,
            "rationale": "Test",
        }
        for index in range(15)
    ]
    ticket = run_publisher(
        {"selections": selections, "combined_odds": 2.0 ** 15, "risk_level": "High"},
        {"decision": "PUBLISH", "final_confidence": 0.001},
        {},
        "Thursday, 05 November 2026",
    )
    assert ticket.count("MATCH:") == 15
    assert "Home 14" in ticket
    assert "formatted deterministically" in ticket
