import unittest
from datetime import date
from unittest.mock import Mock, patch

from backend.db import supabase_client


class SupabaseStorageTests(unittest.TestCase):
    def test_rerun_preserves_outcomes_and_ignores_duplicate_selections(self):
        ticket_response = Mock()
        ticket_response.raise_for_status.return_value = None
        selections_response = Mock()
        selections_response.raise_for_status.return_value = None

        portfolio = {
            "combined_odds": 2.0,
            "risk_level": "Low",
            "selections": [{
                "fixture_id": 12345,
                "home_team": "Home FC",
                "away_team": "Away FC",
                "league": "Premier League",
                "market": "home_win",
                "odds": 2.0,
                "rationale": "Test selection",
            }],
        }

        with (
            patch.object(supabase_client, "_enabled", return_value=True),
            patch.object(
                supabase_client.httpx,
                "post",
                side_effect=[ticket_response, selections_response],
            ) as post,
        ):
            supabase_client.store_ticket(
                date(2026, 11, 4),
                "Ticket text",
                portfolio,
                {"decision": "PUBLISH", "final_confidence": 0.8},
            )

        ticket_call, selections_call = post.call_args_list
        self.assertNotIn("outcome", ticket_call.kwargs["json"])

        self.assertEqual(
            selections_call.kwargs["params"],
            {"on_conflict": "ticket_date,fixture_id,market"},
        )
        self.assertEqual(
            selections_call.kwargs["headers"]["Prefer"],
            "resolution=ignore-duplicates, return=minimal",
        )

    def test_selection_update_returns_true_only_on_success(self):
        response = Mock()
        response.raise_for_status.return_value = None
        with (
            patch.object(supabase_client, "_enabled", return_value=True),
            patch.object(supabase_client.httpx, "patch", return_value=response),
        ):
            updated = supabase_client.update_selection_outcome(
                1, "win", home_score=2, away_score=1
            )
        self.assertTrue(updated)


if __name__ == "__main__":
    unittest.main()
