import unittest
from datetime import date
from unittest.mock import Mock, patch

from backend import result_checker


class FetchResultTests(unittest.TestCase):
    def test_reads_finished_match_from_top_level_response(self):
        response = Mock()
        response.json.return_value = {
            "id": 330299,
            "status": "FINISHED",
            "score": {"fullTime": {"home": 1, "away": 2}},
        }

        with (
            patch.object(result_checker, "FOOTBALL_DATA_KEY", "test-key"),
            patch.object(result_checker, "_fd_get", return_value=response),
        ):
            result = result_checker.fetch_result(330299)

        self.assertEqual(
            result,
            {"home_goals": 1, "away_goals": 2, "status": "FINISHED"},
        )

    def test_leaves_unfinished_match_pending(self):
        response = Mock()
        response.json.return_value = {
            "id": 330299,
            "status": "TIMED",
            "score": {"fullTime": {"home": None, "away": None}},
        }

        with (
            patch.object(result_checker, "FOOTBALL_DATA_KEY", "test-key"),
            patch.object(result_checker, "_fd_get", return_value=response),
        ):
            self.assertIsNone(result_checker.fetch_result(330299))


class ResultRunTests(unittest.TestCase):
    def _selection(self, selection_id):
        return {
            "id": selection_id,
            "fixture_id": 12345,
            "home_team": "Home FC",
            "away_team": "Away FC",
            "market": "home_win",
            "outcome": "pending",
        }

    def test_fetches_duplicate_fixture_once_and_updates_each_row(self):
        with (
            patch.object(
                result_checker.supabase_client,
                "get_pending_tickets",
                return_value=[{"ticket_date": "2026-11-03"}],
            ),
            patch.object(
                result_checker.supabase_client,
                "get_selections_for_date",
                return_value=[self._selection(1), self._selection(2)],
            ),
            patch.object(
                result_checker,
                "fetch_result",
                return_value={"home_goals": 2, "away_goals": 0, "status": "FINISHED"},
            ) as fetch,
            patch.object(
                result_checker.supabase_client,
                "update_selection_outcome",
                return_value=True,
            ) as update_selection,
            patch.object(
                result_checker.supabase_client,
                "update_ticket_outcome",
                return_value=True,
            ) as update_ticket,
        ):
            result_checker.run(today=date(2026, 11, 4))

        fetch.assert_called_once_with(12345)
        self.assertEqual(update_selection.call_count, 2)
        update_ticket.assert_called_once_with(date(2026, 11, 3), "win")

    def test_failed_selection_write_does_not_complete_ticket(self):
        with (
            patch.object(
                result_checker.supabase_client,
                "get_pending_tickets",
                return_value=[{"ticket_date": "2026-11-03"}],
            ),
            patch.object(
                result_checker.supabase_client,
                "get_selections_for_date",
                return_value=[self._selection(1)],
            ),
            patch.object(
                result_checker,
                "fetch_result",
                return_value={"home_goals": 2, "away_goals": 0, "status": "FINISHED"},
            ),
            patch.object(
                result_checker.supabase_client,
                "update_selection_outcome",
                return_value=False,
            ),
            patch.object(
                result_checker.supabase_client,
                "update_ticket_outcome",
            ) as update_ticket,
        ):
            result_checker.run(today=date(2026, 11, 4))

        update_ticket.assert_not_called()


if __name__ == "__main__":
    unittest.main()
