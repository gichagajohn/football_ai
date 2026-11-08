import unittest
from datetime import date
from pathlib import Path
from unittest.mock import Mock, patch

from backend.db import supabase_client


ROOT = Path(__file__).resolve().parents[1]


def selection(fixture_id=12345, market="home_win", odds=2.0):
    return {
        "fixture_id": fixture_id,
        "home_team": "Home FC",
        "away_team": "Away FC",
        "league": "Premier League",
        "kickoff_utc": "2099-01-07T12:00:00Z",
        "market": market,
        "odds": odds,
        "rationale": "Test selection",
    }


class SupabaseStorageTests(unittest.TestCase):
    def test_atomic_publish_uses_one_rpc_with_the_complete_snapshot(self):
        response = Mock()
        response.raise_for_status.return_value = None
        portfolio = {
            "combined_odds": 4.2,
            "risk_level": "Medium",
            "selections": [selection(), selection(67890, "away_win", 1.8)],
            "candidate_evaluations": [{"fixture_id": 12345, "market": "home_win"}],
        }
        decision = {"decision": "PUBLISH", "final_confidence": 0.42, "reason": "Validated"}

        with (
            patch.object(supabase_client, "_enabled", return_value=True),
            patch.object(supabase_client.httpx, "post", return_value=response) as post,
            patch.object(supabase_client, "_store_prediction_logs") as logs,
        ):
            stored = supabase_client.store_ticket(
                date(2026, 11, 7), "Complete ticket", portfolio, decision
            )

        self.assertTrue(stored)
        post.assert_called_once()
        call = post.call_args
        self.assertTrue(call.args[0].endswith("/rest/v1/rpc/store_prediction_ticket"))
        self.assertNotIn("params", call.kwargs)
        payload = call.kwargs["json"]
        self.assertEqual(payload["p_ticket_date"], "2026-11-07")
        self.assertEqual(payload["p_status"], "published")
        self.assertEqual(payload["p_combined_odds"], 4.2)
        self.assertEqual(payload["p_final_confidence"], 0.42)
        self.assertEqual(payload["p_ticket_text"], "Complete ticket")
        self.assertEqual(len(payload["p_selections"]), 2)
        self.assertEqual(payload["p_selections"][1]["odds"], 1.8)
        for row in payload["p_selections"]:
            self.assertEqual(row["kickoff_utc"], "2099-01-07T12:00:00Z")
            self.assertNotIn("ticket_date", row)
            self.assertNotIn("outcome", row)
            self.assertNotIn("home_score", row)
            self.assertNotIn("away_score", row)
        logs.assert_called_once()

    def test_no_bet_uses_the_same_atomic_rpc_with_an_empty_snapshot(self):
        response = Mock()
        response.raise_for_status.return_value = None
        portfolio = {
            "combined_odds": 99,
            "risk_level": "High",
            "selections": [selection()],
        }

        with (
            patch.object(supabase_client, "_enabled", return_value=True),
            patch.object(supabase_client.httpx, "post", return_value=response) as post,
            patch.object(supabase_client, "_store_prediction_logs"),
        ):
            stored = supabase_client.store_ticket(
                date(2026, 11, 7),
                "NO BET",
                portfolio,
                {"decision": "NO_BET", "reason": "No qualifying selections"},
            )

        self.assertTrue(stored)
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["p_status"], "no_bet")
        self.assertEqual(payload["p_selections"], [])
        self.assertIsNone(payload["p_combined_odds"])
        self.assertIsNone(payload["p_risk_level"])

    def test_storage_failure_raises_and_skips_diagnostics(self):
        response = Mock()
        response.raise_for_status.side_effect = RuntimeError("HTTP 500")

        with (
            patch.object(supabase_client, "_enabled", return_value=True),
            patch.object(supabase_client.httpx, "post", return_value=response),
            patch.object(supabase_client, "_store_prediction_logs") as logs,
        ):
            with self.assertRaises(supabase_client.TicketStorageError):
                supabase_client.store_ticket(
                    date(2026, 11, 7),
                    "Ticket",
                    {"selections": [selection()]},
                    {"decision": "PUBLISH"},
                )
        logs.assert_not_called()

    def test_unconfigured_local_dry_run_returns_false_without_http(self):
        with (
            patch.object(supabase_client, "_enabled", return_value=False),
            patch.object(supabase_client.httpx, "post") as post,
        ):
            stored = supabase_client.store_ticket(
                date(2026, 11, 7), "Ticket", {}, {"decision": "NO_BET"}
            )
        self.assertFalse(stored)
        post.assert_not_called()

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


class AtomicStorageMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = (ROOT / "backend/db/migrations/20261107_atomic_ticket_storage.sql").read_text()

    def test_migration_is_atomic_locked_and_service_role_only(self):
        self.assertIn("security definer", self.sql)
        self.assertIn("set search_path = ''", self.sql)
        self.assertIn("pg_catalog.pg_advisory_xact_lock", self.sql)
        self.assertIn(") to service_role;", self.sql)
        self.assertIn(") from anon;", self.sql)
        self.assertIn(") from authenticated;", self.sql)

    def test_migration_replaces_obsolete_legs_and_refreshes_retained_legs(self):
        self.assertIn("delete from public.ticket_selections", self.sql)
        self.assertIn("and not exists (", self.sql)
        self.assertIn("on conflict (ticket_date, fixture_id, market) do update set", self.sql)
        self.assertIn("odds = excluded.odds", self.sql)
        self.assertIn("rationale = excluded.rationale", self.sql)
        self.assertNotIn("outcome = excluded.outcome", self.sql)
        self.assertNotIn("home_score = excluded.home_score", self.sql)
        self.assertNotIn("away_score = excluded.away_score", self.sql)

    def test_migration_protects_settled_rows_and_checks_final_count(self):
        self.assertIn("already has settled results and cannot be rewritten", self.sql)
        self.assertIn("Stored selection count does not match", self.sql)
        self.assertIn("if p_status = 'no_bet' then", self.sql)
        self.assertIn("Each fixture may appear only once", self.sql)

    def test_daily_workflow_serialises_scheduled_and_manual_runs(self):
        workflow = (ROOT / ".github/workflows/daily.yml").read_text()
        self.assertIn("group: football-pulse-daily-production", workflow)
        self.assertIn("cancel-in-progress: false", workflow)


class PredictionIntegrityMigrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sql = (ROOT / "backend/db/migrations/20261108_prediction_integrity.sql").read_text()

    def test_versions_are_private_and_legacy_history_is_preserved(self):
        self.assertIn("create table if not exists public.prediction_ticket_versions", self.sql)
        self.assertIn("enable row level security", self.sql)
        self.assertIn("'legacy_unverified'", self.sql)
        self.assertIn("prediction_ticket_one_current_idx", self.sql)
        self.assertIn("from public.prediction_tickets t", self.sql)
        self.assertIn("revoke all on table public.prediction_ticket_versions from anon", self.sql)
        self.assertIn("revoke all on table public.prediction_ticket_versions from authenticated", self.sql)

    def test_started_and_legacy_selections_cannot_be_rewritten(self):
        self.assertIn("Selection kickoff_utc is required", self.sql)
        self.assertIn("if v_kickoff <= now() then", self.sql)
        self.assertIn("s.kickoff_utc is null or s.kickoff_utc <= now()", self.sql)
        self.assertIn("prediction publication is locked", self.sql)

    def test_pre_kickoff_reruns_supersede_and_append_versions(self):
        self.assertIn("set version_status = 'superseded', superseded_at = now()", self.sql)
        self.assertIn("insert into public.prediction_ticket_versions", self.sql)
        self.assertIn("'official'", self.sql)
        self.assertIn("kickoff_utc = excluded.kickoff_utc", self.sql)
        self.assertNotIn("run_id = excluded.run_id", self.sql)


if __name__ == "__main__":
    unittest.main()
