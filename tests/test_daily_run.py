import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend import daily_run
from backend.db.supabase_client import TicketStorageError


class DailyRunDeliveryTests(unittest.TestCase):
    def test_missing_supabase_configuration_suppresses_pipeline_and_email(self):
        with (
            patch.object(daily_run.supabase_client, "storage_configured", return_value=False),
            patch.object(daily_run, "run_pipeline", new=AsyncMock()) as pipeline,
            patch.object(daily_run, "send_email") as send,
            patch.object(daily_run.result_checker, "run") as results,
        ):
            with self.assertRaises(RuntimeError):
                asyncio.run(daily_run.main())
        pipeline.assert_not_called()
        send.assert_not_called()
        results.assert_not_called()

    def test_storage_failure_suppresses_email(self):
        with (
            patch.object(daily_run.supabase_client, "storage_configured", return_value=True),
            patch.object(
                daily_run,
                "run_pipeline",
                new=AsyncMock(side_effect=TicketStorageError("storage failed")),
            ),
            patch.object(daily_run, "send_email") as send,
            patch.object(daily_run.result_checker, "run"),
        ):
            with self.assertRaises(TicketStorageError):
                asyncio.run(daily_run.main())
        send.assert_not_called()

    def test_email_is_sent_after_successful_pipeline_storage(self):
        with (
            patch.object(daily_run.supabase_client, "storage_configured", return_value=True),
            patch.object(daily_run, "run_pipeline", new=AsyncMock(return_value="Stored ticket")) as pipeline,
            patch.object(daily_run, "send_email") as send,
            patch.object(daily_run.result_checker, "run"),
        ):
            asyncio.run(daily_run.main())
        pipeline.assert_awaited_once()
        send.assert_called_once()
        self.assertEqual(send.call_args.args[1], "Stored ticket")


if __name__ == "__main__":
    unittest.main()
