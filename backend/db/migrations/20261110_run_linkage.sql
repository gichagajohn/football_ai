-- Deterministic run linkage: one opaque run_id per pipeline invocation,
-- stamped on the ticket it publishes and on every prediction log row it
-- writes. No existing row, ticket or version-history entry is modified.
--
-- model_prediction_logs previously carried a unique constraint on
-- (ticket_date, fixture_id, market, model_version) which existed only so
-- the worker could upsert diagnostics on reruns — silently erasing the
-- failed run's evidence. The worker no longer upserts: it appends every
-- run's rows and deduplicates at read time (latest created_at per
-- ticket_date/fixture_id/market), so reruns preserve history. The
-- constraint must therefore be dropped for plain inserts to coexist.
-- Dropping it removes no data.
--
-- Deployment note: if the old pre-patch worker runs between applying this
-- migration and pushing the new code, its diagnostics upsert fails
-- gracefully (it logs the error and continues); ticket storage, settlement,
-- selections and the immutable prediction_ticket_versions history are not
-- affected. ticket_selections' unique index is untouched and the publish
-- RPC is unchanged.

begin;

alter table public.model_prediction_logs
    drop constraint if exists model_prediction_logs_unique;

alter table public.model_prediction_logs
    add column if not exists run_id uuid;

alter table public.prediction_tickets
    add column if not exists run_id uuid;

create index if not exists idx_model_logs_run_id
    on public.model_prediction_logs (run_id);

create index if not exists idx_prediction_tickets_run_id
    on public.prediction_tickets (run_id);

comment on column public.model_prediction_logs.run_id is
    'Opaque id of the pipeline invocation that wrote this row; joins to prediction_tickets.run_id for the same run. Null on pre-2026-11-10 history.';
comment on column public.prediction_tickets.run_id is
    'Opaque id of the pipeline invocation that stored this ticket; joins to model_prediction_logs.run_id. Null on pre-2026-11-10 history.';

alter table public.model_prediction_logs enable row level security;
alter table public.prediction_tickets enable row level security;

commit;
