-- Private conservative-shadow diagnostics and settlement fields.
-- Does not change official tickets, selections, probabilities, or calibration.

begin;

alter table public.model_prediction_logs
    add column if not exists kickoff_utc timestamptz,
    add column if not exists strategy text not null default 'production',
    add column if not exists shadow_selected boolean not null default false,
    add column if not exists robust_model_prob double precision,
    add column if not exists robust_expected_value double precision,
    add column if not exists closing_bookmaker_odds double precision,
    add column if not exists outcome text not null default 'pending',
    add column if not exists home_score smallint,
    add column if not exists away_score smallint;

create index if not exists idx_model_logs_shadow_settlement
    on public.model_prediction_logs (ticket_date, strategy, shadow_selected, outcome);

comment on column public.model_prediction_logs.strategy is
    'production or a pre-registered shadow strategy name; shadow rows remain private.';
comment on column public.model_prediction_logs.shadow_selected is
    'True only for selections in the shadow portfolio; selected remains false to exclude them from official calibration.';
comment on column public.model_prediction_logs.closing_bookmaker_odds is
    'Optional genuine closing quote used only for CLV reporting; null means unavailable, never zero CLV.';

alter table public.model_prediction_logs enable row level security;

commit;
