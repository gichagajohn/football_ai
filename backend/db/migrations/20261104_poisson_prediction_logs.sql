-- Dixon-Coles model diagnostics and future calibration/backtesting support.
-- Safe to run more than once. No existing ticket, selection or result is changed.

create table if not exists model_prediction_logs (
    id bigint generated always as identity primary key,
    ticket_date date not null,
    fixture_id bigint not null,
    home_team text,
    away_team text,
    league text,
    market text not null,
    model_version text not null,
    calibration_version text,
    raw_model_prob double precision,
    model_prob double precision,
    market_implied_prob double precision,
    bookmaker_implied_prob double precision,
    fair_odds double precision,
    bookmaker_odds double precision,
    odds_source text,
    expected_value double precision,
    eligible boolean not null default false,
    selected boolean not null default false,
    rejection_reason text,
    feature_snapshot jsonb not null default '{}'::jsonb,
    created_at timestamptz not null default now(),
    constraint model_prediction_logs_unique unique (ticket_date, fixture_id, market, model_version)
);

create index if not exists idx_model_logs_date
    on model_prediction_logs(ticket_date);
create index if not exists idx_model_logs_selected
    on model_prediction_logs(selected, ticket_date);

alter table model_prediction_logs enable row level security;
-- Deliberately no public SELECT policy. The service_role writer bypasses RLS.

-- Upgrade installations that applied an earlier version of this migration.
alter table model_prediction_logs
    add column if not exists calibration_version text;
alter table model_prediction_logs
    add column if not exists raw_model_prob double precision;
alter table model_prediction_logs
    add column if not exists odds_source text;
