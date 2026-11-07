-- ============================================================
-- FOOTBALL PULSE AI — Supabase Schema (GitHub Actions edition)
-- ============================================================
-- Run this in the Supabase SQL Editor (Project -> SQL Editor -> New Query)
-- before the first GitHub Actions run.

create table if not exists prediction_tickets (
    id bigint generated always as identity primary key,
    ticket_date date not null unique,
    status text not null default 'pending',   -- pending | published | no_bet
    combined_odds numeric,
    selection_count smallint,
    final_confidence numeric,
    risk_level text,
    reason text,
    ticket_text text,
    outcome text default 'pending',           -- pending | win | loss | void
    created_at timestamptz default now()
);

create table if not exists ticket_selections (
    id bigint generated always as identity primary key,
    ticket_date date not null references prediction_tickets(ticket_date),
    fixture_id bigint not null,
    home_team text,
    away_team text,
    league text,
    market text,
    odds numeric,
    rationale text,
    outcome text default 'pending',           -- pending | win | loss | void
    home_score smallint,
    away_score smallint,
    constraint ticket_selections_unique_pick unique (ticket_date, fixture_id, market)
);

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

create index if not exists idx_tickets_date on prediction_tickets(ticket_date);
create index if not exists idx_selections_date on ticket_selections(ticket_date);
create index if not exists idx_model_logs_date on model_prediction_logs(ticket_date);
create index if not exists idx_model_logs_selected on model_prediction_logs(selected, ticket_date);

alter table model_prediction_logs enable row level security;
-- No anon/authenticated policy: detailed features and rejected candidates are
-- private. GitHub Actions uses the service_role key, which bypasses RLS.

-- For databases created with an older version of this schema, run the
-- migrations in backend/db/migrations/. The 20261104 migration adds result
-- storage/deduplication. The 20261107 migration installs the service-role-only
-- store_prediction_ticket RPC used for atomic whole-snapshot publishing.

-- ============================================================
-- Row Level Security (RLS)
-- ============================================================
-- Ticket/selection tables retain their existing public-view setup. The new
-- model_prediction_logs table above has RLS enabled and no public policy.
-- All writes use the service_role key from GitHub Actions, which bypasses RLS.
