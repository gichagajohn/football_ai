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
    outcome text default 'pending'            -- pending | win | loss | void
);

create index if not exists idx_tickets_date on prediction_tickets(ticket_date);
create index if not exists idx_selections_date on ticket_selections(ticket_date);

-- ============================================================
-- Row Level Security (RLS)
-- ============================================================
-- These tables are written to using the service_role key (from GitHub
-- Actions), which bypasses RLS by default. If you also want to read
-- this data from a public dashboard later, enable RLS + add read-only
-- policies for the anon key. For now, RLS stays off for simplicity.
