-- Preserve immutable ticket versions and block publishing at/after kickoff.
-- Builds on the atomic whole-snapshot storage migration.

begin;

alter table public.ticket_selections
    add column if not exists kickoff_utc timestamptz;

create index if not exists idx_selections_kickoff
    on public.ticket_selections (kickoff_utc);

-- Prediction content is retained when a pre-kickoff correction supersedes it.
create table if not exists public.prediction_ticket_versions (
    run_id uuid primary key default gen_random_uuid(),
    ticket_date date not null references public.prediction_tickets(ticket_date) on delete restrict,
    version_number integer not null,
    version_status text not null check (version_status in ('current', 'superseded')),
    integrity_status text not null check (integrity_status in ('official', 'legacy_unverified', 'manual_review')),
    status text not null,
    combined_odds numeric,
    selection_count smallint,
    final_confidence numeric,
    risk_level text,
    reason text,
    ticket_text text not null,
    selections jsonb not null,
    published_at timestamptz not null default now(),
    superseded_at timestamptz,
    note text,
    unique (ticket_date, version_number)
);

create unique index if not exists prediction_ticket_one_current_idx
    on public.prediction_ticket_versions (ticket_date)
    where version_status = 'current';
create index if not exists prediction_ticket_versions_date_idx
    on public.prediction_ticket_versions (ticket_date, version_number desc);

alter table public.prediction_ticket_versions enable row level security;
revoke all on table public.prediction_ticket_versions from public;
revoke all on table public.prediction_ticket_versions from anon;
revoke all on table public.prediction_ticket_versions from authenticated;
grant all on table public.prediction_ticket_versions to service_role;

-- Preserve the current legacy record. Earlier overwritten versions cannot be
-- reconstructed by this migration.
insert into public.prediction_ticket_versions (
    ticket_date, version_number, version_status, integrity_status,
    status, combined_odds, selection_count, final_confidence, risk_level,
    reason, ticket_text, selections, published_at, note
)
select
    t.ticket_date,
    1,
    'current',
    'legacy_unverified',
    t.status,
    t.combined_odds,
    t.selection_count,
    t.final_confidence,
    t.risk_level,
    t.reason,
    t.ticket_text,
    coalesce(
        jsonb_agg(
            jsonb_build_object(
                'fixture_id', s.fixture_id,
                'home_team', s.home_team,
                'away_team', s.away_team,
                'league', s.league,
                'kickoff_utc', s.kickoff_utc,
                'market', s.market,
                'odds', s.odds,
                'rationale', s.rationale,
                'outcome', s.outcome,
                'home_score', s.home_score,
                'away_score', s.away_score
            ) order by s.id
        ) filter (where s.id is not null),
        '[]'::jsonb
    ),
    coalesce(t.created_at, now()),
    'Legacy snapshot; earlier overwritten versions and original kickoff timestamps may be incomplete.'
from public.prediction_tickets t
left join public.ticket_selections s on s.ticket_date = t.ticket_date
where not exists (
    select 1 from public.prediction_ticket_versions v
    where v.ticket_date = t.ticket_date
)
group by t.id;

create or replace function public.store_prediction_ticket(
    p_ticket_date date,
    p_status text,
    p_combined_odds numeric,
    p_final_confidence numeric,
    p_risk_level text,
    p_reason text,
    p_ticket_text text,
    p_selections jsonb
)
returns void
language plpgsql
security definer
set search_path = ''
as $$
declare
    v_selection_count integer;
    v_selection jsonb;
    v_fixture_id bigint;
    v_market text;
    v_kickoff timestamptz;
    v_version_number integer;
begin
    if p_ticket_date is null then
        raise exception 'ticket_date is required' using errcode = '23502';
    end if;

    if p_status not in ('published', 'no_bet') then
        raise exception 'Invalid ticket status: %', p_status using errcode = '23514';
    end if;

    if p_ticket_text is null or pg_catalog.btrim(p_ticket_text) = '' then
        raise exception 'ticket_text is required' using errcode = '23502';
    end if;

    if p_selections is null or pg_catalog.jsonb_typeof(p_selections) <> 'array' then
        raise exception 'p_selections must be a JSON array' using errcode = '23514';
    end if;

    v_selection_count := pg_catalog.jsonb_array_length(p_selections);

    if p_status = 'published' and v_selection_count = 0 then
        raise exception 'A published ticket must contain at least one selection'
            using errcode = '23514';
    end if;

    if p_status = 'no_bet' and v_selection_count <> 0 then
        raise exception 'A no_bet ticket cannot contain selections'
            using errcode = '23514';
    end if;

    -- Validate the complete incoming snapshot before locking or changing rows.
    for v_selection in
        select value from pg_catalog.jsonb_array_elements(p_selections)
    loop
        if pg_catalog.jsonb_typeof(v_selection) <> 'object' then
            raise exception 'Every selection must be a JSON object' using errcode = '23514';
        end if;

        v_fixture_id := (v_selection ->> 'fixture_id')::bigint;
        v_market := nullif(pg_catalog.btrim(v_selection ->> 'market'), '');
        v_kickoff := (v_selection ->> 'kickoff_utc')::timestamptz;

        if v_fixture_id is null or v_fixture_id <= 0 or v_market is null then
            raise exception 'Every selection needs a positive fixture_id and market'
                using errcode = '23502';
        end if;

        if nullif(pg_catalog.btrim(v_selection ->> 'home_team'), '') is null
           or nullif(pg_catalog.btrim(v_selection ->> 'away_team'), '') is null
           or nullif(pg_catalog.btrim(v_selection ->> 'league'), '') is null then
            raise exception 'Selection teams and league are required' using errcode = '23502';
        end if;

        if (v_selection ->> 'odds')::numeric is null
           or (v_selection ->> 'odds')::numeric <= 1 then
            raise exception 'Selection odds must be greater than 1 for fixture %', v_fixture_id
                using errcode = '23514';
        end if;

        if v_kickoff is null then
            raise exception 'Selection kickoff_utc is required for fixture %', v_fixture_id
                using errcode = '23502';
        end if;

        if v_kickoff <= now() then
            raise exception 'Fixture % has already kicked off; prediction publication is locked', v_fixture_id
                using errcode = '55000';
        end if;
    end loop;

    if exists (
        select 1
        from (
            select (value ->> 'fixture_id')::bigint as fixture_id
            from pg_catalog.jsonb_array_elements(p_selections)
        ) incoming
        group by incoming.fixture_id
        having count(*) > 1
    ) then
        raise exception 'Each fixture may appear only once in a ticket' using errcode = '23514';
    end if;

    -- Serialize publishers for this date even outside GitHub Actions. The lock
    -- is held until the surrounding RPC transaction commits or rolls back.
    perform pg_catalog.pg_advisory_xact_lock(
        pg_catalog.hashtextextended('football-pulse-ticket:' || p_ticket_date::text, 0)
    );

    -- Lock existing rows against a concurrent result update before checking
    -- whether this ticket has become part of the settled historical record.
    perform 1
    from public.prediction_tickets t
    where t.ticket_date = p_ticket_date
    for update;

    perform 1
    from public.ticket_selections s
    where s.ticket_date = p_ticket_date
    for update;

    if exists (
        select 1 from public.prediction_tickets t
        where t.ticket_date = p_ticket_date
          and coalesce(t.outcome, 'pending') <> 'pending'
    ) or exists (
        select 1 from public.ticket_selections s
        where s.ticket_date = p_ticket_date
          and (
              coalesce(s.outcome, 'pending') <> 'pending'
              or s.home_score is not null
              or s.away_score is not null
          )
    ) then
        raise exception 'Ticket % already has settled results and cannot be rewritten', p_ticket_date
            using errcode = '55000';
    end if;

    -- Legacy rows have no reliable kickoff evidence and cannot be rewritten
    -- safely. New rows must all remain strictly before kickoff.
    if exists (
        select 1 from public.ticket_selections s
        where s.ticket_date = p_ticket_date
          and (s.kickoff_utc is null or s.kickoff_utc <= now())
    ) then
        raise exception 'Ticket % has missing or started kickoffs and cannot be rewritten', p_ticket_date
            using errcode = '55000';
    end if;

    update public.prediction_ticket_versions v
    set version_status = 'superseded', superseded_at = now()
    where v.ticket_date = p_ticket_date and v.version_status = 'current';

    select coalesce(max(v.version_number), 0) + 1
    into v_version_number
    from public.prediction_ticket_versions v
    where v.ticket_date = p_ticket_date;

    insert into public.prediction_tickets (
        ticket_date,
        status,
        combined_odds,
        selection_count,
        final_confidence,
        risk_level,
        reason,
        ticket_text
    ) values (
        p_ticket_date,
        p_status,
        case when p_status = 'published' then p_combined_odds else null end,
        v_selection_count,
        p_final_confidence,
        case when p_status = 'published' then p_risk_level else null end,
        p_reason,
        p_ticket_text
    )
    on conflict (ticket_date) do update set
        status = excluded.status,
        combined_odds = excluded.combined_odds,
        selection_count = excluded.selection_count,
        final_confidence = excluded.final_confidence,
        risk_level = excluded.risk_level,
        reason = excluded.reason,
        ticket_text = excluded.ticket_text;
        -- outcome is intentionally untouched.

    if p_status = 'no_bet' then
        -- A rerun that becomes NO BET must not leave published selections behind.
        delete from public.ticket_selections s
        where s.ticket_date = p_ticket_date;
    else
        -- Remove selections that are absent from the new complete snapshot.
        delete from public.ticket_selections s
        where s.ticket_date = p_ticket_date
          and not exists (
              select 1
              from pg_catalog.jsonb_array_elements(p_selections) incoming
              where (incoming.value ->> 'fixture_id')::bigint = s.fixture_id
                and incoming.value ->> 'market' = s.market
          );

        -- Insert new legs and refresh retained teams/odds/rationales. Result
        -- columns are deliberately absent so retained outcomes/scores survive.
        insert into public.ticket_selections (
            ticket_date,
            fixture_id,
            home_team,
            away_team,
            league,
            kickoff_utc,
            market,
            odds,
            rationale
        )
        select
            p_ticket_date,
            (incoming.value ->> 'fixture_id')::bigint,
            incoming.value ->> 'home_team',
            incoming.value ->> 'away_team',
            incoming.value ->> 'league',
            (incoming.value ->> 'kickoff_utc')::timestamptz,
            incoming.value ->> 'market',
            (incoming.value ->> 'odds')::numeric,
            incoming.value ->> 'rationale'
        from pg_catalog.jsonb_array_elements(p_selections) incoming
        on conflict (ticket_date, fixture_id, market) do update set
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            league = excluded.league,
            kickoff_utc = excluded.kickoff_utc,
            odds = excluded.odds,
            rationale = excluded.rationale;
            -- outcome/home_score/away_score are intentionally untouched.
    end if;

    -- Defence in depth: any inconsistency aborts the entire transaction.
    if (select count(*) from public.ticket_selections s where s.ticket_date = p_ticket_date)
       <> v_selection_count then
        raise exception 'Stored selection count does not match the incoming snapshot for %', p_ticket_date
            using errcode = '23514';
    end if;

    insert into public.prediction_ticket_versions (
        ticket_date, version_number, version_status, integrity_status,
        status, combined_odds, selection_count, final_confidence, risk_level,
        reason, ticket_text, selections, published_at
    ) values (
        p_ticket_date,
        v_version_number,
        'current',
        'official',
        p_status,
        case when p_status = 'published' then p_combined_odds else null end,
        v_selection_count,
        p_final_confidence,
        case when p_status = 'published' then p_risk_level else null end,
        p_reason,
        p_ticket_text,
        p_selections,
        now()
    );
end;
$$;

-- Only the backend's service-role credentials may publish or replace tickets.
revoke all on function public.store_prediction_ticket(
    date, text, numeric, numeric, text, text, text, jsonb
) from public;
revoke all on function public.store_prediction_ticket(
    date, text, numeric, numeric, text, text, text, jsonb
) from anon;
revoke all on function public.store_prediction_ticket(
    date, text, numeric, numeric, text, text, text, jsonb
) from authenticated;
grant execute on function public.store_prediction_ticket(
    date, text, numeric, numeric, text, text, text, jsonb
) to service_role;

comment on function public.store_prediction_ticket(
    date, text, numeric, numeric, text, text, text, jsonb
) is 'Atomically stores a versioned pre-kickoff ticket snapshot; rejects started or settled rewrites.';

comment on table public.prediction_ticket_versions is
    'Service-role-only immutable prediction snapshots; superseded versions remain available for audit.';

commit;
