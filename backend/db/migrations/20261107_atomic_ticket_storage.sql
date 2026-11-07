-- Atomically replace a daily ticket and its complete selection snapshot.
-- Prevents reruns from combining new summary metadata with stale selections.

begin;

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
            incoming.value ->> 'market',
            (incoming.value ->> 'odds')::numeric,
            incoming.value ->> 'rationale'
        from pg_catalog.jsonb_array_elements(p_selections) incoming
        on conflict (ticket_date, fixture_id, market) do update set
            home_team = excluded.home_team,
            away_team = excluded.away_team,
            league = excluded.league,
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
) is 'Atomically stores one complete pending ticket snapshot; rejects settled-ticket rewrites.';

commit;
