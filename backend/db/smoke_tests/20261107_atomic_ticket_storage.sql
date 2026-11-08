-- Optional live verification for 20261107_atomic_ticket_storage.sql.
-- Run as a trusted database owner in Supabase SQL Editor after the migration.
-- Everything is rolled back. No email or workflow is triggered.

begin;

do $$
declare
    v_date date := '2099-01-07';
begin
    if exists (select 1 from public.prediction_tickets where ticket_date = v_date) then
        raise exception 'Smoke-test date % already exists; choose an unused future date', v_date;
    end if;

    -- Initial snapshot: three legs.
    perform public.store_prediction_ticket(
        v_date,
        'published',
        43.03,
        0.094,
        'High',
        'Smoke test initial snapshot',
        'Initial three-leg ticket',
        '[
          {"fixture_id":900001,"home_team":"A","away_team":"B","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"away_win","odds":2.60,"rationale":"initial A"},
          {"fixture_id":900002,"home_team":"C","away_team":"D","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"home_win","odds":1.65,"rationale":"retained C"},
          {"fixture_id":900003,"home_team":"E","away_team":"F","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"over25","odds":1.52,"rationale":"obsolete E"}
        ]'::jsonb
    );

    if (select count(*) from public.ticket_selections where ticket_date = v_date) <> 3 then
        raise exception 'Initial snapshot did not store exactly three selections';
    end if;

    -- Rerun: retain one leg with a new price/rationale, remove two old legs,
    -- and add one new leg. Summary and selections must change together.
    perform public.store_prediction_ticket(
        v_date,
        'published',
        3.55,
        0.107,
        'High',
        'Smoke test replacement snapshot',
        'Replacement two-leg ticket',
        '[
          {"fixture_id":900002,"home_team":"C","away_team":"D","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"home_win","odds":1.86,"rationale":"updated C"},
          {"fixture_id":900004,"home_team":"G","away_team":"H","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"draw_no_bet_away","odds":1.365,"rationale":"new G"}
        ]'::jsonb
    );

    if (select count(*) from public.ticket_selections where ticket_date = v_date) <> 2 then
        raise exception 'Replacement left stale selections or missed new selections';
    end if;

    if not exists (
        select 1 from public.ticket_selections
        where ticket_date = v_date and fixture_id = 900002
          and odds = 1.86 and rationale = 'updated C'
    ) or not exists (
        select 1 from public.ticket_selections
        where ticket_date = v_date and fixture_id = 900004
    ) then
        raise exception 'Retained/new selection data was not refreshed correctly';
    end if;

    if not exists (
        select 1 from public.prediction_tickets
        where ticket_date = v_date and combined_odds = 3.55
          and selection_count = 2 and ticket_text = 'Replacement two-leg ticket'
    ) then
        raise exception 'Summary and selection snapshot were not updated together';
    end if;

    -- A published-to-NO-BET rerun must clear every old selection.
    perform public.store_prediction_ticket(
        v_date, 'no_bet', null, null, null,
        'Smoke test no bet', 'NO BET', '[]'::jsonb
    );

    if exists (select 1 from public.ticket_selections where ticket_date = v_date)
       or not exists (
           select 1 from public.prediction_tickets
           where ticket_date = v_date and status = 'no_bet' and selection_count = 0
       ) then
        raise exception 'NO BET did not clear the previous selection snapshot';
    end if;

    -- Once settled, the publishing RPC must reject a rewrite.
    perform public.store_prediction_ticket(
        v_date,
        'published',
        2.0,
        0.5,
        'Low',
        'Smoke test settlement setup',
        'One-leg ticket',
        '[{"fixture_id":900005,"home_team":"I","away_team":"J","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"home_win","odds":2.0,"rationale":"settle"}]'::jsonb
    );
    update public.ticket_selections
    set outcome = 'win', home_score = 1, away_score = 0
    where ticket_date = v_date and fixture_id = 900005;

    begin
        perform public.store_prediction_ticket(
            v_date,
            'published',
            2.5,
            0.6,
            'Low',
            'Must be rejected',
            'Forbidden rewrite',
            '[{"fixture_id":900006,"home_team":"K","away_team":"L","league":"Test","kickoff_utc":"2099-01-07T12:00:00Z","market":"away_win","odds":2.5,"rationale":"reject"}]'::jsonb
        );
        raise exception 'Settled-ticket rewrite was incorrectly allowed';
    exception
        when sqlstate '55000' then
            null; -- Expected protection error.
    end;

    raise notice 'Atomic ticket storage smoke test passed; rolling back all test rows.';
end;
$$;

rollback;
