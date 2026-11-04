-- Repair result storage and make ticket selections idempotent.
-- Safe to rerun: score columns and the unique index are created IF NOT EXISTS,
-- and the duplicate deletion affects only rows ranked after the first copy.

begin;

alter table public.ticket_selections
    add column if not exists home_score smallint,
    add column if not exists away_score smallint;

-- Keep one row per logical selection. Prefer a graded/scored copy if an old
-- database contains a mixture, then prefer the lowest id. At the time this
-- migration was authored all duplicate rows were pending and had no scores.
with ranked_selections as (
    select
        id,
        row_number() over (
            partition by ticket_date, fixture_id, market
            order by
                (coalesce(outcome, 'pending') <> 'pending') desc,
                (home_score is not null and away_score is not null) desc,
                id asc
        ) as copy_number
    from public.ticket_selections
)
delete from public.ticket_selections
where id in (select id from ranked_selections where copy_number > 1);

-- PostgREST uses this index for on_conflict=ticket_date,fixture_id,market.
create unique index if not exists ticket_selections_unique_pick_idx
    on public.ticket_selections (ticket_date, fixture_id, market);

commit;
