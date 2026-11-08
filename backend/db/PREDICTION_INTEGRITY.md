# Prediction integrity and ticket versions

Apply `migrations/20261108_prediction_integrity.sql` after the atomic storage migration and before deploying the matching backend patch.

## Guarantees

- Every successful publish appends a `prediction_ticket_versions` row with a UUID `run_id`, version number, timestamp, full ticket text, and complete selection JSON.
- A corrective rerun is accepted only while every incoming and existing selection is strictly before kickoff.
- A rerun marks the prior version `superseded` and creates a new `current` version. Prediction content is not erased.
- A ticket with missing legacy kickoff data cannot be rewritten.
- A ticket with any settled outcome or score cannot be rewritten.
- Summary and live selections still change atomically; failed storage suppresses the production email.
- Anonymous and authenticated browser roles have no access to the private version table. Only the trusted service role and database owner can read/write it.

## Performance semantics

Only the `current` pre-kickoff version is the official live ticket for normal result checking and performance reporting. Superseded versions remain available for audit but are not additional official bets. This prevents duplicate counting while preserving correction history.

The version row stores the prediction as published. Live outcomes and scores continue to be settled in `ticket_selections`; they do not rewrite the original prediction JSON.

## Legacy limitation

The migration archives the currently visible ticket for each existing date as `legacy_unverified`. It cannot reconstruct selections that earlier reruns overwrote or prove original publication/kickoff timestamps. In particular, it does not repair the already contaminated November 7 audit trail.

## Verification

Run `smoke_tests/20261108_prediction_integrity.sql` in Supabase SQL Editor after the migration. It tests complete replacement, immutable versioning, superseding, NO BET, settled-ticket rejection, and post-kickoff rejection on future test dates, then rolls everything back.

Do not manually rerun a real matchday merely to test the lock. Legacy matchdays are intentionally protected from rewrites because their original kickoff timestamps are unavailable.
