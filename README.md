# Football Pulse AI — GitHub Actions Edition

Runs on GitHub Actions, emails a daily ticket (or **NO BET**), stores selections
in Supabase, settles finished matches, and produces performance reports.

## Prediction model

The numerical prediction core is deterministic; Gemini does not invent match
probabilities or add accumulator legs.

1. **Historical fit:** one Dixon–Coles model per competition, using finished
   football-data.org matches. Leagues use the current season plus previous
   history when needed. Domestic cups use three seasons; UEFA tournaments add
   previous-season history. Brazil uses a calendar-year season.
2. **Team strengths:** time-weighted attack/defence ratings, home advantage,
   L2 regularization, and the Dixon–Coles low-score correlation adjustment.
3. **Score probabilities:** a normalized 0–10 score matrix supplies 1X2,
   double-chance, draw-no-bet, BTTS, and totals probabilities.
4. **Market baseline:** complete mutually exclusive bookmaker prices have their
   overround removed. Model probabilities are shrunk toward that baseline
   (default market weight 35%). Raw `1 / decimal_odds` is retained for EV.
5. **Value gate:** `EV = model_probability × decimal_odds - 1`. Missing prices,
   probabilities below 50%, negative EV, insufficient team history, and unknown
   team IDs are rejected.
6. **Unlimited flexible portfolio:** singles are valid and there is **no fixed
   maximum selection count or combined-odds ceiling**. Every priceable fixture
   can be considered, with one market per fixture. A bounded-memory beam search
   limits computation—not legs or odds—and chooses a risk-adjusted portfolio.
   It never pads a ticket to reach a target. Empty qualifying sets return
   **NO BET**.

Direct 1X2 and over-2.5 prices come from The Odds API's featured bulk markets
(`h2h,totals`), with `h2h` as a fallback when the provider rejects a bundled
market request. BTTS probabilities are still computed by Dixon–Coles, and the
parser can accept BTTS prices if supplied, but BTTS is not requested by default:
The Odds API classifies it as an additional market requiring per-event requests.
Double-chance and draw-no-bet prices are reproducibly derived from quoted 1X2
prices and are labeled `derived_from_bookmaker_1x2`. Joint probability assumes
independence between fixtures and is an estimate, not a guarantee. Longer
tickets normally have much lower joint win probability even when every leg
passes the EV gate.

`FORCE_PUBLISH: "true"` bypasses only Gemini Auditor/Decision warnings. It
cannot bypass history, model metadata, probability, EV, unique-fixture, finite
odds, or configured-limit validation.

Gemini is used for injury/weather context, risk review, auditing, and ticket
wording. Provider fixture IDs, names, team IDs, kickoff times, and odds always
overwrite LLM output. Portfolios over 12 legs are formatted deterministically
so no selection is truncated; this is a formatting threshold, not a bet limit.

## Competitions and fixture coverage

The default configuration uses the 10 competitions confirmed accessible on the
current football-data free plan:

```text
PL   Premier League                 ELC  English Championship
PD   La Liga                        BL1  Bundesliga
SA   Serie A                        FL1  Ligue 1
DED  Eredivisie                     PPL  Primeira Liga
BSA  Brazil Série A                 CL   UEFA Champions League
```

These six remain defined but are disabled by default after their fixture feeds
returned plan-related `403` responses on 2026-11-05: `FAC` (FA Cup), `CDR`
(Copa del Rey), `DFB` (DFB-Pokal), `CIT` (Coppa Italia), `EL` (UEFA Europa
League), and `UCL` (UEFA Conference League). After provider coverage changes,
re-enable them through `LEAGUE_IDS`. The override replaces the default list, so
include every desired competition code—not only the newly enabled ones.

Every returned fixture with a matched bookmaker event is processed; the old
12/18-fixture stopping rules are removed. Unpriced fixtures are skipped early
because they cannot pass EV validation. Provider competition/odds coverage
varies by plan and season; failed feeds are logged and skipped safely. The
Scout queries The Odds API sport catalog and resolves provider-specific key
aliases dynamically, with configured keys as fallback. Cross-provider team
matching strips diacritics/punctuation, applies explicit aliases, and removes
only non-identifying whole club tokens; it does not use broad fuzzy matching.
If a fixture remains unmatched, the log lists its canonical names/kickoff and
the available provider event names/kickoffs for diagnosis.

Use a comma-separated `LEAGUE_IDS` override to run a subset when quota is
limited. `SCOUT_HARD_CAP_MATCHES=0` means no fixture cap; a positive value is
available only as an emergency quota control.

### Home-city and weather coverage

The Scout uses a manual club-city map, including current/recent Brazilian Série A
clubs, and then falls back to registered club addresses from football-data's
competition team profiles. The latter costs at most one additional paced request
per competition containing an unmapped, priceable home team and is cached for
the run. Missing or ambiguous addresses skip weather rather than inventing a
location. The selected city and its source are logged.

Weather can add useful heavy-rain, wind, and temperature context, but it is not
automatically evidence of predictive improvement. It remains Scout/risk-review
context and does not alter Dixon–Coles probabilities. A weather adjustment to
the numerical model would require separate holdout/backtest evidence. Neutral
or relocated matches can differ from a club's usual home city.

## Learning from results

There are two feedback loops, not an unsupported claim that an LLM trains
itself from its own text:

1. **Team ratings learn automatically.** Each daily Dixon–Coles fit includes
   newly finished matches, with time decay emphasizing recent performance.
2. **Probability calibration is guarded.** Settled published outputs are joined
   to their actual win/loss outcomes. A recency-weighted logistic calibrator
   trains on raw model probabilities, but activates only after at least 200
   settled outputs and 30 examples per outcome class by default. It must also
   beat the uncalibrated model on a later chronological holdout. Otherwise it
   stays off. Raw and calibrated probabilities are both logged to prevent
   repeatedly calibrating already-calibrated values.

This design reduces self-reinforcement and small-sample overfitting. It does
not guarantee monotonic improvement, future profit, or a universally “best”
model. Gemini itself is not fine-tuned or allowed to rewrite the mathematics.

## Setup

### 1. Supabase

Create a Supabase project and run `backend/db/schema.sql` in SQL Editor. For an
existing installation, preserve the earlier result repair and then apply:

```text
backend/db/migrations/20261104_poisson_prediction_logs.sql
backend/db/migrations/20261107_atomic_ticket_storage.sql
backend/db/migrations/20261108_prediction_integrity.sql
```

The first adds/upgrades private model diagnostics without changing existing
tickets, selections, or settled scores. The second installs a service-role-only
RPC that stores each summary and its complete selection snapshot atomically,
removes stale rerun legs, refreshes retained odds, and rejects settled-ticket
rewrites. The 20261108 migration archives current legacy predictions, adds
private immutable ticket versions, and rejects publication at/after kickoff.
Apply each migration before deploying backend code that depends on it. Optional
rollback-only verification is in `backend/db/smoke_tests/`. Keep service-role
keys secret.

### 2. GitHub Secrets

Configure these under **Settings → Secrets and variables → Actions**:

| Secret | Purpose |
|---|---|
| `GEMINI_API_KEY` | Risk commentary, audit, and ticket wording |
| `FOOTBALL_DATA_KEY` | Fixtures, history, standings, form, and results |
| `ODDS_API_KEY` | Bookmaker prices |
| `OPENWEATHER_KEY` | Kickoff-time weather forecast |
| `SUPABASE_URL` | Supabase project URL |
| `SUPABASE_KEY` | Service-role key for backend writes |
| `SMTP_HOST`, `SMTP_PORT` | Usually `smtp.gmail.com`, `587` |
| `SMTP_USERNAME`, `SMTP_PASSWORD` | Gmail address and app password |
| `EMAIL_TO` | Ticket/report recipient |

### 3. Run

The daily workflow runs at 07:00 EAT (04:00 UTC), or manually through
**Actions → Football Pulse AI — Daily Ticket → Run workflow**. More competitions,
all-fixture processing, historical fitting, and provider pacing can make runs
substantially longer. The timeout is 120 minutes.

Important workflow settings:

```text
MIN_SELECTIONS=1
MAX_SELECTIONS=0                 # zero = no fixed maximum
COMBINED_ODDS_MIN=1.00
COMBINED_ODDS_MAX=0.00           # zero = no fixed maximum
SCOUT_HARD_CAP_MATCHES=0         # zero = every priceable fixture
SCOUT_BATCH_SIZE=12              # processing batch only, not a total limit
PORTFOLIO_BEAM_WIDTH=200         # computational memory only
MIN_EDGE_MARGIN=0.00
MIN_SELECTION_PROBABILITY=0.50
POISSON_MARKET_WEIGHT=0.35
POISSON_TIME_DECAY_DAYS=240
MIN_TEAM_HISTORY=8               # tournament floor is 4 with multi-season data
CALIBRATION_MIN_SAMPLES=200
CALIBRATION_HALF_LIFE_DAYS=240
FORCE_PUBLISH=true
```

## Evaluation

Every evaluated candidate can be logged with model/calibration versions, raw
and calibrated model probabilities, market probabilities, fair odds,
quoted/derived price, EV, eligibility, rejection reason, selection status, and
feature snapshots.

After selections have settled, generate a calibration report locally with the
Supabase environment variables set:

```bat
py -m backend.model_report
```

It reports Brier score, log loss, hit rate, flat-stake ROI, and probability
calibration buckets. `backend.models.backtesting.walk_forward_backtest` also
supports chronological train/test evaluation without fitting on future
matches. Historical odds are optional for calibration but required for
backtested EV/ROI. Small samples and a few large winners can dominate ROI;
none of these statistics guarantees future profit.

## Tests

Windows CMD from the repository root:

```bat
py -m pip install -r requirements.txt
py -m pytest -q
```

GitHub Actions also runs the regression suite on pushes and pull requests.
Tests cover the Dixon–Coles fit/matrix, margin removal, analyst integration,
singles and portfolios above six legs/20 odds, negative-EV rejection,
forced-publish hard gates, 16-competition defaults, cup history, guarded
learning, raw/calibrated logging, Scout odds extraction, walk-forward
evaluation, calibration, and existing result-storage/deduplication behavior.

## Main files

```text
.github/workflows/
  daily.yml                 Daily model/ticket workflow
  weekly.yml                Result settlement and weekly performance report
  result_backfill.yml       Result-only historical repair
  tests.yml                 Regression tests

backend/
  agents/
    scout_agent.py          Multi-competition fixtures, prices, form, weather
    pipeline_agents.py      Deterministic analyst/portfolio + LLM risk/publishing
  models/
    dixon_coles.py          Time-weighted Dixon–Coles fit and score matrix
    markets.py              Margin removal, blending, EV
    poisson_service.py      Multi-season historical acquisition and model fits
    learning.py             Holdout-gated probability self-calibration
    backtesting.py          Walk-forward evaluation
    evaluation.py           Brier/log-loss/ROI metrics
  db/
    supabase_client.py      Ticket, result, and private model-log storage
    schema.sql              Full new-install schema
    migrations/             Apply-ready upgrades for existing databases
  pipeline.py               Orchestration
  result_checker.py         Existing score/result settlement (preserved)
  model_report.py           Settled calibration report
  daily_run.py              Daily entrypoint
```

## Schedules and quotas

- Daily ticket: 07:00 EAT; result and weekly workflows retain their schedules
  and can also be dispatched manually.
- football-data.org and The Odds API quotas vary by plan. The Scout caches
  standings/form within a run and the shared football-data client paces and
  retries fixture/history requests. Cups can require three history requests;
  leagues generally require one or two.
- Gemini model order can be overridden with comma-separated `GEMINI_MODELS`.
- GitHub-hosted runner usage depends on repository visibility and account plan.

**Betting involves risk. Probabilistic output, positive estimated EV, learning
from results, and past performance do not guarantee a win or profit.**
