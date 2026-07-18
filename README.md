# Football Pulse AI — GitHub Actions Edition

Runs entirely on GitHub's free infrastructure. No server, no Docker, no
laptop needed. Sends you a daily prediction ticket via email, tracks
outcomes, and emails a weekly performance report.

---

## What this does

- **Daily (08:00 EAT)**: Scout pulls fixtures from the top 5 European
  leagues + Champions League, runs the full 9-agent pipeline, and emails
  you the resulting ticket (or "NO BET TODAY").
- **Weekly (Sunday 09:00 EAT)**: Checks results of past published
  tickets, calculates hit rate / ROI, and emails a performance report.

---

## Setup — Step by Step

### 1. Create a Supabase project (free)

1. Go to **https://supabase.com** → Sign up → New Project
2. Wait for the project to finish provisioning (~2 min)
3. Go to **SQL Editor** → New Query → paste the contents of
   `backend/db/schema.sql` → Run
4. Go to **Project Settings → API**:
   - Copy **Project URL** → this is `SUPABASE_URL`
   - Copy **service_role key** (NOT the `anon` key) → this is `SUPABASE_KEY`

   ⚠️ The service_role key has full write access — never expose it
   publicly. It only goes into GitHub Secrets (encrypted).

### 2. Set up Gmail App Password (for email)

1. Enable 2-Step Verification on your Google account if not already on
2. Go to **https://myaccount.google.com/apppasswords**
3. Create an app password for "Mail" → copy the 16-character password

This becomes `SMTP_PASSWORD`. Your normal Gmail password will NOT work
with SMTP — you need this app password.

### 3. Push this code to GitHub

```bash
git init
git add .
git commit -m "Football Pulse AI - GitHub Actions edition"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/football-pulse-ai.git
git push -u origin main
```

### 4. Add GitHub Secrets

In your repo: **Settings → Secrets and variables → Actions → New repository secret**

Add each of these:

| Secret name | Value |
|---|---|
| `GROQ_API_KEY` | your Groq API key |
| `API_FOOTBALL_KEY` | your API-Football key |
| `OPENWEATHER_KEY` | your OpenWeatherMap key |
| `SUPABASE_URL` | from Supabase Project Settings → API |
| `SUPABASE_KEY` | service_role key from Supabase |
| `SMTP_HOST` | `smtp.gmail.com` |
| `SMTP_PORT` | `587` |
| `SMTP_USERNAME` | your Gmail address |
| `SMTP_PASSWORD` | the 16-character app password from step 2 |
| `EMAIL_TO` | where you want tickets sent (can be same as SMTP_USERNAME) |

### 5. Test it manually

1. Go to your repo's **Actions** tab
2. Click **"Football Pulse AI — Daily Ticket"** workflow
3. Click **"Run workflow"** → Run workflow
4. Watch the logs — should take 1-3 minutes
5. Check your email for the result

---

## Schedule

- Daily ticket: every day at **08:00 EAT** (05:00 UTC)
- Weekly report: every **Sunday at 09:00 EAT** (06:00 UTC)

Both can also be triggered manually from the Actions tab
(`workflow_dispatch`).

---

## Notes on quotas

- **Groq free tier**: 100,000 tokens/day. One daily run (~10-15 matches
  through Scout, fewer through the rest of the pipeline) should comfortably
  fit within this for a single run.
- **API-Football free tier**: 100 requests/day. ~15 matches × 3 calls
  (odds + 2 injuries) + 6 fixture-list calls (one per league) ≈ 51 calls/day.
- **GitHub Actions free tier**: 2,000 minutes/month for private repos
  (unlimited for public repos). Each run takes ~2-5 minutes, so this is
  far from a concern (daily + weekly ≈ 35 runs/month).

---

## Files

```
.github/workflows/
  daily.yml          — daily ticket workflow (cron + manual trigger)
  weekly.yml         — weekly report workflow

backend/
  agents/
    scout_agent.py       — fixture/odds/injury collection (top-5 leagues)
    pipeline_agents.py   — Analyst/Risk/Portfolio/Auditor/Decision/Publisher
  db/
    supabase_client.py   — outcome storage via Supabase REST API
    schema.sql            — run this in Supabase SQL editor first
  pipeline.py          — orchestrates all 9 agents
  result_checker.py    — grades past tickets against actual results
  weekly_report.py     — generates hit rate / ROI summary
  email_sender.py      — SMTP email delivery
  daily_run.py         — entrypoint for daily workflow
  weekly_run.py        — entrypoint for weekly workflow

requirements.txt
```
