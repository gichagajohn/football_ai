import os
import time
import logging
import httpx
from supabase import create_client, Client

logger = logging.getLogger("backend.result_checker")

FOOTBALL_DATA_KEY = os.getenv("FOOTBALL_DATA_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")

def get_supabase() -> Client:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Supabase URL or Key is missing from environment variables.")
    return create_client(SUPABASE_URL, SUPABASE_KEY)

def evaluate_selection(bet_type: str, home_score: int, away_score: int) -> str:
    """Evaluates whether a selection won ('WON'), lost ('LOST'), or is pending."""
    if home_score is None or away_score is None:
        return "PENDING"
        
    total_goals = home_score + away_score
    bet = bet_type.lower().replace(" ", "_")
    
    if bet in ["home_win", "1"]:
        return "WON" if home_score > away_score else "LOST"
    elif bet in ["away_win", "2"]:
        return "WON" if away_score > home_score else "LOST"
    elif bet in ["draw", "x"]:
        return "WON" if home_score == away_score else "LOST"
    elif bet == "btts_yes":
        return "WON" if (home_score > 0 and away_score > 0) else "LOST"
    elif bet == "btts_no":
        return "WON" if (home_score == 0 or away_score == 0) else "LOST"
    elif bet == "over_2.5":
        return "WON" if total_goals > 2.5 else "LOST"
    elif bet == "under_2.5":
        return "WON" if total_goals < 2.5 else "LOST"
    
    return "PENDING"

def check_pending_tickets():
    logger.info("[RESULT_CHECK] Starting result verification using football-data.org...")
    supabase = get_supabase()

    try:
        response = supabase.table("ticket_selections").select("*").eq("status", "PENDING").execute()
        selections = response.data

        if not selections:
            logger.info("[RESULT_CHECK] No pending selections found.")
            return

        headers = {"X-Auth-Token": FOOTBALL_DATA_KEY}
        client = httpx.Client(headers=headers, timeout=10.0)

        for sel in selections:
            fixture_id = sel.get("fixture_id") or sel.get("match_id")
            if not fixture_id:
                continue

            url = f"https://api.football-data.org/v4/matches/{fixture_id}"
            try:
                res = client.get(url)
                if res.status_code == 429:
                    logger.warning("[RESULT_CHECK] Rate limited by football-data.org. Sleeping...")
                    time.sleep(10)
                    res = client.get(url)

                if res.status_code != 200:
                    logger.error(f"[RESULT_CHECK] Failed to fetch fixture {fixture_id}: HTTP {res.status_code}")
                    continue

                match_data = res.json()
                status = match_data.get("status")

                if status in ["FINISHED", "AET", "PEN"]:
                    score_info = match_data.get("score", {}).get("fullTime", {})
                    home_score = score_info.get("home")
                    away_score = score_info.get("away")

                    if home_score is not None and away_score is not None:
                        bet_type = sel.get("bet_type") or sel.get("selection")
                        outcome = evaluate_selection(bet_type, home_score, away_score)

                        supabase.table("ticket_selections").update({
                            "status": outcome,
                            "home_score": home_score,
                            "away_score": away_score
                        }).eq("id", sel["id"]).execute()

                        logger.info(f"[RESULT_CHECK] Updated selection {sel['id']} for match {fixture_id}: {outcome} ({home_score}-{away_score})")

                # Respect free-tier rate limits (10 requests per minute)
                time.sleep(6)

            except Exception as e:
                logger.error(f"[RESULT_CHECK] Error processing fixture {fixture_id}: {e}")

    except Exception as e:
        logger.error(f"[RESULT_CHECK] Database error during result check: {e}")

if __name__ == "__main__":
    check_pending_tickets()
