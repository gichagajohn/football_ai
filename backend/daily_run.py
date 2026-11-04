import os
import logging
import asyncio
from datetime import date
from backend.pipeline import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("backend.daily_runner")

def main():
    logger.info("Initializing daily football prediction runner...")
    
    # Verify critical environment variables before starting
    required_vars = ["SUPABASE_URL", "SUPABASE_KEY", "FOOTBALL_DATA_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        logger.error(f"Missing critical environment variables: {', '.join(missing)}")
        raise EnvironmentError(f"Missing environment variables: {missing}")

    try:
        # Run your async pipeline
        ticket = asyncio.run(run_pipeline())
        logger.info("Daily pipeline executed successfully.")
        return ticket
    except Exception as e:
        logger.exception(f"Pipeline execution failed: {e}")
        raise e

if __name__ == "__main__":
    main()
