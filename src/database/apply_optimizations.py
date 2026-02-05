import os
import subprocess # nosec B404
from urllib.parse import urlparse

import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()

def apply_database_optimizations():
    """
    Applies the optimized TimescaleDB schema settings from src/database/optimized_schema.sql.
    Includes compression, continuous aggregates, and retention policies.
    """
    load_dotenv()
    
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("database_url_missing", message="DATABASE_URL not found in .env")
        return

    url = urlparse(database_url)
    db_user = url.username
    db_password = url.password or os.getenv("DB_PASSWORD")
    db_host = url.hostname
    db_port = url.port or "5432"
    db_name = url.path[1:]

    if db_password:
        os.environ["PGPASSWORD"] = db_password

    schema_file = "src/database/optimized_schema.sql"
    if not os.path.exists(schema_file):
        logger.error("schema_file_missing", path=schema_file)
        return

    cmd = [
        "psql", 
        "-h", str(db_host), 
        "-p", str(db_port), 
        "-U", str(db_user), 
        "-d", str(db_name), 
        "-f", schema_file
    ]

    logger.info("applying_optimizations", host=db_host, database=db_name)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True) # nosec B603
        logger.info("optimizations_applied_successfully", output=result.stdout[:500])
    except subprocess.CalledProcessError as e:
        logger.error("optimizations_failed", error=e.stderr)
        if "already exists" in e.stderr:
            logger.info("some_optimizations_already_present")

if __name__ == "__main__":
    apply_database_optimizations()
