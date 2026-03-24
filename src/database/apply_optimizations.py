import os
import subprocess
from urllib.parse import urlparse

import structlog
from dotenv import load_dotenv

logger = structlog.get_logger()

def apply_database_optimizations() -> None:
    """
    High-Performance Optimization Wrapper:
    Applies the full-spectrum optimized init-scripts to an existing database.
    """
    load_dotenv()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("database_url_missing")
        return

    url = urlparse(database_url)
    db_user = url.username
    db_password = url.password
    db_host = url.hostname
    db_port = url.port or "5432"
    db_name = url.path[1:]

    if db_password:
        os.environ["PGPASSWORD"] = db_password

    # The new source of truth for tight optimizations
    optimization_scripts = [
        "init-scripts/05-indexes.sql",
        "init-scripts/06-compression-retention.sql",
        "init-scripts/07-continuous-aggregates.sql",
        "init-scripts/08-materialized-views.sql",
        "init-scripts/09-security.sql",
        "init-scripts/10-missing-tables.sql",
        "init-scripts/11-scheduled-jobs.sql",
        "init-scripts/12-performance-dashboard.sql",
        "init-scripts/13-pg16-diagnostics.sql",
    ]

    for script in optimization_scripts:
        if not os.path.exists(script):
            logger.warning("optimization_script_missing", path=script)
            continue

        cmd = [
            "psql",
            "-h",
            str(db_host),
            "-p",
            str(db_port),
            "-U",
            str(db_user),
            "-d",
            str(db_name),
            "-f",
            script,
        ]

        logger.info("applying_optimization_phase", script=script)

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)
        except subprocess.CalledProcessError as e:
            # We ignore 'already exists' errors as scripts are designed to be idempotent
            if "already exists" not in e.stderr:
                logger.error("optimization_phase_failed", script=script, error=e.stderr)

    logger.info("all_optimizations_pressurized", status="_tight")

if __name__ == "__main__":
    apply_database_optimizations()
