import logging

import psycopg2
import requests


def test_database_connectivity_on_startup():
    # Configure logging
    logger = logging.getLogger("test_database_connectivity")
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(formatter)
        logger.addHandler(ch)

    # The service endpoint health check URL
    base_url = "http://localhost:4000"
    health_url = f"{base_url}/"

    try:
        # Step 1: Check service readiness via health endpoint
        health_response = requests.get(health_url, timeout=30)
        assert (
            health_response.status_code == 200
        ), f"Health check failed with status code {health_response.status_code}; service might not be ready."
    except (requests.RequestException, AssertionError) as e:
        logger.error(f"Service health check failed or service not ready: {e}")
        assert False, "Service is not healthy or not running."

    # Step 2: Verify DB connection by making a direct query to PostgreSQL
    # Assume standard PostgreSQL connection env variables or defaults; adjust as needed.

    # Typical connection params (these would ideally be configured or discovered)
    pg_config = {
        "host": "localhost",
        "port": 5432,
        "dbname": "postgres",
        "user": "postgres",
        "password": "postgres",
    }

    try:
        conn = psycopg2.connect(
            host=pg_config["host"],
            port=pg_config["port"],
            dbname=pg_config["dbname"],
            user=pg_config["user"],
            password=pg_config["password"],
            connect_timeout=10,
        )
        cur = conn.cursor()
        cur.execute("SELECT 1;")
        result = cur.fetchone()
        assert result == (1,), f"Unexpected query result: {result}"
        cur.close()
        conn.close()
    except Exception as e:
        logger.error(f"Database connectivity or query failed: {e}")
        # Additional check: if DB fails, service should not mark itself ready
        # Cross-check health again - it should fail or status != 200 to confirm behavior.
        try:
            health_response = requests.get(health_url, timeout=30)
            if health_response.status_code == 200:
                logger.error(
                    "Service returned status 200 despite DB connection failure."
                )
                assert False, "Service marked ready despite DB connectivity failure."
        except requests.RequestException:
            pass
        assert False, "Database connectivity or simple query failed."


test_database_connectivity_on_startup()
