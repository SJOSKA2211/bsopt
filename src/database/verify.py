import os
import subprocess

from dotenv import load_dotenv


def verify_postgres_connection():
    # 1. Load .env variables
    load_dotenv()

    # 2. Get DB_PASSWORD from .env
    db_password = os.getenv("DB_PASSWORD")
    if not db_password:
        print("Error: DB_PASSWORD not found in .env")
        return

    # 3. Set PGPASSWORD env var (Exporting for this process and its children)
    os.environ["PGPASSWORD"] = db_password

    # Connection details (based on docker-compose.yml mappings)
    db_user = "admin"
    db_host = "127.0.0.1"
    db_port = "5434"
    db_name = "bsopt"

    # 4. Execute psql -c '\l'
    cmd = ["psql", "-h", db_host, "-p", db_port, "-U", db_user, "-d", db_name, "-c", "\\l"]

    print(f"Running: PGPASSWORD=[HIDDEN] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print("\n--- Connection Verified ---")
        print(result.stdout)
    except subprocess.CalledProcessError as e:
        print("\n--- Connection Failed ---")
        print(f"Exit Code: {e.returncode}")
        print(f"Error:\n{e.stderr}")
        if "Connection refused" in e.stderr:
            print("\nNote: Ensure the PostgreSQL container is running.")
            print("Run: 'docker-compose up -d postgres'")


if __name__ == "__main__":
    verify_postgres_connection()
