import os
import subprocess
from urllib.parse import urlparse # Import for parsing DATABASE_URL

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

    from urllib.parse import urlparse # Import for parsing DATABASE_URL

    # 4. Get DATABASE_URL from .env
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        print("Error: DATABASE_URL not found in .env")
        return
        
    # Parse DATABASE_URL
    url = urlparse(database_url)
    db_user = url.username
    db_password_url = url.password # Use this if PGPASSWORD is not set globally
    db_host = url.hostname
    db_port = url.port if url.port else "5432" # Default PostgreSQL port
    db_name = url.path[1:] # Remove leading '/'

    # Set PGPASSWORD from parsed URL or existing DB_PASSWORD
    if db_password_url:
        os.environ["PGPASSWORD"] = db_password_url
    elif db_password: # Fallback to DB_PASSWORD from .env
        os.environ["PGPASSWORD"] = db_password
    else:
        print("Error: PostgreSQL password not found in DATABASE_URL or DB_PASSWORD.")
        return

    # 5. Execute psql -c '\l'
    cmd = ["psql", "-h", str(db_host), "-p", str(db_port), "-U", str(db_user), "-d", str(db_name), "-c", "\\l"]

    print(f"Running: PGPASSWORD=[HIDDEN] {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)  # nosec
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
