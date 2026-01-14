#!/bin/bash
# Load .env variables and export PGPASSWORD for psql

# Use python-dotenv to read the password from .env
# This ensures we handle potential formatting issues in .env correctly
export PGPASSWORD=$(python3 -c "import dotenv, os; dotenv.load_dotenv(); print(os.getenv('DB_PASSWORD', ''))")

if [ -z "$PGPASSWORD" ]; then
    echo "Error: DB_PASSWORD not found in .env"
    exit 1
fi

echo "PGPASSWORD exported successfully."

# Execute verification command
echo "Verifying connection..."
psql -h 127.0.0.1 -p 5434 -U admin -d bsopt -c '\l'
