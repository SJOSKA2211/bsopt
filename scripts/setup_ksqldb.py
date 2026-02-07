import json
import os
import sys

import requests


def run_ksql_query(url, ksql_text):
    print("Executing ksqlDB query...")
    print(f"Query: {ksql_text}")
    
    payload = {
        "ksql": ksql_text,
        "streamsProperties": {
            "ksql.streams.auto.offset.reset": "earliest"
        }
    }
    
    try:
        response = requests.post(
            f"{url}/ksql",
            json=payload,
            headers={"Content-Type": "application/vnd.ksql.v1+json"}
        )
        
        if response.status_code == 200:
            print("Successfully executed query.")
            print(f"Response: {json.dumps(response.json(), indent=2)}")
            return True
        else:
            print(f"Failed to execute query. Status code: {response.status_code}")
            print(f"Response: {response.text}")
            return False
    except Exception as e:
        print(f"Error connecting to ksqlDB: {e}")
        return False

if __name__ == "__main__":
    ksqldb_url = os.environ.get("KSQLDB_URL", "http://localhost:8088")
    
    # 1. Create Stream from market-data topic (Optimized with Protobuf)
    create_stream_query = """
    CREATE STREAM IF NOT EXISTS market_data_stream (
        symbol VARCHAR KEY,
        timestamp BIGINT,
        bid DOUBLE,
        ask DOUBLE,
        last DOUBLE,
        volume BIGINT,
        open_interest BIGINT,
        implied_volatility DOUBLE,
        delta DOUBLE,
        gamma DOUBLE,
        vega DOUBLE,
        theta DOUBLE
    ) WITH (
        KAFKA_TOPIC='market-data',
        VALUE_FORMAT='PROTOBUF',
        PARTITIONS=8
    );
    """
    
    # 2. Create Persistent Table for High IV Options (Optimized with 1-minute Tumbling Window)
    # Reduces downstream noise by aggregating metrics per minute per symbol
    create_high_iv_query = """
    CREATE TABLE IF NOT EXISTS high_iv_options_agg AS
    SELECT 
        symbol,
        WINDOWSTART AS window_start,
        AVG(implied_volatility) AS avg_iv,
        MAX(implied_volatility) AS max_iv,
        AVG(delta) AS avg_delta,
        COUNT(*) AS tick_count
    FROM market_data_stream 
    WINDOW TUMBLING (SIZE 1 MINUTE)
    WHERE implied_volatility > 0.5
    GROUP BY symbol;
    """
    
    success = True
    if not run_ksql_query(ksqldb_url, create_stream_query):
        success = False
        
    if success and not run_ksql_query(ksqldb_url, create_high_iv_query):
        success = False
        
    if success:
        print("\nksqlDB streams and queries setup SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("\nksqlDB setup FAILED.")
        sys.exit(1)
