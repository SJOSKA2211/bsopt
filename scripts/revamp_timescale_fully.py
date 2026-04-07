from sqlalchemy import text

from src.database import db_manager


def revamp_timescale():
    db_manager.initialize()
    engine = db_manager.engine # Use sync engine
    
    with engine.connect() as conn:
        # 1. Convert model_predictions to hypertable
        print("Revamping model_predictions...")
        try:
            # Drop native partition stuff if exists
            conn.execute(text("DROP TABLE IF EXISTS model_predictions_default CASCADE;"))
            conn.execute(text("DROP TABLE IF EXISTS model_predictions CASCADE;"))
            
            # Recreate as hypertable
            conn.execute(text("""
                CREATE TABLE model_predictions (
                    timestamp TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    id UUID NOT NULL DEFAULT gen_random_uuid(),
                    model_id UUID,
                    symbol TEXT NOT NULL,
                    input_features JSONB NOT NULL,
                    predicted_price NUMERIC(12, 4) NOT NULL,
                    actual_price NUMERIC(12, 4),
                    prediction_error NUMERIC(12, 4),
                    actual_value NUMERIC,
                    PRIMARY KEY (timestamp, id)
                );
            """))
            conn.execute(text("SELECT create_hypertable('model_predictions', 'timestamp', if_not_exists => TRUE);"))
            conn.execute(text("SELECT set_chunk_time_interval('model_predictions', INTERVAL '1 day');"))
            conn.execute(text("ALTER TABLE model_predictions SET (timescaledb.compress, timescaledb.compress_segmentby = 'symbol', timescaledb.compress_orderby = 'timestamp DESC');"))
            conn.execute(text("SELECT add_compression_policy('model_predictions', INTERVAL '7 days', if_not_exists => TRUE);"))
            print("✅ model_predictions is now a compressed hypertable.")
        except Exception as e:
            print(f"❌ Failed to revamp model_predictions: {e}")

        # 2. Ensure other tables exist and are hypertables
        tables = [
            ("rl_episodes", "created_at"),
            ("calibration_results", "created_at")
        ]
        
        for table, time_col in tables:
            print(f"Checking {table}...")
            try:
                # Check if table exists
                res = conn.execute(text(f"SELECT count(*) FROM pg_tables WHERE tablename = '{table}'"))
                if res.scalar() == 0:
                    print(f"Creating {table}...")
                    if table == "rl_episodes":
                        conn.execute(text(f"""
                            CREATE TABLE {table} (
                                id UUID DEFAULT gen_random_uuid(),
                                agent_id TEXT NOT NULL,
                                episode_reward DOUBLE PRECISION NOT NULL,
                                steps INTEGER NOT NULL,
                                hyperparameters JSONB,
                                {time_col} TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                PRIMARY KEY ({time_col}, id)
                            );
                        """))
                    elif table == "calibration_results":
                        conn.execute(text(f"""
                            CREATE TABLE {table} (
                                id UUID DEFAULT gen_random_uuid(),
                                symbol TEXT NOT NULL,
                                v0 DOUBLE PRECISION NOT NULL,
                                kappa DOUBLE PRECISION NOT NULL,
                                theta DOUBLE PRECISION NOT NULL,
                                sigma DOUBLE PRECISION NOT NULL,
                                rho DOUBLE PRECISION NOT NULL,
                                rmse DOUBLE PRECISION NOT NULL,
                                r_squared DOUBLE PRECISION NOT NULL,
                                num_options INTEGER NOT NULL,
                                svi_params JSONB,
                                {time_col} TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                                PRIMARY KEY ({time_col}, id)
                            );
                        """))
                
                conn.execute(text(f"SELECT create_hypertable('{table}', '{time_col}', if_not_exists => TRUE);"))
                conn.execute(text(f"ALTER TABLE {table} SET (timescaledb.compress, timescaledb.compress_orderby = '{time_col} DESC');"))
                conn.execute(text(f"SELECT add_compression_policy('{table}', INTERVAL '30 days', if_not_exists => TRUE);"))
                print(f"✅ {table} is now a hypertable.")
            except Exception as e:
                print(f"❌ Failed to optimize {table}: {e}")

        # 3. Create health view for TimescaleDB
        print("Creating timescale_health_overview view...")
        try:
            conn.execute(text("""
                CREATE OR REPLACE VIEW timescale_health_overview AS
                SELECT
                    h.hypertable_name,
                    h.num_chunks,
                    h.compression_enabled,
                    pg_size_pretty(COALESCE(s.before_compression_total_bytes, 0)) as uncompressed_size,
                    pg_size_pretty(COALESCE(s.after_compression_total_bytes, 0)) as compressed_size,
                    CASE 
                        WHEN s.before_compression_total_bytes > 0 
                        THEN round(100.0 * (s.before_compression_total_bytes - s.after_compression_total_bytes) / s.before_compression_total_bytes, 2)
                        ELSE 0 
                    END as compression_ratio_pct
                FROM timescaledb_information.hypertables h
                LEFT JOIN LATERAL hypertable_compression_stats(h.hypertable_name::regclass) s ON TRUE;
            """))
            print("✅ timescale_health_overview view created.")
        except Exception as e:
            print(f"❌ Failed to create timescale health view: {e}")
            
        conn.commit()

if __name__ == "__main__":
    revamp_timescale()
