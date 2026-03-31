import msgspec
from src.ml.aiops.schemas import MLHealthReport, MLflowStatus, PrometheusMetrics, RedisAnomaly, RabbitMQStatus

def verify_schema():
    report = MLHealthReport(
        status="healthy",
        mlflow=MLflowStatus(stage="prod", drift_detected=False),
        prometheus=PrometheusMetrics(error_rate_5xx=0.01, p95_latency=0.1, cpu_usage=0.2, memory_usage=0.5),
        redis_anomalies=[],
        rabbitmq=RabbitMQStatus(
            connected=True,
            queue_depths={"market_ticks": 10},
            consumer_counts={"market_ticks": 2}
        ),
        timestamp="2026-03-31T17:30:00Z"
    )
    
    encoded = msgspec.json.encode(report)
    print(f"Encoded Report: {encoded.decode()}")
    
    decoded = msgspec.json.decode(encoded, type=MLHealthReport)
    assert decoded.rabbitmq.connected is True
    assert decoded.rabbitmq.queue_depths["market_ticks"] == 10
    print("✅ Schema Verification Successful")

if __name__ == "__main__":
    try:
        verify_schema()
    except Exception as e:
        print(f"❌ Schema Verification Failed: {str(e)}")
        exit(1)
