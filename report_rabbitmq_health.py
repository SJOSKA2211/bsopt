import asyncio
import sys
import os
from src.shared.rabbitmq import get_rabbitmq
from src.shared.utils.broker import broker
import structlog

# Mocking structlog for basic output if needed
structlog.configure(
    processors=[structlog.processors.JSONRenderer()]
)

async def report_health():
    print("--- RabbitMQ Health Report ---")
    rmq = get_rabbitmq()
    masked_url = rmq.url.replace(rmq.url.split('@')[0].split(':')[-1], "******") if rmq.url else "None"
    print(f"Target URL: {masked_url}")
    try:
        await rmq.connect()
        print(f"Connection: Connected to {rmq.url.split('@')[-1]}")
        
        queues = [rmq.queue_name, rmq.audit_queue, rmq.dlq_name, rmq.news_topic, rmq.signal_topic]
        for q_name in queues:
            try:
                q = await rmq.channel.get_queue(q_name)
                print(f"Queue '{q_name}':")
                print(f"  Messages: {q.declaration_result.message_count}")
                print(f"  Consumers: {q.declaration_result.consumer_count}")
            except Exception as e:
                print(f"Queue '{q_name}': Error fetching stats - {e}")
        
        # Also check the "Optimized" broker health
        print("\n--- Optimized Broker Health ---")
        health = await broker.health_check()
        print(f"Status: {health.get('status')}")
        if "error" in health:
            print(f"Error: {health['error']}")
        
    except Exception as e:
        print(f"CRITICAL: Failed to connect to RabbitMQ: {e}")
        sys.exit(1)
    finally:
        await rmq.close()
        await broker.close()

if __name__ == "__main__":
    # Ensure environment variables are set for the script
    # In a real scenario, we'd use scripts/utils_env.sh load_decrypted_secrets
    # but here we assume the calling environment has them or we can run via a wrapper
    asyncio.run(report_health())
