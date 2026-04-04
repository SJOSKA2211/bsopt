import asyncio
import os
import sys

# Add src to path
sys.path.append(os.getcwd())

from src.shared.config import settings


async def main():
    print(f"RABBITMQ_USER: {settings.RABBITMQ_USER}")
    # print(f"RABBITMQ_PASSWORD: {settings.RABBITMQ_PASSWORD}") # Don't print secret
    print(f"RABBITMQ_HOST: {settings.RABBITMQ_HOST}")
    print(f"RABBITMQ_PORT: {settings.RABBITMQ_PORT}")
    print(f"Constructed URL: {settings.RABBITMQ_URL}")
    
    from src.shared.rabbitmq import get_rabbitmq
    rmq = get_rabbitmq()
    try:
        await rmq.connect()
        print("✅ RabbitMQ Connection Successful!")
        
        # Check queue stats
        stats = await rmq.get_queue_stats("market_ticks")
        print(f"📊 Queue 'market_ticks' stats: {stats}")
        
        await rmq.close()
        print("🔌 Connection closed.")
    except Exception as e:
        print(f"❌ RabbitMQ Connection Failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
