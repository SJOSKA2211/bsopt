import asyncio

import structlog

from src.ml.aiops.autonomous_engine import AutonomousEngine

logger = structlog.get_logger()


async def verify():
    engine = AutonomousEngine()
    print("🚀 Triggering Autonomous Engine ML Inference Health Check...")
    try:
        ready = await engine._check_ml_inference_ready()
        if ready:
            print("✅ ML Inference is READY according to AutonomousEngine.")
        else:
            print("❌ ML Inference is NOT READY.")
    except Exception as e:
        print(f"❌ Error during health check: {e}")


if __name__ == "__main__":
    asyncio.run(verify())
