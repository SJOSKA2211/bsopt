#!/usr/bin/env python3
import asyncio
import os
import sys
import structlog
import subprocess
from datetime import datetime

# Import AIOps Core
from src.ml.aiops.autonomous_engine import AutonomousEngine
from src.shared.config import settings

# Setup logging
try:
    from src.shared.observability import setup_logging
    setup_logging()
except ImportError:
    pass

logger = structlog.get_logger(__name__)

async def run_diagnostic(script_name: str) -> bool:
    """Runs a specific health check script and returns its success."""
    print(f"[*] Running Diagnostic: {script_name}...")
    try:
        # Use sys.executable to ensure we use the same Python interpreter
        process = await asyncio.create_subprocess_exec(
            sys.executable, f"scripts/{script_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            print(f"    ✅ {script_name} PASSED")
            return True
        else:
            print(f"    ❌ {script_name} FAILED")
            if stderr:
                print(f"       Error: {stderr.decode().splitlines()[-1] if stderr.decode().splitlines() else 'Unknown'}")
            return False
    except Exception as e:
        print(f"    🚨 Error executing {script_name}: {str(e)}")
        return False

async def verify_infrastructure_stack():
    """Sequentially verifies all layers of the manifold stack."""
    diagnostics = [
        "run_postgres_healthy.py",
        "run_redis_healthy.py",
        "run_rabbitmq_healthy.py",
        "run_auth_healthy.py",
        "run_api_healthy.py",
        "run_ingestion_healthy.py",
        "run_portfolio_healthy.py",
        "run_math_kernel_healthy.py",
        "run_ml_inference_healthy.py",
        "run_worker_healthy.py"
    ]
    
    print("\n" + "="*60)
    print("🚀 AIOPS MANIFOLD: PRE-FLIGHT READINESS CHECK")
    print("="*60)
    
    results = []
    for diag in diagnostics:
        results.append(await run_diagnostic(diag))
    
    success_rate = sum(results) / len(results)
    print(f"\n📊 Infrastructure Readiness: {success_rate:.0%}")
    
    if all(results):
        print("🟢 ALL SYSTEMS GO. Manifold is ready for autonomous operation.")
        return True
    elif success_rate >= 0.7:
        print("⚠️  SYSTEM DEGRADED. Proceeding with caution (Safe Mode likely).")
        return True
    else:
        print("🔴 CRITICAL FAILURE. Multi-layer infrastructure loss detected.")
        return False

async def start_manifold():
    """Main entry point to start the autonomous manifold."""
    print("\n" + "="*60)
    print(f"🔱 STARTING AUTONOMOUS MANIFOLD CYCLE v5.1")
    print(f"🕒 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    # 1. Verify Infrastructure
    if not await verify_infrastructure_stack():
        print("\n❌ ABORTING MANIFOLD START: Infrastructure requirements not met.")
        sys.exit(1)

    # 2. Instantiate Engine
    print("\n[*] Initializing Autonomous Engine...")
    engine = AutonomousEngine(
        config={
            "prometheus_url": settings.PROMETHEUS_URL,
            "api_service_name": "bsopt-api"
        }
    )

    # 3. Start Engine Loop
    print("[*] Engine Loop starting. Press Ctrl+C to stop.")
    try:
        # We need a data source to poll. 
        # For now, we'll use a placeholder that might be provided by the user later,
        # or we just let it run in 'Infrastructure Monitoring' mode.
        class InfrastructureDataSource:
            def get_latest_metrics(self):
                import pandas as pd
                return pd.DataFrame() # Empty but valid
                
        await engine.start(InfrastructureDataSource())
    except KeyboardInterrupt:
        print("\n[*] Shutdown signal received.")
    except Exception as e:
        print(f"\n🚨 Engine Loop crashed: {str(e)}")
    finally:
        engine.stop()
        print("🏁 Manifold Loop stopped.")

if __name__ == "__main__":
    try:
        asyncio.run(start_manifold())
    except KeyboardInterrupt:
        pass
