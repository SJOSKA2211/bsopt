import random
import subprocess
import time

import structlog

logger = structlog.get_logger(__name__)

CONTAINERS = ["api", "auth-service", "neural-pricing", "worker", "redis", "rabbitmq"]

def chaos_monkey():
    """Randomly kill containers to test system resilience."""
    logger.info("chaos_monkey_unleashed", target_containers=CONTAINERS)
    
    while True:
        target = random.choice(CONTAINERS)
        logger.warning("chaos_event_triggered", container=target)
        
        try:
            # Podman kill
            subprocess.run(["podman", "kill", target], check=True)
            logger.info("chaos_action_success", killed=target)
            
            # Wait for self-healing (podman-compose restart policy)
            time.sleep(10)
            
            # Verify recovery
            status = subprocess.run(["podman", "inspect", "-f", "{{.State.Running}}", target], 
                                     capture_output=True, text=True)
            if "true" in status.stdout.lower():
                logger.info("self_healing_verified", container=target)
            else:
                logger.error("self_healing_failed", container=target)
                
        except Exception as e:
            logger.error("chaos_monkey_error", error=str(e))
            
        time.sleep(random.randint(60, 300))

if __name__ == "__main__":
    chaos_monkey()
