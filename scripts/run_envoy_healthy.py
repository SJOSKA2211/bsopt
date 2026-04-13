#!/usr/bin/env python3
import subprocess
import sys
import time

import requests
from rich.console import Console
from rich.panel import Panel

console = Console()


def get_container_engine():
    """Detect container engine, strictly prioritizing docker."""
    return "docker", "docker compose"


def check_envoy_ready(ready_url: str, health_url: str, timeout: int = 120, interval: int = 5):
    console.print(f"[bold blue][*][/bold blue] Monitoring Envoy Readiness at {ready_url}...")
    console.print(f"[bold blue][*][/bold blue] Monitoring Upstream Health at {health_url}...")

    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            # 1. Check Envoy Admin /ready endpoint
            ready_resp = requests.get(ready_url, timeout=2)
            is_live = ready_resp.status_code == 200 and "LIVE" in ready_resp.text

            # 2. Check Gateway health (which depends on API)
            health_resp = requests.get(health_url, timeout=2)
            is_healthy = health_resp.status_code == 200

            if is_live and is_healthy:
                console.print(
                    Panel(
                        "[bold green] ENVOY GATEWAY IS LIVE AND UPSTREAM IS HEALTHY[/bold green]"
                    )
                )
                return True
            else:
                status = "READY" if is_live else "NOT_READY"
                up_status = "HEALTHY" if is_healthy else "UNHEALTHY"
                console.print(
                    f"[yellow][-][/yellow] Envoy: {status} | Upstream: {up_status}, retrying..."
                )
        except requests.exceptions.RequestException as e:
            console.print(f"[dim][-] Connectivity gap: {e}[/dim]")

        time.sleep(interval)
    return False


def start_envoy(compose_cmd):
    console.print(f"[bold blue][*][/bold blue] Starting Envoy via [cyan]{compose_cmd}[/cyan]...")
    cmd = compose_cmd.split() + [
        "-f",
        "infrastructure/orchestration/docker-compose.yml",
        "up",
        "-d",
        "envoy",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            console.print("[bold green][+][/bold green] Envoy start command issued successfully.")
            return True
        else:
            console.print(f"[bold red][!][/bold red] Failed to start Envoy: {result.stderr}")
            return False
    except Exception as e:
        console.print(f"[bold red][!][/bold red] Exception during start: {e}")
        return False


if __name__ == "__main__":
    # Standardized ports
    ready_url = "http://localhost:9901/ready"
    health_url = "http://localhost:8081/health"

    engine, compose = get_container_engine()
    console.print(
        f"[bold blue][*][/bold blue] Environment: [cyan]Engine={engine}[/cyan], [cyan]Compose={compose}[/cyan]"
    )

    if start_envoy(compose):
        if check_envoy_ready(ready_url, health_url):
            console.print(
                Panel(
                    "[bold green] ENVOY API GATEWAY IS FULLY OPTIMIZED AND HEALTHY[/bold green]",
                    title="Success",
                )
            )
            sys.exit(0)
        else:
            console.print(
                Panel(
                    "[bold red] ERROR: Envoy failed to reach healthy state within timeout.[/bold red]",
                    title="Failure",
                )
            )
            # Dump logs for diagnostics
            log_cmd = compose.split() + [
                "-f",
                "infrastructure/orchestration/docker-compose.yml",
                "logs",
                "--tail=50",
                "envoy",
            ]
            subprocess.run(log_cmd)
            sys.exit(1)
    else:
        sys.exit(1)
