#!/usr/bin/env python3
import json
import os
import subprocess
import sys
import time

from rich.console import Console
from rich.panel import Panel

console = Console()


def get_container_engine():
    """Detect container engine, strictly prioritizing docker."""
    return "docker", "docker compose"


def check_heartbeat(path, timeout=300, interval=10, exit_early_if_healthy=False):
    console.print(f"[bold blue][*][/bold blue] Monitoring Heartbeat at {path}...")
    start_time = time.time()
    # If we just want to check once and exit
    limit = timeout if not exit_early_if_healthy else 1
    
    while True:
        if os.path.exists(path):
            try:
                with open(path) as f:
                    content = f.read().strip()

                # Parse JSON heartbeat
                data = json.loads(content)
                ts = data.get("time", 0.0)
                metrics = data.get("metrics", {})
                processed = metrics.get("processed", 0)

                delta = time.time() - ts
                if delta < 60:  # Recent enough
                    if processed > 0:
                        console.print(
                            Panel(
                                f"[bold green] SCRAPER IS HEALTHY & PRODUCING DATA[/bold green]\n"
                                f"Metrics: {processed} ticks ingested"
                            )
                        )
                        return True
                    else:
                        if not exit_early_if_healthy:
                            console.print(
                                "[yellow][-][/yellow] Scraper active but 0 ticks processed, waiting..."
                            )
                else:
                    if not exit_early_if_healthy:
                        console.print(
                            f"[dim][-] Heartbeat is stale ({delta:.1f}s ago), waiting...[/dim]"
                        )
            except (json.JSONDecodeError, ValueError, KeyError):
                # Fallback to legacy timestamp if it's not JSON yet
                try:
                    ts = float(content)
                    if time.time() - ts < 60:
                        console.print(
                            Panel(
                                "[bold yellow]️ Legacy Heartbeat Detected - Scraper is LIVE but metrics missing[/bold yellow]"
                            )
                        )
                        return True
                except ValueError:
                    pass
                if not exit_early_if_healthy:
                    console.print("[dim][-] Corrupt heartbeat file, waiting...[/dim]")
        else:
            if not exit_early_if_healthy:
                console.print(
                    "[dim][-] Heartbeat file missing, waiting for scraper and ingestion-service...[/dim]"
                )

        if time.time() - start_time >= limit:
            break
            
        if not exit_early_if_healthy:
            time.sleep(interval)
        else:
            break
    return False


def start_scraper(compose_cmd, service_name):
    abs_path = os.path.abspath("infrastructure/orchestration/docker-compose.yml")
    console.print(
        f"[bold blue][*][/bold blue] Starting {service_name} via [cyan]{compose_cmd}[/cyan]..."
    )

    # Scrapers depend on ingestion-service, so we bring them up together
    cmd_up = compose_cmd.split() + ["-f", abs_path, "up", "-d", "ingestion-service", service_name]

    try:
        process = subprocess.Popen(
            cmd_up, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in process.stdout:
            console.print(f"[dim]{line.strip()}[/dim]")
        process.wait()

        if process.returncode == 0:
            console.print(
                f"[bold green][+][/bold green] {service_name} start command issued successfully."
            )
            return True
        else:
            console.print(
                f"[bold red][!][/bold red] Failed to start {service_name} (Exit {process.returncode})"
            )
            return False
    except Exception as e:
        console.print(f"[bold red][!][/bold red] Exception: {e}")
        return False


if __name__ == "__main__":
    # Default to nse-scraper if no args provided
    service_name = sys.argv[1] if len(sys.argv) > 1 else "nse-scraper"
    heartbeat_path = "/tmp/scraper_heartbeat"

    engine, compose = get_container_engine()
    console.print(
        f"[bold blue][*][/bold blue] Target: [cyan]{service_name}[/cyan] | Env: [cyan]{engine}[/cyan]"
    )

    # CHECK FIRST
    console.print(" Checking if scraper is already healthy...")
    if check_heartbeat(heartbeat_path, exit_early_if_healthy=True):
        console.print(Panel(f" [bold green]{service_name.upper()} IS ALREADY ONLINE[/bold green]"))
        sys.exit(0)

    if start_scraper(compose, service_name):
        if check_heartbeat(heartbeat_path):
            console.print(
                Panel(
                    f" [bold green]{service_name.upper()} DATA FLOW IS ONLINE[/bold green]",
                    title="Success",
                )
            )
            sys.exit(0)

    console.print(Panel(f" FAILED to reach healthy state for {service_name}"))
    sys.exit(1)
