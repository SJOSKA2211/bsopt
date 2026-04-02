#!/usr/bin/env python3
import time
import json
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_container_engine():
    """Detect container engine with toolbox/Silverblue support."""
    if os.path.exists("/run/.containerenv"):
        return "flatpak-spawn", "flatpak-spawn --host podman compose"
    return "podman", "podman compose"

def check_heartbeat(path, timeout=300, interval=10):
    console.print(f"[bold blue][*][/bold blue] Monitoring Heartbeat at {path}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        if os.path.exists(path):
            try:
                with open(path, "r") as f:
                    content = f.read().strip()
                
                # Parse JSON heartbeat
                data = json.loads(content)
                ts = data.get("time", 0.0)
                metrics = data.get("metrics", {})
                processed = metrics.get("processed", 0)
                
                delta = time.time() - ts
                if delta < 60: # Recent enough
                    if processed > 0:
                        console.print(Panel(f"[bold green]✅ SCRAPER IS HEALTHY & PRODUCING DATA[/bold green]\n"
                                             f"Metrics: {processed} ticks ingested"))
                        return True
                    else:
                        console.print(f"[yellow][-][/yellow] Scraper active but 0 ticks processed, waiting...")
                else:
                    console.print(f"[dim][-] Heartbeat is stale ({delta:.1f}s ago), waiting...[/dim]")
            except (json.JSONDecodeError, ValueError, KeyError):
                # Fallback to legacy timestamp if it's not JSON yet
                try:
                    ts = float(content)
                    if time.time() - ts < 60:
                        console.print(Panel("[bold yellow]⚠️ Legacy Heartbeat Detected - Scraper is LIVE but metrics missing[/bold yellow]"))
                        return True
                except ValueError:
                    pass
                console.print("[dim][-] Corrupt heartbeat file, waiting...[/dim]")
        else:
            console.print("[dim][-] Heartbeat file missing, waiting for scraper and ingestion-service...[/dim]")
        
        time.sleep(interval)
    return False

def start_scraper(compose_cmd, service_name):
    abs_path = os.path.abspath("infrastructure/orchestration/docker-compose.yml")
    console.print(f"[bold blue][*][/bold blue] Starting {service_name} via [cyan]{compose_cmd}[/cyan]...")
    
    # Scrapers depend on ingestion-service, so we bring them up together
    cmd_up = compose_cmd.split() + ["-f", abs_path, "up", "-d", "ingestion-service", service_name]
    
    try:
        process = subprocess.Popen(
            cmd_up,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1
        )
        for line in process.stdout:
            console.print(f"[dim]{line.strip()}[/dim]")
        process.wait()
        
        if process.returncode == 0:
            console.print(f"[bold green][+][/bold green] {service_name} start command issued successfully.")
            return True
        else:
            console.print(f"[bold red][!][/bold red] Failed to start {service_name} (Exit {process.returncode})")
            return False
    except Exception as e:
        console.print(f"[bold red][!][/bold red] Exception: {e}")
        return False

if __name__ == "__main__":
    # Default to nse-scraper if no args provided
    service_name = sys.argv[1] if len(sys.argv) > 1 else "nse-scraper"
    heartbeat_path = "/tmp/scraper_heartbeat"
    
    engine, compose = get_container_engine()
    console.print(f"[bold blue][*][/bold blue] Target: [cyan]{service_name}[/cyan] | Env: [cyan]{engine}[/cyan]")
    
    if start_scraper(compose, service_name):
        if check_heartbeat(heartbeat_path):
            console.print(Panel(f"🚀 [bold green]{service_name.upper()} DATA FLOW IS ONLINE[/bold green]", title="Success"))
            sys.exit(0)
    
    console.print(Panel(f"❌ FAILED to reach healthy state for {service_name}"))
    sys.exit(1)
