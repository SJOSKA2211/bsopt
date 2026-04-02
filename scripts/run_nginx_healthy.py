#!/usr/bin/env python3
import time
import requests
import os
import sys
import subprocess
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_container_engine():
    """Detect container engine with toolbox/Silverblue support."""
    # 1. Check if we're in a toolbox (standard method)
    if os.path.exists("/run/.containerenv"):
        console.print("[dim][*] Detected Toolbox environment, using flatpak-spawn...[/dim]")
        return "flatpak-spawn", "flatpak-spawn --host podman compose"

    # 2. Check for Podman socket if NOT in toolbox
    sock_path = "/run/user/1000/podman/podman.sock"
    if os.path.exists(sock_path):
        return "podman", "podman compose"
        
    return "podman", "podman compose"

def check_health(url, timeout=120, interval=5):
    console.print(f"[bold blue][*][/bold blue] Monitoring NGINX Health at {url}...")
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            resp = requests.get(url, timeout=2)
            if resp.status_code == 200:
                console.print(Panel("[bold green]✅ NGINX GATEWAY & UPSTREAM ARE HEALTHY[/bold green]"))
                return True
            else:
                console.print(f"[yellow][-][/yellow] NGINX Status: {resp.status_code}, retrying...")
        except requests.exceptions.RequestException:
            console.print("[dim][-] Connection refused, waiting for NGINX...[/dim]")
        
        time.sleep(interval)
    return False

def start_nginx(compose_cmd):
    abs_path = os.path.abspath("infrastructure/orchestration/docker-compose.yml")
    console.print(f"[bold blue][*][/bold blue] Starting NGINX via [cyan]{compose_cmd}[/cyan] with file [cyan]{abs_path}[/cyan]...")
    
    # Explicitly pull/build first to avoid timeout issues in check_health
    cmd_up = compose_cmd.split() + ["-f", abs_path, "up", "-d", "nginx"]
    
    try:
        # Use Popen to stream output directly to console for visibility in toolbox
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
            console.print("[bold green][+][/bold green] NGINX start command issued successfully.")
            return True
        else:
            console.print(f"[bold red][!][/bold red] Failed to start NGINX (Exit {process.returncode})")
            return False
    except Exception as e:
        console.print(f"[bold red][!][/bold red] Exception: {e}")
        return False

if __name__ == "__main__":
    health_url = "http://localhost:8081/health"
    
    engine, compose = get_container_engine()
    console.print(f"[bold blue][*][/bold blue] Env: [cyan]{engine}[/cyan] | [cyan]{compose}[/cyan]")
    
    if start_nginx(compose):
        if check_health(health_url):
            console.print(Panel("[bold green]🚀 NGINX API GATEWAY (OPTIMIZED) IS ONLINE[/bold green]", title="Success"))
            sys.exit(0)
    
    console.print(Panel("[bold red]❌ FAILED to reach healthy state[/bold red]"))
    sys.exit(1)
