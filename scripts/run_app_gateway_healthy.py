import subprocess
import time
import sys
import os
import json
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_container_engine():
    """Detect if we should use podman or docker."""
    # On Silverblue/Toolbox, podman is standard.
    try:
        subprocess.run(["podman", "--version"], check=True, capture_output=True)
        return "podman"
    except:
        try:
            subprocess.run(["docker", "--version"], check=True, capture_output=True)
            return "docker"
        except:
            # Fallback to podman as it's likely a toolbox
            return "podman"

def run_command(cmd):
    """Run a host command via flatpak-spawn if in a toolbox/container, otherwise run directly."""
    # Check for common toolbox/flatpak indicators
    is_toolbox = os.path.exists("/run/.toolboxenv") or os.path.exists("/run/.containerenv")
    is_flatpak = os.path.exists("/.flatpak-info")
    
    if is_toolbox or is_flatpak:
        return ["flatpak-spawn", "--host"] + cmd
    return cmd

def is_port_open(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_heartbeat(path):
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r") as f:
            data = json.load(f)
            ts = data.get("time", 0)
            # Heartbeat must be within last 15 seconds
            if time.time() - ts < 15:
                return True
    except:
        pass
    return False

def main():
    engine = get_container_engine()
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    compose_file = os.path.join(project_root, "infrastructure/orchestration/docker-compose.yml")
    heartbeat_path = "/tmp/frontend_heartbeat"

    console.print(Panel(f"[bold cyan]🚀 Starting App Gateway (Frontend) Sequence[/bold cyan]\nEngine: {engine}\nRoot: {project_root}"))

    # Cleanup old heartbeat
    if os.path.exists(heartbeat_path):
        os.remove(heartbeat_path)

    # Start the service
    console.print("[yellow]Starting frontend service...[/yellow]")
    cmd = run_command([engine, "compose", "-f", compose_file, "up", "-d", "frontend"])
    subprocess.run(cmd, check=True)

    # Phase 1: Port Gating
    console.print("[yellow]Waiting for port 5173 (Vite)...[/yellow]")
    start_time = time.time()
    while time.time() - start_time < 120:
        if is_port_open(5173):
            console.print("[green]✅ Port 5173 is OPEN[/green]")
            break
        time.sleep(2)
    else:
        console.print("[red]❌ Timed out waiting for port 5173[/red]")
        sys.exit(1)

    # Phase 2: Flow Gating (AIOps Heartbeat)
    console.print("[yellow]Waiting for AIOps Flow-Positive Heartbeat...[/yellow]")
    start_time = time.time()
    while time.time() - start_time < 60:
        if check_heartbeat(heartbeat_path):
            console.print("[bold green]✅ App Gateway is HEALTHY and REPORTING FLOW[/bold green]")
            break
        time.sleep(2)
    else:
        console.print("[red]❌ Timed out waiting for AIOps Healthy state[/red]")
        sys.exit(1)

    console.print(Panel("[bold green]🎉 App Gateway Startup Complete![/bold green]"))

if __name__ == "__main__":
    main()
