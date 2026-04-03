import subprocess
import time
import sys
import os
import json
from rich.console import Console
from rich.panel import Panel

console = Console()

def get_container_engine():
    """Detect container engine, strictly prioritizing docker."""
    return "docker"

def is_port_open(port):
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def check_heartbeat(path):
    try:
        import subprocess
        result = subprocess.run(["docker", "compose", "-f", "infrastructure/orchestration/docker-compose.yml", "exec", "frontend", "cat", path], capture_output=True, text=True)
        if result.returncode != 0:
            return False
        data = json.loads(result.stdout)
        ts = data.get("time", 0)
        # Heartbeat must be within last 15 seconds
        if time.time() - ts < 15:
            return True
    except Exception as e:
        print(f"Heartbeat error: {e}")
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
    cmd = [engine, "compose", "-f", compose_file, "up", "-d", "frontend"]
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
