import asyncio
import os
import time
import httpx
import sys
from rich.console import Console
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

# Service configurations
SERVICES = {
    "API Gateway": {"url": "http://localhost:8000/health", "type": "http"},
    "Auth Service": {"url": "http://localhost:3001/health", "type": "http"},
    "ML Inference": {"url": "http://localhost:5002/health", "type": "http"},
    "Portfolio": {"url": "http://localhost:8003/health", "type": "http"},
    "Ingestion": {"heartbeat": "/tmp/ingestion_heartbeat", "type": "heartbeat"},
    "NSE Scraper": {"heartbeat": "/tmp/scraper_heartbeat", "type": "heartbeat"},
    "yfinance Scraper": {"heartbeat": "/tmp/yfinance_heartbeat", "type": "heartbeat"},
    "Transformer": {"heartbeat": "/tmp/transformer_heartbeat", "type": "heartbeat"},
}

console = Console()

async def check_http(url):
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return "healthy", str(data.get("status", "ok"))
            return "unhealthy", f"HTTP {resp.status_code}"
    except Exception as e:
        return "down", str(e)

def check_heartbeat(path):
    if not os.path.exists(path):
        return "missing", "Heartbeat file not found"
    try:
        with open(path, "r") as f:
            ts = float(f.read().strip())
        delta = time.time() - ts
        if delta < 60:
            return "healthy", f"Active ({delta:.1f}s ago)"
        return "stale", f"Last active {delta:.1f}s ago"
    except Exception as e:
        return "error", str(e)

async def get_health_data():
    tasks = []
    for name, config in SERVICES.items():
        if config["type"] == "http":
            tasks.append(check_http(config["url"]))
        else:
            # Heartbeat check is synchronous but we wrap it for consistency
            tasks.append(asyncio.to_thread(check_heartbeat, config["heartbeat"]))
    
    results = await asyncio.gather(*tasks)
    return dict(zip(SERVICES.keys(), results))

def generate_table(health_data):
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Service", style="cyan", no_wrap=True)
    table.add_column("Status", width=12)
    table.add_column("Details", style="dim")

    all_healthy = True
    for name, (status, details) in health_data.items():
        style = "green" if status == "healthy" else "red"
        if status in ["stale", "missing", "down"]:
            all_healthy = False
            style = "yellow" if status == "stale" else "red"
        
        table.add_row(name, f"[{style}]{status.upper()}[/{style}]", details)
    
    return table, all_healthy

async def main():
    if "--simulate" in sys.argv:
        console.print(Panel("[bold green]✅ SIMULATED: All systems are GO! Engine is healthy.[/bold green]"))
        health_data = {k: ("healthy", "Simulated Operational") for k in SERVICES.keys()}
        table, _ = generate_table(health_data)
        console.print(table)
        return

    if len(sys.argv) > 1 and sys.argv[1] == "--wait":
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Waiting for all services to be healthy...", total=None)
            
            while True:
                health_data = await get_health_data()
                _, all_healthy = generate_table(health_data)
                
                if all_healthy:
                    console.print(Panel("[bold green]✅ All systems are GO! Engine is healthy.[/bold green]"))
                    table, _ = generate_table(health_data)
                    console.print(table)
                    break
                
                await asyncio.sleep(5)
    else:
        health_data = await get_health_data()
        table, all_healthy = generate_table(health_data)
        
        title = "🚀 Engine Health Report" if all_healthy else "⚠️ Engine Health Warning"
        console.print(Panel(table, title=title, expand=False))
        
        if not all_healthy:
            sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
