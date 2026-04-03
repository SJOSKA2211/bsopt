import asyncio
import sys

import httpx
from rich.console import Console
from rich.table import Table


async def report_health():
    console = Console()
    base_url = "http://localhost:8000"

    console.print(f"Scraping health and metrics from {base_url}...")

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            # Health Check
            try:
                health_resp = await client.get(f"{base_url}/health")
                health_data = health_resp.json()
            except Exception as e:
                health_data = {"status": "error", "error": str(e)}

            # Metrics Check
            try:
                metrics_resp = await client.get(f"{base_url}/metrics")
                metrics_text = metrics_resp.text
            except Exception as e:
                metrics_text = f"Error: {e}"

            # Summary Table
            table = Table(title="BS-OPT Health Report")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="bold")
            table.add_column("Details", style="magenta")

            # API Status
            api_status = health_data.get("status", "unknown")
            api_style = "green" if api_status == "healthy" else "red"
            table.add_row("API", f"[{api_style}]{api_status}[/{api_style}]", "")

            # Database Status
            db_status = health_data.get("database", {})
            db_healthy = db_status.get("status") == "healthy"
            db_style = "green" if db_healthy else "red"
            table.add_row(
                "Database",
                f"[{db_style}]{db_status.get('status', 'unknown')}[/{db_style}]",
                str(db_status),
            )

            # Rust Core Status
            rust_status = health_data.get("rust_core", {})
            rust_available = rust_status.get("available", False)
            rust_healthy = rust_status.get("status") == "healthy"
            rust_style = "green" if rust_healthy else "yellow"
            table.add_row(
                "Rust Core",
                f"[{rust_style}]{rust_status.get('status', 'unknown')}[/{rust_style}]",
                f"Available: {rust_available}",
            )

            # Metrics Summary
            rust_metrics = [line for line in metrics_text.splitlines() if line.startswith("rust_")]
            rust_metrics_count = len(rust_metrics)
            metrics_style = "green" if rust_metrics_count > 0 else "yellow"
            table.add_row(
                "Metrics",
                f"[{metrics_style}]Available[/{metrics_style}]",
                f"{rust_metrics_count} Rust metrics found",
            )

            console.print(table)

            if rust_metrics_count > 0:
                console.print("\n[bold]Sample Rust Metrics:[/bold]")
                for m in rust_metrics[:5]:
                    console.print(f"  {m}")

    except Exception as e:
        console.print(f"[red]Fatal error reporting health: {e}[/red]")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(report_health())
