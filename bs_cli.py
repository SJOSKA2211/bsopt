#!/usr/bin/env python3
"""
BS-OPT Singularity Command-Line Interface
=========================================
Precision control for high-performance quantitative finance.
"""

import sys
import asyncio
import click
import time
import structlog
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box

# Enforce Venv
try:
    from scripts.enforce_venv import check_venv
    check_venv()
except ImportError:
    pass

from src.config import settings
from src.services.pricing_service import PricingService
from src.pricing.black_scholes import BSParameters

console = Console()
logger = structlog.get_logger()

@click.group()
@click.version_option(version='3.0.0', prog_name='bsopt-singularity')
def cli():
    """🚀 God-Mode Control for the Black-Scholes Advanced Platform."""
    pass

# ============================================================================
# Core Pricing Commands
# ============================================================================

@cli.command()
@click.option('--spot', type=float, required=True)
@click.option('--strike', type=float, required=True)
@click.option('--maturity', type=float, required=True)
@click.option('--volatility', type=float, required=True)
@click.option('--rate', type=float, required=True)
@click.option('--dividend', type=float, default=0.0)
@click.option('--option-type', type=click.Choice(['call', 'put']), default='call')
@click.option('--model', default='black_scholes')
def price(spot, strike, maturity, volatility, rate, dividend, option_type, model):
    """Price an option using the Singularity engine."""
    params = BSParameters(spot=spot, strike=strike, maturity=maturity, volatility=volatility, rate=rate, dividend=dividend)
    service = PricingService()
    
    with console.status(f"[bold green]Computing {model} price..."):
        result = asyncio.run(service.price_option(params, option_type, model))
    
    table = Table(title=f"Pricing Result: {model.upper()}", box=box.ROUNDED)
    table.add_column("Field", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Theoretical Price", f"${result.price:.4f}")
    table.add_row("Execution Time", f"{result.computation_time_ms:.2f}ms")
    table.add_row("Model", result.model)
    console.print(table)

# ============================================================================
# Singularity Commands
# ============================================================================

@cli.command()
@click.option('--timesteps', default=10000, help='Total training steps')
@click.option('--shm-name', default='rl_weights', help='SHM segment for weights')
def train_transformer(timesteps, shm_name):
    """Trigger a high-performance Transformer-RL training run."""
    console.print(Panel("[bold magenta]🚀 INITIALIZING TRANSFORMER SINGULARITY TRAINING[/bold magenta]"))
    from src.ml.reinforcement_learning.train import train_td3
    
    with console.status("[bold cyan]Spinning up 8-head Attention Encoders..."):
        # This will run the real training loop we refactored
        result = train_td3(total_timesteps=timesteps)
        
    console.print(f"[bold green]Training Complete![/bold green]")
    console.print(f"MLflow Run ID: [cyan]{result['run_id']}[/cyan]")
    console.print(f"Model saved to: [dim]{result['model_path']}[/dim]")

@cli.command()
def mesh_status():
    """Inspect the Shared Memory Market Mesh health."""
    console.print(Panel("[bold yellow]MESH DIAGNOSTICS[/bold yellow]"))
    from src.shared.shm_manager import SHMManager
    
    # Check Market Mesh
    try:
        shm = SHMManager("market_mesh", dict)
        data = shm.read()
        console.print(f"[green]✔ Market Mesh SHM:[/green] Active")
        console.print(f"  - Segment Name: {shm.name}")
        console.print(f"  - Tickers Tracked: {len(data)}")
    except Exception as e:
        console.print(f"[red]✘ Market Mesh SHM:[/red] Offline ({str(e)})")

    # Check RL Weights Mesh
    try:
        shm = SHMManager("rl_weights", dict)
        console.print(f"[green]✔ RL Weights SHM:[/green] Active")
    except Exception:
        console.print(f"[yellow]! RL Weights SHM:[/yellow] Inactive")

if __name__ == '__main__':
    cli()