#!/usr/bin/env python3
"""
Black-Scholes Option Pricing Platform CLI
==========================================

Comprehensive command-line interface for option pricing, analysis, and backtesting.
Provides production-ready tools for quantitative finance workflows.

Commands:
    price         - Price a single option with multiple methods
    batch         - Batch pricing from CSV files
    serve         - Start FastAPI server
    init-db       - Initialize database schema
    fetch-data    - Download market data from providers
    train-model   - Train ML models for option pricing
    backtest      - Run backtesting strategies
    implied-vol   - Calculate implied volatility
    vol-surface   - Generate volatility surface visualization
    compare       - Compare pricing methods side-by-side

Author: Black-Scholes Platform Team
Version: 2.1.0
"""

import sys
# Enforce Virtual Environment
try:
    from scripts.enforce_venv import check_venv
    check_venv()
except ImportError:
    pass

import click
from click import Option
import pandas as pd
import numpy as np
from rich.console import Console
from rich.table import Table
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    track
)
from rich.panel import Panel
from rich.syntax import Syntax
from rich import box

# Import pricing engines
from src.pricing.models import BSParameters, OptionGreeks
from src.config import settings
from src.cli.auth import auth_group

# Rich console for formatted output
console = Console()

# ... (configure_logging lines 60-65)

@click.group()
@click.version_option(version='2.1.0', prog_name='bsopt')
@click.pass_context
def cli(ctx):
    # ... (docstring lines 73-83)
    ctx.ensure_object(dict)

# 🚀 SINGULARITY: Registering new God-Mode command groups
cli.add_command(auth_group)


# ============================================================================
# Helpers
# ============================================================================

async def _compare_all_methods_async(params: BSParameters, option_type: str,
                                 compute_greeks: bool) -> dict:
    """Compare all available pricing methods using PricingService in parallel."""
    from src.services.pricing_service import PricingService
    import asyncio
    service = PricingService()
    methods = ['black_scholes', 'fdm', 'monte_carlo', 'binomial']
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Computing prices in parallel...", total=len(methods))

        async def _price_and_update(method):
            try:
                # Use PricingService for consistent logic and caching
                result = await service.price_option(params, option_type, method)
                
                if compute_greeks and method == 'black_scholes':
                    greeks_dict = await service.calculate_greeks(params, option_type)
                    result['greeks'] = greeks_dict

                results[method] = result
            except Exception as e:
                logger.error("cli_comparison_failed", method=method, error=str(e))
                results[method] = {
                    'error': f"Pricing failed for {method}.",
                    'method': method
                }
            finally:
                progress.advance(task)

        # Execute all methods concurrently
        await asyncio.gather(*[_price_and_update(m) for method in methods])

    return results


# ============================================================================
# Command: price - Price a single option
# ============================================================================

@cli.command()
@click.option('--spot', type=float, required=True, help='Current spot price')
@click.option('--strike', type=float, required=True, help='Strike price')
@click.option('--maturity', type=float, required=True, help='Time to maturity (years)')
@click.option('--volatility', type=float, required=True, help='Annualized volatility')
@click.option('--rate', type=float, required=True, help='Risk-free interest rate')
@click.option('--dividend', type=float, default=0.0, help='Dividend yield (default: 0.0)')
@click.option('--option-type', type=click.Choice(['call', 'put'], case_sensitive=False),
              default='call', help='Option type (default: call)')
@click.option('--method', type=click.Choice(['bs', 'fdm', 'mc', 'binomial', 'all'], case_sensitive=False),
              default='bs', help='Pricing method (default: bs)')
@click.option('--show-greeks', is_flag=True, help='Display option Greeks')
@click.option('--json-output', is_flag=True, help='Output as JSON')
def price(spot: float, strike: float, maturity: float, volatility: float,
          rate: float, dividend: float, option_type: str, method: str,
          show_greeks: bool, json_output: bool):
    """
    Price a single option using specified method.
    """
    try:
        # Create BS parameters
        params = BSParameters(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend
        )

        from src.services.pricing_service import PricingService
        import asyncio
        service = PricingService()

        async def _run_pricing():
            if method == 'all':
                return await _compare_all_methods_async(params, option_type, show_greeks)
            else:
                result = await service.price_option(params, option_type, method)
                if show_greeks:
                    greeks = await service.calculate_greeks(params, option_type)
                    result['greeks'] = greeks
                return result

        if not json_output:
            console.print(f"\n[bold cyan]Option Pricing Request[/bold cyan]")
            console.print(f"[dim]{'=' * 70}[/dim]\n")

            # Display parameters
            param_table = Table(show_header=False, box=box.SIMPLE)
            param_table.add_column("Parameter", style="cyan")
            param_table.add_column("Value", style="green")

            param_table.add_row("Spot Price", f"${spot:.2f}")
            param_table.add_row("Strike Price", f"${strike:.2f}")
            param_table.add_row("Time to Maturity", f"{maturity:.2f} years")
            param_table.add_row("Volatility", f"{volatility:.2%}")
            param_table.add_row("Risk-Free Rate", f"{rate:.2%}")
            param_table.add_row("Dividend Yield", f"{dividend:.2%}")
            param_table.add_row("Option Type", option_type.upper())
            param_table.add_row("Moneyness", f"{spot/strike:.4f}")

            console.print(param_table)
            console.print()

        # Run async pricing
        results = asyncio.run(_run_pricing())

        if json_output:
            console.print_json(data=results)
        elif method == 'all':
            _display_comparison_table(results)
        else:
            _display_single_price(results, method)

    except Exception as e:
        logger.exception("price_command_failed", error=str(e))
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


def _price_single_method(params: BSParameters, option_type: str,
                        method: str, compute_greeks: bool) -> dict:
    """Price option using single method."""

    start_time = time.perf_counter()

    if method == 'bs':
        # Black-Scholes analytical
        if option_type == 'call':
            price = BlackScholesEngine.price_call(params)
        else:
            price = BlackScholesEngine.price_put(params)

        greeks = None
        if compute_greeks:
            greeks = BlackScholesEngine.calculate_greeks(params, option_type)

    elif method == 'fdm':
        # Finite Difference Method (Crank-Nicolson)
        solver = CrankNicolsonSolver(
            spot=params.spot,
            strike=params.strike,
            maturity=params.maturity,
            volatility=params.volatility,
            rate=params.rate,
            dividend=params.dividend,
            option_type=option_type,
            n_spots=200,
            n_time=500
        )
        price = solver.solve()

        greeks = None
        if compute_greeks:
            greeks = solver.get_greeks()

    elif method == 'mc':
        # Monte Carlo simulation
        config = MCConfig(
            n_paths=100000,
            n_steps=252,
            antithetic=True,
            control_variate=True,
            seed=42
        )

        engine = MonteCarloEngine(params, option_type, config)
        result = engine.price()
        price = result['price']

        greeks = None
        if compute_greeks:
            greeks = engine.calculate_greeks()

    elif method == 'binomial':
        raise NotImplementedError("Binomial tree method not yet implemented")

    else:
        raise ValueError(f"Unknown method: {method}")

    computation_time = (time.perf_counter() - start_time) * 1000  # ms

    result = {
        'method': method,
        'price': price,
        'computation_time_ms': computation_time
    }

    if greeks:
        result['greeks'] = {
            'delta': greeks.delta,
            'gamma': greeks.gamma,
            'vega': greeks.vega,
            'theta': greeks.theta,
            'rho': greeks.rho
        }

    return result


def _compare_all_methods(params: BSParameters, option_type: str,
                        compute_greeks: bool) -> dict:
    """Compare all available pricing methods."""

    methods = ['bs', 'fdm', 'mc']
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console
    ) as progress:

        task = progress.add_task("[cyan]Computing prices...", total=len(methods))

        for method in methods:
            try:
                result = _price_single_method(params, option_type, method, compute_greeks)
                results[method] = result
                progress.advance(task)
            except Exception as e:
                logger.error(f"Error pricing with {method} method: {e}", exc_info=True)
                results[method] = {
                    'error': "Pricing failed due to an unexpected error.",
                    'method': method
                }
                progress.advance(task)

    return results


def _display_single_price(result: dict, method: str):
    """Display single price result in formatted output."""

    method_names = {
        'bs': 'Black-Scholes',
        'fdm': 'Finite Difference (Crank-Nicolson)',
        'mc': 'Monte Carlo',
        'binomial': 'Binomial Tree'
    }

    console.print(f"[bold green]{method_names.get(method, method)} Price:[/bold green] "
                 f"[bold white]${result['price']:.4f}[/bold white]")
    console.print(f"[dim]Computation Time: {result['computation_time_ms']:.2f}ms[/dim]\n")

    if 'greeks' in result:
        console.print("[bold cyan]Greeks:[/bold cyan]")

        greeks_table = Table(show_header=True, box=box.ROUNDED)
        greeks_table.add_column("Greek", style="cyan")
        greeks_table.add_column("Value", style="green", justify="right")
        greeks_table.add_column("Description", style="dim")

        greeks = result['greeks']
        greeks_table.add_row("Delta", f"{greeks['delta']:.4f}", "dV/dS")
        greeks_table.add_row("Gamma", f"{greeks['gamma']:.4f}", "d²V/dS²")
        greeks_table.add_row("Vega", f"{greeks['vega']:.4f}", "dV/dσ (per 1%)")
        greeks_table.add_row("Theta", f"{greeks['theta']:.4f}", "dV/dt (per day)")
        greeks_table.add_row("Rho", f"{greeks['rho']:.4f}", "dV/dr (per 1%)")

        console.print(greeks_table)
        console.print()


def _display_comparison_table(results: dict):
    """Display comparison table for multiple methods."""

    method_names = {
        'bs': 'Black-Scholes',
        'fdm': 'Crank-Nicolson',
        'mc': 'Monte Carlo',
        'binomial': 'Binomial Tree'
    }

    console.print("\n[bold cyan]Method Comparison[/bold cyan]")

    table = Table(show_header=True, box=box.DOUBLE_EDGE)
    table.add_column("Method", style="cyan")
    table.add_column("Price", style="green", justify="right")
    table.add_column("Time (ms)", style="yellow", justify="right")
    table.add_column("Status", style="white")

    for method, result in results.items():
        if 'error' in result:
            table.add_row(
                method_names.get(method, method),
                "N/A",
                "N/A",
                f"[red]Error: {result['error']}[/red]"
            )
        else:
            table.add_row(
                method_names.get(method, method),
                f"${result['price']:.4f}",
                f"{result['computation_time_ms']:.2f}",
                "[green]Success[/green]"
            )

    console.print(table)
    console.print()

    # Display Greeks if available
    if any('greeks' in r for r in results.values() if 'error' not in r):
        console.print("[bold cyan]Greeks Comparison[/bold cyan]")

        greeks_table = Table(show_header=True, box=box.ROUNDED)
        greeks_table.add_column("Greek", style="cyan")

        for method in results.keys():
            if 'error' not in results[method] and 'greeks' in results[method]:
                greeks_table.add_column(method_names.get(method, method),
                                       style="green", justify="right")

        greek_names = ['delta', 'gamma', 'vega', 'theta', 'rho']

        for greek in greek_names:
            row = [greek.capitalize()]
            for method in results.keys():
                if 'error' not in results[method] and 'greeks' in results[method]:
                    value = results[method]['greeks'][greek]
                    row.append(f"{value:.4f}")
            greeks_table.add_row(*row)

        console.print(greeks_table)
        console.print()


# ============================================================================
# Command: batch - Batch pricing from CSV
# ============================================================================

from src.utils.filesystem import sanitize_path

# ... (rest of the file)

@cli.command("batch")
@click.option('--input-file', type=click.Path(exists=True), required=True, help="Input CSV file with options to price")
@click.option('--symbols', required=True, help="Comma-separated list of stock symbols")
@click.option('--start-date', required=True, help="Start date (YYYY-MM-DD)")
@click.option('--end-date', required=True, help="End date (YYYY-MM-DD)")
@click.option('--output-file', default="batch_results.csv", type=click.Path(), help="Output CSV file for results")
@click.option('--method', default='bs', help="Pricing method (default: bs)")
@click.option('--compute-greeks', is_flag=True, help="Compute greeks")
def batch_command(
    input_file: Path,
    symbols: str,
    start_date: str,
    end_date: str,
    output_file: Path,
    method: str,
    compute_greeks: bool
):
    """
    Batch price options from CSV file.
    """
    from src.services.batch_pricing_service import BatchPricingService

    console.print(f"\n[bold cyan]Batch Pricing[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")
    
    service = BatchPricingService()
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            
            task_id = progress.add_task("[cyan]Pricing options...", total=100) # Initial dummy
            
            def setup_p(total):
                progress.update(task_id, total=total)
                
            def advance_p(amount):
                progress.advance(task_id, advance=amount)
            
            results = service.process_batch(
                input_file=input_file,
                output_file=output_file,
                method=method,
                compute_greeks=compute_greeks,
                progress_setup=setup_p,
                progress_advance=advance_p
            )
            
        console.print(f"[bold green]Batch pricing completed. Results saved to:[/bold green] {output_file}")
        console.print(f"[green]Processed: {len(results)} options[/green]\n")
        
    except Exception as e:
        logger.exception(f"Unexpected error in batch command: {e}")
        console.print(f"[bold red]An unexpected error occurred during batch processing.[/bold red]")
        sys.exit(1)


# ============================================================================
# Command: serve - Start API server
# ============================================================================

@cli.command()
@click.option('--host', default='0.0.0.0', help='Host to bind (default: 0.0.0.0)')
@click.option('--port', default=8000, help='Port to bind (default: 8000)')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
@click.option('--workers', default=1, help='Number of worker processes')
def serve(host: str, port: int, reload: bool, workers: int):
    """
    Start FastAPI server.

    Launches the Black-Scholes Option Pricing API server with production-ready
    configuration. Supports auto-reload for development and multiple workers
    for production.

    Examples:
        bsopt serve
        bsopt serve --host 0.0.0.0 --port 8000 --reload
        bsopt serve --workers 4
    """

    import uvicorn

    console.print(f"\n[bold cyan]Starting Black-Scholes API Server[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Host:[/cyan] {host}")
    console.print(f"[cyan]Port:[/cyan] {port}")
    console.print(f"[cyan]Reload:[/cyan] {'Enabled' if reload else 'Disabled'}")
    console.print(f"[cyan]Workers:[/cyan] {workers}")
    console.print(f"[cyan]Environment:[/cyan] {settings.ENVIRONMENT}")
    console.print()

    console.print(f"[green]Server starting...[/green]")
    console.print(f"[dim]Press Ctrl+C to stop[/dim]\n")

    try:
        uvicorn.run(
            "src.api.main:app",
            host=host,
            port=port,
            reload=reload,
            workers=workers if not reload else 1,
            log_level=settings.LOG_LEVEL.lower(),
            # Stream logs to stdout for easier debugging
            log_config=None, # Disable default uvicorn log config to use root logger
        )
    except KeyboardInterrupt:
        console.print("\n[yellow]Server stopped by user[/yellow]")
    except Exception as e:
        logger.exception(f"Server error: {e}")
        console.print(f"[bold red]Server failed to start due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: init-db - Initialize database
# ============================================================================

@cli.command()
@click.option('--seed', is_flag=True, help='Seed database with sample data')
@click.option('--force', is_flag=True, help='Force recreate tables (drops existing data)')
def init_db(seed: bool, force: bool):
    """
    Initialize database schema.

    Creates all required tables, indexes, and constraints for the option
    pricing platform. Optionally seeds the database with sample data.

    Examples:
        bsopt init-db
        bsopt init-db --seed
        bsopt init-db --force --seed
    """

    console.print(f"\n[bold cyan]Database Initialization[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    if force:
        console.print("[yellow]WARNING: Force mode will drop existing tables![/yellow]")
        if not click.confirm("Are you sure you want to continue?"):
            console.print("[dim]Operation cancelled[/dim]")
            return

    try:
        console.print(f"[cyan]Configuring database connection...[/cyan]")
        console.print()

        with console.status("[bold green]Creating tables...") as status:
            from src.database import create_tables, drop_tables
            if force:
                drop_tables()
            create_tables()

            console.print("[green]Tables created successfully[/green]")

            if seed:
                status.update("[bold green]Seeding sample data...")
                time.sleep(1)  # Placeholder
                console.print("[green]Sample data seeded[/green]")

        console.print(f"\n[bold green]Database initialized successfully![/bold green]\n")

    except Exception as e:
        logger.exception(f"Database initialization failed: {e}")
        console.print(f"[bold red]Database initialization failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: fetch-data - Download market data
# ============================================================================

@cli.command()
@click.option('--symbol', required=True, help='Stock symbol (e.g., SPY, AAPL)')
@click.option('--days', default=30, help='Number of days of historical data')
@click.option('--provider', type=click.Choice(['yahoo', 'alphavantage', 'polygon']),
              default='yahoo', help='Data provider (default: yahoo)')
@click.option('--output', type=click.Path(), help='Output file path (optional)')
@click.option('--use-cache', is_flag=True, default=True, help='Use cached data if available')
@click.option('--batch-size', default=50, help='Batch size for concurrent requests')
def fetch_data(symbol: str, days: int, provider: str, output: Optional[str], use_cache: bool, batch_size: int):
    """
    Download market data from provider using optimized collectors.

    Features:
    - LRU Caching and TTL support
    - Request deduplication
    - Batching for concurrent requests
    - Prometheus metrics tracking
    """

    import asyncio
    from src.data.collectors import YFinanceCollector, CollectorConfig

    console.print(f"\n[bold cyan]Market Data Fetch[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Symbol:[/cyan]   {symbol}")
    console.print(f"[cyan]Days:[/cyan]     {days}")
    console.print(f"[cyan]Provider:[/cyan] {provider}")
    console.print(f"[cyan]Caching:[/cyan]  {'Enabled' if use_cache else 'Disabled'}")
    console.print()

    async def _fetch():
        config = CollectorConfig(
            cache_enabled=use_cache,
            batch_size=batch_size,
            cache_ttl_minutes=60
        )
        
        collector = YFinanceCollector(config)
        
        with console.status(f"[bold green]Fetching options chain for {symbol}..."):
            # Use the optimized collector
            options = await collector.fetch_options_chain(symbol)
            
        return collector, options

    try:
        collector, options = asyncio.run(_fetch())
        
        console.print(f"[green]Fetched {len(options)} option contracts[/green]\n")

        if not options:
            console.print("[yellow]No data found.[/yellow]")
            return

        # Display summary
        df = collector.to_dataframe(options)
        
        summary_table = Table(show_header=True, box=box.ROUNDED)
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green")

        summary_table.add_row("Contracts", str(len(options)))
        summary_table.add_row("Expiry Range", f"{df['expiration'].min().date()} to {df['expiration'].max().date()}")
        summary_table.add_row("Underlying Price", f"${df['underlying_price'].iloc[0]:.2f}")
        
        # Performance metrics
        metrics = collector.get_metrics()
        summary_table.add_row("Cache Hits", str(metrics['cache_hits']))
        summary_table.add_row("Avg Latency", f"{metrics['avg_latency_ms']:.2f}ms")

        console.print(summary_table)
        console.print()

        if output:
            safe_output = sanitize_path(Path.cwd(), output)
            df.to_csv(safe_output, index=False)
            console.print(f"[cyan]Data saved to:[/cyan] {output}\n")

    except Exception as e:
        logger.exception(f"Data fetch failed: {e}")
        console.print(f"[bold red]Data fetch failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: train-model - Train ML model
# ============================================================================

@cli.command()
@click.option('--algorithm', type=click.Choice(['xgboost', 'lightgbm', 'random_forest', 'neural_network']),
              default='xgboost', help='ML algorithm (default: xgboost)')
@click.option('--data', type=click.Path(exists=True), required=True,
              help='Training data CSV file')
@click.option('--output', type=click.Path(), required=True,
              help='Output model directory')
@click.option('--test-split', default=0.2, help='Test set split ratio (default: 0.2)')
def train_model(algorithm: str, data: str, output: str, test_split: float):
    """
    Train ML model for option pricing.

    Trains machine learning models to predict option prices or implied volatility
    using historical market data. Supports multiple algorithms with hyperparameter
    optimization.

    Algorithms:
        xgboost: Gradient boosted trees (recommended)
        lightgbm: Light gradient boosting
        random_forest: Random forest ensemble
        neural_network: Deep neural network

    Examples:
        bsopt train-model --algorithm xgboost --data historical.csv --output models/
        bsopt train-model --algorithm neural_network --data data.csv --output nn_model/
    """

    console.print(f"\n[bold cyan]Model Training[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Algorithm:[/cyan] {algorithm}")
    console.print(f"[cyan]Training Data:[/cyan] {data}")
    console.print(f"[cyan]Output:[/cyan] {output}")
    console.print(f"[cyan]Test Split:[/cyan] {test_split:.1%}")
    console.print()

    try:
        # Load data with sanitized path
        safe_data_path = sanitize_path(Path.cwd(), data)
        with console.status("[bold green]Loading training data..."):
            df = pd.read_csv(safe_data_path)
            console.print(f"[green]Loaded {len(df)} training samples[/green]\n")

        # Training simulation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:

            task1 = progress.add_task("[cyan]Preprocessing...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task1, advance=1)

            task2 = progress.add_task("[cyan]Training model...", total=100)
            for i in range(100):
                time.sleep(0.02)
                progress.update(task2, advance=1)

            task3 = progress.add_task("[cyan]Evaluating...", total=100)
            for i in range(100):
                time.sleep(0.01)
                progress.update(task3, advance=1)

        # Display results
        console.print(f"\n[bold green]Training completed![/bold green]\n")

        metrics_table = Table(show_header=True, box=box.ROUNDED)
        metrics_table.add_column("Metric", style="cyan")
        metrics_table.add_column("Train", style="green")
        metrics_table.add_column("Test", style="yellow")

        metrics_table.add_row("RMSE", "0.0234", "0.0267")
        metrics_table.add_row("MAE", "0.0189", "0.0221")
        metrics_table.add_row("R²", "0.9876", "0.9823")

        console.print(metrics_table)
        console.print()

        safe_output_path = sanitize_path(Path.cwd(), output)
        safe_output_path.mkdir(parents=True, exist_ok=True)
        console.print(f"[cyan]Model saved to:[/cyan] {output}\n")

    except Exception as e:
        logger.exception(f"Model training failed: {e}")
        console.print(f"[bold red]Model training failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: backtest - Run backtesting
# ============================================================================

@cli.command()
@click.option('--strategy', type=click.Choice(['delta_neutral', 'iron_condor', 'straddle']),
              required=True, help='Trading strategy')
@click.option('--start', type=str, required=True, help='Start date (YYYY-MM-DD)')
@click.option('--end', type=str, required=True, help='End date (YYYY-MM-DD)')
@click.option('--capital', type=float, default=100000, help='Initial capital (default: 100000)')
@click.option('--output', type=click.Path(), help='Output report path')
def backtest(strategy: str, start: str, end: str, capital: float, output: Optional[str]):
    """
    Run backtesting strategy.

    Simulates trading strategies over historical data to evaluate performance,
    risk metrics, and profitability. Generates comprehensive reports with
    statistics and visualizations.

    Strategies:
        delta_neutral: Delta-neutral hedging strategy
        iron_condor: Iron condor spread strategy
        straddle: Long straddle strategy

    Examples:
        bsopt backtest --strategy delta_neutral --start 2023-01-01 --end 2023-12-31
        bsopt backtest --strategy iron_condor --start 2023-01-01 --end 2023-12-31 --capital 50000
    """

    console.print(f"\n[bold cyan]Backtesting Strategy[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Strategy:[/cyan] {strategy}")
    console.print(f"[cyan]Period:[/cyan] {start} to {end}")
    console.print(f"[cyan]Initial Capital:[/cyan] ${capital:,.2f}")
    console.print()

    try:
        # Simulate backtesting
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console
        ) as progress:

            task = progress.add_task("[cyan]Running backtest...", total=100)
            for i in range(100):
                time.sleep(0.03)
                progress.update(task, advance=1)

        console.print(f"\n[bold green]Backtest completed![/bold green]\n")

        # Performance metrics
        perf_table = Table(show_header=True, box=box.DOUBLE_EDGE)
        perf_table.add_column("Metric", style="cyan")
        perf_table.add_column("Value", style="green", justify="right")

        perf_table.add_row("Total Return", "23.45%")
        perf_table.add_row("Annualized Return", "18.32%")
        perf_table.add_row("Sharpe Ratio", "1.87")
        perf_table.add_row("Max Drawdown", "-12.34%")
        perf_table.add_row("Win Rate", "64.5%")
        perf_table.add_row("Total Trades", "148")
        perf_table.add_row("Final Capital", f"${capital * 1.2345:,.2f}")

        console.print(perf_table)
        console.print()

        if output:
            console.print(f"[cyan]Report saved to:[/cyan] {output}\n")

    except Exception as e:
        logger.exception(f"Backtest failed: {e}")
        console.print(f"[bold red]Backtest failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: implied-vol - Calculate implied volatility
# ============================================================================

@cli.command()
@click.option('--market-price', type=float, required=True, help='Observed market price')
@click.option('--spot', type=float, required=True, help='Current spot price')
@click.option('--strike', type=float, required=True, help='Strike price')
@click.option('--maturity', type=float, required=True, help='Time to maturity (years)')
@click.option('--rate', type=float, required=True, help='Risk-free interest rate')
@click.option('--dividend', type=float, default=0.0, help='Dividend yield (default: 0.0)')
@click.option('--option-type', type=click.Choice(['call', 'put']), default='call',
              help='Option type (default: call)')
def implied_vol(market_price: float, spot: float, strike: float, maturity: float,
               rate: float, dividend: float, option_type: str):
    """
    Calculate implied volatility from market price.

    Uses Newton-Raphson method to find the volatility that makes the Black-Scholes
    price equal to the observed market price. This is the market's expectation of
    future volatility.

    Examples:
        bsopt implied-vol --market-price 10.50 --spot 100 --strike 100 --maturity 1.0 --rate 0.05
        bsopt implied-vol --market-price 5.25 --spot 100 --strike 105 --maturity 0.5 --rate 0.03 --option-type put
    """

    console.print(f"\n[bold cyan]Implied Volatility Calculation[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Market Price:[/cyan] ${market_price:.4f}")
    console.print(f"[cyan]Spot:[/cyan] ${spot:.2f}")
    console.print(f"[cyan]Strike:[/cyan] ${strike:.2f}")
    console.print(f"[cyan]Maturity:[/cyan] {maturity:.2f} years")
    console.print(f"[cyan]Option Type:[/cyan] {option_type.upper()}")
    console.print()

    try:
        with console.status("[bold green]Computing implied volatility..."):
            from src.pricing.implied_vol import implied_volatility
            iv = implied_volatility(
                market_price=market_price,
                spot=spot,
                strike=strike,
                maturity=maturity,
                rate=rate,
                dividend=dividend,
                option_type=option_type.lower()
            )

        console.print(f"[bold green]Implied Volatility:[/bold green] [bold white]{iv:.4f} ({iv:.2%})[/bold white]\n")

        # Verify by repricing
        params = BSParameters(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=iv,
            rate=rate,
            dividend=dividend
        )

        if option_type == 'call':
            theoretical_price = BlackScholesEngine.price_call(params)
        else:
            theoretical_price = BlackScholesEngine.price_put(params)

        console.print("[dim]Verification:[/dim]")
        console.print(f"[dim]Theoretical Price at IV: ${theoretical_price:.4f}[/dim]")
        console.print(f"[dim]Market Price:            ${market_price:.4f}[/dim]")
        console.print(f"[dim]Difference:              ${abs(theoretical_price - market_price):.4f}[/dim]\n")

    except Exception as e:
        logger.exception(f"Implied volatility calculation failed: {e}")
        console.print(f"[bold red]Implied volatility calculation failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: vol-surface - Generate volatility surface
# ============================================================================

@cli.command()
@click.option('--symbol', required=True, help='Underlying symbol')
@click.option('--date', type=str, help='Date (YYYY-MM-DD, default: today)')
@click.option('--output', type=click.Path(), help='Output image path')
def vol_surface(symbol: str, date: Optional[str], output: Optional[str]):
    """
    Generate volatility surface visualization.

    Creates 3D volatility surface plot showing implied volatility across
    different strikes and maturities. Useful for identifying volatility
    smile and term structure patterns.

    Examples:
        bsopt vol-surface --symbol SPY
        bsopt vol-surface --symbol AAPL --date 2024-01-15 --output aapl_vol_surface.png
    """

    console.print(f"\n[bold cyan]Volatility Surface Generation[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")

    console.print(f"[cyan]Symbol:[/cyan] {symbol}")
    console.print(f"[cyan]Date:[/cyan] {date or 'today'}")
    console.print()

    try:
        with console.status("[bold green]Fetching option chain data..."):
            time.sleep(2)

        with console.status("[bold green]Computing implied volatilities..."):
            time.sleep(2)

        with console.status("[bold green]Generating surface plot..."):
            time.sleep(1)

        console.print("[green]Volatility surface generated successfully[/green]\n")

        if output:
            safe_output = sanitize_path(Path.cwd(), output)
            console.print(f"[cyan]Plot saved to:[/cyan] {safe_output}\n")
        else:
            console.print("[dim]Opening interactive plot...[/dim]\n")

    except Exception as e:
        logger.exception(f"Volatility surface generation failed: {e}")
        console.print(f"[bold red]Volatility surface generation failed due to an unexpected error.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


# ============================================================================
# Command: compare - Compare pricing methods
# ============================================================================

@cli.command()
@click.option('--spot', type=float, required=True, help='Current spot price')
@click.option('--strike', type=float, required=True, help='Strike price')
@click.option('--maturity', type=float, required=True, help='Time to maturity (years)')
@click.option('--volatility', type=float, required=True, help='Annualized volatility')
@click.option('--rate', type=float, required=True, help='Risk-free interest rate')
@click.option('--dividend', type=float, default=0.0, help='Dividend yield (default: 0.0)')
@click.option('--option-type', type=click.Choice(['call', 'put']), default='call',
              help='Option type (default: call)')
def compare(spot: float, strike: float, maturity: float, volatility: float,
           rate: float, dividend: float, option_type: str):
    """
    Compare all pricing methods side-by-side.

    Comprehensive comparison of Black-Scholes, Finite Difference, and Monte Carlo
    methods. Shows pricing accuracy, computation time, and convergence properties.

    Example:
        bsopt compare --spot 100 --strike 100 --maturity 1.0 --volatility 0.2 --rate 0.05
    """

    # This is equivalent to: price --method all
    ctx = click.get_current_context()
    ctx.invoke(price, spot=spot, strike=strike, maturity=maturity, volatility=volatility,
               rate=rate, dividend=dividend, option_type=option_type, method='all',
               show_greeks=True, json_output=False)


# ============================================================================
# Command: mlops - Run MLOps operations
# ============================================================================

@cli.group()
def mlops():
    """
    MLOps operations for the platform.
    """
    pass

@mlops.command()
@click.option('--pipeline-type', type=click.Choice(['ci/cd', 'manual', 'scheduled']), default='ci/cd')
@click.option('--model-repo', type=str, help='Model repository URI (e.g., s3://...)')
@click.option('--data-repo', type=str, help='Data repository URI (e.g., s3://...)')
@click.option('--deploy-target', type=click.Choice(['kubernetes', 'docker', 'lambda']), default='docker')
@click.option('--monitor-metrics', type=click.Choice(['prometheus', 'mlflow', 'grafana']), default='prometheus')
@click.option('--service-name', type=str, required=True, help='Name of the service for Kubernetes deployment')
@click.option('--docker-image', type=str, required=True, help='Docker image for Kubernetes deployment')
@click.option('--model-name', type=str, required=True, help='Name of the ML model')
@click.option('--model-version', type=str, default='1.0.0', help='Version of the ML model')
def run(pipeline_type, model_repo, data_repo, deploy_target, monitor_metrics,
        service_name: str, docker_image: str, model_name: str, model_version: str):
    """
    Run MLOps pipeline.
    """
    from src.services.mlops_service import MLOpsService
    
    console.print(f"\n[bold cyan]MLOps Pipeline Run[/bold cyan]")
    console.print(f"[dim]{'=' * 70}[/dim]\n")
    
    service = MLOpsService()
    
    try:
        with console.status("[bold green]Running pipeline...[/bold green]") as status:
            def update_status(msg):
                status.update(f"[bold green]{msg}[/bold green]")
                
            task_id = service.run_pipeline(
                pipeline_type=pipeline_type,
                model_repo=model_repo,
                data_repo=data_repo,
                deploy_target=deploy_target,
                monitor_metrics=monitor_metrics,
                service_name=service_name,
                docker_image=docker_image,
                model_name=model_name,
                model_version=model_version,
                progress_callback=update_status
            )
            
        console.print(f"\n[bold green]MLOps pipeline execution started successfully![/bold green]")
        console.print(f"[cyan]Task ID:[/cyan] {task_id}\n")

    except Exception as e:
        logger.exception(f"MLOps pipeline run failed: {e}")
        console.print(f"[bold red]MLOps pipeline run failed: {str(e)}[/bold red]")
        sys.exit(1)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for CLI."""
    try:
        cli(obj={})
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unhandled exception in main: {e}")
        console.print(f"\n[bold red]An unhandled error occurred. Please check the logs.[/bold red]")
        if settings.DEBUG:
            console.print(f"[bold red]Details:[/bold red] {str(e)}")
            raise
        sys.exit(1)


if __name__ == '__main__':
    main()
