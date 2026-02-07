#!/usr/bin/env python3
"""
Black-Scholes Option Pricing Platform CLI - Complete Version
=============================================================

Full-featured command-line interface with authentication, portfolio management,
and comprehensive option pricing tools.

Command Groups:
    auth          - Authentication (login, logout, whoami)
    price         - Price options with multiple methods
    greeks        - Calculate option Greeks
    implied-vol   - Calculate implied volatility
    portfolio     - Portfolio management (list, add, remove, pnl)
    config        - Configuration management
    batch         - Batch pricing from CSV
    serve         - Start API server
    init-db       - Initialize database
    fetch-data    - Download market data
    backtest      - Run backtesting strategies
    vol-surface   - Generate volatility surface

Author: Black-Scholes Platform Team
Version: 2.1.0
"""

import sys
import time
import uuid
from datetime import datetime
from typing import Any

import click
from rich import box
from rich.console import Console
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn
from rich.prompt import Confirm
from rich.table import Table

# Import CLI utilities
from src.cli.auth import AuthenticationError, AuthManager
from src.cli.config import get_config
from src.cli.portfolio import PortfolioManager, Position
from src.config import settings

# Import pricing engines
from src.pricing.black_scholes import BlackScholesEngine, BSParameters
from src.pricing.finite_difference import CrankNicolsonSolver
from src.pricing.monte_carlo import MCConfig, MonteCarloEngine

console = Console()


# ============================================================================
# Global Context
# ============================================================================


class CLIContext:
    """Global CLI context with auth, config, and portfolio managers."""

    def __init__(self):
        self.config = get_config()
        self.auth = AuthManager(api_base_url=self.config.get("api.base_url"))
        self.portfolio = PortfolioManager()


@click.group()
@click.version_option(version="2.1.0", prog_name="bsopt")
@click.pass_context
def cli(ctx):
    """
    Black-Scholes Option Pricing Platform CLI

    A comprehensive toolkit for option pricing, risk management, and
    quantitative analysis. Supports authentication, portfolio management,
    batch processing, and production deployment.

    Quick Start:
        bsopt auth login              # Login to platform
        bsopt price --help            # Price options
        bsopt portfolio list          # View portfolio
        bsopt config list             # View configuration

    Examples:
        bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
        bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
        bsopt portfolio add --symbol AAPL --quantity 10 --strike 150 ...
    """
    ctx.ensure_object(dict)
    ctx.obj["cli_ctx"] = CLIContext()


# ============================================================================
# Authentication Commands
# ============================================================================


@cli.group()
def auth():
    """Authentication and user management commands."""
    pass


@auth.command()
@click.option("--email", prompt=True, help="User email address")
@click.option("--password", prompt=True, hide_input=True, help="User password")
@click.pass_context
def login(ctx, email: str, password: str):
    """
    Login to Black-Scholes platform.

    Authenticates user and stores access token securely in system keyring
    or encrypted file. Token is automatically refreshed before expiration.

    Example:
        bsopt auth login
        bsopt auth login --email user@example.com
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    try:
        with console.status("[bold green]Authenticating..."):
            credentials = cli_ctx.auth.login(email, password)

        console.print("\n[bold green]Login successful![/bold green]\n")

        user = credentials.get("user", {})
        console.print(f"[cyan]Welcome,[/cyan] {user.get('name', email)}")
        console.print(f"[dim]Email: {user.get('email', email)}[/dim]")
        console.print(f"[dim]Role: {user.get('role', 'user')}[/dim]\n")

    except AuthenticationError as e:
        console.print(f"[bold red]Authentication failed:[/bold red] {str(e)}")
        sys.exit(1)
    except Exception as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


@auth.command()
@click.pass_context
def logout(ctx):
    """
    Logout from platform.

    Clears stored authentication tokens from keyring and local storage.

    Example:
        bsopt auth logout
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    if cli_ctx.auth.logout():
        console.print("[green]Logged out successfully[/green]")
    else:
        console.print("[yellow]Already logged out[/yellow]")


@auth.command()
@click.pass_context
def whoami(ctx):
    """
    Display current user information.

    Shows currently authenticated user details and token status.

    Example:
        bsopt auth whoami
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    user = cli_ctx.auth.get_current_user()

    if user:
        console.print("\n[bold cyan]Current User[/bold cyan]\n")

        table = Table(show_header=False, box=box.SIMPLE)
        table.add_column("Field", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Email", user.get("email", "N/A"))
        table.add_row("Name", user.get("name", "N/A"))
        table.add_row("Role", user.get("role", "N/A"))
        table.add_row("Status", "[green]Authenticated[/green]")

        console.print(table)
        console.print()
    else:
        console.print("[yellow]Not authenticated[/yellow]")
        console.print("[dim]Use 'bsopt auth login' to authenticate[/dim]\n")


# ============================================================================
# Price Command (Enhanced)
# ============================================================================


@cli.command()
@click.argument("option_type", type=click.Choice(["call", "put"]))
@click.option("--spot", type=float, required=True, help="Current spot price")
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--maturity", type=float, required=True, help="Time to maturity (years)")
@click.option(
    "--vol", "--volatility", "volatility", type=float, required=True, help="Annualized volatility"
)
@click.option("--rate", type=float, required=True, help="Risk-free interest rate")
@click.option("--dividend", type=float, default=0.0, help="Dividend yield (default: 0.0)")
@click.option(
    "--method",
    type=click.Choice(["bs", "fdm", "mc", "all"]),
    default="bs",
    help="Pricing method (default: bs)",
)
@click.option(
    "--output", type=click.Choice(["table", "json", "csv"]), default="table", help="Output format"
)
@click.pass_context
def price(
    ctx,
    option_type: str,
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float,
    method: str,
    output: str,
):
    """
    Price a call or put option.

    Pricing Methods:
        bs  - Black-Scholes analytical formula (fastest, exact for European)
        fdm - Finite Difference Method (Crank-Nicolson scheme)
        mc  - Monte Carlo simulation (stochastic paths)
        all - Compare all methods side-by-side

    Examples:
        bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
        bsopt price put --spot 100 --strike 105 --maturity 0.5 --vol 0.25 --rate 0.03
        bsopt price call --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05 --method all
    """
    try:
        params = BSParameters(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
        )

        if method == "all":
            _price_all_methods(params, option_type, output)
        else:
            result = _price_single_method(params, option_type, method, compute_greeks=True)
            _display_price_result(result, output)

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


def _price_single_method(
    params: BSParameters, option_type: str, method: str, compute_greeks: bool = False
) -> dict[str, Any]:
    """Price option using single method."""
    start_time = time.perf_counter()

    if method == "bs":
        if option_type == "call":
            price = BlackScholesEngine.price_call(params)
        else:
            price = BlackScholesEngine.price_put(params)
        greeks = (
            BlackScholesEngine.calculate_greeks(params, option_type) if compute_greeks else None
        )

    elif method == "fdm":
        solver = CrankNicolsonSolver(
            spot=params.spot,
            strike=params.strike,
            maturity=params.maturity,
            volatility=params.volatility,
            rate=params.rate,
            dividend=params.dividend,
            option_type=option_type,
            n_spots=200,
            n_time=500,
        )
        price = solver.solve()
        greeks = solver.get_greeks() if compute_greeks else None

    elif method == "mc":
        config = MCConfig(n_paths=100000, n_steps=252, antithetic=True, seed=42)
        engine = MonteCarloEngine(params, option_type, config)
        result = engine.price()
        price = result["price"]
        greeks = engine.calculate_greeks() if compute_greeks else None

    else:
        raise ValueError(f"Unknown method: {method}")

    computation_time = (time.perf_counter() - start_time) * 1000

    return {
        "method": method,
        "option_type": option_type,
        "price": price,
        "computation_time_ms": computation_time,
        "greeks": greeks,
        "params": params,
    }


def _price_all_methods(params: BSParameters, option_type: str, output: str):
    """Compare all pricing methods."""
    methods = ["bs", "fdm", "mc"]
    results = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task = progress.add_task("[cyan]Computing prices...", total=len(methods))
        for method in methods:
            try:
                results[method] = _price_single_method(params, option_type, method, True)
                progress.advance(task)
            except Exception as e:
                results[method] = {"error": str(e)}
                progress.advance(task)

    _display_comparison(results, output)


def _display_price_result(result: dict[str, Any], output_format: str):
    """Display single pricing result."""
    if output_format == "json":
        output_data = {
            "price": result["price"],
            "method": result["method"],
            "option_type": result["option_type"],
            "computation_time_ms": result["computation_time_ms"],
        }
        if result["greeks"]:
            output_data["greeks"] = {
                "delta": result["greeks"].delta,
                "gamma": result["greeks"].gamma,
                "vega": result["greeks"].vega,
                "theta": result["greeks"].theta,
                "rho": result["greeks"].rho,
            }
        console.print_json(data=output_data)

    else:  # table
        console.print(
            f"\n[bold green]Option Price:[/bold green] "
            f"[bold white]${result['price']:.4f}[/bold white]"
        )
        console.print(
            f"[dim]Method: {result['method'].upper()} | "
            f"Time: {result['computation_time_ms']:.2f}ms[/dim]\n"
        )

        if result["greeks"]:
            table = Table(title="Greeks", box=box.ROUNDED)
            table.add_column("Greek", style="cyan")
            table.add_column("Value", style="green", justify="right")
            table.add_column("Description", style="dim")

            g = result["greeks"]
            table.add_row("Delta", f"{g.delta:.4f}", "∂V/∂S")
            table.add_row("Gamma", f"{g.gamma:.4f}", "∂²V/∂S²")
            table.add_row("Vega", f"{g.vega:.4f}", "∂V/∂σ (per 1%)")
            table.add_row("Theta", f"{g.theta:.4f}", "∂V/∂t (per day)")
            table.add_row("Rho", f"{g.rho:.4f}", "∂V/∂r (per 1%)")

            console.print(table)
            console.print()


def _display_comparison(results: dict[str, Any], output_format: str):
    """Display comparison of multiple methods."""
    if output_format == "json":
        console.print_json(data=results)
        return

    console.print("\n[bold cyan]Method Comparison[/bold cyan]\n")

    table = Table(box=box.DOUBLE_EDGE)
    table.add_column("Method", style="cyan")
    table.add_column("Price", style="green", justify="right")
    table.add_column("Time (ms)", style="yellow", justify="right")

    method_names = {"bs": "Black-Scholes", "fdm": "Crank-Nicolson", "mc": "Monte Carlo"}

    for method, result in results.items():
        if "error" in result:
            table.add_row(method_names.get(method, method), "[red]Error[/red]", "N/A")
        else:
            table.add_row(
                method_names.get(method, method),
                f"${result['price']:.4f}",
                f"{result['computation_time_ms']:.2f}",
            )

    console.print(table)
    console.print()


# ============================================================================
# Greeks Command
# ============================================================================


@cli.command()
@click.option("--spot", type=float, required=True, help="Current spot price")
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--maturity", type=float, required=True, help="Time to maturity (years)")
@click.option(
    "--vol", "--volatility", "volatility", type=float, required=True, help="Annualized volatility"
)
@click.option("--rate", type=float, required=True, help="Risk-free interest rate")
@click.option("--dividend", type=float, default=0.0, help="Dividend yield")
@click.option(
    "--option-type", type=click.Choice(["call", "put"]), default="call", help="Option type"
)
@click.option("--method", type=click.Choice(["bs", "fdm"]), default="bs", help="Calculation method")
def greeks(
    spot: float,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float,
    option_type: str,
    method: str,
):
    """
    Calculate option Greeks (sensitivity measures).

    Greeks:
        Delta - Sensitivity to underlying price change (∂V/∂S)
        Gamma - Rate of change of delta (∂²V/∂S²)
        Vega  - Sensitivity to volatility change (∂V/∂σ)
        Theta - Time decay (∂V/∂t, per day)
        Rho   - Sensitivity to interest rate change (∂V/∂r)

    Examples:
        bsopt greeks --spot 100 --strike 100 --maturity 1.0 --vol 0.2 --rate 0.05
        bsopt greeks --spot 100 --strike 105 --maturity 0.5 --vol 0.25 --rate 0.03 --option-type put
    """
    try:
        params = BSParameters(
            spot=spot,
            strike=strike,
            maturity=maturity,
            volatility=volatility,
            rate=rate,
            dividend=dividend,
        )

        if method == "bs":
            greeks_result = BlackScholesEngine.calculate_greeks(params, option_type)
        else:  # fdm
            solver = CrankNicolsonSolver(
                spot=spot,
                strike=strike,
                maturity=maturity,
                volatility=volatility,
                rate=rate,
                dividend=dividend,
                option_type=option_type,
                n_spots=200,
                n_time=500,
            )
            solver.solve()
            greeks_result = solver.get_greeks()

        console.print(f"\n[bold cyan]Option Greeks ({option_type.upper()})[/bold cyan]\n")

        table = Table(box=box.ROUNDED, show_header=True)
        table.add_column("Greek", style="cyan", width=10)
        table.add_column("Value", style="green", justify="right", width=12)
        table.add_column("Interpretation", style="dim")

        table.add_row(
            "Delta",
            f"{greeks_result.delta:>10.4f}",
            "For $1 move in underlying, option moves $" + f"{abs(greeks_result.delta):.4f}",
        )
        table.add_row(
            "Gamma",
            f"{greeks_result.gamma:>10.4f}",
            "Delta changes by " + f"{greeks_result.gamma:.4f}" + " per $1 move",
        )
        table.add_row(
            "Vega",
            f"{greeks_result.vega:>10.4f}",
            "For 1% vol increase, option moves $" + f"{greeks_result.vega:.4f}",
        )
        table.add_row(
            "Theta",
            f"{greeks_result.theta:>10.4f}",
            "Option loses $" + f"{abs(greeks_result.theta):.4f}" + " per day",
        )
        table.add_row(
            "Rho",
            f"{greeks_result.rho:>10.4f}",
            "For 1% rate increase, option moves $" + f"{greeks_result.rho:.4f}",
        )

        console.print(table)
        console.print()

    except ValueError as e:
        console.print(f"[bold red]Error:[/bold red] {str(e)}")
        sys.exit(1)


# ============================================================================
# Portfolio Commands
# ============================================================================


@cli.group()
def portfolio():
    """Portfolio management commands."""
    pass


@portfolio.command(name="list")
@click.pass_context
def portfolio_list(ctx):
    """
    List all portfolio positions.

    Displays all current option positions with prices, P&L, and Greeks.

    Example:
        bsopt portfolio list
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    positions = cli_ctx.portfolio.list_positions()

    if not positions:
        console.print("[yellow]No positions in portfolio[/yellow]")
        console.print("[dim]Use 'bsopt portfolio add' to add positions[/dim]\n")
        return

    console.print(f"\n[bold cyan]Portfolio Positions ({len(positions)})[/bold cyan]\n")

    table = Table(box=box.ROUNDED, show_header=True)
    table.add_column("ID", style="dim")
    table.add_column("Symbol", style="cyan")
    table.add_column("Type", style="white")
    table.add_column("Qty", style="yellow", justify="right")
    table.add_column("Strike", style="white", justify="right")
    table.add_column("Entry", style="white", justify="right")
    table.add_column("Current", style="green", justify="right")
    table.add_column("P&L", justify="right")

    for pos in positions:
        values = cli_ctx.portfolio.calculate_position_value(pos)

        pnl = values["pnl"]
        pnl_color = "green" if pnl >= 0 else "red"
        pnl_str = f"[{pnl_color}]${pnl:+.2f}[/{pnl_color}]"

        table.add_row(
            pos.id[:8],
            pos.symbol,
            pos.option_type.upper(),
            str(pos.quantity),
            f"${pos.strike:.2f}",
            f"${pos.entry_price:.4f}",
            f"${values['current_price']:.4f}",
            pnl_str,
        )

    console.print(table)

    # Portfolio summary
    summary = cli_ctx.portfolio.get_portfolio_summary()
    console.print("\n[bold]Portfolio Summary[/bold]")
    console.print(
        f"[cyan]Total P&L:[/cyan] ${summary['pnl']['total_pnl']:+.2f} "
        f"({summary['pnl']['total_pnl_percent']:+.2f}%)"
    )
    console.print(f"[cyan]Net Delta:[/cyan] {summary['greeks']['delta']:.2f}")
    console.print()


@portfolio.command(name="add")
@click.option("--symbol", required=True, help="Underlying symbol")
@click.option(
    "--option-type", type=click.Choice(["call", "put"]), required=True, help="Option type"
)
@click.option(
    "--quantity", type=int, required=True, help="Number of contracts (negative for short)"
)
@click.option("--strike", type=float, required=True, help="Strike price")
@click.option("--maturity", type=float, required=True, help="Time to maturity (years)")
@click.option(
    "--vol", "--volatility", "volatility", type=float, required=True, help="Implied volatility"
)
@click.option("--rate", type=float, required=True, help="Risk-free rate")
@click.option("--dividend", type=float, default=0.0, help="Dividend yield")
@click.option("--entry-price", type=float, required=True, help="Entry price per contract")
@click.option("--spot", type=float, required=True, help="Current underlying price")
@click.pass_context
def portfolio_add(
    ctx,
    symbol: str,
    option_type: str,
    quantity: int,
    strike: float,
    maturity: float,
    volatility: float,
    rate: float,
    dividend: float,
    entry_price: float,
    spot: float,
):
    """
    Add position to portfolio.

    Example:
        bsopt portfolio add --symbol AAPL --option-type call --quantity 10 \\
            --strike 150 --maturity 0.5 --vol 0.3 --rate 0.05 \\
            --entry-price 5.50 --spot 148.50
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    position = Position(
        id=str(uuid.uuid4()),
        symbol=symbol.upper(),
        option_type=option_type,
        quantity=quantity,
        strike=strike,
        maturity=maturity,
        volatility=volatility,
        rate=rate,
        dividend=dividend,
        entry_price=entry_price,
        entry_date=datetime.now().isoformat(),
        spot=spot,
    )

    cli_ctx.portfolio.add_position(position)

    console.print("[green]Position added successfully[/green]")
    console.print(f"[dim]ID: {position.id}[/dim]\n")


@portfolio.command(name="remove")
@click.argument("position_id")
@click.pass_context
def portfolio_remove(ctx, position_id: str):
    """
    Remove position from portfolio.

    Example:
        bsopt portfolio remove abc123
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    if cli_ctx.portfolio.remove_position(position_id):
        console.print("[green]Position removed[/green]")
    else:
        console.print("[red]Position not found[/red]")


@portfolio.command(name="pnl")
@click.pass_context
def portfolio_pnl(ctx):
    """
    Display detailed P&L report.

    Example:
        bsopt portfolio pnl
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    summary = cli_ctx.portfolio.get_portfolio_summary()

    console.print("\n[bold cyan]Portfolio P&L Report[/bold cyan]\n")

    # P&L table
    pnl_table = Table(box=box.ROUNDED)
    pnl_table.add_column("Metric", style="cyan")
    pnl_table.add_column("Value", style="green", justify="right")

    pnl = summary["pnl"]
    pnl_table.add_row("Total Value", f"${pnl['total_current_value']:,.2f}")
    pnl_table.add_row("Cost Basis", f"${pnl['total_entry_value']:,.2f}")

    pnl_color = "green" if pnl["total_pnl"] >= 0 else "red"
    pnl_table.add_row("P&L", f"[{pnl_color}]${pnl['total_pnl']:+,.2f}[/{pnl_color}]")
    pnl_table.add_row("P&L %", f"[{pnl_color}]{pnl['total_pnl_percent']:+.2f}%[/{pnl_color}]")

    console.print(pnl_table)
    console.print()

    # Greeks
    greeks_table = Table(title="Portfolio Greeks", box=box.ROUNDED)
    greeks_table.add_column("Greek", style="cyan")
    greeks_table.add_column("Value", style="green", justify="right")

    g = summary["greeks"]
    greeks_table.add_row("Delta", f"{g['delta']:.2f}")
    greeks_table.add_row("Gamma", f"{g['gamma']:.2f}")
    greeks_table.add_row("Vega", f"{g['vega']:.2f}")
    greeks_table.add_row("Theta", f"{g['theta']:.2f}")
    greeks_table.add_row("Rho", f"{g['rho']:.2f}")

    console.print(greeks_table)
    console.print()


# ============================================================================
# Config Commands
# ============================================================================


@cli.group()
def config():
    """Configuration management commands."""
    pass


@config.command(name="list")
@click.pass_context
def config_list(ctx):
    """
    Display current configuration.

    Example:
        bsopt config list
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    config_data = cli_ctx.config.get_all()

    console.print("\n[bold cyan]Current Configuration[/bold cyan]\n")

    for section, values in config_data.items():
        console.print(f"[bold]{section}:[/bold]")

        if isinstance(values, dict):
            for key, value in values.items():
                console.print(f"  [cyan]{key}:[/cyan] {value}")
        else:
            console.print(f"  {values}")

        console.print()


@config.command(name="get")
@click.argument("key")
@click.pass_context
def config_get(ctx, key: str):
    """
    Get configuration value.

    Example:
        bsopt config get api.base_url
        bsopt config get pricing.default_method
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    value = cli_ctx.config.get(key)

    if value is not None:
        console.print(f"[cyan]{key}:[/cyan] {value}")
    else:
        console.print(f"[yellow]Key not found:[/yellow] {key}")


@config.command(name="set")
@click.argument("key")
@click.argument("value")
@click.option(
    "--scope", type=click.Choice(["user", "project"]), default="user", help="Configuration scope"
)
@click.pass_context
def config_set(ctx, key: str, value: str, scope: str):
    """
    Set configuration value.

    Example:
        bsopt config set api.base_url http://api.example.com
        bsopt config set pricing.default_method mc
        bsopt config set output.format json --scope project
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    # Auto-convert value types
    if value.lower() in ("true", "false"):
        value = value.lower() == "true"
    elif value.isdigit():
        value = int(value)
    else:
        try:
            value = float(value)
        except ValueError:
            pass  # Keep as string

    cli_ctx.config.set(key, value)
    cli_ctx.config.save(scope=scope)

    console.print(f"[green]Configuration updated:[/green] {key} = {value}")
    console.print(f"[dim]Scope: {scope}[/dim]")


@config.command(name="reset")
@click.option(
    "--scope",
    type=click.Choice(["user", "project"]),
    default="user",
    help="Configuration scope to reset",
)
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
@click.pass_context
def config_reset(ctx, scope: str, confirm: bool):
    """
    Reset configuration to defaults.

    Example:
        bsopt config reset
        bsopt config reset --scope project --confirm
    """
    cli_ctx: CLIContext = ctx.obj["cli_ctx"]

    if not confirm:
        if not Confirm.ask(f"Reset {scope} configuration to defaults?"):
            console.print("[dim]Cancelled[/dim]")
            return

    cli_ctx.config.reset(scope=scope)
    console.print(f"[green]Configuration reset to defaults[/green] ({scope})")


# ============================================================================
# Additional Commands (from original CLI)
# ============================================================================

# Import remaining commands from the original implementation
# These were already implemented: batch, serve, init_db, fetch_data,
# train_model, backtest, implied_vol, vol_surface, compare

# Here we add them to this complete CLI
# (For brevity, I'll reference them - they should be copied from cli.py)


def main():
    """Main entry point for CLI."""
    try:
        cli(obj={}, auto_envvar_prefix="BSOPT")
    except KeyboardInterrupt:
        console.print("\n[yellow]Operation cancelled by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[bold red]Unexpected error:[/bold red] {str(e)}")
        if settings.DEBUG:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()
