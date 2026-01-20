import pytest
from click.testing import CliRunner
from cli import cli

@pytest.fixture
def runner():
    return CliRunner()

def test_cli_version(runner):
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "2.1.0" in result.output

def test_cli_price_bs(runner):
    result = runner.invoke(cli, [
        "price",
        "--spot", "100",
        "--strike", "100",
        "--maturity", "1.0",
        "--volatility", "0.2",
        "--rate", "0.05",
        "--method", "bs"
    ])
    assert result.exit_code == 0
    assert "Black-Scholes Price" in result.output
    assert "10.45" in result.output

def test_cli_compare(runner):
    # Mock other methods if they take too long, but here we just check if it runs
    result = runner.invoke(cli, [
        "compare",
        "--spot", "100",
        "--strike", "100",
        "--maturity", "0.1", # short for speed
        "--volatility", "0.2",
        "--rate", "0.05"
    ])
    assert result.exit_code == 0
    assert "Method Comparison" in result.output

def test_cli_price_invalid_params(runner):
    # Pass an invalid value that BSParameters or Click will fail on
    # spot=0 should trigger ValueError in BSParameters which calls sys.exit(1)
    result = runner.invoke(cli, [
        "price",
        "--spot", "0", 
        "--strike", "100",
        "--maturity", "1.0",
        "--volatility", "0.2",
        "--rate", "0.05"
    ])
    assert result.exit_code != 0
    assert "Error" in result.output
