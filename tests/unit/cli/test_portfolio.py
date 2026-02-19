from unittest.mock import patch

import pytest

from src.cli.portfolio import PortfolioManager, Position


@pytest.fixture
def portfolio_manager(tmp_path):
    with patch("src.cli.portfolio.Path.home", return_value=tmp_path):
        yield PortfolioManager()


def test_add_list_remove_position(portfolio_manager):
    pos = Position(
        id="test-id",
        symbol="AAPL",
        option_type="call",
        quantity=1,
        strike=150.0,
        maturity=0.5,
        volatility=0.2,
        rate=0.05,
        dividend=0.0,
        entry_price=10.0,
        entry_date="2025-01-01",
        spot=155.0,
    )

    portfolio_manager.add_position(pos)
    assert len(portfolio_manager.list_positions()) == 1

    assert portfolio_manager.remove_position("test-id") is True
    assert len(portfolio_manager.list_positions()) == 0


def test_calculate_position_value(portfolio_manager):
    pos = Position(
        id="test-id",
        symbol="AAPL",
        option_type="call",
        quantity=1,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.0,
        entry_price=5.0,
        entry_date="2025-01-01",
        spot=100.0,
    )

    val = portfolio_manager.calculate_position_value(pos)
    assert val["current_price"] > 0
    assert val["pnl"] is not None


def test_get_portfolio_summary(portfolio_manager):
    pos = Position(
        id="test-id",
        symbol="AAPL",
        option_type="call",
        quantity=1,
        strike=100.0,
        maturity=1.0,
        volatility=0.2,
        rate=0.05,
        dividend=0.0,
        entry_price=5.0,
        entry_date="2025-01-01",
        spot=100.0,
    )
    portfolio_manager.add_position(pos)

    summary = portfolio_manager.get_portfolio_summary()
    assert "pnl" in summary
    assert "greeks" in summary
    assert summary["pnl"]["total_entry_value"] == 5.0 * 100
