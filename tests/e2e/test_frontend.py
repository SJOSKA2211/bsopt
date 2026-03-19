import pytest
from playwright.sync_api import Page, expect

@pytest.mark.e2e
def test_dashboard_load(page: Page):
    """Verify the primary dashboard loads with all institutional components."""
    # Note: Using localhost:8000 as a placeholder for the integrated frontend/API
    page.goto("http://localhost:8000")
    
    # Check title
    expect(page).to_have_title("EquaFlow")
    
    # Verify core UI components
    expect(page.get_by_role("heading", name="Quantitative Terminal")).to_be_visible()
    expect(page.get_by_text("Real-time Tickers")).to_be_visible()
    expect(page.get_by_text("Option Chain")).to_be_visible()

@pytest.mark.e2e
def test_authentication_flow(page: Page):
    """Verify the zero-trust authentication flow."""
    page.goto("http://localhost:8000/login")
    
    # Fill login form
    page.get_by_label("Email").fill("quant@equaflow.io")
    page.get_by_label("Password").fill("argon2_secure_password")
    page.get_by_role("button", name="Institutional Login").click()
    
    # Should redirect to dashboard
    expect(page).to_have_url("http://localhost:8000/dashboard")
    expect(page.get_by_text("Session Active")).to_be_visible()

@pytest.mark.e2e
def test_realtime_websocket_updates(page: Page):
    """Verify Protobuf-encoded WebSocket updates are rendered correctly."""
    page.goto("http://localhost:8000/dashboard")
    
    # Wait for the first ticker update
    ticker_row = page.get_by_role("row", name="AAPL")
    expect(ticker_row).to_be_visible(timeout=10000)
    
    # Verify price is updating (look for flash or value change)
    price_cell = ticker_row.locator("td.price-cell")
    initial_price = price_cell.inner_text()
    
    # Wait for change
    expect(price_cell).not_to_have_text(initial_price, timeout=5000)
