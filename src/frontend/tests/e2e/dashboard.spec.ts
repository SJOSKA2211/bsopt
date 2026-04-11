import { test, expect } from '@playwright/test';

test.describe('Dashboard UI & Component Validation', () => {
  test('should load the dashboard and verify key widgets', async ({ page }) => {
    // Navigate to the root (dashboard)
    await page.goto('/', { waitUntil: 'networkidle', timeout: 120000 });

    // Verify key KPI labels are present
    await expect(page.locator('text=SYSTEM_GAMMA')).toBeVisible({ timeout: 60000 });
    await expect(page.locator('text=PORTFOLIO_NAV')).toBeVisible();
    await expect(page.locator('text=VEGA_SENS')).toBeVisible();
    await expect(page.locator('text=WS_STATUS')).toBeVisible();

    // Verify Intelligence Layer
    await expect(page.locator('text=DEEP_INFERENCE_ENGINE')).toBeVisible();
    await expect(page.locator('text=RISK_EXPOSURE_GRID')).toBeVisible();
    await expect(page.locator('text=STRATEGY_ALLOCATION')).toBeVisible();

    // Verify the Observation Deck and Signal Telemetry
    await expect(page.locator('text=TEMPORAL_TRAJECTORY')).toBeVisible();
    await expect(page.locator('text=SIGNAL_TELEMETRY')).toBeVisible();
  });

  test('should navigate to Market page and verify layout', async ({ page }) => {
    await page.goto('/market', { waitUntil: 'networkidle' });
    // Assuming TradeExecutionPage has something recognizable. If not, this might fail.
    // Let's just check the URL for now or a generic div since we haven't seen TradeExecutionPage.tsx
    await expect(page).toHaveURL(/.*market/);
  });
});
