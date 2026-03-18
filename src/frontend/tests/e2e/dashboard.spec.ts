import { test, expect } from '@playwright/test';

test.describe('Dashboard UI & Component Validation', () => {
  test('should load the dashboard and verify key widgets', async ({ page }) => {
    // Navigate to the root (dashboard)
    await page.goto('/', { waitUntil: 'networkidle', timeout: 120000 });

    // Verify the "Salutations" text
    await expect(page.locator('text=Salutations')).toBeVisible({ timeout: 60000 });
    
    // Verify the "Arch-Quant" text with shimmer effect
    await expect(page.locator('text=Arch-Quant')).toBeVisible();

    // Verify key KPI cards are present as placeholders/empty states or with simulated data
    await expect(page.locator('text=Portfolio Oracle')).toBeVisible();
    await expect(page.locator('text=Systemic Gamma')).toBeVisible();
    await expect(page.locator('text=Predictive Accuracy')).toBeVisible();

    // verify the "Human vs Machine" Comparison Dashboard exists
    await expect(page.locator('text=Human vs Machine')).toBeVisible({ timeout: 20000 });
    await expect(page.locator('text=Real-time Alpha Execution Comparison')).toBeVisible();

    // Verify the AI Oracle section in the comparison table
    await expect(page.locator('text=AI ORACLE')).toBeVisible();
    await expect(page.locator('text=YOUR STRATEGY')).toBeVisible();
  });

  test('should navigate to Market page and verify layout', async ({ page }) => {
    await page.goto('/', { waitUntil: 'networkidle' });
    
    // Find the navigation or use direct URL for now if nav is lazy/async
    await page.goto('/market', { waitUntil: 'networkidle' });
    
    await expect(page.locator('text=Market Data')).toBeVisible({ timeout: 60000 });
    await expect(page.locator('text=Options Chain')).toBeVisible();
    await expect(page.locator('text=Greeks')).toBeVisible();
  });
});
