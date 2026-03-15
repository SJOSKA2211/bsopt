import { test, expect } from '@playwright/test';

test.describe('EquaFlow Institutional Auth Flow', () => {
    test('should allow a quant user to login and access the dashboard', async ({ page }) => {
        // 1. Navigate to Entry Point (Envoy Gateway)
        await page.goto('http://localhost:8080/login');

        // 2. Perform Login
        await page.fill('input[name="username"]', 'quant_admin');
        await page.fill('input[name="password"]', 'EquaFlow2026!');
        await page.click('button[type="submit"]');

        // 3. Verify ZTNA Token Cookie / Redirect
        await expect(page).toHaveURL(/.*dashboard/);
        
        // 4. Verify gRPC-backed data is loading on dashboard
        const tickerGrid = page.locator('#ticker-firehose');
        await expect(tickerGrid).toBeVisible();
    });

    test('should block unauthorized access to math kernels', async ({ page }) => {
        const response = await page.goto('http://localhost:8080/api/v1/compute/black-scholes');
        expect(response?.status()).toBe(401);
    });
});
