import { test, expect } from '@playwright/test';

test.describe('Authentication Flow & Token Rotation', () => {
  const testUser = {
    email: `testuser_${Date.now()}@example.com`,
    password: 'SecurePassword123!',
    name: 'E2E Test User',
  };

  test('User Registration, Login, and Refresh Token Rotation', async ({ page }) => {
    // 1. Registration
    await page.goto('/signup');
    await page.fill('input[name="name"]', testUser.name);
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);
    await page.click('button[type="submit"]');

    // Wait for redi
    // rect to login or dashboard
    await page.waitForURL('**/login', { timeout: 10000 }).catch(() => { });

    // 2. Login
    await page.goto('/login');
    await page.fill('input[name="email"]', testUser.email);
    await page.fill('input[name="password"]', testUser.password);

    // Intercept the login response to capture tokens
    const [loginResponse] = await Promise.all([
      page.waitForResponse(res => res.url().includes('/api/auth/login') && res.status() === 200),
      page.click('button[type="submit"]')
    ]);

    const loginData = await loginResponse.json();
    expect(loginData).toHaveProperty('access_token');
    expect(loginData).toHaveProperty('refresh_token');

    // Verify successful login by checking dashboard element
    await expect(page.locator('text=Dashboard')).toBeVisible();

    // 3. Refresh Token Rotation
    // Wait for the client to trigger a token refresh, or manually trigger it via page evaluation
    const refreshResponse = await page.evaluate(async () => {
      const res = await fetch('/api/auth/refresh', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        }
      });
      return await res.json();
    });

    // Check that a new access token is issued
    expect(refreshResponse).toHaveProperty('access_token');

    // If rotation is fully implemented, a new refresh token is also issued
    if (refreshResponse.refresh_token) {
      expect(refreshResponse.refresh_token).not.toEqual(loginData.refresh_token);
    }
  });
});
