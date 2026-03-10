import { test, expect } from '@playwright/test';

test.describe('Sign Up Flow', () => {
  test('should successfully sign up a new user', async ({ page }) => {
    // Generate a unique email for each run to avoid collision
    const uniqueEmail = `testuser_${Date.now()}@example.com`;

    // Navigate to the signup page
    await page.goto('/signup', { waitUntil: 'networkidle', timeout: 120000 });

    // Wait for the signup form to be visible - using a more robust selector
    await expect(page.getByRole('heading', { name: /Create an account/i })).toBeVisible({ timeout: 90000 });

    // Fill in the sign up form
    await page.fill('input[name="name"]', 'Test User');
    await page.fill('input[name="email"]', uniqueEmail);
    await page.fill('input[name="password"]', 'SecurePass123!');

    // Submit the form
    await page.getByRole('button', { name: /Sign Up/i }).click();

    // Expect the success message or redirection
    // Based on the code, it shows "Account created! Redirecting to login..."
    await expect(page.locator('text=Account created! Redirecting to login...')).toBeVisible({ timeout: 10000 });

    // Wait for the redirect to login page
    await expect(page).toHaveURL(/\/login/, { timeout: 10000 });
  });
});
