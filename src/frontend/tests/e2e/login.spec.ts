import { test, expect } from '@playwright/test';

test.describe('Login Flow', () => {
  test('should successfully log in with existing credentials', async ({ page }) => {
    // Generate a unique email and password for the user we will create first
    const uniqueEmail = `testuser_login_${Date.now()}@example.com`;
    const password = 'SecurePass123!';

    // 1. Sign up first to ensure the user exists
    await page.goto('/signup', { waitUntil: 'networkidle', timeout: 120000 });
    await expect(page.getByRole('heading', { name: /Create an account/i })).toBeVisible({ timeout: 90000 });
    await page.fill('input[name="name"]', 'Login Test User');
    await page.fill('input[name="email"]', uniqueEmail);
    await page.fill('input[name="password"]', password);
    await page.getByRole('button', { name: /Sign Up/i }).click();

    // Wait for the redirect to login page after successful signup
    await expect(page).toHaveURL(/\/login/, { timeout: 20000 });

    // 2. Perform login
    await expect(page.getByRole('heading', { name: /Welcome back/i })).toBeVisible({ timeout: 20000 });

    // Based on SignIn.tsx:
    // Email TextField has id="email"
    // Password TextField has id="password"
    await page.fill('#email', uniqueEmail);
    await page.fill('#password', password);

    // Click the Sign In button
    await page.getByRole('button', { name: /Sign In/i }).click();

    // Expect success message or redirect to dashboard
    // SignIn.tsx shows "Signed in successfully!" Alert on success
    await expect(page.locator('text=Signed in successfully!')).toBeVisible({ timeout: 10000 });

    // Check for redirection to dashboard (root or /dashboard)
    // For now we check if we're not on /login anymore or if we see a dashboard indicator
    await expect(page).not.toHaveURL(/\/login/, { timeout: 10000 });
  });

  test('should show error on invalid credentials', async ({ page }) => {
    await page.goto('/login', { waitUntil: 'networkidle' });

    await page.fill('#email', 'nonexistent@example.com');
    await page.fill('#password', 'WrongPassword123!');
    await page.getByRole('button', { name: /Sign In/i }).click();

    // Based on authClient.signIn.email callbacks in SignIn.tsx
    // The error message is displayed in an Alert
    await expect(page.getByRole('alert')).toBeVisible({ timeout: 10000 });
  });
});
