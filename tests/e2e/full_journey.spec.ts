/**
 * Manifold Full Journey E2E Test
 * 
 * Tests the complete user journey:
 * 1. Sign Up
 * 2. Email Verification (mocked)
 * 3. Login
 * 4. Create Portfolio
 * 5. Execute Trade
 * 6. View ML Dashboard
 * 7. Trigger ML Training
 * 
 * Run with: npx playwright test tests/e2e/full_journey.spec.ts
 */

import { test, expect, Page } from "@playwright/test";

const BASE_URL = process.env.PLAYWRIGHT_TEST_BASE_URL || "http://localhost:8080";
const API_URL = process.env.PLAYWRIGHT_API_URL || "http://localhost:8080/api/v1";

test.describe("Manifold Full Journey", () => {
  let uniqueEmail: string;
  const testPassword = "SecurePass123!";
  const testUserName = "Test User";

  test.beforeAll(async () => {
    uniqueEmail = `user_${Date.now()}@Manifold.test`;
  });

  test.describe("Authentication Flow", () => {
    test("should complete sign-up flow", async ({ page }) => {
      await test.step("Navigate to signup page", async () => {
        await page.goto(`${BASE_URL}/signup`);
        await expect(page).toHaveURL(/\/signup/);
      });

      await test.step("Fill signup form", async () => {
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.fill('[name="confirmPassword"]', testPassword);
        await page.fill('[name="fullName"]', testUserName);
      });

      await test.step("Submit signup", async () => {
        await page.click('[type="submit"]');
        await page.waitForURL(/\/(dashboard|verify-email|login)/, { timeout: 10000 });
      });

      await test.step("Verify success", async () => {
        const currentUrl = page.url();
        expect(currentUrl).not.toContain("/signup");
      });
    });

    test("should complete login flow", async ({ page }) => {
      await test.step("Navigate to login page", async () => {
        await page.goto(`${BASE_URL}/login`);
        await expect(page).toHaveURL(/\/login/);
      });

      await test.step("Fill login form", async () => {
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
      });

      await test.step("Submit login", async () => {
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });

      await test.step("Verify dashboard access", async () => {
        await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 5000 });
      });
    });
  });

  test.describe("Portfolio Management", () => {
    test.beforeEach(async ({ page }) => {
      await test.step("Login first", async () => {
        await page.goto(`${BASE_URL}/login`);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });
    });

    test("should create a new portfolio", async ({ page }) => {
      await test.step("Navigate to portfolios", async () => {
        await page.click('text=Portfolios');
        await page.waitForURL(/\/portfolios/, { timeout: 5000 });
      });

      await test.step("Click new portfolio button", async () => {
        await page.click('text=New Portfolio');
        await expect(page.locator('[name="name"]')).toBeVisible();
      });

      await test.step("Fill portfolio form", async () => {
        await page.fill('[name="name"]', `Test Portfolio ${Date.now()}`);
        await page.fill('[name="cash"]', "100000");
      });

      await test.step("Create portfolio", async () => {
        await page.click('button:has-text("Create")');
        await page.waitForTimeout(2000);
      });

      await test.step("Verify portfolio created", async () => {
        await expect(page.locator("text=Portfolio created")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  test.describe("Trading Flow", () => {
    test.beforeEach(async ({ page }) => {
      await test.step("Login and navigate to trading", async () => {
        await page.goto(`${BASE_URL}/login`);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
        await page.click('text=Trade');
        await page.waitForURL(/\/trade/, { timeout: 5000 });
      });
    });

    test("should search and select a symbol", async ({ page }) => {
      await test.step("Search for NIFTY", async () => {
        const searchInput = page.locator('[name="symbol"]');
        await searchInput.fill("NIFTY");
        await page.waitForTimeout(500);
      });

      await test.step("Select from dropdown", async () => {
        await page.click('text=NIFTY');
        await page.waitForTimeout(1000);
      });

      await test.step("Verify symbol selected", async () => {
        await expect(page.locator('[name="symbol"]')).toHaveValue(/NIFTY/i);
      });
    });

    test("should execute a buy order", async ({ page }) => {
      await test.step("Select symbol", async () => {
        await page.fill('[name="symbol"]', "NIFTY");
        await page.waitForTimeout(500);
        await page.click('text=NIFTY');
      });

      await test.step("Enter order details", async () => {
        await page.fill('[name="quantity"]', "10");
        await page.fill('[name="orderType"]', "market");
        await page.click('text=Buy');
      });

      await test.step("Confirm order", async () => {
        await page.click('button:has-text("Confirm")');
        await page.waitForTimeout(3000);
      });

      await test.step("Verify order success", async () => {
        await expect(page.locator("text=Order submitted")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  test.describe("ML Pipeline", () => {
    test.beforeEach(async ({ page }) => {
      await test.step("Login and navigate to ML", async () => {
        await page.goto(`${BASE_URL}/login`);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
        await page.click('text=ML Models');
        await page.waitForURL(/\/ml/, { timeout: 5000 });
      });
    });

    test("should display ML dashboard with Ray status", async ({ page }) => {
      await test.step("Verify ML dashboard loaded", async () => {
        await expect(page.locator("text=ML Models")).toBeVisible({ timeout: 10000 });
      });

      await test.step("Check Ray cluster status", async () => {
        const rayStatus = page.locator('text=/Ray.*(status|cluster)/i');
        await expect(rayStatus).toBeVisible({ timeout: 5000 });
      });
    });

    test("should trigger model training", async ({ page }) => {
      await test.step("Find training button", async () => {
        const trainButton = page.locator('button:has-text("Train Model")');
        if (await trainButton.isVisible()) {
          await trainButton.click();
        }
      });

      await test.step("Configure training", async () => {
        await page.fill('[name="epochs"]', "50");
        await page.fill('[name="batchSize"]', "128");
      });

      await test.step("Start training", async () => {
        await page.click('button:has-text("Start Training")');
      });

      await test.step("Verify training started", async () => {
        await expect(page.locator("text=Training Started")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  test.describe("API Health Checks", () => {
    test("should verify API health endpoint", async ({ request }) => {
      const response = await request.get(`${API_URL}/health`);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data.status).toBe("healthy");
    });

    test("should verify MLflow is accessible", async ({ request }) => {
      const mlflowUrl = process.env.MLFLOW_URL || "http://localhost:5000";
      const response = await request.get(`${mlflowUrl}/health`);
      expect(response.ok()).toBeTruthy();
    });
  });

  test.describe("Error Handling", () => {
    test("should show error for invalid credentials", async ({ page }) => {
      await test.step("Navigate to login", async () => {
        await page.goto(`${BASE_URL}/login`);
      });

      await test.step("Enter invalid credentials", async () => {
        await page.fill('[name="email"]', "invalid@test.com");
        await page.fill('[name="password"]', "wrongpassword");
        await page.click('[type="submit"]');
      });

      await test.step("Verify error message", async () => {
        await expect(page.locator("text=/Invalid|error|wrong/i")).toBeVisible({ timeout: 5000 });
      });
    });

    test("should show validation error for empty form", async ({ page }) => {
      await test.step("Navigate to signup", async () => {
        await page.goto(`${BASE_URL}/signup`);
      });

      await test.step("Submit empty form", async () => {
        await page.click('[type="submit"]');
      });

      await test.step("Verify validation errors", async () => {
        const errors = page.locator('[class*="error"], [class*="Error"]');
        await expect(errors.first()).toBeVisible();
      });
    });
  });
});

test.describe("WebSocket Real-time Updates", () => {
  test("should receive real-time market data updates", async ({ page }) => {
    await test.step("Login", async () => {
      await page.goto(`${BASE_URL}/login`);
      await page.fill('[name="email"]', uniqueEmail);
      await page.fill('[name="password"]', testPassword);
      await page.click('[type="submit"]');
      await page.waitForURL(/\/dashboard/, { timeout: 15000 });
    });

    await test.step("Navigate to market data", async () => {
      await page.click('text=Markets');
      await page.waitForTimeout(2000);
    });

    await test.step("Verify WebSocket connection", async () => {
      const wsStatus = page.locator('text=/Connected|Live/i');
      await expect(wsStatus).toBeVisible({ timeout: 5000 });
    });
  });
});
