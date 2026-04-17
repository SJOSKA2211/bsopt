/**
 * Manifold Full Journey E2E Test
 * 
 * Tests the complete user journey:
 * 1. Sign Up
 * 2. Login
 * 3. Create Portfolio
 * 4. Execute Trade
 * 5. View ML Dashboard
 * 6. Trigger ML Training
 * 
 * Run with: npx playwright test tests/e2e/full_journey.spec.ts
 */

import { test, expect, Page } from "@playwright/test";

const BASE_URL = process.env.PLAYWRIGHT_TEST_BASE_URL || "http://localhost:8000"; // Adjusted base URL to match API service
const API_URL = process.env.PLAYWRIGHT_API_URL || "http://localhost:8000/api/v1";

test.describe("Manifold Full Journey", () => {
  let uniqueEmail: string;
  const testPassword = "SecurePass123!";
  const testUserName = "Test User";

  test.beforeAll(async () => {
    uniqueEmail = `user_${Date.now()}@Manifold.test`;
  });

  test.describe("Authentication Flow", () => {
    test("should complete sign-up and login flow", async ({ page }) => {
      await test.step("Navigate to signup page", async () => {
        await page.goto(`${BASE_URL}/signup`); // Assuming signup is handled by frontend
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
        // After signup, assume direct redirect to dashboard for test users,
        // bypassing explicit email verification for test automation.
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });

      await test.step("Verify dashboard access", async () => {
        await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 10000 });
      });

      await test.step("Login with newly created user", async () => {
        // If the signup automatically logs in, this step might be redundant or need adjustment.
        // Assuming a logout/re-login for a clean test of the login flow itself.
        // If signup auto-logs in, this block could be removed or adapted.
        await page.goto(`${BASE_URL}/logout`); // Assuming logout endpoint exists
        await page.waitForURL(/\/login/);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });

      await test.step("Verify successful login on dashboard", async () => {
        await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 10000 });
        // Check for specific UI indicators of gRPC-backend connectivity
        const meshStatus = page.locator('[data-testid="mesh-status-indicator"]');
        if (await meshStatus.isVisible()) {
          await expect(meshStatus).toHaveAttribute("data-status", "healthy");
        }
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
      // Assume navigation to portfolios is handled within the page content or a fixture
    });

    test("should create a new portfolio", async ({ page }) => {
      await test.step("Navigate to portfolios", async () => {
        // Use a more robust selector if 'text=Portfolios' is not stable
        await page.click('a:has-text("Portfolios")');
        await page.waitForURL(/\/portfolios/, { timeout: 5000 });
      });

      await test.step("Click new portfolio button", async () => {
        await page.click('button:has-text("New Portfolio")');
        await expect(page.locator('[name="name"]')).toBeVisible();
      });

      await test.step("Fill portfolio form", async () => {
        await page.fill('[name="name"]', `Test Portfolio ${Date.now()}`);
        await page.fill('[name="cash"]', "100000");
      });

      await test.step("Create portfolio", async () => {
        await page.click('button:has-text("Create")');
        // Wait for success message or navigation to portfolio list/detail
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
        await page.click('a:has-text("Trade")');
        await page.waitForURL(/\/trade/, { timeout: 5000 });
      });
    });

    test("should search and select a symbol", async ({ page }) => {
      await test.step("Search for NIFTY", async () => {
        const searchInput = page.locator('[name="symbol"]');
        await searchInput.fill("NIFTY");
        // Wait for search results to appear (e.g., a dropdown)
        await page.waitForSelector('text=NIFTY', { state: 'visible', timeout: 5000 });
      });

      await test.step("Select from dropdown", async () => {
        await page.click('text=NIFTY');
        await page.waitForFunction(
            () => document.querySelector('[name="symbol"]')?.value?.includes('NIFTY'),
            { timeout: 5000 }
        );
      });

      await test.step("Verify symbol selected", async () => {
        await expect(page.locator('[name="symbol"]')).toHaveValue(/NIFTY/i);
      });
    });

    test("should execute a buy order", async ({ page }) => {
      await test.step("Select symbol", async () => {
        await page.fill('[name="symbol"]', "NIFTY");
        await page.waitForSelector('text=NIFTY', { state: 'visible', timeout: 5000 });
        await page.click('text=NIFTY');
        await page.waitForFunction(
            () => document.querySelector('[name="symbol"]')?.value?.includes('NIFTY'),
            { timeout: 5000 }
        );
      });

      await test.step("Enter order details", async () => {
        await page.fill('[name="quantity"]', "10");
        await page.fill('[name="orderType"]', "market"); // Assuming 'market' is a valid type
        await page.click('button:has-text("Buy")');
      });

      await test.step("Confirm order", async () => {
        await page.click('button:has-text("Confirm")');
        // Wait for success message or navigation to order history
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
        await page.click('a:has-text("ML Models")');
        await page.waitForURL(/\/ml/, { timeout: 5000 });
      });
    });

    test("should display ML dashboard with Ray status", async ({ page }) => {
      await test.step("Verify ML dashboard loaded", async () => {
        await expect(page.locator("text=ML Models")).toBeVisible({ timeout: 10000 });
      });

      await test.step("Check Ray cluster status", async () => {
        // This check assumes a specific UI element or text indicating Ray status.
        // Adjust selector based on actual frontend implementation.
        const rayStatus = page.locator('text=/Ray.*(status|cluster)/i');
        await expect(rayStatus).toBeVisible({ timeout: 5000 });
      });
    });

    test("should trigger model training", async ({ page }) => {
      await test.step("Find training button", async () => {
        const trainButton = page.locator('button:has-text("Train Model")');
        await expect(trainButton).toBeVisible({ timeout: 5000 });
        await trainButton.click();
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

    // This test assumes MLflow is running and accessible externally or internally.
    // If MLflow is part of the docker-compose, it should be available via service discovery.
    test("should verify MLflow is accessible", async ({ request }) => {
      // Dynamically determine MLflow URL. Assume it's in env or a known service name.
      const mlflowUrl = process.env.MLFLOW_URL || "http://localhost:5000"; // Default if not set
      try {
        const response = await request.get(`${mlflowUrl}/api/2.0/ml/health`); // MLflow health endpoint
        expect(response.ok()).toBeTruthy();
      } catch (error) {
        // If MLflow is not running or accessible, this test might fail.
        // Depending on requirements, this could be ignored or marked as skipped.
        console.warn(`MLflow health check failed: ${error.message}`);
        // expect(true).toBe(true); // Mark test as passed if it's optional or expected to fail in some environments
        throw error; // Re-throw to fail test if MLflow is critical and expected to be up
      }
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
        // Use a more specific selector for error messages if possible
        await expect(page.locator("text=/Invalid.*credentials|error/i")).toBeVisible({ timeout: 5000 });
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
        // Look for common validation error indicators
        const errors = page.locator('input:invalid, [aria-invalid="true"], .error-message');
        await expect(errors.first()).toBeVisible({ timeout: 5000 });
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
      await page.click('a:has-text("Markets")');
      // Wait for WebSocket connection indicator or initial data load
      await page.waitForSelector('text=/Connected|Live Data|Market Feed/i', { state: 'visible', timeout: 10000 });
    });

    await test.step("Verify WebSocket connection indicator", async () => {
      const wsStatus = page.locator('text=/Connected|Live|Online/i');
      await expect(wsStatus).toBeVisible({ timeout: 5000 });
    });
  });
});
