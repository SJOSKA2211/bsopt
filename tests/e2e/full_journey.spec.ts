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
 * 7. Perform ML Prediction
 * 8. Deploy ML Model (simulated)
 * 9. Calculate Portfolio Value
 * 10. View Market Data
 * 
 * Run with: npx playwright test tests/e2e/full_journey.spec.ts
 */

import { test, expect, Page } from "@playwright/test";
import time from 'time'; // Assuming time library for date manipulation in tests

// --- Configuration ---
const BASE_URL = process.env.PLAYWRIGHT_TEST_BASE_URL || "http://localhost:8000"; // API service URL
const API_URL = process.env.PLAYWRIGHT_API_URL || "http://localhost:8000/api/v1";

// --- Test Data ---
const testPassword = "SecurePass123!";
const testUserName = "Test User";

// --- Test Suite ---
test.describe("Manifold Full Journey", () => {
  let uniqueEmail: string;
  let portfolio_id: string = ''; // To store created portfolio ID for reuse

  test.beforeAll(async () => {
    uniqueEmail = `user_${Date.now()}@Manifold.test`;
  });

  // --- Authentication Flow ---
  test.describe("Authentication Flow", () => {
    test("should complete sign-up and login flow", async ({ page }) => {
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
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });

      await test.step("Verify dashboard access", async () => {
        await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 10000 });
        const meshStatus = page.locator('[data-testid="mesh-status-indicator"]');
        if (await meshStatus.isVisible()) {
          await expect(meshStatus).toHaveAttribute("data-status", "healthy");
        }
      });

      await test.step("Logout and re-login", async () => {
        await page.goto(`${BASE_URL}/logout`); 
        await page.waitForURL(/\/login/);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });

      await test.step("Verify successful login on dashboard", async () => {
        await expect(page.locator("text=Dashboard")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  // --- Portfolio Management ---
  test.describe("Portfolio Management", () => {
    test.beforeEach(async ({ page }) => {
      await test.step("Login for portfolio tests", async () => {
        await page.goto(`${BASE_URL}/login`);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
      });
    });

    test("should create, get, list, update, and get value of a portfolio", async ({ page, request }) => {
      const initial_cash = 10000.0;
      const updated_cash = 10500.50;
      const portfolio_name_base = `E2E Portfolio ${Date.now()}`;

      await test.step("Create portfolio", async () => {
        await page.click('a:has-text("Portfolios")');
        await page.waitForURL(/\/portfolios/, { timeout: 5000 });
        await page.click('button:has-text("New Portfolio")');
        await expect(page.locator('[name="name"]')).toBeVisible();
        await page.fill('[name="name"]', portfolio_name_base);
        await page.fill('[name="cash"]', initial_cash.toString());
        await page.click('button:has-text("Create")');
        await expect(page.locator("text=Portfolio created")).toBeVisible({ timeout: 10000 });

        const listResponse = await request.get(`${API_URL}/portfolios/`, { headers: { Authorization: `Bearer ${await page.request.storageState().cookies.find(c => c.name === 'access_token')?.value || ''}` }}); // Needs actual token retrieval
        expect(listResponse.ok()).toBeTruthy();
        const portfolios = await listResponse.json();
        const created = portfolios.find(p => p.name.startsWith(portfolio_name_base));
        expect(created).toBeDefined();
        portfolio_id = created.id;
      });

      await test.step("Get portfolio by ID", async () => {
        await page.goto(`${BASE_URL}/portfolios/${portfolio_id}`); 
        await expect(page.locator(`[data-testid="portfolio-name"]`)).toHaveText(portfolio_name);
        await expect(page.locator(`[data-testid="portfolio-cash"]`)).toHaveText(initial_cash.toString());
      });

      await test.step("Update portfolio", async () => {
        await page.click('button:has-text("Edit")'); 
        await page.fill('[name="cash"]', updated_cash.toString());
        await page.click('button:has-text("Save")');
        await expect(page.locator("text=Portfolio updated")).toBeVisible({ timeout: 10000 });
        await page.reload(); 
        await expect(page.locator(`[data-testid="portfolio-cash"]`)).toHaveText(updated_cash.toString());
      });

      await test.step("Get portfolio value", async () => {
        await page.goto(`${BASE_URL}/portfolios/value/${portfolio_id}`); // Navigate to portfolio value page
        await expect(page.locator("h1")).toHaveText("Portfolio Value"); 
        await expect(page.locator("text=total_value")).toBeVisible(); 
        const valueText = await page.locator("text=total_value").textContent(); 
        expect(valueText).toContain("$"); 
      });

      await test.step("List portfolios and verify update", async () => {
        await page.click('a:has-text("Portfolios")');
        await page.waitForURL(/\/portfolios/, { timeout: 5000 });
        await expect(page.locator(`div.portfolio-item:has-text("${portfolio_name} - Updated")`)).toBeVisible();
        await expect(page.locator(`div.portfolio-item:has-text("${updated_cash}")`)).toBeVisible();
      });
    });
  });

  // --- Trading Flow ---
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

    test("should search, select symbol, and execute a buy order", async ({ page }) => {
      await test.step("Search and select symbol", async () => {
        const searchInput = page.locator('[name="symbol"]');
        await searchInput.fill("NIFTY");
        await page.waitForSelector('text=NIFTY', { state: 'visible', timeout: 5000 });
        await page.click('text=NIFTY');
        await page.waitForFunction(
            () => document.querySelector('[name="symbol"]')?.value?.includes('NIFTY'),
            { timeout: 5000 }
        );
      });

      await test.step("Enter order details", async () => {
        await page.fill('[name="quantity"]', "10");
        await page.selectOption('[name="orderType"]', "market"); 
        await page.click('button:has-text("Buy")');
      });

      await test.step("Confirm order", async () => {
        await page.click('button:has-text("Confirm")');
        await expect(page.locator("text=Order submitted")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  // --- ML Pipeline ---
  test.describe("ML Pipeline", () => {
    test.beforeEach(async ({ page }) => {
      await test.step("Login and navigate to ML section", async () => {
        await page.goto(`${BASE_URL}/login`);
        await page.fill('[name="email"]', uniqueEmail);
        await page.fill('[name="password"]', testPassword);
        await page.click('[type="submit"]');
        await page.waitForURL(/\/dashboard/, { timeout: 15000 });
        await page.click('a:has-text("ML Models")');
        await page.waitForURL(/\/ml\/models/, { timeout: 5000 });
      });
    });

    test("should create, predict with, trigger training for, and deploy an ML model", async ({ page, request }) => {
      let model_id = '';
      const model_name = `E2E Model ${Date.now()}`;
      const model_version = '1.0.0';
      const model_description = 'E2E Test Model';
      
      await test.step("Create an ML model", async () => {
        await page.click('button:has-text("New ML Model")');
        await expect(page.locator('[name="name"]')).toBeVisible();
        await page.fill('[name="name"]', model_name);
        await page.fill('[name="version"]', model_version);
        await page.fill('[name="description"]', model_description);
        await page.click('button:has-text("Create")');
        await expect(page.locator(`text=${model_name}`)).toBeVisible({ timeout: 10000 });

        const modelsResponse = await request.get(`${API_URL}/ml/models`, { headers: { Authorization: `Bearer ${await page.request.storageState().cookies.find(c => c.name === 'access_token')?.value || ''}` }}); // Needs actual token
        expect(modelsResponse.ok()).toBeTruthy();
        const models = await modelsResponse.json();
        const created_model = models.find(m => m.name === model_name && m.version === model_version);
        expect(created_model).toBeDefined();
        model_id = created_model.id;
      });

      await test.step("Predict using the created model", async () => {
        await page.goto(`${BASE_URL}/ml/predict/${model_id}`); 
        await expect(page.locator("h2")).toHaveText("Predict"); 
        await page.fill('[name="inputValue"]', "123.45"); 
        await page.click('button:has-text("Predict")');
        await expect(page.locator("text=prediction")).toBeVisible({ timeout: 10000 });
        const predictionResult = await page.locator('pre').textContent();
        expect(predictionResult).toContain("simulated_result");
      });

      await test.step("Trigger model training", async () => {
        await page.click('a:has-text("ML Models")'); 
        await page.waitForURL(/\/ml\/models/, { timeout: 5000 });
        await page.click(`button:has-text("Train") >> nth=0`); 
        await expect(page.locator('[name="epochs"]')).toBeVisible();
        await page.fill('[name="epochs"]', "50");
        await page.fill('[name="batchSize"]', "128");
        await page.click('button:has-text("Start Training")');
        await expect(page.locator("text=ML training task enqueued")).toBeVisible({ timeout: 10000 });
      });

      await test.step("Trigger model deployment", async () => {
        await page.click('button:has-text("Deploy")'); 
        await expect(page.locator('[name="version"]')).toBeVisible();
        await page.fill('[name="version"]', "1.0.0"); 
        await page.selectOption('[name="targetEnvironment"]', "production"); 
        await page.click('button:has-text("Deploy Model")');
        await expect(page.locator("text=ML model deployment task enqueued")).toBeVisible({ timeout: 10000 });
      });
    });
  });

  // --- API Health Checks ---
  test.describe("API Health Checks", () => {
    test("should verify API health endpoint", async ({ request }) => {
      const response = await request.get(`${API_URL}/health`);
      expect(response.ok()).toBeTruthy();

      const data = await response.json();
      expect(data.status).toBe("healthy");
    });

    test("should verify ML service health (simulated)", async ({ request }) => {
      const response = await request.get(`${API_URL}/ml/models`); 
      expect(response.ok()).toBeTruthy();
    });

    test("should verify Market Data service health", async ({ request }) => {
        const response = await request.get(`${API_URL}/market/historical?symbol=TEST&startDate=2023-01-01&endDate=2023-01-01`);
        // Expecting success or potentially a specific health check endpoint if available
        // For now, checking if the endpoint is reachable and returns some response
        expect(response.ok()).toBeTruthy(); 
    });
  });

  // --- Error Handling ---
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
        const errors = page.locator('input:invalid, [aria-invalid="true"], .error-message');
        await expect(errors.first()).toBeVisible({ timeout: 5000 });
      });
    });
  });

  // --- WebSocket Real-time Updates ---
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
        await page.waitForSelector('text=/Connected|Live Data|Market Feed/i', { state: 'visible', timeout: 10000 });
      });

      await test.step("Verify WebSocket connection indicator", async () => {
        const wsStatus = page.locator('text=/Connected|Live|Online/i');
        await expect(wsStatus).toBeVisible({ timeout: 5000 });
      });
    });
  });
});
