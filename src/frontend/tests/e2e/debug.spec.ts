import { test, expect } from '@playwright/test';

test('debug frontend crash', async ({ page }) => {
  page.on('console', msg => console.log('BROWSER CONSOLE:', msg.text()));
  page.on('pageerror', err => console.log('BROWSER ERROR:', err.message));

  await page.goto('/', { waitUntil: 'load', timeout: 60000 });
  await page.waitForTimeout(5000);
  await page.screenshot({ path: 'debug-screenshot.png' });
  
  const content = await page.content();
  console.log('PAGE CONTENT LENGTH:', content.length);
  // console.log('PAGE CONTENT:', content);
});
