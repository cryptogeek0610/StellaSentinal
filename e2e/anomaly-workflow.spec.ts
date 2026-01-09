import { test, expect } from '@playwright/test';

test.describe('Anomaly Detection Workflow', () => {
  test('complete anomaly triage flow', async ({ page }) => {
    await page.goto('/dashboard');
    await expect(page.locator('[data-testid="dashboard-stats"]')).toBeVisible();

    await page.click('[data-testid="view-anomalies"]');
    await expect(page).toHaveURL(/\/investigations/);

    await page.click('[data-testid="anomaly-row"]:first-child');
    await expect(page.locator('[data-testid="investigation-panel"]')).toBeVisible();

    await page.click('[data-testid="resolve-button"]');
    await page.fill('[data-testid="resolution-notes"]', 'Test resolution');
    await page.click('[data-testid="confirm-resolve"]');

    await expect(page.locator('[data-testid="status-badge"]')).toHaveText('Resolved');
  });

  test('cost management workflow', async ({ page }) => {
    await page.goto('/costs');
    await expect(page.locator('[data-testid="cost-summary"]')).toBeVisible();

    await page.click('[data-testid="add-hardware-cost"]');
    await page.fill('[data-testid="cost-amount"]', '250.00');
    await page.selectOption('[data-testid="cost-category"]', 'repair');
    await page.click('[data-testid="save-cost"]');

    await expect(page.locator('text=250.00')).toBeVisible();
  });
});
