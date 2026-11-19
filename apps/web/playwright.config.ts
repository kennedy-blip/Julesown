import path from 'node:path';
import { defineConfig, devices } from '@playwright/test';

/**
 * The dashboard in a real browser, against the real API on local stand-ins
 * (apps/api/tests/standin_server.py): no account, no key, nothing leaves the
 * machine. `npm run test:e2e` builds the dashboard, starts both and runs the
 * tests; servers that are already running on the two ports are used as they are.
 */
const API_PORT = Number(process.env.E2E_API_PORT || 8011);
const WEB_PORT = Number(process.env.E2E_WEB_PORT || 3011);
const repositoryRoot = path.resolve(__dirname, '../..');
// The interpreter that has apps/api/requirements-dev.txt installed.
const python = process.env.E2E_PYTHON || 'python3';
// A Chromium that is already on the machine, where Playwright's own cannot be downloaded.
const chromium = process.env.PLAYWRIGHT_CHROMIUM_PATH;

export default defineConfig({
  testDir: './tests/e2e',
  // The tests share one backend and start it from scratch each time.
  fullyParallel: false,
  workers: 1,
  forbidOnly: !!process.env.CI,
  retries: process.env.CI ? 1 : 0,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
  timeout: 60_000,
  expect: { timeout: 10_000 },
  outputDir: './test-results',
  use: {
    baseURL: `http://127.0.0.1:${WEB_PORT}`,
    trace: 'retain-on-failure',
    screenshot: 'only-on-failure',
  },
  projects: [
    {
      name: 'dashboard',
      use: {
        ...devices['Desktop Chrome'],
        launchOptions: chromium ? { executablePath: chromium } : {},
      },
    },
  ],
  webServer: [
    {
      command: `${python} -m apps.api.tests.standin_server --port ${API_PORT}`,
      cwd: repositoryRoot,
      url: `http://127.0.0.1:${API_PORT}/api/v1/health`,
      reuseExistingServer: !process.env.CI,
      timeout: 60_000,
    },
    {
      // The address of the API is part of the build (next.config.mjs).
      command: `npm run build && npm run start -- -p ${WEB_PORT} -H 127.0.0.1`,
      cwd: __dirname,
      env: { API_BASE_URL: `http://127.0.0.1:${API_PORT}` },
      url: `http://127.0.0.1:${WEB_PORT}`,
      reuseExistingServer: !process.env.CI,
      timeout: 300_000,
    },
  ],
});
