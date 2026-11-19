import { defineConfig } from '@playwright/test';

/** The logic in lib/, without a browser and without a server. */
export default defineConfig({
  testDir: './tests/unit',
  fullyParallel: true,
  forbidOnly: !!process.env.CI,
  reporter: process.env.CI ? [['github'], ['list']] : 'list',
});
