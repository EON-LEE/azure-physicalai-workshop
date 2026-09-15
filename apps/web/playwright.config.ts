import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/browser',
  testMatch: '*.spec.ts',
  fullyParallel: false,
  workers: 1,
  retries: 0,
  timeout: 30_000,
  expect: { timeout: 8000 },
  reporter: [['list'], ['html', { open: 'never' }]],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:4178',
    viewport: { width: 1440, height: 1060 },
    locale: 'ko-KR',
    timezoneId: 'Asia/Seoul',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run test:fixture-server',
    url: 'http://127.0.0.1:4178',
    timeout: 120_000,
    reuseExistingServer: false,
  },
});
