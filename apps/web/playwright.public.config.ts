import { defineConfig, devices } from '@playwright/test';

export default defineConfig({
  testDir: './tests/public-browser',
  workers: 1,
  retries: 0,
  timeout: 30000,
  expect: { timeout: 8000 },
  outputDir: 'test-results/public-browser',
  reporter: [['list'], ['junit', { outputFile: 'test-results/public-browser.junit.xml' }]],
  use: {
    ...devices['Desktop Chrome'],
    baseURL: 'http://127.0.0.1:4179',
    viewport: { width: 1440, height: 1060 },
    locale: 'ko-KR',
    screenshot: 'only-on-failure',
    trace: 'retain-on-failure',
  },
  webServer: {
    command: 'npm run build && vite preview --host 127.0.0.1 --port 4179 --strictPort',
    url: 'http://127.0.0.1:4179',
    timeout: 120000,
    reuseExistingServer: false,
  },
});
