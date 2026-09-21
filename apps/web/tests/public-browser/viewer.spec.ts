import { expect, test } from '@playwright/test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const reference = JSON.parse(readFileSync(join(process.cwd(), '../../examples/inspection-cell.json'), 'utf8'));
const snapshot = {
  api_version: 'public-demo-v1', access: 'public_read_only', deployment: 'azure',
  mode: 'reference', observed_at: new Date().toISOString(),
  scene: { id: 'inspection-cell-v1', name: 'Inspection and sorting cell', length_unit: 'm', stations: reference.stations, robot: 'Franka reference arm', data_origin: 'synthetic_reference_configuration' },
  simulation: { status: 'not_published', live_available: false, message_code: 'live_not_published', frame_url: null },
  agent: { provider: 'microsoft_foundry', connectivity: 'configured', verified_at: null, verification_scope: 'connectivity_only' },
  learning: { status: 'cpu_smoke_verified', execution_location: 'azure_acr', data_kind: 'test_fixture', optimizer_steps: 1, quality_verified: false },
  capabilities: { anonymous_control: false, anonymous_editing: false, public_live_video: false },
};

test('the actual production entry opens a public viewer with no auth, private requests or writes', async ({ page }) => {
  const requests: string[] = [];
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('request', (request) => requests.push(`${request.method()} ${new URL(request.url()).pathname}`));
  await page.route('**/api/demo', (route) => route.fulfill({ json: snapshot }));
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '비전 검사 · 부품 분류' })).toBeVisible();
  await expect(page.getByRole('button', { name: 'Microsoft로 로그인' })).toHaveCount(0);
  await expect(page.getByText('작업 셀 구성도 · 실제 시뮬레이션 아님')).toBeVisible();
  const normal = page.getByRole('button', { name: '양품 흐름' });
  await normal.focus();
  await page.keyboard.press('Enter');
  await expect(normal).toHaveAttribute('aria-pressed', 'true');
  await page.getByRole('button', { name: /04.*결과를 다시/ }).click();
  await expect(page.getByText('설명 경로: 검사 → 다음 공정')).toBeVisible();
  expect(requests.filter((item) => item.includes('/api/'))).toEqual(['GET /api/demo']);
  expect(requests.some((item) => /msal-|OperatorEntry-/.test(item))).toBe(false);
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/public-viewer-reference-preview.png', fullPage: true });
});

test('public navigation remains usable at mobile width and reduced motion', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.route('**/api/demo', (route) => route.fulfill({ json: snapshot }));
  await page.goto('/?path=rejected&step=sort');
  await expect(page.getByRole('button', { name: '불량 격리' })).toHaveAttribute('aria-pressed', 'true');
  await expect(page.getByText('설명 경로: 검사 → 불량 격리 트레이')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/public-viewer-mobile-reference-preview.png', fullPage: true });
});

test('operator entry is still explicitly authenticated', async ({ page }) => {
  const requests: string[] = [];
  page.on('request', (request) => requests.push(new URL(request.url()).pathname));
  await page.route('**/api/config', (route) => route.fulfill({ status: 503, json: { error: { code: 'offline', message: 'Test-only operator configuration failure.' } } }));
  await page.goto('/operator');
  await expect.poll(() => requests.includes('/api/config')).toBe(true);
  expect(requests.includes('/api/demo')).toBe(false);
  await expect(page.getByRole('heading', { name: '비전 검사 · 부품 분류' })).toHaveCount(0);
});
