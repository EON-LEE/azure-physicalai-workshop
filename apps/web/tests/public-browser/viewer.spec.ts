import { expect, test, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
import type { DemoSnapshot } from '../../src/public/api';
import { epoch, makePresentation, makeSnapshot, nextEpoch, recordedCases, referenceSnapshot } from '../fixtures/public';

async function fixturePage(page: Page, readSnapshot: () => DemoSnapshot, options: { wrongEpoch?: boolean; wrongEvidence?: boolean } = {}) {
  const requests: Array<{ method: string; path: string; authorization?: string }> = [];
  const errors: string[] = [];
  let servedSnapshot = readSnapshot();
  const records = recordedCases();
  const png = await page.evaluate(() => {
    const canvas = document.createElement('canvas');
    canvas.width = 1280;
    canvas.height = 720;
    const context = canvas.getContext('2d');
    if (!context) throw new Error('Test PNG canvas unavailable');
    context.fillStyle = '#172840';
    context.fillRect(0, 0, 1280, 720);
    context.strokeStyle = '#345575';
    context.lineWidth = 2;
    context.strokeRect(70, 70, 1140, 580);
    context.fillStyle = '#d0e3ff';
    context.textAlign = 'center';
    context.font = 'bold 30px sans-serif';
    context.fillText('PUBLIC_PRESENTATION_TEST_FIXTURE', 640, 310);
    context.font = '20px sans-serif';
    context.fillText('NOT AZURE / GPU / FOUNDRY VERIFICATION', 640, 365);
    context.font = '17px sans-serif';
    context.fillText('Static test PNG. Production has no fixture image or replacement camera.', 640, 408);
    return canvas.toDataURL('image/png').split(',')[1] ?? '';
  });
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('request', (request) => {
    const url = new URL(request.url());
    if (url.protocol === 'http:' || url.protocol === 'https:') requests.push({ method: request.method(), path: url.pathname, authorization: request.headers().authorization });
  });
  await page.route('https://**/*', (route) => route.abort('blockedbyclient'));
  await page.route('**/*', async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    if (path === '/test-only-csp.js') {
      return route.fulfill({
        contentType: 'application/javascript',
        body: "try { new Function('return 1')(); document.body.dataset.cspEvalBlocked = 'false'; } catch { document.body.dataset.cspEvalBlocked = 'true'; }",
      });
    }
    if (path === '/test-only-fixture.css') {
      return route.fulfill({
        contentType: 'text/css',
        body: '.public-fixture-banner{background:#ffedb7;color:#633a08;border-bottom:2px solid #c68f21;text-align:center;padding:10px 14px;font:600 12px/1.6 system-ui,sans-serif;overflow-wrap:anywhere}',
      });
    }
    if (request.resourceType() !== 'document') return route.continue();
    const response = await route.fetch();
    const html = (await response.text()).replace('</head>', '<link rel="stylesheet" href="/test-only-fixture.css"></head>')
      .replace('<body>', '<body><div class="public-fixture-banner" role="note">TEST-ONLY FIXTURES · 브라우저 동작 검증용 · Azure / GPU / 실제 Foundry 검증 아님</div>');
    return route.fulfill({ response, body: html });
  });
  await page.route('**/api/**', async (route) => {
    const url = new URL(route.request().url());
    if (url.pathname === '/api/demo') {
      servedSnapshot = readSnapshot();
      return route.fulfill({ json: servedSnapshot, headers: { 'Cache-Control': 'no-store' } });
    }
    if (url.pathname === '/api/demo/cases') return route.fulfill({ json: records });
    if (url.pathname === '/api/demo/cases/evidence') {
      const item = records.cases.find(value => value.observation_id === url.searchParams.get('observation_id'));
      if (!item || !records.presentation_id || url.searchParams.get('presentation_id') !== records.presentation_id) {
        return route.fulfill({ status: 409, json: { error: { code: 'public_case_changed', message: 'TEST ONLY: different publication' } } });
      }
      return route.fulfill({
        body: Buffer.from(png, 'base64'), contentType: 'image/png',
        headers: { 'X-Frame-Id': item.observation_id, 'X-Captured-At': item.captured_at, 'X-Presentation-Id': records.presentation_id },
      });
    }
    const decision = servedSnapshot.presentation?.decision;
    if (url.pathname === '/api/demo/frame') {
      const now = new Date().toISOString();
      return route.fulfill({
        body: Buffer.from(png, 'base64'), contentType: 'image/png',
        headers: {
          'Cache-Control': 'no-store',
          'X-Frame-Id': 'test-only-public-live-frame',
          'X-Scene-Epoch': options.wrongEpoch ? nextEpoch : url.searchParams.get('epoch') ?? epoch,
          'X-Captured-At': now, 'X-Server-Time': now, 'X-Physics-Steps': '125',
        },
      });
    }
    if (url.pathname === '/api/demo/evidence' && decision) {
      return route.fulfill({
        body: Buffer.from(png, 'base64'), contentType: 'image/png',
        headers: { 'Cache-Control': 'no-store', 'X-Frame-Id': options.wrongEvidence ? nextEpoch : decision.observation_id, 'X-Captured-At': decision.captured_at },
      });
    }
    return route.fulfill({ status: 503, json: { error: { code: 'test_only_unavailable', message: 'TEST ONLY: protected configuration not supplied.' } } });
  });
  return { requests, errors };
}

test('production public entry shows same-cycle evidence and live state without auth, writes or private modules', async ({ page }) => {
  const presentation = makePresentation();
  const { requests, errors } = await fixturePage(page, () => makeSnapshot({ presentation }));
  await page.goto('/?view=demo');
  await expect(page.getByRole('note')).toContainText('실제 Foundry 검증 아님');
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toBeVisible();
  await expect(page.getByRole('img', { name: /현재 Foundry 판단에 실제 사용된/ })).toBeVisible();
  await expect(page.getByText('서버 입력: 표면 흠집 부품')).toBeVisible();
  await expect(page.getByText('불량으로 판단', { exact: true })).toBeVisible();
  await expect(page.getByText('대상 트레이로 이동', { exact: true })).toBeVisible();
  await expect(page.getByText('종합 성공 확인', { exact: true })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '양품 흐름' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Microsoft로 로그인' })).toHaveCount(0);
  for (const request of requests.filter((item) => item.path.startsWith('/api/'))) {
    expect(['/api/demo', '/api/demo/frame', '/api/demo/evidence']).toContain(request.path);
    expect(request.method).toBe('GET');
    expect(request.authorization).toBeUndefined();
  }
  expect(requests.some((item) => /msal-|OperatorEntry-/.test(item.path))).toBe(false);
  expect(errors).toEqual([]);
  await page.screenshot({ path: 'test-results/public-moving-TEST-FIXTURE-not-azure.png', fullPage: true });
});

test('customer can compare recorded business outcomes and download a real experiment without controlling the robot', async ({ page }) => {
  const { requests, errors } = await fixturePage(page, () => makeSnapshot({ presentation: makePresentation({ status: 'failed' }) }));
  await page.goto('/?viewing=paused');
  await expect(page.getByRole('heading', { name: '알림이 아니라 작업 완료' })).toBeVisible();
  await page.getByRole('button', { name: '실제 실행 기록 3가지 비교' }).click();
  await expect(page.getByText('정상 트레이 도착 측정됨')).toBeVisible();
  await expect(page.getByText('격리 트레이 도착 측정됨')).toBeVisible();
  await expect(page.getByText('검사 불일치 · 이동 승인 및 실행 없음')).toBeVisible();
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toHaveCount(0);
  await page.getByRole('radio', { name: /격리 트레이를 10 cm/ }).check();
  await page.getByRole('radio', { name: /표면 결함 표식/ }).check();
  const downloadPromise = page.waitForEvent('download');
  await page.getByRole('link', { name: '환경 JSON 다운로드' }).click();
  const download = await downloadPromise;
  const path = await download.path();
  if (!path) throw new Error('The local JSON download did not complete');
  const document = JSON.parse(await readFile(path, 'utf8'));
  expect(document.stations.find((station: { role: string }) => station.role === 'rejected').position_m).toEqual([0.32, -0.38, 0.2]);
  expect(document.scene.seed).toBe(43);
  expect(document.execution.max_step_seconds).toBe(30);
  await expect(page.getByRole('link', { name: '운영자에게 이 실험 전달' })).toHaveAttribute('href', '/operator?view=studio&experiment=relocate-quarantine&sample=surface_defect');
  expect(requests.filter(item => item.path.startsWith('/api/')).every(item => item.method === 'GET' && !item.authorization)).toBe(true);
  expect(errors).toEqual([]);
  await page.setViewportSize({ width: 390, height: 844 });
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});

test('pause is keyboard accessible and stops viewing requests, not a robot command', async ({ page }) => {
  const { requests } = await fixturePage(page, () => makeSnapshot());
  await page.goto('/');
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toBeVisible();
  await page.getByRole('button', { name: '관람 일시 정지' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByRole('heading', { name: '영상 관람을 일시 정지했습니다' })).toBeVisible();
  const reads = requests.filter((item) => item.path.startsWith('/api/')).length;
  await page.waitForTimeout(2400);
  expect(requests.filter((item) => item.path.startsWith('/api/')).length).toBe(reads);
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toHaveCount(0);
  expect(requests.every((request) => request.method === 'GET')).toBe(true);
  await page.getByRole('button', { name: '관람 재개' }).focus();
  await page.keyboard.press('Enter');
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toBeVisible();
});

test('actual outcome requires correct inspection as well as completed physical movement', async ({ page }) => {
  let presentation = makePresentation();
  await fixturePage(page, () => makeSnapshot({ presentation }));
  await page.goto('/');
  await expect(page.getByText('최종 결과 대기')).toBeVisible();
  presentation = makePresentation({
    status: 'completed',
    motion: { status: 'succeeded', phase: 'complete', part_position_m: [0.22, -0.38, 0.2], target_position_m: [0.22, -0.38, 0.2] },
    result: { status: 'succeeded', physical_success: true, inspection_correct: false, final_position_m: [0.22, -0.38, 0.2], completed_at: new Date().toISOString(), message: '테스트 표본: 이동은 완료되었지만 검사 정답이 다릅니다.' },
  });
  await expect(page.getByText('종합 성공 미확인', { exact: true })).toBeVisible();
  await expect(page.getByText('입력 정답과 불일치', { exact: true })).toBeVisible();
  await expect(page.getByText('물리 목표 도달 확인', { exact: true })).toBeVisible();
  await expect(page.getByText('종합 성공 확인', { exact: true })).toHaveCount(0);
  const finalResult = presentation.result;
  if (!finalResult) throw new Error('Test result required');
  presentation = { ...presentation, result: { ...finalResult, inspection_correct: true } };
  await expect(page.getByText('종합 성공 확인', { exact: true })).toBeVisible();
});

test('wrong scene/evidence responses never appear as current images; strict CSP remains effective', async ({ page }) => {
  const { errors } = await fixturePage(page, () => makeSnapshot(), { wrongEpoch: true, wrongEvidence: true });
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '관측과 게시 회차가 일치하지 않습니다' })).toBeVisible();
  await expect(page.getByText('불량으로 판단', { exact: true })).toHaveCount(0);
  await expect(page.locator('main img')).toHaveCount(0);
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toHaveCount(0);
  expect(errors).toEqual([]);
  await page.addScriptTag({ url: '/test-only-csp.js' });
  expect(await page.evaluate(() => document.body.dataset.cspEvalBlocked)).toBe('true');
});

test('new cycle clears old classification and images instead of reusing prior outcome', async ({ page }) => {
  let presentation = makePresentation();
  await fixturePage(page, () => makeSnapshot({ presentation }));
  await page.goto('/');
  await expect(page.getByText('불량으로 판단', { exact: true })).toBeVisible();
  presentation = makePresentation({
    cycle: 2, scenario: 'normal', status: 'preparing', scene_epoch: null, run_id: null, decision: null, motion: null, result: null,
  });
  await expect(page.getByText('서버 입력: 정상 부품')).toBeVisible();
  await expect(page.getByRole('heading', { name: '실제 시연을 준비하고 있습니다' })).toBeVisible();
  await expect(page.getByText('불량으로 판단', { exact: true })).toHaveCount(0);
  await expect(page.locator('main img')).toHaveCount(0);
  await expect(page.getByText('종합 성공 확인', { exact: true })).toHaveCount(0);
});

test('reference and missing-presentation responses have no schematic, replay or camera substitute', async ({ page }) => {
  const { requests } = await fixturePage(page, referenceSnapshot);
  await page.goto('/');
  await expect(page.getByRole('heading', { name: '게시된 자동 시연이 없습니다' })).toBeVisible();
  await expect(page.getByRole('button', { name: '시연 상태 다시 확인' })).toBeVisible();
  await expect(page.getByRole('navigation', { name: '시나리오 단계' })).toHaveCount(0);
  await expect(page.locator('main img, .demo-cell-svg')).toHaveCount(0);
  expect(requests.filter((item) => item.path.startsWith('/api/')).every((item) => item.path === '/api/demo')).toBe(true);
  await page.screenshot({ path: 'test-results/public-unavailable-TEST-FIXTURE-not-azure.png', fullPage: true });
});

test('390px layout, long copy, reduced motion and focus remain accessible', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  const presentation = makePresentation({ instruction: '테스트 전용 긴 작업 설명입니다. '.repeat(40) });
  if (presentation.decision) presentation.decision.summary = '테스트 전용 긴 판단 요약이며 실제 모델의 검증 결과가 아닙니다. '.repeat(30);
  await fixturePage(page, () => makeSnapshot({ presentation }));
  await page.goto('/');
  await expect(page.getByRole('button', { name: '관람 재개' })).toBeVisible();
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toHaveCount(0);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.getByRole('button', { name: '관람 재개' }).focus();
  expect(await page.getByRole('button', { name: '관람 재개' }).evaluate((element) => getComputedStyle(element).outlineStyle)).not.toBe('none');
  await page.keyboard.press('Enter');
  await expect(page.getByText('LIVE · 실제 카메라 수신')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
  await page.screenshot({ path: 'test-results/public-mobile-TEST-FIXTURE-not-azure.png', fullPage: true });
});

test('operator path still requires config/auth even with public query parameters', async ({ page }) => {
  const { requests } = await fixturePage(page, referenceSnapshot);
  await page.goto('/operator?view=demo');
  await expect(page.getByRole('heading', { name: '작업 공간에 로그인' })).toBeVisible();
  await expect(page.getByRole('alert')).toContainText('콘솔을 시작할 수 없습니다');
  expect(requests.some((request) => request.path === '/api/config')).toBe(true);
  expect(requests.some((request) => request.path === '/api/demo')).toBe(false);
  await expect(page.getByRole('heading', { name: '실제 작업 셀 카메라' })).toHaveCount(0);
});
