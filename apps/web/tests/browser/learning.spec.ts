import { expect, test } from '@playwright/test';

test('reference-only stage uses deliberate keyboard approval without manual or paid motion', async ({ page }) => {
  await page.goto('/?scenario=learning-reference&view=learning');
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  await page.getByLabel('학습 프로젝트', { exact: true }).selectOption({ label: 'TEST-ONLY REFERENCE 작업' });
  const start = page.getByRole('button', { name: '승인한 REFERENCE 시연 생성' });
  await expect(start).toBeDisabled();
  await page.getByLabel('기준 시연 배치').selectOption('train-anchor');
  await page.getByRole('checkbox', { name: '선택한 배치의 검토된 기준 제어기 이동을 승인합니다' }).focus();
  await page.keyboard.press('Space');
  await page.keyboard.press('Tab');
  await expect(start).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByText('REFERENCE · running · 캡처 pending')).toBeVisible();
  await expect(page.getByText('실제 WALL 시간 (ms)')).toBeVisible();
  await expect(page.getByText('실제 SIM 시간 (초)')).toBeVisible();
  await expect(page.getByRole('button', { name: '새 직접 시연 세션 시작' })).toHaveCount(0);
  await expect(page.getByRole('button', { name: '검증된 데이터로 제한된 학습 제출' })).toBeDisabled();
  const trace = await page.evaluate(() => window.__learningFixture);
  expect(trace.referenceRequests).toEqual([{
    request_id: expect.any(String), case_id: 'train-anchor', motion_approved: true,
  }]);
  expect(trace.calls.some((call) => ['teach', 'arm', 'jog', 'train', 'evaluate'].includes(call))).toBe(false);
});

test('seventy owner environments remain browsable with bounded case rows and retained selection', async ({ page }) => {
  await page.goto('/?scenario=learning-pages&view=learning');
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  await expect(page.getByText('불러온 저장 환경 50개')).toBeVisible();
  await page.getByRole('button', { name: '새 학습 작업 정의' }).click();
  expect(await page.locator('input[name="teaching-case"]').count()).toBeLessThanOrEqual(20);
  await page.getByRole('searchbox', { name: '시연 배치 검색' }).fill('case-040');
  const selection = page.getByRole('checkbox', { name: /case-040/ });
  await selection.focus();
  await page.keyboard.press('Space');
  await expect(selection).toBeChecked();
  await page.getByRole('button', { name: '저장 환경 더 불러오기' }).click();
  await expect(page.getByText('불러온 저장 환경 70개')).toBeVisible();
  await expect(selection).toBeChecked();
  await page.getByRole('searchbox', { name: '시연 배치 검색' }).fill('case-000');
  await expect(page.getByRole('checkbox', { name: /case-000/ })).toBeVisible();
  expect(await page.evaluate(() => window.__learningFixture.calls.includes('createProject'))).toBe(false);
});

test('paused-simulation budgets are explicit and cannot submit unadmitted work', async ({ page }) => {
  await page.goto('/?scenario=learning-pages&view=learning');
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  await page.getByRole('button', { name: '새 학습 작업 정의' }).click();
  await page.getByLabel('실행 시간 모드').selectOption('paused_simulation');
  await expect(page.getByText('한 회차 최대 30 SIM초 · 600 WALL초')).toBeVisible();
  await expect(page.getByLabel('전체 평가 WALL 예산 (초)')).toHaveValue('7200');
  const profile = page.getByLabel('시뮬레이션 예산 버전');
  await profile.focus();
  await profile.selectOption('franka-position-hold-10hz-paused-v2');
  await expect(profile).toBeFocused();
  await expect(page.getByText('한 회차 최대 60 SIM초 · 600 WALL초')).toBeVisible();
  await expect(page.getByText(/기존 v1 실행·결과를 v2 통과로 바꾸지 않습니다/)).toBeVisible();
  await page.getByLabel('전체 평가 WALL 예산 (초)').fill('9000');
  await expect(page.getByRole('button', { name: '불변 작업 정의 저장' })).toBeDisabled();
  await expect(page.getByText(/실시간 100ms\/80ms 통과가 아닙니다/)).toBeVisible();
  expect(await page.evaluate(() => window.__learningFixture.calls.some((call) => ['teach', 'train', 'evaluate'].includes(call)))).toBe(false);
});

test('teaching remains blocked without verified dependencies and never creates a paid job', async ({ page }) => {
  await page.goto('/?scenario=learning-off&view=learning');
  await expect(page.getByRole('heading', { name: '학습 기능이 아직 활성화되지 않았습니다' })).toBeVisible();
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  expect(await page.evaluate(() => window.__learningFixture.calls)).toEqual(['capabilities']);
  await page.screenshot({ path: 'test-results/learning-disabled-TEST-FIXTURE-not-azure.png', fullPage: true });
});

test('explicit teaching grants, upload state and no-improvement evaluation remain truthful', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/?scenario=learning&view=learning');
  await page.getByLabel('학습 프로젝트', { exact: true }).selectOption({ label: 'TEST-ONLY 작업 가르치기' });
  const start = page.getByRole('button', { name: '새 직접 시연 세션 시작' });
  await expect(start).toBeDisabled();
  await page.getByRole('checkbox', { name: '저속 시연 조작과 서버의 제한된 이동 권한을 승인합니다' }).check();
  await start.click();
  const jog = page.getByRole('button', { name: 'X 양의 방향 5mm' });
  await jog.focus();
  await page.keyboard.down('Space');
  await expect.poll(() => page.evaluate(() => window.__learningFixture.inputs.some((input) => input.deadman))).toBe(true);
  await page.keyboard.up('Space');
  await expect.poll(() => page.evaluate(() => window.__learningFixture.inputs.some((input) => !input.deadman))).toBe(true);
  const inputs = await page.evaluate(() => window.__learningFixture.inputs);
  expect(inputs.find((input) => input.deadman)?.grant_id).toBeTruthy();
  expect(inputs.some((input) => 'expires_at' in input)).toBe(false);
  await page.getByRole('button', { name: '시연 마감 및 캡처 검증 요청' }).click();
  await expect(page.getByText(/물리 상태: succeeded · 캡처 상태: uploading/)).toBeVisible();
  await page.locator('.learning-record-row').filter({ hasText: 'evaluation' }).getByRole('button', { name: '기록 보기' }).click();
  await expect(page.getByText('개선 미확인', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '검토한 정책 게시' })).toBeDisabled();
  await expect(page.locator('.policy-comparison tbody tr')).toHaveCount(40);
  expect(await page.evaluate(() => window.__learningFixture.calls.includes('train'))).toBe(false);
  expect(errors).toEqual([]);
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: 'test-results/teaching-comparison-TEST-FIXTURE-not-azure.png', fullPage: true });
});

test('learning navigation and real-record presentation stay usable at 390px', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await page.goto('/?scenario=learning&view=learning');
  await page.getByLabel('학습 프로젝트', { exact: true }).selectOption({ label: 'TEST-ONLY 작업 가르치기' });
  await expect(page.getByRole('button', { name: '승인한 smolvla 학습 제출' })).toBeDisabled();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true);
});
