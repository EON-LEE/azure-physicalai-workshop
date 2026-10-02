import { expect, test } from '@playwright/test';
import { environment, fixtureDocument, inspectionResponseId, pendingRun } from '../fixtures/data';
import { skillInstruction, skillPlanId, skillRun } from '../fixtures/released-skill';

test('strict CSP, keyboard approval, confirmed completion and a labeled fixture screenshot', async ({ page }) => {
  const errors: string[] = [];
  const externalRequests: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  page.on('console', (message) => { if (message.type() === 'error') errors.push(message.text()); });
  page.on('request', (request) => { if (request.url().startsWith('https://')) externalRequests.push(request.url()); });
  await page.goto('/?scenario=approval');
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  await expect(page.getByText('LIVE · 프레임 수신 중', { exact: true })).toBeVisible();
  await expect(page.getByText('승인 대기', { exact: true })).toBeVisible();
  await expect(page.getByRole('button', { name: '계획 승인 및 실행' })).toBeDisabled();
  expect(await page.evaluate(() => window.__factoryFixture.approvals)).toEqual([]);
  await page.screenshot({ path: 'test-results/factory-live-TEST-FIXTURE-not-azure.png', fullPage: true });
  const confirmation = page.getByRole('checkbox', { name: '표시된 계획과 이동 대상을 확인했습니다' });
  await confirmation.focus();
  await page.keyboard.press('Space');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: '계획 승인 및 실행' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByText('물리 실행 확인 중', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => window.__factoryFixture.approvals)).toEqual([{ id: pendingRun.id, planResponseId: inspectionResponseId }]);
  await expect(page.getByText('서버가 실행 성공을 확인했습니다', { exact: true })).toBeVisible();
  expect(errors).toEqual([]);
  expect(externalRequests).toEqual([]);
});

test('a released skill is not classified by CV and requires its own deliberate keyboard approval', async ({ page }) => {
  await page.goto('/?scenario=skill');
  await expect(page.getByRole('note')).toContainText('Azure / GPU 동작 검증이 아닙니다');
  await expect(page.getByRole('heading', { name: '게시된 작업 계획과 실행' })).toBeVisible();
  await expect(page.getByText('CV 검사 수행 안 함', { exact: true })).toBeVisible();
  await expect(page.getByText(skillInstruction, { exact: true }).first()).toBeVisible();
  await expect(page.getByText('model_response_id', { exact: true })).toHaveCount(0);
  await expect(page.getByText(/분류: (정상|불량) 후보/)).toHaveCount(0);
  await expect(page.getByRole('button', { name: '계획 승인 및 실행' })).toBeDisabled();
  expect(await page.evaluate(() => window.__factoryFixture.approvals)).toEqual([]);
  await page.getByRole('checkbox', { name: '표시된 계획과 이동 대상을 확인했습니다' }).focus();
  await page.keyboard.press('Space');
  await page.keyboard.press('Tab');
  await expect(page.getByRole('button', { name: '계획 승인 및 실행' })).toBeFocused();
  await page.keyboard.press('Enter');
  await expect(page.getByText('물리 실행 확인 중', { exact: true })).toBeVisible();
  expect(await page.evaluate(() => window.__factoryFixture.approvals)).toEqual([{ id: skillRun.id, skillPlanId }]);
  await expect(page.getByText('서버가 실행 성공을 확인했습니다', { exact: true })).toHaveCount(0);
});

test('raw JSON and revision survive conflict under CSP; navigation preserves the draft', async ({ page }) => {
  const cspErrors: string[] = [];
  page.on('console', (message) => { if (message.type() === 'error') cspErrors.push(message.text()); });
  await page.goto('/?scenario=conflict');
  await page.getByRole('navigation', { name: '콘솔 화면' }).getByRole('button', { name: /Environment Studio/ }).click();
  const raw = ` \n${JSON.stringify({ ...fixtureDocument, display_name: '원본 텍스트 보존 테스트' }, null, 4)}\n`;
  await page.getByLabel('고객 환경 JSON 원본').fill(raw);
  await page.getByRole('button', { name: 'JSON 저장' }).click();
  await expect(page.getByRole('alert')).toContainText('저장 버전 충돌');
  expect(await page.evaluate(() => window.__factoryFixture.saves)).toEqual([{ documentJson: raw, expectedRevision: environment.revision }]);
  await expect(page.getByLabel('고객 환경 JSON 원본')).toHaveValue(raw);
  await page.getByRole('button', { name: /Run History/ }).click();
  await expect(page.getByText('아직 실행 기록이 없습니다')).toBeVisible();
  await page.getByRole('navigation', { name: '콘솔 화면' }).getByRole('button', { name: /Environment Studio/ }).click();
  await expect(page.getByLabel('고객 환경 JSON 원본')).toHaveValue(raw);
  await page.getByRole('button', { name: 'JSON 저장' }).click();
  await expect(page.getByText(/원본 JSON을 저장했습니다/)).toBeVisible();
  await expect(page.getByRole('button', { name: '저장된 씬 활성화' })).toBeEnabled();
  expect(await page.evaluate(() => window.__factoryFixture.calls.includes('activate'))).toBe(false);
  expect(cspErrors).toEqual([]);
});

test('camera polling stops on navigation and unavailable runtime never gets a replacement frame', async ({ page }) => {
  await page.goto('/');
  await expect(page.getByText('LIVE · 프레임 수신 중', { exact: true })).toBeVisible();
  await page.getByRole('button', { name: /Run History/ }).click();
  const frames = await page.evaluate(() => window.__factoryFixture.calls.filter((call) => call === 'frame').length);
  await page.waitForTimeout(2200);
  expect(await page.evaluate(() => window.__factoryFixture.calls.filter((call) => call === 'frame').length)).toBe(frames);
  await page.goto('/?scenario=unavailable');
  await expect(page.getByText('활성화된 씬이 필요합니다')).toBeVisible();
  await expect(page.getByRole('button', { name: '관측하고 계획 요청' })).toBeDisabled();
  expect(await page.evaluate(() => window.__factoryFixture.calls.includes('frame'))).toBe(false);
  await expect(page.locator('img')).toHaveCount(0);
});

test('narrow viewport stays usable and labels the fixture instead of claiming a GPU session', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/?scenario=approval');
  await expect(page.getByRole('heading', { name: 'Factory Live', exact: true })).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
  await page.getByRole('navigation', { name: '콘솔 화면' }).getByRole('button', { name: /Environment Studio/ }).click();
  await expect(page.getByLabel('고객 환경 JSON 원본')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth)).toBe(true);
});
