import document from '../../../../examples/inspection-cell.json' with { type: 'json' };
import schema from '../../../../contracts/customer-environment.schema.json' with { type: 'json' };
import type { EnvironmentRecord, PublicConfig, RunRecord, RuntimeInfo } from '../../src/api/contracts';
import type { SignedInAccount } from '../../src/auth/types';

export const FIXTURE_MARKER = 'TEST_ONLY_FACTORY_FIXTURE';
export const environmentSchema = schema;
export const fixtureDocument = document;
export const account: SignedInAccount = {
  id: 'test-only-account',
  name: '테스트 운영자',
  username: 'fixture-operator@example.invalid',
};
export const config: PublicConfig = {
  api_version: 'v1',
  deployment: 'azure',
  auth: {
    tenant_id: '11111111-1111-4111-8111-111111111111',
    client_id: '22222222-2222-4222-8222-222222222222',
    scope: 'api://33333333-3333-4333-8333-333333333333/access_as_user',
  },
};
export const environment: EnvironmentRecord = {
  environment_id: document.environment_id,
  display_name: '참조 검사 셀 · 테스트 전용',
  revision: 'a'.repeat(64),
  document,
  created_at: '2026-09-15T02:00:00+00:00',
  updated_at: '2026-09-15T02:00:00+00:00',
};
export const runtime: RuntimeInfo = {
  deployment: 'azure',
  simulation: {
    backend: 'isaac_sim', status: 'ready', environment_id: environment.environment_id, revision: environment.revision,
    epoch: 'test-only-world-epoch', physics_steps: 125, message: null,
  },
  agent: { provider: 'microsoft_foundry', configured: true },
  storage: { provider: 'azure_cosmos_blob' },
  release_ready: false,
};
export const inspectionResponseId = 'test-only-foundry-response-id';
export const pendingRun: RunRecord = {
  id: 'test-only-run-awaiting-approval',
  environment_id: environment.environment_id,
  revision: environment.revision,
  instruction: '테스트 전용: 부품의 결함을 검사하고 불량 스테이션으로 분류하는 계획을 작성하세요.',
  status: 'awaiting_approval',
  created_at: '2026-09-15T02:10:00+00:00',
  updated_at: '2026-09-15T02:10:01+00:00',
  plan: {
    classification: 'rejected', target_station_id: 'rejected', object_id: 'part-001',
    summary: '테스트 픽스처의 계획 요약입니다. 실제 Foundry 응답이 아닙니다.',
    observation_id: 'test-only-observation-id', epoch: 'test-only-world-epoch', state_revision: 1,
    model_response_id: inspectionResponseId,
  },
  execution: null, error: null,
  events: [{ kind: 'planning', message: '테스트 전용 관측 이벤트 · Azure 검증 아님', at: '2026-09-15T02:10:00+00:00' }],
};
export const runningRun: RunRecord = {
  ...pendingRun, status: 'running',
  execution: { command_id: 'test-only-command-id', status: 'accepted' },
};
export const succeededRun: RunRecord = {
  ...runningRun, status: 'succeeded',
  execution: { command_id: 'test-only-command-id', status: 'succeeded', final_position: [0.5, -0.4, 0.2], completed_at: '2026-09-15T02:10:05+00:00' },
};

export const pngBytes = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLbtAAAAABJRU5ErkJggg=='), (char) => char.charCodeAt(0));
export const fixturePng = () => new Blob([pngBytes], { type: 'image/png' });
