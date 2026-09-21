import type { DemoSnapshot, Presentation, PublicFrame, PublicEvidence } from '../../src/public/api';
import { fixtureDocument, fixturePng } from './data';

export const epoch = '11000000-1111-4111-8111-111111111111';
export const nextEpoch = '22000000-2222-4222-8222-222222222222';
export const observationId = '33000000-3333-4333-8333-333333333333';
export const runId = '44000000-4444-4444-8444-444444444444';

export function makePresentation(overrides: Partial<Presentation> = {}): Presentation {
  return {
    id: 'test-only-public-presentation',
    status: 'moving',
    cycle: 1,
    total_cycles: 4,
    scenario: 'surface_defect',
    instruction: '테스트 전용: 표면에 흠집이 있는 부품을 후공정으로 보내지 않고 격리합니다.',
    updated_at: new Date().toISOString(),
    expires_at: new Date(Date.now() + 120_000).toISOString(),
    scene_epoch: epoch,
    run_id: runId,
    decision: {
      classification: 'rejected',
      summary: 'TEST-ONLY FIXTURE: 입력 표본의 표면 흠집을 근거로 격리하는 판단입니다. 실제 Foundry 호출이 아닙니다.',
      target_station_id: 'rejected',
      observation_id: observationId,
      captured_at: new Date(Date.now() - 30_000).toISOString(),
      image_url: '/api/demo/evidence',
    },
    motion: { status: 'running', phase: 'transporting', part_position_m: [0.35, 0, 0.25], target_position_m: [0.22, -0.38, 0.2] },
    result: null,
    counts: { attempted: 1, succeeded: 0, failed: 0, inspected_correctly: 0, physically_completed: 0 },
    ...overrides,
  };
}

export function makeSnapshot(overrides: Partial<DemoSnapshot> = {}): DemoSnapshot {
  return {
    api_version: 'public-demo-v1', access: 'public_read_only', deployment: 'azure',
    mode: 'live', observed_at: new Date().toISOString(),
    scene: {
      id: 'inspection-cell-v1', name: 'Test-only reference inspection cell', length_unit: 'm',
      stations: fixtureDocument.stations.map((item) => ({
        id: item.id,
        role: item.role === 'source' || item.role === 'inspection' || item.role === 'accepted' ? item.role : 'rejected',
        position_m: [item.position_m[0] ?? 0, item.position_m[1] ?? 0, item.position_m[2] ?? 0],
      })),
      robot: 'Franka reference arm', data_origin: 'synthetic_reference_configuration',
    },
    simulation: { status: 'ready', live_available: true, frame_url: '/api/demo/frame', message_code: 'ready' },
    agent: { provider: 'microsoft_foundry', connectivity: 'verified', verified_at: new Date().toISOString(), verification_scope: 'connectivity_only' },
    learning: { status: 'cpu_smoke_verified', execution_location: 'azure_acr', data_kind: 'test_fixture', optimizer_steps: 1, quality_verified: false },
    capabilities: { anonymous_control: false, anonymous_editing: false, public_live_video: true },
    presentation: makePresentation(),
    ...overrides,
  };
}

export function referenceSnapshot(): DemoSnapshot {
  return makeSnapshot({
    mode: 'reference',
    simulation: { status: 'not_published', live_available: false, frame_url: null, message_code: 'not_published' },
    capabilities: { anonymous_control: false, anonymous_editing: false, public_live_video: false },
    presentation: null,
  });
}

export function frame(overrides: Partial<PublicFrame> = {}): PublicFrame {
  return {
    blob: fixturePng(), frameId: 'test-only-public-frame', capturedAt: new Date().toISOString(),
    physicsSteps: 125, sceneEpoch: epoch, expiresAtMonotonicMs: performance.now() + 5000, ...overrides,
  };
}
export function evidence(presentation: Presentation): PublicEvidence {
  if (!presentation.decision) throw new Error('This fixture requires a decision.');
  return { blob: fixturePng(), frameId: presentation.decision.observation_id, capturedAt: presentation.decision.captured_at };
}
