import { createRoot } from 'react-dom/client';
import { ConsoleApp } from '../../src/ConsoleApp';
import { ApiError } from '../../src/api/errors';
import type { ConsoleApi, EnvironmentRecord, RunRecord, RuntimeInfo } from '../../src/api/contracts';
import { account, environment, environmentSchema, fixtureDocument, FIXTURE_MARKER, pendingRun, runtime } from '../fixtures/data';
import '../../src/styles.css';
import './fixture.css';
import { browserLearning, type LearningTrace } from './learning-fixture';
import { skillRun } from '../fixtures/released-skill';
import { environmentCursor, seventyEnvironments } from '../fixtures/environment-pages';

interface FixtureTrace {
  marker: string;
  calls: string[];
  saves: Array<{ documentJson: string; expectedRevision: string | null }>;
  approvals: Array<{ id: string; planResponseId: string } | { id: string; skillPlanId: string }>;
}

declare global {
  interface Window {
    __factoryFixture: FixtureTrace;
    __learningFixture: LearningTrace;
  }
}

const trace: FixtureTrace = { marker: FIXTURE_MARKER, calls: [], saves: [], approvals: [] };
window.__factoryFixture = trace;
const scenario = new URLSearchParams(window.location.search).get('scenario') ?? 'ready';
const learningTrace: LearningTrace = { calls: [], inputs: [] };
window.__learningFixture = learningTrace;
let activeRuntime: RuntimeInfo = scenario === 'unavailable'
  ? { ...runtime, simulation: { ...runtime.simulation, status: 'unavailable', epoch: null, physics_steps: null, message: 'TEST ONLY: Azure/GPU 런타임에 연결하지 않은 테스트입니다.' } }
  : scenario === 'skill' ? { ...runtime, agent: { ...runtime.agent, configured: false } } : structuredClone(runtime);
let record = structuredClone(environment);
let run: RunRecord | null = scenario === 'approval' ? structuredClone(pendingRun) : scenario === 'skill' ? structuredClone(skillRun) : null;
let runPolls = 0;
let activationPolls = 0;
let frameCount = 0;
let saveConflict = scenario === 'conflict';

function frameFixture(): Promise<Blob> {
  const canvas = document.createElement('canvas');
  canvas.width = 960;
  canvas.height = 600;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Test canvas is unavailable');
  context.fillStyle = '#132138';
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.strokeStyle = '#243b59';
  context.lineWidth = 1;
  for (let column = 0; column < 960; column += 60) {
    context.beginPath(); context.moveTo(column, 0); context.lineTo(column, 600); context.stroke();
  }
  for (let row = 0; row < 600; row += 60) {
    context.beginPath(); context.moveTo(0, row); context.lineTo(960, row); context.stroke();
  }
  context.fillStyle = '#bdd6f6';
  context.textAlign = 'center';
  context.font = '600 24px sans-serif';
  context.fillText('TEST-ONLY CAMERA FIXTURE', 480, 275);
  context.font = '15px sans-serif';
  context.fillStyle = '#8daacb';
  context.fillText('NO AZURE CONNECTION / NO GPU VERIFICATION', 480, 313);
  context.fillText('This static PNG exists only in the browser test harness.', 480, 347);
  return new Promise((resolve, reject) => canvas.toBlob((blob) => blob ? resolve(blob) : reject(new Error('PNG fixture encoding failed')), 'image/png'));
}

const png = await frameFixture();
function call(name: string, signal?: AbortSignal) {
  signal?.throwIfAborted();
  trace.calls.push(name);
}

const api: ConsoleApi = {
  learning: scenario.startsWith('learning') ? browserLearning(learningTrace, scenario !== 'learning-off', scenario === 'learning-reference') : undefined,
  async getRuntime(signal) {
    call('runtime', signal);
    if (activeRuntime.simulation.status === 'loading' && ++activationPolls >= 2) {
      activeRuntime = { ...runtime, simulation: { ...runtime.simulation, environment_id: record.environment_id, revision: record.revision } };
    }
    return structuredClone(activeRuntime);
  },
  async getEnvironments(signal, cursor) {
    call('environments', signal);
    if (scenario === 'learning-pages') return structuredClone(cursor
      ? { items: seventyEnvironments.slice(50), next_cursor: null }
      : { items: seventyEnvironments.slice(0, 50), next_cursor: environmentCursor });
    return { items: [structuredClone(record)], next_cursor: null };
  },
  async getEnvironmentSchema(signal) { call('schema', signal); return environmentSchema; },
  async getTemplates(signal) { call('templates', signal); return { items: [{ name: '참조 검사 셀 (테스트 전용)', document: fixtureDocument }] }; },
  async saveEnvironment(documentJson, expectedRevision, signal) {
    call('save', signal);
    trace.saves.push({ documentJson, expectedRevision });
    if (saveConflict) {
      saveConflict = false;
      throw new ApiError('revision_conflict', '테스트 전용: 서버의 저장 버전이 변경되었습니다.', 409);
    }
    const document: Record<string, unknown> = JSON.parse(documentJson);
    record = { ...record, document, revision: 'b'.repeat(64) } satisfies EnvironmentRecord;
    return structuredClone(record);
  },
  async activateEnvironment(id, revision, signal) {
    call('activate', signal);
    activeRuntime = { ...runtime, simulation: { ...runtime.simulation, status: 'loading', environment_id: id, revision } };
    return { activation_id: 'test-only-activation', environment_id: id, revision, status: 'loading' };
  },
  async getFrame(_id, _revision, _camera, signal) {
    call('frame', signal);
    return { blob: png, frameId: `test-only-frame-${++frameCount}`, capturedAt: new Date().toISOString(), physicsSteps: 125 };
  },
  async getRuns(signal) { call('runs', signal); return { items: run ? [structuredClone(run)] : [] }; },
  async getRun(id, signal) {
    call(`run:${id}`, signal);
    if (!run) throw new ApiError('not_found', 'Test run not found', 404);
    if (scenario !== 'skill' && run.status === 'running' && ++runPolls >= 2) run = { ...run, status: 'succeeded', execution: { command_id: 'test-only-command', status: 'succeeded', final_position: [0.5, -0.4, 0.2] } };
    if (run.status === 'cancelling') run = { ...run, status: 'cancelled' };
    return structuredClone(run);
  },
  async createRun(input, signal) {
    call('createRun', signal);
    run = input.execution_mode === 'released_skill'
      ? { ...skillRun, environment_id: input.environment_id, revision: input.revision }
      : { ...pendingRun, environment_id: input.environment_id, revision: input.revision, instruction: input.instruction };
    return structuredClone(run);
  },
  async approveRun(id, approval, signal) {
    call('approve', signal);
    trace.approvals.push(typeof approval === 'string' ? { id, planResponseId: approval } : { id, skillPlanId: approval.skill_plan_id });
    if (!run) throw new ApiError('not_found', 'Test run not found', 404);
    run = { ...run, status: 'running', execution: { command_id: 'test-only-command', status: 'accepted' } };
    return structuredClone(run);
  },
  async cancelRun(_id, signal) {
    call('cancel', signal);
    if (!run) throw new ApiError('not_found', 'Test run not found', 404);
    run = { ...run, status: 'cancelling' };
    return structuredClone(run);
  },
  async getObservation(_id, signal) { call('observation', signal); return png; },
};

const root = document.getElementById('root');
if (!root) throw new Error('Fixture root missing');
createRoot(root).render(<>
  <div className="fixture-warning" role="note">TEST-ONLY FIXTURES · 테스트 전용 데이터 · Azure / GPU 동작 검증이 아닙니다</div>
  <ConsoleApp api={api} account={account} />
</>);
