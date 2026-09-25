import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import { datasetSchema, projectSchema, referenceCollectionSchema, trainingSchema } from '../src/learning/contracts';
import { referenceFixture } from './fixtures/reference-collection';
import { LearningJobPanel } from '../src/learning/LearningJobPanel';
import { learningApi, learningFixture } from './fixtures/learning';
import { seventyEnvironments } from './fixtures/environment-pages';

const simulationLearning = {
  execution_timing: 'paused_simulation' as const, real_time_admission: false as const,
  supported: true as const, enabled: false as const, status: 'producer_verifier_unavailable' as const,
  message: 'Test only: paused model/report adapters are not admitted.',
  reference_generation_enabled: false, training_enabled: false, evaluation_enabled: false, release_enabled: false,
};

function projectRecord() {
  const fixture = learningFixture().project.item;
  const plan = fixture.evaluation_plan;
  return {
    ...fixture, control_profile_id: 'franka-position-hold-10hz-paused-v1',
    execution_timing: 'paused_simulation', real_time_admission: false,
    control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    budget: { ...fixture.budget, evaluation_seconds: 7200 },
    evaluation_plan: {
      id: plan.id, seeds: plan.seeds, held_out_episode_ids: plan.held_out_episode_ids, cases: plan.cases,
      execution_timing: 'paused_simulation', real_time_admission: false,
      minimum_success_rate: .9, minimum_absolute_improvement: .05,
      maximum_axis_error_m: .04, max_cartesian_speed_m_s: .2,
      max_simulation_seconds: 30, max_wall_seconds: 600,
      max_observation_wall_ms: 2000, max_policy_wall_ms: 2000, max_hold_wall_ms: 2000,
      max_interval_wall_ms: 5000, max_heartbeat_wall_ms: 2000,
    },
  };
}

describe('explicit simulation-only learning mode', () => {
  it('decodes a separate paused project without real-time gate fields or qualification', () => {
    const input = projectRecord();
    const parsed = projectSchema.parse(input);
    expect(parsed).toMatchObject({
      execution_timing: 'paused_simulation', real_time_admission: false,
      control_profile_id: 'franka-position-hold-10hz-paused-v1',
      budget: { evaluation_seconds: 7200 },
      evaluation_plan: { max_simulation_seconds: 30, max_wall_seconds: 600 },
    });
    expect(parsed.evaluation_plan).not.toHaveProperty('maximum_inference_p95_ms');
    expect(parsed.evaluation_plan).not.toHaveProperty('max_step_seconds');
  });

  it.each([
    { execution_timing: undefined },
    { real_time_admission: true },
    { criteria_sha256: undefined },
    { control_profile_id: 'franka-position-hold-10hz-v1' },
  ])('rejects mixed timing or missing immutable pins: %j', (override) => {
    expect(() => projectSchema.parse({ ...projectRecord(), ...override })).toThrow();
  });

  it('keeps scripted source counts and exact paused provenance on verified capture records', () => {
    const fixture = learningFixture().dataset.item;
    const timing = {
      execution_timing: 'paused_simulation', real_time_admission: false,
      control_profile_id: 'franka-position-hold-10hz-paused-v1',
      control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    };
    const parsed = datasetSchema.parse({
      ...fixture, ...timing, captures: fixture.captures.map((capture) => ({ ...capture, ...timing })),
    });
    expect(parsed.reference_controller_count).toBe(1);
    expect(parsed.human_teleop_count).toBe(0);
    expect(parsed.captures[0]?.source).toBe('reference_controller');
    expect(parsed.captures[0]?.real_time_admission).toBe(false);
    expect(() => datasetSchema.parse({ ...parsed, criteria_sha256: undefined })).toThrow();
  });

  it('labels a recorded paused training job without turning it into real-time qualification', async () => {
    const fixture = learningFixture().training;
    const item = trainingSchema.parse({
      ...fixture.item, execution_timing: 'paused_simulation', real_time_admission: false,
      control_profile_id: 'franka-position-hold-10hz-paused-v1',
      control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    });
    const api = learningApi();
    api.job.mockResolvedValue({ ...fixture, item });
    render(<LearningJobPanel api={api} initial={{ ...fixture, item }} />);
    expect(await screen.findByText('NON_REALTIME_SIMULATION · 실시간 제어 승인 아님')).toBeInTheDocument();
    expect(screen.getByText('optimizer step 미수신')).toBeInTheDocument();
    expect(api.train).not.toHaveBeenCalled();
  });

  it('lets the operator review separate simulation and total wall budgets without enabling unready work', async () => {
    const api = learningApi();
    const current = await api.capabilities();
    api.capabilities.mockResolvedValue({ ...current, simulation_learning: simulationLearning });
    render(<TeachingStudio api={api} environments={seventyEnvironments} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    await userEvent.selectOptions(screen.getByLabelText('실행 시간 모드'), 'paused_simulation');
    expect(screen.getAllByText(/NON_REALTIME_SIMULATION/).length).toBeGreaterThan(0);
    expect(screen.getByText(/한 회차 최대 30 SIM초 · 600 WALL초/)).toBeInTheDocument();
    const total = screen.getByLabelText('전체 평가 WALL 예산 (초)');
    expect(total).toHaveValue(7200);
    expect(total).toHaveAttribute('max', '21600');
    fireEvent.change(total, { target: { value: '9000' } });
    expect(total).toHaveValue(9000);
    expect(screen.getByText(/실시간 100ms\/80ms 통과가 아닙니다/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '불변 작업 정의 저장' })).toBeDisabled();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.teach).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
    await userEvent.selectOptions(screen.getByLabelText('실행 시간 모드'), 'legacy');
    expect(screen.queryByLabelText('전체 평가 WALL 예산 (초)')).not.toBeInTheDocument();
  });

  it('shows an existing simulation-only project as blocked rather than exposing old real-time controls', async () => {
    const api = learningApi();
    const current = await api.capabilities();
    api.capabilities.mockResolvedValue({ ...current, simulation_learning: simulationLearning });
    const item = projectSchema.parse(projectRecord());
    api.projects.mockResolvedValue({ items: [{ item, etag: 'test-only-etag' }] });
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${item.id}`);
    render(<TeachingStudio api={api} environments={seventyEnvironments} />);
    expect(await screen.findByText(/전체 평가 WALL 예산: 7,200초/)).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '새 직접 시연 세션 시작' })).not.toBeInTheDocument();
    expect(api.train).not.toHaveBeenCalled();
    expect(api.evaluate).not.toHaveBeenCalled();
  });

  it('accepts a sixty-second plan only under an explicitly matching v2 project/profile', () => {
    const original = projectRecord();
    const v2 = {
      ...original, control_profile_id: 'franka-position-hold-10hz-paused-v2',
      evaluation_plan: { ...original.evaluation_plan, control_profile_id: 'franka-position-hold-10hz-paused-v2', max_simulation_seconds: 60 },
    };
    expect(projectSchema.parse(v2).evaluation_plan).toHaveProperty('max_simulation_seconds', 60);
    expect(() => projectSchema.parse({ ...v2, control_profile_id: original.control_profile_id })).toThrow();
    expect(() => projectSchema.parse({ ...v2, evaluation_plan: { ...original.evaluation_plan, max_simulation_seconds: 60 } })).toThrow();
    expect(projectSchema.parse(original).evaluation_plan).not.toHaveProperty('control_profile_id');
  });

  it('switches to v2 only by explicit selection, preserving the original default and wall budget', async () => {
    const api = learningApi();
    api.capabilities.mockResolvedValue({ ...await api.capabilities(), simulation_learning: simulationLearning });
    render(<TeachingStudio api={api} environments={seventyEnvironments} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    await userEvent.selectOptions(screen.getByLabelText('실행 시간 모드'), 'paused_simulation');
    const profile = screen.getByLabelText('시뮬레이션 예산 버전');
    expect(profile).toHaveValue('franka-position-hold-10hz-paused-v1');
    expect(screen.getByText('한 회차 최대 30 SIM초 · 600 WALL초')).toBeInTheDocument();
    const provenanceLabels = ['검토된 control profile SHA256', '고정된 평가 기준 SHA256', '모델 독립 scene conditions SHA256'];
    for (const label of provenanceLabels) {
      fireEvent.change(screen.getByLabelText(label), { target: { value: 'a'.repeat(64) } });
    }
    await userEvent.selectOptions(profile, 'franka-position-hold-10hz-paused-v2');
    expect(profile).toHaveFocus();
    for (const label of provenanceLabels) expect(screen.getByLabelText(label)).toHaveValue('');
    expect(screen.getByText('한 회차 최대 60 SIM초 · 600 WALL초')).toBeInTheDocument();
    expect(screen.getByText(/기존 v1 실행·결과를 v2 통과로 바꾸지 않습니다/)).toBeInTheDocument();
    expect(screen.getByLabelText('전체 평가 WALL 예산 (초)')).toHaveValue(7200);
    expect(screen.getByRole('button', { name: '불변 작업 정의 저장' })).toBeDisabled();
    expect(api.createProject).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
  });

  it('decodes actual sixty-SIM-second reference metrics only under the same explicit v2 profile', () => {
    const old = referenceFixture().collection.item;
    const v2 = {
      ...old, control_profile_id: 'franka-position-hold-10hz-paused-v2',
      command: { ...old.command, profile_id: 'franka-position-hold-10hz-paused-v2', max_simulation_steps: 3600 },
      execution: { ...old.execution, simulation_runtime: {
        ...old.execution!.simulation_runtime, profile_id: 'franka-position-hold-10hz-paused-v2',
        simulation_steps: 3600, applied_action_count: 3600, simulation_elapsed_seconds: 60,
      } },
    };
    expect(referenceCollectionSchema.parse(v2).execution?.simulation_runtime.simulation_elapsed_seconds).toBe(60);
    expect(() => referenceCollectionSchema.parse({ ...v2, control_profile_id: old.control_profile_id })).toThrow();
    expect(() => referenceCollectionSchema.parse({ ...v2, command: old.command })).toThrow();
  });

  it('submits only the selected version cases when v1 and v2 have the same physical seeds', async () => {
    const api = learningApi();
    api.capabilities.mockResolvedValue({
      ...await api.capabilities(), simulation_learning: { ...simulationLearning, reference_generation_enabled: true },
    });
    api.createProject.mockRejectedValue(new Error('TEST ONLY no live project created.'));
    const environments = [1, 2].flatMap((version) => seventyEnvironments.map((item) => ({
      ...item, environment_id: `v${version}-${item.environment_id}`, display_name: `TEST v${version} ${item.environment_id}`,
      revision: `${version}${item.revision.slice(1)}`,
      document: { ...item.document, environment_id: `v${version}-${item.environment_id}`, learning_execution: {
        schema: `physicalai.paused-simulation/v${version}`, execution_timing: 'paused_simulation',
        profile_id: `franka-position-hold-10hz-paused-v${version}`,
        max_simulation_seconds: version * 30, max_wall_seconds: 600,
      } },
    })));
    render(<TeachingStudio api={api} environments={environments} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    await userEvent.selectOptions(screen.getByLabelText('실행 시간 모드'), 'paused_simulation');
    await userEvent.selectOptions(screen.getByLabelText('시뮬레이션 예산 버전'), 'franka-position-hold-10hz-paused-v2');
    await userEvent.selectOptions(screen.getByLabelText('저장된 LIVE 환경'), 'v2-case-000');
    fireEvent.change(screen.getByLabelText('프로젝트 이름'), { target: { value: 'TEST ONLY v2 definition' } });
    fireEvent.change(screen.getByLabelText('운영자가 검토·등록한 P0 release ID'), { target: { value: '90000000-1111-4111-8111-111111111111' } });
    fireEvent.change(screen.getByLabelText('검토된 control profile SHA256'), { target: { value: 'a'.repeat(64) } });
    fireEvent.change(screen.getByLabelText('고정된 평가 기준 SHA256'), { target: { value: 'b'.repeat(64) } });
    fireEvent.change(screen.getByLabelText('모델 독립 scene conditions SHA256'), { target: { value: 'c'.repeat(64) } });
    fireEvent.change(screen.getByLabelText('작업별 최대 승인 금액 (USD)'), { target: { value: '10' } });
    fireEvent.change(screen.getByLabelText('학습에서 제외할 seed 20~100개 (쉼표 구분)'), { target: { value: Array.from({ length: 20 }, (_, index) => 30001 + index).join(',') } });
    fireEvent.change(screen.getByRole('searchbox', { name: '시연 배치 검색' }), { target: { value: 'case-000' } });
    expect(screen.queryByRole('checkbox', { name: /v1-case-000/ })).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('checkbox', { name: /v2-case-000/ }));
    await userEvent.click(screen.getByRole('button', { name: '불변 작업 정의 저장' }));
    await waitFor(() => expect(api.createProject).toHaveBeenCalledTimes(1));
    const input = api.createProject.mock.calls[0]![0];
    expect(input.control_profile_id).toBe('franka-position-hold-10hz-paused-v2');
    expect(input.evaluation_plan).toMatchObject({ control_profile_id: input.control_profile_id, max_simulation_seconds: 60, max_wall_seconds: 600 });
    expect(input.evaluation_plan.cases).toHaveLength(20);
    expect(input.evaluation_plan.cases.every((item) => item.environment_id.startsWith('v2-'))).toBe(true);
    expect(input.teaching_cases[0]?.environment_id).toBe('v2-case-000');
    expect(api.train).not.toHaveBeenCalled();
  });
});
