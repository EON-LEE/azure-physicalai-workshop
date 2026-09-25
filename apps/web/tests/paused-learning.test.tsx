import { fireEvent, render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it } from 'vitest';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import { datasetSchema, projectSchema, trainingSchema } from '../src/learning/contracts';
import { LearningJobPanel } from '../src/learning/LearningJobPanel';
import { learningApi, learningFixture } from './fixtures/learning';
import { seventyEnvironments } from './fixtures/environment-pages';

const simulationLearning = {
  execution_timing: 'paused_simulation' as const, real_time_admission: false as const,
  supported: true as const, enabled: false as const, status: 'producer_verifier_unavailable' as const,
  message: 'Test only: paused model/report adapters are not admitted.',
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
});
