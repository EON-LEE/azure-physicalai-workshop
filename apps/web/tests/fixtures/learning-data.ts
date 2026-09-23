import type {
  Dataset, Evaluation, Grant, Job, Project, Resource, Teaching,
} from '../../src/learning/contracts';

const uuid = (value: number) => `10000000-1111-4111-8111-${String(value).padStart(12, '0')}`;
const sha = (value: string) => value.repeat(64);
const base = (value: number) => ({
  id: uuid(value), actor_id: uuid(90),
  created_at: '2026-09-23T00:00:00Z', updated_at: '2026-09-23T00:00:00Z',
});

export function learningFixture() {
  const project: Resource<Project> = { etag: '"test-project-etag"', item: {
    ...base(1), kind: 'project', project_kind: 'adaptation', display_name: 'TEST-ONLY 작업 가르치기',
    task_id: 'part-kitting-v1', instruction: 'TEST-ONLY: 부품을 키팅 트레이에 놓습니다.',
    policy_type: 'smolvla',
    goal_station_id: 'accepted', environment_id: 'reference-cell', revision: sha('a'),
    baseline_release_id: uuid(2), pretrained_artifact_id: null, control_profile_id: 'franka-position-hold-10hz-v1',
    evaluation_plan_sha256: sha('b'),
    teaching_cases: [
      { case_id: 'train-anchor', environment_id: 'reference-cell', revision: sha('a'), seed: 10001, split: 'train' },
      { case_id: 'validation-20001', environment_id: 'validation-20001', revision: sha('b'), seed: 20001, split: 'validation' },
    ],
    evaluation_plan: {
      id: uuid(3), seeds: Array.from({ length: 20 }, (_, index) => 200 + index), held_out_episode_ids: [],
      cases: Array.from({ length: 20 }, (_, index) => ({ seed: 200 + index, environment_id: `test-held-out-${index}`, revision: sha('c') })),
      minimum_success_rate: .9, maximum_axis_error_m: .04, maximum_inference_p95_ms: 80,
      max_step_seconds: 30, max_cartesian_speed_m_s: .2,
    },
    budget: { teaching_seconds: 120, training_seconds: 3600, evaluation_seconds: 1800, optimizer_steps: 100, maximum_cost_usd: '10.00' },
  } };
  const dataset: Resource<Dataset> = { etag: '"test-dataset-etag"', item: {
    ...base(4), kind: 'dataset', project_id: project.item.id, status: 'ready', artifact_id: uuid(5),
    manifest_sha256: sha('d'), episode_ids: [uuid(6)], seeds: [10001],
    human_teleop_count: 0, reference_controller_count: 1, learned_policy_count: 0,
    evaluation_plan_sha256: project.item.evaluation_plan_sha256,
    captures: [{
      episode_id: uuid(6), artifact_id: uuid(5), manifest_sha256: sha('d'), frame_count: 20,
      source: 'reference_controller', seed: 10001, task_id: project.item.task_id,
      control_profile_id: project.item.control_profile_id, source_model_sha256: null,
      case_id: 'train-anchor', environment_id: 'reference-cell', revision: sha('a'), split: 'train',
    }],
  } };
  const teaching: Resource<Teaching> = { etag: '"test-teaching-etag"', item: {
    ...base(7), kind: 'teaching', project_id: project.item.id, source: 'reference_controller',
    teaching_case: project.item.teaching_cases[0]!,
    status: 'recording', lease_id: uuid(8), epoch: uuid(9), command_id: uuid(10),
    expires_at: new Date(Date.now() + 120000).toISOString(), last_sequence: 0, input_expires_at: null,
    physical_status: 'running', capture: null, error_code: null, message: null,
  } };
  const grant: Resource<Grant> = { etag: '"test-grant-etag"', item: {
    ...base(11), kind: 'control_grant', session_id: teaching.item.id, lease_id: teaching.item.lease_id,
    epoch: teaching.item.epoch, sequence: 1, delta_xyz_m: [.005, 0, 0], gripper: 'hold',
    expires_at: new Date(Date.now() + 1000).toISOString(), consumed_by: null,
  } };
  const training: Resource<Job> = { etag: '"test-training-etag"', item: {
    ...base(12), kind: 'training', project_id: project.item.id, status: 'submitted',
    policy_type: project.item.policy_type,
    backend_job_name: 'learning-test-only-job',
    azure_job_id: '/subscriptions/test-only/resourceGroups/test/providers/Microsoft.MachineLearningServices/workspaces/test/jobs/learning-test-only-job',
    deadline: new Date(Date.now() + 3600000).toISOString(), approved_cost_usd: '10.00', specification_sha256: sha('e'),
    metrics: { optimizer_steps: null, loss: null, measured_at: null }, error_code: null, message: null,
    dataset_id: dataset.item.id, parent_release_id: project.item.baseline_release_id,
    pretrained_artifact_id: null, optimizer_steps: 100, candidate_id: null,
  } };
  const evaluation: Resource<Evaluation> = { etag: '"test-evaluation-etag"', item: {
    ...base(13), kind: 'evaluation', project_id: project.item.id, status: 'succeeded', backend_job_name: 'learning-test-only-evaluation',
    policy_type: project.item.policy_type,
    azure_job_id: '/subscriptions/test-only/resourceGroups/test/providers/Microsoft.MachineLearningServices/workspaces/test/jobs/learning-test-only-evaluation',
    deadline: new Date(Date.now() + 3600000).toISOString(), approved_cost_usd: '10.00', specification_sha256: sha('e'),
    metrics: { optimizer_steps: null, loss: null, measured_at: null }, error_code: null, message: null,
    candidate_id: uuid(14), baseline_release_id: project.item.baseline_release_id,
    comparison_kind: 'paired_policy', evaluation_plan_sha256: sha('b'),
    report: {
      comparison_kind: 'paired_policy', evaluation_plan_sha256: sha('b'),
      before_model_sha256: sha('1'), after_model_sha256: sha('2'),
      conclusion: 'not_improved', quality_gate_passed: false, report_sha256: sha('3'), artifact_id: uuid(15),
      trials: (['before', 'after'] as const).flatMap((policy) => Array.from({ length: 20 }, (_, index) => ({
        seed: 200 + index, environment_id: `test-held-out-${index}`, revision: sha('c'),
        attempt: 1, policy, status: index < 10 ? 'succeeded' as const : 'failed' as const,
        model_sha256: policy === 'before' ? sha('1') : sha('2'), physical_success: index < 10,
        axis_error_m: [index < 10 ? .01 : .1, 0, 0] as [number, number, number],
        duration_seconds: 20, safety_violations: 0, inference_p95_ms: 70,
        applied_action_count: 20, policy_predict_calls: 20, reference_route_calls: 0 as const,
        recording_id: null, message: 'TEST-ONLY paired trial, not Azure/GPU validation.',
      }))),
    },
  } };
  return { project, dataset, teaching, grant, training, evaluation };
}
