import { projectSchema, type ReferenceCollection, type Resource } from '../../src/learning/contracts';
import { learningFixture } from './learning-data';

export function referenceFixture() {
  const original = learningFixture().project;
  if ('execution_timing' in original.item.evaluation_plan) throw new Error('Expected legacy fixture plan.');
  const { maximum_inference_p95_ms: _latency, max_step_seconds: _time, ...plan } = original.item.evaluation_plan;
  const project = { ...original, item: projectSchema.parse({
    ...original.item, display_name: 'TEST-ONLY REFERENCE 작업',
    execution_timing: 'paused_simulation', real_time_admission: false,
    control_profile_id: 'franka-position-hold-10hz-paused-v1',
    control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    evaluation_plan: {
      ...plan, execution_timing: 'paused_simulation', real_time_admission: false,
      minimum_absolute_improvement: .05, max_simulation_seconds: 30, max_wall_seconds: 600,
      max_observation_wall_ms: 2000, max_policy_wall_ms: 2000, max_hold_wall_ms: 2000,
      max_interval_wall_ms: 5000, max_heartbeat_wall_ms: 2000,
    },
  }) };
  const id = '30000000-1111-4111-8111-111111111111';
  const collection: Resource<ReferenceCollection> = { etag: 'test-only-reference-etag', item: {
    id, actor_id: project.item.actor_id, created_at: project.item.created_at, updated_at: project.item.updated_at,
    kind: 'reference_collection', project_id: project.item.id,
    teaching_case: project.item.teaching_cases[0]!, source: 'reference_controller',
    execution_timing: 'paused_simulation', real_time_admission: false,
    control_profile_id: 'franka-position-hold-10hz-paused-v1',
    control_profile_sha256: 'a'.repeat(64), criteria_sha256: 'b'.repeat(64), frozen_plan_sha256: 'c'.repeat(64),
    command_id: id, epoch: '40000000-1111-4111-8111-111111111111',
    command: {
      schema: 'physicalai.simulation-episode-command/v1', execution_timing: 'paused_simulation',
      real_time_admission: false, profile_id: 'franka-position-hold-10hz-paused-v1',
      controller: 'reference_controller', authorization_kind: 'reference_collection',
      authorization_id: '50000000-1111-4111-8111-111111111111',
      wall_expires_at: new Date(Date.now() + 600000).toISOString(), max_simulation_steps: 1800,
    },
    target_position_m: [.5, .2, .1], goal_tolerance_m: .04,
    runtime_catalog_record_sha256: 'd'.repeat(64), source_revision: 'e'.repeat(40),
    simulator_image_digest: `sha256:${'f'.repeat(64)}`, status: 'running',
    execution: { command_id: id, status: 'running', final_position: null, simulation_runtime: {
      execution_timing: 'paused_simulation', real_time_admission: false, controller: 'reference_controller',
      control_profile_sha256: 'a'.repeat(64), phase: 'observing', wall_elapsed_ms: 2000,
      simulation_steps: 6, simulation_elapsed_seconds: .1, policy_predict_calls: 0,
      applied_model_sha256: null, applied_action_count: 6, reference_route_calls: 1,
    } },
    capture_status: 'pending', capture: null, artifact_operation_id: null,
    error_code: null, message: 'TEST ONLY · reference processing; not human input or real GPU proof.',
  } };
  return { project, collection };
}
