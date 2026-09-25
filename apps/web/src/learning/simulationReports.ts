import { z } from 'zod';
import { pausedProfileSchema, pausedProfileSteps } from './pausedProfiles';

const sha = z.string().regex(/^[a-f0-9]{64}$/);
const count = z.number().int().nonnegative();
const duration = z.number().nonnegative();
const position = z.tuple([z.number(), z.number(), z.number()]);
const role = z.enum(['before', 'after', 'reference', 'candidate']);
const wall = z.object({
  samples: count, p50: duration.nullable(), p95: duration.nullable(), max: duration.nullable(),
}).strict();
const task = z.object({
  predicate_version: z.literal('physicalai.measured-grasp-transport/v1'),
  predicate_source: z.object({
    commit: z.string(), path: z.string(), git_blob: z.string(), sha256: sha, symbol: z.literal('TaskWatchdog'),
  }).strict(),
  grasp_evidence_kind: z.literal('measured_lift_proximity_finger_gap_no_contact_sensor'),
  grasp_verified: z.boolean(), settled: z.boolean(), settled_simulation_seconds: duration,
  final_goal_error_m: duration, maximum_tcp_speed_m_s: duration, safety_violation_count: count,
}).strict();
const trial = z.object({
  episode_id: z.string(), seed: count, attempt: z.literal(0), policy: role, model_sha256: sha.nullable(),
  environment_id: z.string(), revision: sha, observed_initial_pose_m: position, scene_builder_sha256: sha,
  final_pose_m: position, destination_id: z.string(), terminated: z.boolean(), truncated: z.boolean(),
  failure_reason: z.string().nullable(), safety_violation_count: count,
  policy_predict_calls: count, applied_action_count: count.max(3600), reference_route_calls: count,
  final_images: z.object({ inspection: sha, overview: sha }).strict(),
  phase_wall_ms: z.object({ policy: wall, observation: wall, hold: wall, interval: wall, heartbeat: wall }).strict(),
  wall_duration_ms: duration, simulation_duration_ms: duration, task_evidence: task,
  physical_success: z.boolean(), position_error_m: position,
}).strict();

export const simulationReportBody = z.object({
  execution_timing: z.literal('paused_simulation'), real_time_admission: z.literal(false),
  native_schema: z.enum(['physicalai.smolvla-paired-report/v2', 'physicalai.smolvla-bootstrap-report/v2']),
  comparison_kind: z.enum(['paired_policy', 'reference_bootstrap']),
  control_profile_id: pausedProfileSchema,
  control_profile_sha256: sha, criteria_sha256: sha, frozen_plan_sha256: sha,
  evaluation_plan_sha256: sha, native_plan_sha256: sha, results_sha256: sha, runtime_sha256: sha,
  report_sha256: sha, artifact_id: z.uuid(),
  before_model_sha256: sha.nullable(), after_model_sha256: sha.nullable(),
  candidate_model_sha256: sha.nullable(), reference_controller_sha256: sha.nullable(),
  trials: z.array(trial).length(40),
  counts: z.partialRecord(role, z.object({ total: z.literal(20), success: count.max(20) })),
  success_rates: z.partialRecord(role, z.number().min(0).max(1)),
  absolute_success_rate_improvement: z.number().min(-1).max(1),
  latency_wall_ms: z.partialRecord(role, wall),
  total_wall_duration_ms: duration, total_simulation_duration_ms: duration,
  safety_violation_count: count, resource_violation_count: count, total_trial_count: z.literal(40),
  quality_gate_passed: z.boolean(), conclusion: z.enum(['improved', 'not_improved', 'inconclusive']),
  live_gpu_verified: z.literal(true),
}).strict();
export function consistentSimulationReport(report: Omit<z.infer<typeof simulationReportBody>, 'artifact_id'>, context: z.RefinementCtx) {
  if (report.trials.some((trial) => trial.applied_action_count > pausedProfileSteps[report.control_profile_id] ||
    Math.abs(trial.simulation_duration_ms - trial.applied_action_count * 1000 / 60) > 1e-6)) {
    context.addIssue({ code: 'custom', message: 'Trial ticks/duration exceed the explicit report profile.' });
  }
  const expected = report.comparison_kind === 'reference_bootstrap' ? ['candidate', 'reference'] : ['after', 'before'];
  for (const values of [report.counts, report.success_rates, report.latency_wall_ms]) {
    if (JSON.stringify(Object.keys(values).sort()) !== JSON.stringify(expected)) {
      context.addIssue({ code: 'custom', message: 'Both exact evaluated controller roles are required.' });
    }
  }
}
export const simulationReportSchema = simulationReportBody.superRefine(consistentSimulationReport);
export const publicSimulationReportSchema = simulationReportBody.omit({ artifact_id: true }).superRefine(consistentSimulationReport);

export type SimulationReport = z.infer<typeof simulationReportSchema>;
