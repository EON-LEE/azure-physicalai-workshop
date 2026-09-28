import { z } from 'zod';
import { MANAGED_REPORT_SCHEMA, managedImportReceiptSchema, simulationReportSchema } from './simulationReports';
import { PAUSED_PROFILE_V1, PAUSED_PROFILE_V2, isPausedProfile, pausedProfileSchema, pausedProfileSteps } from './pausedProfiles';

const id = z.uuid();
const sha = z.string().regex(/^[a-f0-9]{64}$/);
const date = z.iso.datetime({ offset: true });
const count = z.number().int().nonnegative();
const base = { id, actor_id: id, created_at: date, updated_at: date };
const profile = z.literal('franka-position-hold-10hz-v1');
const timingProfile = z.enum(['franka-position-hold-10hz-v1', PAUSED_PROFILE_V1, PAUSED_PROFILE_V2]);
const timingMetadata = {
  execution_timing: z.literal('paused_simulation').optional(), real_time_admission: z.literal(false).optional(),
  control_profile_id: timingProfile.optional(), control_profile_sha256: sha.optional(),
  criteria_sha256: sha.optional(), frozen_plan_sha256: sha.optional(),
};
type TimingMetadata = z.infer<z.ZodObject<typeof timingMetadata>>;
function consistentTiming(value: TimingMetadata, context: z.RefinementCtx) {
  const paused = value.execution_timing === 'paused_simulation';
  if ((paused && (value.real_time_admission !== false || !isPausedProfile(value.control_profile_id) ||
    !value.control_profile_sha256 || !value.criteria_sha256 || !value.frozen_plan_sha256)) ||
    (!paused && (isPausedProfile(value.control_profile_id) ||
      [value.real_time_admission, value.control_profile_sha256, value.criteria_sha256, value.frozen_plan_sha256].some((item) => item !== undefined)))) {
    context.addIssue({ code: 'custom', message: 'Artifact timing/profile provenance is incomplete or mixed.' });
  }
}
const policyType = z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla']);
export const teachingCaseSchema = z.object({
  case_id: z.string().min(1), environment_id: z.string().min(1), revision: sha,
  seed: count, split: z.enum(['train', 'validation']),
});
const captureSchema = z.object({
  ...timingMetadata,
  episode_id: id, manifest_sha256: sha, artifact_id: id, frame_count: count,
  source: z.enum(['human_teleop', 'reference_controller', 'learned']), seed: count,
  task_id: z.string(), control_profile_id: timingProfile, source_model_sha256: sha.nullable(),
  case_id: z.string().nullable().default(null), environment_id: z.string().nullable().default(null),
  revision: sha.nullable().default(null), split: z.enum(['train', 'validation']).nullable().default(null),
}).superRefine((value, context) => {
  consistentTiming(value, context);
  if (value.execution_timing === 'paused_simulation' && isPausedProfile(value.control_profile_id) &&
    value.frame_count > pausedProfileSteps[value.control_profile_id] / 6) {
    context.addIssue({ code: 'custom', message: 'Capture exceeds its declared profile frame budget.' });
  }
});
export const budgetSchema = z.object({
  teaching_seconds: z.number().int().min(5).max(300),
  training_seconds: z.number().int().positive().max(86400),
  evaluation_seconds: z.number().int().positive().max(21600),
  optimizer_steps: z.number().int().positive().max(100000),
  maximum_cost_usd: z.string(),
});
export const evaluationPlanSchema = z.object({
  id, seeds: z.array(count).min(20).max(100), held_out_episode_ids: z.array(id),
  cases: z.array(z.object({ seed: count, environment_id: z.string(), revision: sha })).min(20).max(100),
  minimum_success_rate: z.number().min(.9).max(1),
  maximum_axis_error_m: z.number().positive().max(.04),
  maximum_inference_p95_ms: z.number().positive().max(80),
  max_step_seconds: z.number().int().positive().max(30),
  max_cartesian_speed_m_s: z.number().positive().max(.2),
}).strict();
export const pausedEvaluationPlanSchema = z.object({
  execution_timing: z.literal('paused_simulation'), real_time_admission: z.literal(false),
  control_profile_id: pausedProfileSchema.optional(),
  id, seeds: z.array(count).length(20), held_out_episode_ids: z.array(id),
  cases: z.array(z.object({ seed: count, environment_id: z.string(), revision: sha })).length(20),
  minimum_success_rate: z.number().min(.9).max(1),
  minimum_absolute_improvement: z.number().min(.05).max(1),
  maximum_axis_error_m: z.number().positive().max(.04),
  max_cartesian_speed_m_s: z.number().positive().max(.2),
  max_simulation_seconds: z.number().int().min(1).max(60),
  max_wall_seconds: z.number().int().min(1).max(600),
  max_observation_wall_ms: z.literal(2000), max_policy_wall_ms: z.literal(2000),
  max_hold_wall_ms: z.literal(2000), max_interval_wall_ms: z.literal(5000),
  max_heartbeat_wall_ms: z.literal(2000),
}).strict().refine((value) => value.max_simulation_seconds <= pausedProfileSteps[value.control_profile_id ?? PAUSED_PROFILE_V1] / 60);
export const projectSchema = z.object({
  ...base, kind: z.literal('project'), display_name: z.string(), task_id: z.string(),
  policy_type: policyType,
  instruction: z.string(), goal_station_id: z.string(), environment_id: z.string(), revision: sha,
  project_kind: z.enum(['adaptation', 'bootstrap']), baseline_release_id: id.nullable(),
  pretrained_artifact_id: id.nullable(), control_profile_id: timingProfile,
  execution_timing: z.literal('paused_simulation').optional(),
  real_time_admission: z.literal(false).optional(),
  control_profile_sha256: sha.optional(), criteria_sha256: sha.optional(), frozen_plan_sha256: sha.optional(),
  evaluation_plan: z.union([evaluationPlanSchema, pausedEvaluationPlanSchema]), evaluation_plan_sha256: sha, budget: budgetSchema,
  teaching_cases: z.array(teachingCaseSchema).max(1000).default([]),
}).superRefine((project, context) => {
  const paused = project.execution_timing === 'paused_simulation';
  if (paused !== isPausedProfile(project.control_profile_id) ||
    paused !== ('execution_timing' in project.evaluation_plan) ||
    (paused && (project.real_time_admission !== false || !project.control_profile_sha256 || !project.criteria_sha256 || !project.frozen_plan_sha256 || project.policy_type !== 'smolvla')) ||
    (!paused && [project.real_time_admission, project.control_profile_sha256, project.criteria_sha256, project.frozen_plan_sha256].some((value) => value !== undefined))) {
    context.addIssue({ code: 'custom', message: 'Project timing, profile and immutable provenance must match.' });
  }
  if ('execution_timing' in project.evaluation_plan &&
    (project.evaluation_plan.control_profile_id ?? PAUSED_PROFILE_V1) !== project.control_profile_id) {
    context.addIssue({ code: 'custom', message: 'Project and plan must declare the same paused profile version.' });
  }
});
export const datasetSchema = z.object({
  ...timingMetadata,
  ...base, kind: z.literal('dataset'), project_id: id, status: z.literal('ready'),
  artifact_id: id, manifest_sha256: sha, episode_ids: z.array(id).min(1), seeds: z.array(count),
  human_teleop_count: count, reference_controller_count: count, learned_policy_count: count,
  evaluation_plan_sha256: sha,
  captures: z.array(captureSchema).max(1000).default([]),
}).superRefine(consistentTiming);
export const artifactOperationSchema = z.object({
  ...base, kind: z.literal('artifact_operation'), project_id: id, target_id: id,
  operation: z.enum(['capture', 'dataset', 'managed_evaluation', 'external_training']), work_sha256: sha,
  deadline: date, max_bytes: count, max_files: count,
  status: z.enum(['queued', 'running', 'ready', 'failed', 'timed_out', 'uncertain']),
  phase: z.enum(['queued', 'processing', 'manifest_committed', 'report_committed', 'import_committed', 'stopped']),
  result: z.object({
    artifact_id: id, manifest_sha256: sha, capture: captureSchema.nullable(),
    managed_evaluation: managedImportReceiptSchema.optional(),
    external_training: z.lazy(() => externalTrainingResultSchema).optional(),
  }).nullable(),
  error_code: z.string().nullable(), message: z.string().nullable(),
}).refine((value) => (value.status === 'ready') === (value.result !== null))
  .refine((value) => !value.result || (value.operation === 'managed_evaluation'
    ? value.result.managed_evaluation?.evaluation_run_id === value.target_id &&
      value.result.managed_evaluation.operation_id === value.id &&
      value.result.managed_evaluation.report.report_sha256 === value.result.manifest_sha256 &&
      value.result.managed_evaluation.report.artifact_id === value.result.artifact_id &&
      value.result.capture === null
    : value.result.managed_evaluation === undefined))
  .refine((value) => !value.result || (value.operation === 'external_training'
    ? value.result.external_training?.record.id === value.target_id &&
      value.result.external_training.record.project_id === value.project_id &&
      value.result.external_training.candidate.artifact_id === value.result.artifact_id &&
      value.result.external_training.candidate.manifest_sha256 === value.result.manifest_sha256 &&
      value.result.capture === null
    : value.result.external_training === undefined));
export type ArtifactOperation = z.infer<typeof artifactOperationSchema>;
export const teachingSchema = z.object({
  ...timingMetadata,
  ...base, kind: z.literal('teaching'), project_id: id,
  teaching_case: teachingCaseSchema.nullable().default(null),
  source: z.enum(['human_teleop', 'reference_controller']),
  status: z.enum(['starting', 'recording', 'finishing', 'finalizing', 'uploading', 'ready', 'cancelling', 'cancelled', 'invalid', 'blocked']),
  lease_id: id, epoch: id, command_id: id, expires_at: date, last_sequence: count,
  input_expires_at: date.nullable(), physical_status: z.string().nullable(),
  capture: captureSchema.nullable(),
  artifact_operation_id: id.nullable().optional(),
  verification_status: z.enum(['queued', 'running', 'ready', 'failed', 'timed_out', 'uncertain']).nullable().optional(),
  error_code: z.string().nullable(), message: z.string().nullable(),
}).superRefine(consistentTiming);
export const referenceCollectionSchema = z.object({
  ...base, ...timingMetadata, kind: z.literal('reference_collection'),
  project_id: id, teaching_case: teachingCaseSchema, source: z.literal('reference_controller'),
  command_id: id, epoch: id,
  command: z.object({
    schema: z.literal('physicalai.simulation-episode-command/v1'),
    execution_timing: z.literal('paused_simulation'), real_time_admission: z.literal(false),
    profile_id: pausedProfileSchema,
    controller: z.literal('reference_controller'), authorization_kind: z.literal('reference_collection'),
    authorization_id: id, wall_expires_at: date, max_simulation_steps: z.number().int().min(6).max(3600).multipleOf(6),
  }).refine((value) => value.max_simulation_steps <= pausedProfileSteps[value.profile_id]),
  target_position_m: z.tuple([z.number(), z.number(), z.number()]), goal_tolerance_m: z.number().positive().max(.04),
  runtime_catalog_record_sha256: sha, source_revision: z.string(), simulator_image_digest: z.string(),
  status: z.enum(['starting', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'timed_out', 'unconfirmed']),
  execution: z.object({
    command_id: id, status: z.string(),
    final_position: z.tuple([z.number(), z.number(), z.number()]).nullable().optional(),
    simulation_runtime: z.object({
      execution_timing: z.literal('paused_simulation'), real_time_admission: z.literal(false),
      profile_id: pausedProfileSchema.optional(),
      controller: z.literal('reference_controller'), control_profile_sha256: sha,
      phase: z.enum(['queued', 'observing', 'predicting', 'applying', 'idle', 'stopped']), wall_elapsed_ms: z.number().nonnegative(), simulation_steps: count.max(3600),
      simulation_elapsed_seconds: z.number().min(0).max(60), policy_predict_calls: z.literal(0),
      applied_model_sha256: z.null(), applied_action_count: count, reference_route_calls: count,
    }).refine((value) => value.simulation_steps <= pausedProfileSteps[value.profile_id ?? PAUSED_PROFILE_V1] &&
      value.simulation_elapsed_seconds === value.simulation_steps / 60),
  }).nullable(),
  capture_status: z.enum(['pending', 'verifying', 'ready', 'invalid']),
  capture: captureSchema.nullable(), artifact_operation_id: id.nullable(),
  error_code: z.string().nullable(), message: z.string().nullable(),
}).superRefine((value, context) => {
  consistentTiming(value, context);
  if (value.execution_timing !== 'paused_simulation' || value.real_time_admission !== false ||
    value.command.profile_id !== value.control_profile_id ||
    (value.execution && (value.execution.command_id !== value.command_id ||
      (value.execution.simulation_runtime.profile_id ?? PAUSED_PROFILE_V1) !== value.control_profile_id ||
      value.execution.simulation_runtime.control_profile_sha256 !== value.control_profile_sha256)) ||
    (value.capture_status === 'ready' && !value.capture)) {
    context.addIssue({ code: 'custom', message: 'Reference timing, command and evidence must match.' });
  }
});
export type ReferenceCollection = z.infer<typeof referenceCollectionSchema>;
const jobStatus = z.enum(['awaiting_import', 'submitting', 'submission_unknown', 'submitted', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'timed_out', 'blocked']);
const jobBase = {
  ...timingMetadata,
  ...base, project_id: id, status: jobStatus, backend_job_name: z.string(),
  policy_type: policyType,
  azure_job_id: z.string().nullable(), deadline: date, approved_cost_usd: z.string(),
  job_deadline_utc: date.nullable().optional(),
  backend_status: z.enum(['submitted', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'timed_out']).nullable().optional(),
  azure_status: z.string().max(64).nullable().optional(),
  cancellation: z.object({
    request_id: id, reason: z.enum(['user', 'deadline']), requested_at: date,
    state: z.enum(['claimed', 'acknowledged', 'uncertain', 'forbidden']),
    error_code: z.string().nullable(),
  }).nullable().optional(),
  specification_sha256: sha,
  metrics: z.object({ optimizer_steps: count.nullable(), loss: z.number().nonnegative().nullable(), measured_at: date.nullable() }),
  error_code: z.string().nullable(), message: z.string().nullable(),
};
export const trainingSchema = z.object({
  ...jobBase, kind: z.literal('training'), dataset_id: id, parent_release_id: id.nullable(),
  pretrained_artifact_id: id.nullable(),
  optimizer_steps: count, candidate_id: id.nullable(),
}).superRefine(consistentTiming).refine((value) => value.status !== 'awaiting_import');
export const trialSchema = z.object({
  seed: count, attempt: z.number().int().positive(), policy: z.enum(['before', 'after']),
  environment_id: z.string(), revision: sha,
  status: z.enum(['succeeded', 'failed', 'cancelled', 'timed_out']), model_sha256: sha,
  physical_success: z.boolean(), axis_error_m: z.tuple([z.number(), z.number(), z.number()]).nullable(),
  duration_seconds: z.number().nonnegative(), safety_violations: count,
  inference_p95_ms: z.number().nonnegative().nullable(), applied_action_count: count,
  policy_predict_calls: count, reference_route_calls: z.literal(0), recording_id: id.nullable(),
  message: z.string(),
});
export const reportSchema = z.object({
  comparison_kind: z.literal('paired_policy'),
  evaluation_plan_sha256: sha, before_model_sha256: sha, after_model_sha256: sha,
  trials: z.array(trialSchema).min(40).max(1000),
  conclusion: z.enum(['improved', 'not_improved', 'inconclusive']), quality_gate_passed: z.boolean(),
  report_sha256: sha, artifact_id: id,
});
export const bootstrapReportSchema = z.object({
  comparison_kind: z.literal('reference_bootstrap'),
  evaluation_plan_sha256: sha, candidate_model_sha256: sha, reference_controller_sha256: sha,
  trials: z.array(trialSchema.extend({
    policy: z.enum(['reference', 'candidate']), model_sha256: sha.nullable(),
    reference_route_calls: count,
  })).min(40).max(1000),
  quality_gate_passed: z.boolean(), report_sha256: sha, artifact_id: id,
});
export const evaluationSchema = z.object({
  ...jobBase, kind: z.literal('evaluation'), candidate_id: id, baseline_release_id: id.nullable(),
  provider: z.enum(['azure_ml', 'managed_batch']).optional(),
  import_operation_id: id.optional(), import_receipt: managedImportReceiptSchema.optional(),
  before_candidate_id: id.optional(),
  comparison_kind: z.enum(['paired_policy', 'reference_bootstrap']),
  evaluation_plan_sha256: sha, report: z.union([simulationReportSchema, reportSchema, bootstrapReportSchema]).nullable(),
}).superRefine((value, context) => {
  consistentTiming(value, context);
  const managed = value.provider === 'managed_batch';
  if (managed ? (
    value.execution_timing !== 'paused_simulation' || value.comparison_kind !== 'paired_policy' ||
    value.azure_job_id !== null || value.azure_status != null || value.backend_status != null || value.cancellation != null ||
    !['awaiting_import', 'succeeded'].includes(value.status) ||
    (value.baseline_release_id === null) === (value.before_candidate_id === undefined) ||
    (value.status === 'succeeded' ? (
      !value.import_receipt || !value.import_operation_id || !value.report ||
      value.import_receipt.operation_id !== value.import_operation_id ||
      value.import_receipt.evaluation_run_id !== value.id ||
      value.import_receipt.project_id !== value.project_id ||
      value.import_receipt.specification_sha256 !== value.specification_sha256 ||
      JSON.stringify(value.import_receipt.report) !== JSON.stringify(value.report)
    ) : (value.report !== null || value.import_receipt !== undefined))
  ) : (value.status === 'awaiting_import' || value.import_operation_id !== undefined ||
    value.import_receipt !== undefined || value.before_candidate_id !== undefined)) {
    context.addIssue({ code: 'custom', message: 'Managed artifact imports must not claim an Azure ML identity or unverified completion.' });
  }
  if (value.report && 'native_schema' in value.report &&
    (value.report.native_schema === MANAGED_REPORT_SCHEMA) !== managed) {
    context.addIssue({ code: 'custom', message: 'Report and original provider must match.' });
  }
  if (value.report && ('execution_timing' in value.report) !== (value.execution_timing === 'paused_simulation')) {
    context.addIssue({ code: 'custom', message: 'Report and evaluation job timing must match.' });
  }
});
export const candidateSchema = z.object({
  ...timingMetadata,
  ...base, kind: z.literal('candidate'), project_id: id, dataset_id: id, training_run_id: id.nullable(),
  training_origin: z.enum(['api_training_run', 'external_native_import']).optional(),
  external_import_id: id.optional(),
  parent_release_id: id.nullable(), pretrained_artifact_id: id.nullable(), policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla', 'act_auxiliary']),
  model_sha256: sha, parent_model_sha256: sha, processor_sha256: sha, manifest_sha256: sha,
  artifact_id: id, optimizer_steps: count, azure_job_id: z.string(),
  source_commit: z.string(), model_revision: z.string(), control_profile_id: timingProfile,
}).superRefine(consistentTiming).refine((value) => value.training_origin === 'external_native_import'
  ? value.training_run_id === null && value.external_import_id !== undefined &&
    value.parent_release_id === null && value.pretrained_artifact_id === null &&
    value.policy_type === 'smolvla' && value.execution_timing === 'paused_simulation'
  : value.training_run_id !== null && value.external_import_id === undefined);
export const externalTrainingImportSchema = z.object({
  ...base, ...timingMetadata, kind: z.literal('external_import'),
  schema: z.literal('physicalai.external-training-import/v1'),
  project_id: id, imported_at: date, native_job_name: id, azure_job_id: z.string(),
  azure_job_type: z.literal('command'), native_created_at: date, job_deadline_utc: date,
  approval_sha256: sha, plan_sha256: sha, plan_archive_sha256: sha, configuration_sha256: sha,
  specification_sha256: sha, image_qualification_sha256: sha, environment_image: z.string(),
  managed_identity_client_id: id, code_snapshot_sha256: sha, static_source_sha256: sha,
  completion_sha256: sha, result_sha256: sha, transfer_sha256: sha, model_sha256: sha, parent_model_sha256: sha,
  backbone_sha256: sha, raw_manifest_sha256: sha, candidate_id: id, dataset_id: id,
  optimizer_steps: count.positive(), learning_quality_verified: z.literal(false),
}).superRefine(consistentTiming).refine((value) =>
  value.execution_timing === 'paused_simulation' &&
  Date.parse(value.imported_at) > Date.parse(value.native_created_at) &&
  value.created_at === value.imported_at && value.updated_at === value.imported_at &&
  value.azure_job_id.endsWith(`/jobs/${value.native_job_name}`));
const externalTrainingResultSchema = z.object({
  record: externalTrainingImportSchema, candidate: candidateSchema, dataset: datasetSchema,
}).refine((value) => value.candidate.external_import_id === value.record.id &&
  value.candidate.training_origin === 'external_native_import' &&
  value.candidate.id === value.record.candidate_id && value.dataset.id === value.record.dataset_id &&
  value.candidate.dataset_id === value.dataset.id && value.candidate.azure_job_id === value.record.azure_job_id &&
  value.candidate.model_sha256 === value.record.model_sha256 &&
  value.dataset.manifest_sha256 === value.record.raw_manifest_sha256);
export const releaseSchema = z.object({
  ...timingMetadata,
  ...base, kind: z.literal('release'), project_id: id, candidate_id: id, evaluation_run_id: id,
  policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla', 'act_auxiliary']), model_sha256: sha, processor_sha256: sha,
  manifest_sha256: sha, artifact_id: id, environment_id: z.string(), revision: sha,
  task_id: z.string(), goal_station_id: z.string(), instruction: z.string(),
  control_profile_id: timingProfile, evaluation_plan_sha256: sha, reviewed_by: id,
  comparison_kind: z.enum(['paired_policy', 'reference_bootstrap']),
  environment_cases: z.array(z.object({ environment_id: z.string(), revision: sha })),
}).superRefine(consistentTiming);
export const grantSchema = z.object({
  ...base, kind: z.literal('control_grant'), session_id: id, lease_id: id, epoch: id,
  sequence: z.number().int().positive(), delta_xyz_m: z.tuple([z.number(), z.number(), z.number()]),
  gripper: z.enum(['open', 'close', 'hold']), expires_at: date, consumed_by: id.nullable(),
});
export const simulationLearningSchema = z.object({
  execution_timing: z.literal('paused_simulation'), real_time_admission: z.literal(false),
  supported: z.literal(true), enabled: z.boolean(),
  status: z.string(), message: z.string(),
  reference_generation_enabled: z.boolean().default(false),
  training_enabled: z.boolean().default(false),
  evaluation_enabled: z.boolean().default(false),
  release_enabled: z.boolean().default(false),
});
export type SimulationLearningCapability = z.infer<typeof simulationLearningSchema>;
export const capabilitiesSchema = z.object({
  enabled: z.boolean(), status: z.enum(['disabled', 'blocked', 'configured']), message: z.string(),
  policy_types: z.array(policyType), control_profiles: z.array(profile),
  training_verified: z.literal(false), coach_configured: z.boolean(),
  bootstrap_allowed: z.boolean().default(false),
  model_admission: z.enum(['configured_not_verified', 'license_or_hardware_unapproved']).optional(),
  simulation_learning: simulationLearningSchema.optional(),
});
export const coachSchema = z.object({
  request_id: id, model_response_id: z.string().min(1), authority: z.literal('proposal_only'),
  proposal: z.object({
    project_id: id,
    action: z.enum(['define_task', 'review_demonstrations', 'propose_training', 'explain_evaluation', 'select_release']),
    summary: z.string(), dataset_id: id.nullable(), optimizer_steps: count.nullable(),
    selected_release_id: id.nullable(),
  }),
});
export const jobSchema = z.discriminatedUnion('kind', [trainingSchema, evaluationSchema]);
export const recordSchema = z.discriminatedUnion('kind', [projectSchema, teachingSchema, referenceCollectionSchema, datasetSchema, trainingSchema, evaluationSchema, candidateSchema, releaseSchema, externalTrainingImportSchema]);
export const resource = <T extends z.ZodType>(item: T) => z.object({ item, etag: z.string().min(1) });
export const resourceList = <T extends z.ZodType>(item: T) => z.object({ items: z.array(resource(item)).max(50) });
export type Project = z.infer<typeof projectSchema>;
export type Dataset = z.infer<typeof datasetSchema>;
export type Teaching = z.infer<typeof teachingSchema>;
export type TeachingCase = z.infer<typeof teachingCaseSchema>;
export type Job = z.infer<typeof jobSchema>;
export type Evaluation = z.infer<typeof evaluationSchema>;
export type Candidate = z.infer<typeof candidateSchema>;
export type PolicyRelease = z.infer<typeof releaseSchema>;
export type Grant = z.infer<typeof grantSchema>;
export type Capabilities = z.infer<typeof capabilitiesSchema>;
export type CoachResponse = z.infer<typeof coachSchema>;
export type LearningRecord = z.infer<typeof recordSchema>;
export interface Resource<T> { item: T; etag: string }
export interface JogBody {
  request_id: string; lease_id: string; epoch: string; sequence: number;
  deadman: boolean; delta_xyz_m: [number, number, number]; gripper: 'open' | 'close' | 'hold';
  grant_id?: string;
}
export interface TrainingBody {
  request_id: string; dataset_id: string; parent_release_id: string | null; policy_type: z.infer<typeof policyType>; pretrained_artifact_id?: string | null;
  optimizer_steps: number; paid_approved: true; maximum_cost_usd: string;
}
export type CreateProjectBody = Omit<Project, 'id' | 'actor_id' | 'created_at' | 'updated_at' | 'kind' | 'evaluation_plan_sha256'> & { request_id: string };
export interface LearningApi {
  capabilities(signal?: AbortSignal): Promise<Capabilities>;
  projects(signal?: AbortSignal): Promise<{ items: Resource<Project>[] }>;
  createProject(body: CreateProjectBody, signal?: AbortSignal): Promise<Resource<Project>>;
  records(projectId: string, kind: LearningRecord['kind'], signal?: AbortSignal): Promise<{ items: Resource<LearningRecord>[] }>;
  teach(projectId: string, body: { request_id: string; source: 'human_teleop' | 'reference_controller'; motion_approved: true; case_id?: string }, etag: string, signal?: AbortSignal): Promise<Resource<Teaching>>;
  teaching(id: string, signal?: AbortSignal): Promise<Resource<Teaching>>;
  arm(id: string, body: JogBody, etag: string, signal?: AbortSignal): Promise<Resource<Grant>>;
  jog(id: string, body: JogBody, etag: string, signal?: AbortSignal): Promise<Resource<Teaching>>;
  teachingControl(id: string, action: 'finish' | 'cancel', body: { request_id: string; lease_id: string; epoch: string }, etag: string, signal?: AbortSignal): Promise<Resource<Teaching>>;
  startReference(projectId: string, body: { request_id: string; case_id: string; motion_approved: true }, etag: string, signal?: AbortSignal): Promise<Resource<ReferenceCollection>>;
  reference(id: string, signal?: AbortSignal): Promise<Resource<ReferenceCollection>>;
  cancelReference(id: string, signal?: AbortSignal): Promise<Resource<ReferenceCollection>>;
  seal(projectId: string, body: { request_id: string; teaching_session_ids: string[]; reference_collection_ids?: string[] }, etag: string, signal?: AbortSignal): Promise<Resource<Dataset | ArtifactOperation>>;
  artifactOperation(id: string, signal?: AbortSignal): Promise<Resource<ArtifactOperation>>;
  dataset(id: string, signal?: AbortSignal): Promise<Resource<Dataset>>;
  train(projectId: string, body: TrainingBody, etag: string, signal?: AbortSignal): Promise<Resource<Job>>;
  evaluate(projectId: string, body: { request_id: string; candidate_id: string; baseline_release_id: string | null; comparison_kind?: 'paired_policy' | 'reference_bootstrap'; evaluation_plan_sha256: string; motion_approved: true; paid_approved: true; maximum_cost_usd: string }, etag: string, signal?: AbortSignal): Promise<Resource<Evaluation>>;
  job(id: string, signal?: AbortSignal): Promise<Resource<Job>>;
  reportDocument(id: string, signal?: AbortSignal): Promise<Blob>;
  cancelJob(id: string, requestId: string, etag: string, signal?: AbortSignal): Promise<Resource<Job>>;
  release(body: { request_id: string; candidate_id: string; evaluation_run_id: string; release_approved: true }, etag: string, signal?: AbortSignal): Promise<Resource<PolicyRelease>>;
  coach(projectId: string, body: { request_id: string; instruction: string; dataset_id: string | null; evaluation_run_id: string | null }, signal?: AbortSignal): Promise<CoachResponse>;
}
export const terminalJob = (job: Job) => ['succeeded', 'failed', 'cancelled', 'timed_out', 'blocked'].includes(job.status);
export const sourceLabel = (source: string) => source === 'human_teleop' ? '사람의 직접 시연' : source === 'reference_controller' ? '기준 제어기 시연' : '정책 생성 시연';
