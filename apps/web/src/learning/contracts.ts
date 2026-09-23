import { z } from 'zod';

const id = z.uuid();
const sha = z.string().regex(/^[a-f0-9]{64}$/);
const date = z.iso.datetime({ offset: true });
const count = z.number().int().nonnegative();
const base = { id, actor_id: id, created_at: date, updated_at: date };
const profile = z.literal('franka-position-hold-10hz-v1');
const policyType = z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla']);
export const teachingCaseSchema = z.object({
  case_id: z.string().min(1), environment_id: z.string().min(1), revision: sha,
  seed: count, split: z.enum(['train', 'validation']),
});
const captureSchema = z.object({
  episode_id: id, manifest_sha256: sha, artifact_id: id, frame_count: count,
  source: z.enum(['human_teleop', 'reference_controller', 'learned']), seed: count,
  task_id: z.string(), control_profile_id: profile, source_model_sha256: sha.nullable(),
  case_id: z.string().nullable().default(null), environment_id: z.string().nullable().default(null),
  revision: sha.nullable().default(null), split: z.enum(['train', 'validation']).nullable().default(null),
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
});
export const projectSchema = z.object({
  ...base, kind: z.literal('project'), display_name: z.string(), task_id: z.string(),
  policy_type: policyType,
  instruction: z.string(), goal_station_id: z.string(), environment_id: z.string(), revision: sha,
  project_kind: z.enum(['adaptation', 'bootstrap']), baseline_release_id: id.nullable(),
  pretrained_artifact_id: id.nullable(), control_profile_id: profile,
  evaluation_plan: evaluationPlanSchema, evaluation_plan_sha256: sha, budget: budgetSchema,
  teaching_cases: z.array(teachingCaseSchema).max(1000).default([]),
});
export const datasetSchema = z.object({
  ...base, kind: z.literal('dataset'), project_id: id, status: z.literal('ready'),
  artifact_id: id, manifest_sha256: sha, episode_ids: z.array(id).min(1), seeds: z.array(count),
  human_teleop_count: count, reference_controller_count: count, learned_policy_count: count,
  evaluation_plan_sha256: sha,
  captures: z.array(captureSchema).max(1000).default([]),
});
export const teachingSchema = z.object({
  ...base, kind: z.literal('teaching'), project_id: id,
  teaching_case: teachingCaseSchema.nullable().default(null),
  source: z.enum(['human_teleop', 'reference_controller']),
  status: z.enum(['starting', 'recording', 'finishing', 'finalizing', 'uploading', 'ready', 'cancelling', 'cancelled', 'invalid', 'blocked']),
  lease_id: id, epoch: id, command_id: id, expires_at: date, last_sequence: count,
  input_expires_at: date.nullable(), physical_status: z.string().nullable(),
  capture: captureSchema.nullable(),
  error_code: z.string().nullable(), message: z.string().nullable(),
});
const jobStatus = z.enum(['submitting', 'submission_unknown', 'submitted', 'running', 'cancelling', 'succeeded', 'failed', 'cancelled', 'timed_out', 'blocked']);
const jobBase = {
  ...base, project_id: id, status: jobStatus, backend_job_name: z.string(),
  policy_type: policyType,
  azure_job_id: z.string().nullable(), deadline: date, approved_cost_usd: z.string(),
  specification_sha256: sha,
  metrics: z.object({ optimizer_steps: count.nullable(), loss: z.number().nonnegative().nullable(), measured_at: date.nullable() }),
  error_code: z.string().nullable(), message: z.string().nullable(),
};
export const trainingSchema = z.object({
  ...jobBase, kind: z.literal('training'), dataset_id: id, parent_release_id: id.nullable(),
  pretrained_artifact_id: id.nullable(),
  optimizer_steps: count, candidate_id: id.nullable(),
});
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
  comparison_kind: z.enum(['paired_policy', 'reference_bootstrap']),
  evaluation_plan_sha256: sha, report: z.discriminatedUnion('comparison_kind', [reportSchema, bootstrapReportSchema]).nullable(),
});
export const candidateSchema = z.object({
  ...base, kind: z.literal('candidate'), project_id: id, dataset_id: id, training_run_id: id,
  parent_release_id: id.nullable(), pretrained_artifact_id: id.nullable(), policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla', 'act_auxiliary']),
  model_sha256: sha, parent_model_sha256: sha, processor_sha256: sha, manifest_sha256: sha,
  artifact_id: id, optimizer_steps: count, azure_job_id: z.string(),
  source_commit: z.string(), model_revision: z.string(), control_profile_id: profile,
});
export const releaseSchema = z.object({
  ...base, kind: z.literal('release'), project_id: id, candidate_id: id, evaluation_run_id: id,
  policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla', 'act_auxiliary']), model_sha256: sha, processor_sha256: sha,
  manifest_sha256: sha, artifact_id: id, environment_id: z.string(), revision: sha,
  task_id: z.string(), goal_station_id: z.string(), instruction: z.string(),
  control_profile_id: profile, evaluation_plan_sha256: sha, reviewed_by: id,
  comparison_kind: z.enum(['paired_policy', 'reference_bootstrap']),
  environment_cases: z.array(z.object({ environment_id: z.string(), revision: sha })),
});
export const grantSchema = z.object({
  ...base, kind: z.literal('control_grant'), session_id: id, lease_id: id, epoch: id,
  sequence: z.number().int().positive(), delta_xyz_m: z.tuple([z.number(), z.number(), z.number()]),
  gripper: z.enum(['open', 'close', 'hold']), expires_at: date, consumed_by: id.nullable(),
});
export const capabilitiesSchema = z.object({
  enabled: z.boolean(), status: z.enum(['disabled', 'blocked', 'configured']), message: z.string(),
  policy_types: z.array(policyType), control_profiles: z.array(profile),
  training_verified: z.literal(false), coach_configured: z.boolean(),
  bootstrap_allowed: z.boolean().default(false),
  model_admission: z.enum(['configured_not_verified', 'license_or_hardware_unapproved']).optional(),
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
export const recordSchema = z.discriminatedUnion('kind', [projectSchema, teachingSchema, datasetSchema, trainingSchema, evaluationSchema, candidateSchema, releaseSchema]);
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
  seal(projectId: string, body: { request_id: string; teaching_session_ids: string[] }, etag: string, signal?: AbortSignal): Promise<Resource<Dataset>>;
  train(projectId: string, body: TrainingBody, etag: string, signal?: AbortSignal): Promise<Resource<Job>>;
  evaluate(projectId: string, body: { request_id: string; candidate_id: string; baseline_release_id: string | null; comparison_kind?: 'paired_policy' | 'reference_bootstrap'; evaluation_plan_sha256: string; motion_approved: true; paid_approved: true; maximum_cost_usd: string }, etag: string, signal?: AbortSignal): Promise<Resource<Evaluation>>;
  job(id: string, signal?: AbortSignal): Promise<Resource<Job>>;
  cancelJob(id: string, requestId: string, etag: string, signal?: AbortSignal): Promise<Resource<Job>>;
  release(body: { request_id: string; candidate_id: string; evaluation_run_id: string; release_approved: true }, etag: string, signal?: AbortSignal): Promise<Resource<PolicyRelease>>;
  coach(projectId: string, body: { request_id: string; instruction: string; dataset_id: string | null; evaluation_run_id: string | null }, signal?: AbortSignal): Promise<CoachResponse>;
}
export const terminalJob = (job: Job) => ['succeeded', 'failed', 'cancelled', 'timed_out', 'blocked'].includes(job.status);
export const sourceLabel = (source: string) => source === 'human_teleop' ? '사람의 직접 시연' : source === 'reference_controller' ? '기준 제어기 시연' : '정책 생성 시연';
