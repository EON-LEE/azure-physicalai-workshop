import { z } from 'zod';
import type { LearningApi } from '../learning/contracts';

// Decode API responses without runtime code generation under the production CSP.
z.config({ jitless: true });

const id = z.string().min(1);
const timestamp = z.string().refine((value) => Number.isFinite(Date.parse(value)), 'Invalid timestamp');
const nullableId = id.nullable();

export const configSchema = z.object({
  api_version: z.literal('v1'),
  deployment: z.literal('azure'),
  auth: z.object({
    tenant_id: z.uuid(),
    client_id: z.uuid(),
    scope: z.string().regex(/^api:\/\/[^\s]+\/[^\s/]+$/),
  }),
});

export const runtimeSchema = z.object({
  deployment: z.literal('azure'),
  simulation: z.object({
    backend: z.literal('isaac_sim'),
    status: z.enum(['ready', 'loading', 'unavailable', 'occupied']),
    environment_id: nullableId,
    revision: nullableId,
    epoch: nullableId,
    physics_steps: z.number().int().nonnegative().nullable(),
    message: z.string().nullable(),
  }),
  agent: z.object({
    provider: z.literal('microsoft_foundry'),
    configured: z.boolean(),
  }),
  storage: z.object({ provider: z.literal('azure_cosmos_blob') }),
  release_ready: z.boolean(),
});

export const environmentSchema = z.object({
  environment_id: id,
  display_name: id,
  revision: id,
  document: z.record(z.string(), z.unknown()),
  created_at: timestamp,
  updated_at: timestamp,
});

export const environmentsSchema = z.object({
  items: z.array(environmentSchema).max(50),
  next_cursor: z.uuid().nullable().optional(),
});
export const templatesSchema = z.object({
  items: z.array(z.object({ name: id, document: z.record(z.string(), z.unknown()) })),
});
export const jsonSchemaSchema = z.record(z.string(), z.unknown());
export const activationSchema = z.object({
  activation_id: id,
  environment_id: id,
  revision: id,
  status: z.literal('loading'),
});

export const runStatusSchema = z.enum([
  'planning',
  'awaiting_approval',
  'running',
  'cancelling',
  'succeeded',
  'failed',
  'cancelled',
  'timed_out',
]);

const releasedPolicySchema = z.object({
  policy_release_id: z.uuid(), policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla']),
  model_sha256: id, processor_sha256: id, manifest_sha256: id,
  control_profile_id: id, task_id: id, goal_station_id: id, instruction: z.string(),
});
const inspectionPlanSchema = z.object({
  kind: z.literal('inspection').optional(),
  classification: z.enum(['accepted', 'rejected']),
  target_station_id: id, object_id: id, summary: z.string(),
  observation_id: id, epoch: id, state_revision: z.number().int().nonnegative(),
  model_response_id: id, expires_at: timestamp.optional(),
});
const skillPlanSchema = z.object({
  kind: z.literal('released_skill'), skill_plan_id: z.uuid(), policy_release_id: z.uuid(),
  policy_type: z.enum(['gr00t_n1_5', 'gr00t_n1_7', 'smolvla']), model_sha256: id,
  task_id: id, instruction: z.string(), target_station_id: id, object_id: id,
  summary: z.string(), observation_id: id, epoch: id,
  state_revision: z.number().int().nonnegative(), expires_at: timestamp,
}).strict();

export const runSchema = z.object({
  id,
  environment_id: id,
  revision: id,
  instruction: z.string(),
  execution_mode: z.enum(['inspection', 'released_skill']).optional(),
  policy: releasedPolicySchema.nullable().optional(),
  status: runStatusSchema,
  created_at: timestamp,
  updated_at: timestamp,
  plan: z.union([inspectionPlanSchema, skillPlanSchema]).nullable(),
  execution: z.object({
    command_id: id,
    status: id,
    // The HTTP contract does not yet constrain the final_position representation.
    final_position: z.unknown().optional(),
    completed_at: timestamp.nullable().optional(),
    policy_runtime: z.object({
      execution_mode: z.literal('learned'), policy_release_id: z.uuid(),
      policy_type: z.enum(['smolvla', 'gr00t_n1_5', 'gr00t_n1_7']),
      applied_model_sha: id.nullable(), control_profile_id: id.nullable(),
      policy_predict_calls: z.number().int().nonnegative(),
      applied_action_count: z.number().int().nonnegative(), reference_route_calls: z.literal(0),
    }).nullable().optional(),
  }).nullable(),
  error: z.object({
    code: id,
    message: z.string(),
    retryable: z.boolean(),
  }).nullable(),
  events: z.array(z.object({ kind: id, message: z.string(), at: timestamp })),
}).superRefine((run, context) => {
  if (run.plan && (run.plan.kind === 'released_skill') !== (run.execution_mode === 'released_skill')) {
    context.addIssue({ code: 'custom', message: 'The execution mode and plan kind do not match.' });
  }
  if (run.execution_mode === 'released_skill' && !run.policy) {
    context.addIssue({ code: 'custom', message: 'Released-skill execution requires the reviewed policy binding.' });
  }
});

export const runsSchema = z.object({ items: z.array(runSchema) });

export type PublicConfig = z.infer<typeof configSchema>;
export type RuntimeInfo = z.infer<typeof runtimeSchema>;
export type EnvironmentRecord = z.infer<typeof environmentSchema>;
export type Environments = z.infer<typeof environmentsSchema>;
export type EnvironmentTemplates = z.infer<typeof templatesSchema>;
export type Activation = z.infer<typeof activationSchema>;
export type RunRecord = z.infer<typeof runSchema>;
export type Runs = z.infer<typeof runsSchema>;
export type RunStatus = RunRecord['status'];
export type EnvironmentJsonSchema = z.infer<typeof jsonSchemaSchema>;
export type Camera = 'overview' | 'inspection';

interface RunInputScope {
  request_id: string;
  environment_id: string;
  revision: string;
}
export type CreateRunInput = RunInputScope & (
  { execution_mode?: 'inspection'; instruction: string; policy_release_id?: never }
  | { execution_mode: 'released_skill'; policy_release_id: string; instruction?: string }
);
export type RunApproval = string | { skill_plan_id: string };

export interface FrameImage {
  blob: Blob;
  frameId: string;
  capturedAt: string;
  physicsSteps: number;
}

export interface ConsoleApi {
  readonly learning?: LearningApi;
  getRuntime(signal?: AbortSignal): Promise<RuntimeInfo>;
  getEnvironments(signal?: AbortSignal, cursor?: string): Promise<Environments>;
  getEnvironmentSchema(signal?: AbortSignal): Promise<EnvironmentJsonSchema>;
  getTemplates(signal?: AbortSignal): Promise<EnvironmentTemplates>;
  saveEnvironment(documentJson: string, expectedRevision: string | null, signal?: AbortSignal): Promise<EnvironmentRecord>;
  activateEnvironment(id: string, revision: string, signal?: AbortSignal): Promise<Activation>;
  getFrame(id: string, revision: string, camera: Camera, signal?: AbortSignal): Promise<FrameImage>;
  getRuns(signal?: AbortSignal): Promise<Runs>;
  getRun(id: string, signal?: AbortSignal): Promise<RunRecord>;
  createRun(input: CreateRunInput, signal?: AbortSignal): Promise<RunRecord>;
  approveRun(id: string, approval: RunApproval, signal?: AbortSignal): Promise<RunRecord>;
  cancelRun(id: string, signal?: AbortSignal): Promise<RunRecord>;
  getObservation(id: string, signal?: AbortSignal): Promise<Blob>;
}

export function isTerminal(status: RunStatus): boolean {
  return ['succeeded', 'failed', 'cancelled', 'timed_out'].includes(status);
}

export function environmentMode(record: EnvironmentRecord): 'live' | 'replay' | null {
  const execution = record.document.execution;
  if (typeof execution !== 'object' || execution === null || !('mode' in execution)) return null;
  return execution.mode === 'live' || execution.mode === 'replay' ? execution.mode : null;
}

export function matchesRuntime(
  runtime: RuntimeInfo | null,
  environment: Pick<EnvironmentRecord, 'environment_id' | 'revision'> | null,
): boolean {
  return Boolean(
    runtime && environment &&
    runtime.simulation.status === 'ready' &&
    runtime.simulation.environment_id === environment.environment_id &&
    runtime.simulation.revision === environment.revision &&
    runtime.simulation.epoch !== null &&
    runtime.simulation.physics_steps !== null,
  );
}
