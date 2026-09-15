import { z } from 'zod';

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

export const environmentsSchema = z.object({ items: z.array(environmentSchema) });
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

export const runSchema = z.object({
  id,
  environment_id: id,
  revision: id,
  instruction: z.string(),
  status: runStatusSchema,
  created_at: timestamp,
  updated_at: timestamp,
  plan: z.object({
    classification: z.enum(['accepted', 'rejected']),
    target_station_id: id,
    object_id: id,
    summary: z.string(),
    observation_id: id,
    epoch: id,
    state_revision: z.number().int().nonnegative(),
    model_response_id: id,
  }).nullable(),
  execution: z.object({
    command_id: id,
    status: id,
    // The HTTP contract does not yet constrain the final_position representation.
    final_position: z.unknown().optional(),
    completed_at: timestamp.nullable().optional(),
  }).nullable(),
  error: z.object({
    code: id,
    message: z.string(),
    retryable: z.boolean(),
  }).nullable(),
  events: z.array(z.object({ kind: id, message: z.string(), at: timestamp })),
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

export interface CreateRunInput {
  request_id: string;
  environment_id: string;
  revision: string;
  instruction: string;
}

export interface FrameImage {
  blob: Blob;
  frameId: string;
  capturedAt: string;
  physicsSteps: number;
}

export interface ConsoleApi {
  getRuntime(signal?: AbortSignal): Promise<RuntimeInfo>;
  getEnvironments(signal?: AbortSignal): Promise<Environments>;
  getEnvironmentSchema(signal?: AbortSignal): Promise<EnvironmentJsonSchema>;
  getTemplates(signal?: AbortSignal): Promise<EnvironmentTemplates>;
  saveEnvironment(documentJson: string, expectedRevision: string | null, signal?: AbortSignal): Promise<EnvironmentRecord>;
  activateEnvironment(id: string, revision: string, signal?: AbortSignal): Promise<Activation>;
  getFrame(id: string, revision: string, camera: Camera, signal?: AbortSignal): Promise<FrameImage>;
  getRuns(signal?: AbortSignal): Promise<Runs>;
  getRun(id: string, signal?: AbortSignal): Promise<RunRecord>;
  createRun(input: CreateRunInput, signal?: AbortSignal): Promise<RunRecord>;
  approveRun(id: string, planResponseId: string, signal?: AbortSignal): Promise<RunRecord>;
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
