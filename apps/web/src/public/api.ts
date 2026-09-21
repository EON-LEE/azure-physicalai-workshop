import { z } from 'zod';
import { ApiError, isAbort } from '../api/errors';

z.config({ jitless: true });
const timestamp = z.iso.datetime({ offset: true });
const point = z.tuple([z.number().finite(), z.number().finite(), z.number().finite()]);
const counter = z.number().int().nonnegative().safe();
const station = z.object({
  id: z.string().min(1),
  role: z.enum(['source', 'inspection', 'accepted', 'rejected']),
  position_m: point,
});
const decisionSchema = z.object({
  classification: z.enum(['accepted', 'rejected']),
  summary: z.string(),
  target_station_id: z.string().min(1),
  observation_id: z.uuid(),
  captured_at: timestamp,
  image_url: z.literal('/api/demo/evidence'),
});
export const presentationSchema = z.object({
  id: z.string().min(1),
  status: z.enum(['preparing', 'inspecting', 'awaiting_motion', 'moving', 'completed', 'stopped', 'failed']),
  cycle: counter,
  total_cycles: counter,
  scenario: z.enum(['normal', 'surface_defect']),
  instruction: z.string().min(1),
  updated_at: timestamp,
  expires_at: timestamp,
  scene_epoch: z.uuid().nullable(),
  run_id: z.uuid().nullable(),
  decision: decisionSchema.nullable(),
  motion: z.object({
    status: z.enum(['queued', 'running', 'succeeded', 'failed', 'cancelled', 'timed_out', 'cancelling']),
    phase: z.enum(['idle', 'approaching', 'grasping', 'lifting', 'inspection_station', 'transporting', 'releasing', 'returning', 'complete', 'stopped']).nullable(),
    part_position_m: point.nullable(),
    target_position_m: point,
  }).nullable(),
  result: z.object({
    status: z.enum(['succeeded', 'failed', 'cancelled', 'timed_out']),
    physical_success: z.boolean(),
    inspection_correct: z.boolean().nullable(),
    final_position_m: point.nullable(),
    completed_at: timestamp,
    message: z.string(),
  }).nullable(),
  counts: z.object({
    attempted: counter,
    succeeded: counter,
    failed: counter,
    inspected_correctly: counter,
    physically_completed: counter,
  }),
});

export const demoSchema = z.object({
  api_version: z.literal('public-demo-v1'),
  access: z.literal('public_read_only'),
  deployment: z.literal('azure'),
  mode: z.enum(['reference', 'live']),
  observed_at: timestamp,
  scene: z.object({
    id: z.literal('inspection-cell-v1'),
    name: z.string().min(1),
    length_unit: z.literal('m'),
    stations: z.array(station).length(4),
    robot: z.string().min(1),
    data_origin: z.literal('synthetic_reference_configuration'),
  }),
  simulation: z.object({
    status: z.enum(['not_published', 'loading', 'unavailable', 'ready']),
    live_available: z.boolean(),
    message_code: z.string(),
    frame_url: z.literal('/api/demo/frame').nullable(),
  }),
  agent: z.object({
    provider: z.literal('microsoft_foundry'),
    connectivity: z.enum(['configured', 'verified', 'unavailable']),
    verified_at: timestamp.nullable(),
    verification_scope: z.literal('connectivity_only'),
  }),
  learning: z.object({
    status: z.enum(['not_published', 'cpu_smoke_verified']),
    execution_location: z.literal('azure_acr'),
    data_kind: z.literal('test_fixture'),
    optimizer_steps: counter,
    quality_verified: z.literal(false),
  }),
  capabilities: z.object({
    anonymous_control: z.literal(false),
    anonymous_editing: z.literal(false),
    public_live_video: z.boolean(),
  }),
  presentation: presentationSchema.nullable().optional(),
}).superRefine((value, context) => {
  if (value.simulation.live_available &&
    (value.mode !== 'live' || value.simulation.status !== 'ready' || !value.simulation.frame_url || !value.capabilities.public_live_video)) {
    context.addIssue({ code: 'custom', message: 'Inconsistent live publication state.' });
  }
  if (new Set(value.scene.stations.map((item) => item.id)).size !== value.scene.stations.length ||
    new Set(value.scene.stations.map((item) => item.role)).size !== 4) {
    context.addIssue({ code: 'custom', message: 'The reference scene needs distinct stations and all four roles.' });
  }
  if (value.presentation && value.presentation.cycle > value.presentation.total_cycles) {
    context.addIssue({ code: 'custom', message: 'The current cycle exceeds the published cycle bound.' });
  }
});

export type DemoSnapshot = z.infer<typeof demoSchema>;
export type Presentation = z.infer<typeof presentationSchema>;
export type PublicDecision = z.infer<typeof decisionSchema>;
export type DemoStation = DemoSnapshot['scene']['stations'][number];
export interface PublicEvidence {
  blob: Blob;
  frameId: string;
  capturedAt: string;
}
export interface PublicFrame extends PublicEvidence {
  physicsSteps: number;
  sceneEpoch: string;
}
export const FRAME_MAX_AGE_MS = 5000;
export const SNAPSHOT_MAX_AGE_MS = 10_000;
const MAX_IMAGE_BYTES = 5 * 1024 * 1024;
const errorEnvelope = z.object({ error: z.object({ code: z.string(), message: z.string() }) });

async function publicRequest<T>(path: string, signal: AbortSignal, image: boolean, read: (response: Response) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  signal.throwIfAborted();
  signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => controller.abort(), 15_000);
  try {
    const response = await fetch(path, {
      method: 'GET', credentials: 'omit', cache: 'no-store', signal: controller.signal, redirect: 'error',
      headers: { Accept: image ? 'image/png' : 'application/json' },
    });
    if (!response.ok) {
      const body: unknown = await response.json().catch((error: unknown) => {
        if (isAbort(error)) throw error;
        return null;
      });
      const parsed = errorEnvelope.safeParse(body);
      throw new ApiError(parsed.success ? parsed.data.error.code : 'public_demo_unavailable',
        parsed.success ? parsed.data.error.message : '공개 시연 연결을 확인할 수 없습니다. 다시 확인해 주세요.', response.status);
    }
    return await read(response);
  } catch (error) {
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (controller.signal.aborted) throw new ApiError('request_timeout', '공개 시연 응답 시간이 초과됐습니다. 다시 확인해 주세요.');
    if (error instanceof ApiError) throw error;
    if (error instanceof TypeError) throw new ApiError('network_error', '공개 시연에 연결할 수 없습니다. 네트워크를 확인해 주세요.');
    throw error;
  } finally {
    controller.abort();
    clearTimeout(timer);
    signal.removeEventListener('abort', onAbort);
  }
}

export async function getDemo(signal: AbortSignal): Promise<DemoSnapshot> {
  return publicRequest('/api/demo', signal, false, async (response) => {
    let body: unknown;
    try {
      body = await response.json();
    } catch (error) {
      if (isAbort(error)) throw error;
      throw new ApiError('invalid_snapshot', '공개 시연 정보 형식이 올바르지 않습니다. 다시 확인해 주세요.');
    }
    const parsed = demoSchema.safeParse(body);
    if (!parsed.success) throw new ApiError('invalid_snapshot', '공개 시연 정보가 API 계약과 일치하지 않습니다. 다시 확인해 주세요.');
    return parsed.data;
  });
}

async function readImage(response: Response, signal: AbortSignal): Promise<PublicEvidence> {
  const frameId = response.headers.get('X-Frame-Id');
  const capturedAt = response.headers.get('X-Captured-At');
  if (response.headers.get('Content-Type')?.split(';')[0]?.trim().toLowerCase() !== 'image/png' ||
    !frameId || !capturedAt || !timestamp.safeParse(capturedAt).success) {
    throw new ApiError('invalid_image_metadata', '영상의 PNG 형식 또는 촬영 정보를 확인할 수 없습니다.');
  }
  const contentLength = response.headers.get('Content-Length');
  if (contentLength && Number(contentLength) > MAX_IMAGE_BYTES) {
    throw new ApiError('invalid_image_size', '공개 영상이 허용된 크기를 초과했습니다.');
  }
  const reader = response.body?.getReader();
  if (!reader) throw new ApiError('invalid_image', '영상 응답이 비어 있습니다.');
  const chunks: Uint8Array<ArrayBuffer>[] = [];
  let size = 0;
  try {
    while (true) {
      const item = await reader.read();
      signal.throwIfAborted();
      if (item.done) break;
      size += item.value.byteLength;
      if (size > MAX_IMAGE_BYTES) throw new ApiError('invalid_image_size', '공개 영상이 허용된 크기를 초과했습니다.');
      chunks.push(new Uint8Array(item.value));
    }
  } catch (error) {
    await reader.cancel().catch(() => undefined);
    throw error;
  } finally {
    reader.releaseLock();
  }
  const blob = new Blob(chunks, { type: 'image/png' });
  const signature = new Uint8Array(await blob.slice(0, 8).arrayBuffer());
  if (![137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => signature[index] === value)) {
    throw new ApiError('invalid_image', '유효한 PNG 카메라 관측이 아닙니다.');
  }
  return { blob, frameId, capturedAt };
}

export async function getDemoFrame(epoch: string, signal: AbortSignal): Promise<PublicFrame> {
  if (!z.uuid().safeParse(epoch).success) throw new ApiError('invalid_epoch', '현재 시연의 씬 정보를 확인할 수 없습니다.');
  const query = new URLSearchParams({ camera: 'overview', epoch });
  return publicRequest(`/api/demo/frame?${query}`, signal, true, async (response) => {
    const sceneEpoch = response.headers.get('X-Scene-Epoch');
    if (!sceneEpoch || sceneEpoch.toLowerCase() !== epoch.toLowerCase()) {
      throw new ApiError('scene_changed', '다른 회차의 카메라 응답입니다. 현재 시연을 다시 확인합니다.', 409);
    }
    const steps = response.headers.get('X-Physics-Steps');
    if (steps === null || !/^\d+$/.test(steps) || !Number.isSafeInteger(Number(steps))) {
      throw new ApiError('invalid_frame_metadata', '프레임의 실제 물리 스텝을 확인할 수 없습니다.');
    }
    const image = await readImage(response, signal);
    const age = Date.now() - Date.parse(image.capturedAt);
    if (age < -FRAME_MAX_AGE_MS || age > FRAME_MAX_AGE_MS) {
      throw new ApiError('stale_frame', '카메라 촬영 시각이 최신이 아닙니다. 연결을 다시 확인합니다.', 503);
    }
    return { ...image, physicsSteps: Number(steps), sceneEpoch };
  });
}

export async function getDemoEvidence(decision: PublicDecision, signal: AbortSignal): Promise<PublicEvidence> {
  if (decision.image_url !== '/api/demo/evidence' || !z.uuid().safeParse(decision.observation_id).success) {
    throw new ApiError('invalid_evidence_reference', '공개된 검사 입력 이미지 경로를 확인할 수 없습니다.');
  }
  const query = new URLSearchParams({ observation_id: decision.observation_id });
  return publicRequest(`/api/demo/evidence?${query}`, signal, true, async (response) => {
    const image = await readImage(response, signal);
    if (image.frameId.toLowerCase() !== decision.observation_id.toLowerCase() ||
      Date.parse(image.capturedAt) !== Date.parse(decision.captured_at)) {
      throw new ApiError('evidence_changed', '현재 판단과 다른 입력 이미지입니다. 시연 정보를 다시 확인합니다.', 409);
    }
    return image;
  });
}

export function needsSnapshotRefresh(error: unknown): boolean {
  return error instanceof ApiError && error.status === 409;
}

export function canShowCamera(snapshot: DemoSnapshot): boolean {
  return snapshot.mode === 'live' && snapshot.simulation.status === 'ready' &&
    snapshot.simulation.live_available && snapshot.capabilities.public_live_video &&
    snapshot.simulation.frame_url === '/api/demo/frame';
}
