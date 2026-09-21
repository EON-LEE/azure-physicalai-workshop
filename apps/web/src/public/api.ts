import { z } from 'zod';
import { ApiError } from '../api/errors';

z.config({ jitless: true });
const timestamp = z.string().refine((value) => Number.isFinite(Date.parse(value)));
const station = z.object({
  id: z.string().min(1),
  role: z.enum(['source', 'inspection', 'accepted', 'rejected']),
  position_m: z.tuple([z.number().finite(), z.number().finite(), z.number().finite()]),
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
    optimizer_steps: z.number().int().nonnegative(),
    quality_verified: z.literal(false),
  }),
  capabilities: z.object({
    anonymous_control: z.literal(false),
    anonymous_editing: z.literal(false),
    public_live_video: z.boolean(),
  }),
}).superRefine((value, context) => {
  const live = value.mode === 'live';
  if (live !== value.simulation.live_available || live !== value.capabilities.public_live_video ||
      (live && (value.simulation.status !== 'ready' || !value.simulation.frame_url))) {
    context.addIssue({ code: 'custom', message: 'Inconsistent live publication state.' });
  }
  if (new Set(value.scene.stations.map((item) => item.id)).size !== value.scene.stations.length) {
    context.addIssue({ code: 'custom', message: 'Duplicate station IDs.' });
  }
  if (new Set(value.scene.stations.map((item) => item.role)).size !== 4) {
    context.addIssue({ code: 'custom', message: 'The reference scene needs all four station roles.' });
  }
});

export type DemoSnapshot = z.infer<typeof demoSchema>;
export type DemoStation = DemoSnapshot['scene']['stations'][number];
export interface PublicFrame {
  blob: Blob;
  frameId: string;
  capturedAt: string;
  physicsSteps: number;
}

async function publicRequest<T>(path: string, signal: AbortSignal, read: (response: Response) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  const onAbort = () => controller.abort();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  signal.addEventListener('abort', onAbort, { once: true });
  const timer = setTimeout(() => controller.abort(), 20000);
  let receivedResponse = false;
  try {
    const response = await fetch(path, {
      method: 'GET', credentials: 'omit', signal: controller.signal, redirect: 'error',
      headers: { Accept: path.startsWith('/api/demo/frame') ? 'image/png' : 'application/json' },
    });
    receivedResponse = true;
    if (!response.ok) throw new ApiError('public_demo_unavailable', '공개 데모 연결을 확인해 주세요.', response.status);
    return await read(response);
  } catch (error) {
    if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
    if (controller.signal.aborted) throw new ApiError('request_timeout', '데모 연결 시간이 초과됐습니다. 다시 시도해 주세요.');
    if (!receivedResponse && error instanceof TypeError) throw new ApiError('network_error', '네트워크 연결을 확인해 주세요.');
    throw error;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', onAbort);
  }
}

export async function getDemo(signal: AbortSignal): Promise<DemoSnapshot> {
  return publicRequest('/api/demo', signal, async (response) => {
    const parsed = demoSchema.safeParse(await response.json());
    if (!parsed.success) throw new Error('데모 정보 형식이 올바르지 않습니다. 새로고침해 주세요.');
    return parsed.data;
  });
}

export async function getDemoFrame(signal: AbortSignal): Promise<PublicFrame> {
  return publicRequest('/api/demo/frame?camera=overview', signal, async (response) => {
  const frameId = response.headers.get('X-Frame-Id');
  const capturedAt = response.headers.get('X-Captured-At');
  const steps = response.headers.get('X-Physics-Steps');
  if (!response.headers.get('content-type')?.startsWith('image/png') ||
      !frameId || !capturedAt || !Number.isFinite(Date.parse(capturedAt)) ||
      steps === null || !/^\d+$/.test(steps) || !Number.isSafeInteger(Number(steps))) {
    throw new Error('실제 카메라 프레임을 확인할 수 없습니다.');
  }
  const bytes = await response.arrayBuffer();
  if (signal.aborted) throw new DOMException('Aborted', 'AbortError');
  if (!bytes.byteLength || bytes.byteLength > 5 * 1024 * 1024) throw new Error('카메라 이미지 크기가 올바르지 않습니다.');
  const signature = new Uint8Array(bytes, 0, Math.min(bytes.byteLength, 8));
  if (signature.length !== 8 || ![137, 80, 78, 71, 13, 10, 26, 10].every((value, index) => signature[index] === value)) {
    throw new Error('유효한 PNG 카메라 관측이 아닙니다.');
  }
  const age = Date.now() - Date.parse(capturedAt);
  if (age < -500 || age > 5000) throw new ApiError('stale_frame', '오래된 카메라 관측입니다. 다시 연결해 주세요.', 503);
  const blob = new Blob([bytes], { type: 'image/png' });
  return { blob, frameId, capturedAt, physicsSteps: Number(steps) };
  });
}
