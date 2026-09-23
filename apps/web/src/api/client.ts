import { z } from 'zod';
import {
  activationSchema, configSchema, environmentsSchema, environmentSchema,
  jsonSchemaSchema, runSchema, runsSchema, runtimeSchema, templatesSchema,
  type Camera, type ConsoleApi, type CreateRunInput, type FrameImage, type PublicConfig,
} from './contracts';
import { ApiError, AuthenticationRequiredError, isAbort } from './errors';
import { LearningClient } from '../learning/client';

export type FetchTransport = (input: RequestInfo | URL, init?: RequestInit) => Promise<Response>;
export type TokenProvider = () => Promise<string>;

const errorSchema = z.object({
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.array(z.unknown()).optional(),
  }),
});
const fetchTransport: FetchTransport = (input, init) => fetch(input, init);

async function readJson(response: Response): Promise<unknown> {
  try {
    return await response.json();
  } catch (error) {
    if (isAbort(error)) throw error;
    throw new ApiError('invalid_response', '서버가 올바른 JSON 응답을 반환하지 않았습니다.', response.status);
  }
}

async function checkResponse(response: Response): Promise<void> {
  if (response.ok) return;
  const body: unknown = await response.json().catch((error: unknown) => {
    if (isAbort(error)) throw error;
    return null;
  });
  const parsed = errorSchema.safeParse(body);
  if (!parsed.success) {
    throw new ApiError('http_error', `API 요청이 실패했습니다 (HTTP ${response.status}).`, response.status);
  }
  throw new ApiError(parsed.data.error.code, parsed.data.error.message, response.status, parsed.data.error.details);
}

async function decode<T>(response: Response, schema: z.ZodType<T>): Promise<T> {
  await checkResponse(response);
  const body = await readJson(response);
  const result = schema.safeParse(body);
  if (!result.success) {
    throw new ApiError('invalid_response', 'API 응답이 v1 계약과 일치하지 않습니다. 서버 구성을 확인하세요.', response.status,
      result.error.issues.map((issue) => ({ path: issue.path.join('.'), message: issue.message })));
  }
  return result.data;
}

async function send<T>(transport: FetchTransport, path: string, init: RequestInit, consume: (response: Response) => Promise<T>): Promise<T> {
  const controller = new AbortController();
  let timedOut = false;
  const abort = () => controller.abort();
  init.signal?.throwIfAborted();
  init.signal?.addEventListener('abort', abort, { once: true });
  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, init.method === 'POST' ? 120_000 : 15_000);
  try {
    const response = await transport(path, {
      ...init,
      signal: controller.signal,
      credentials: 'omit',
      cache: 'no-store',
      redirect: 'error',
    });
    return await consume(response);
  } catch (error) {
    if (timedOut) throw new ApiError('request_timeout', 'API 응답 시간이 초과되었습니다. 요청이 접수되었을 수 있으므로 서버 상태를 먼저 확인하세요.');
    if (isAbort(error)) throw error;
    if (error instanceof ApiError) throw error;
    throw new ApiError('network_error', 'API에 연결할 수 없습니다. 네트워크와 같은 출처의 API 서버를 확인하세요.');
  } finally {
    clearTimeout(timer);
    init.signal?.removeEventListener('abort', abort);
  }
}

export async function getPublicConfig(signal?: AbortSignal, transport: FetchTransport = fetchTransport): Promise<PublicConfig> {
  return send(transport, '/api/config', { signal, headers: { Accept: 'application/json' } }, (response) => decode(response, configSchema));
}

async function readPng(response: Response): Promise<Blob> {
  await checkResponse(response);
  if (response.headers.get('Content-Type')?.split(';')[0]?.trim().toLowerCase() !== 'image/png') {
    throw new ApiError('invalid_image', '카메라 API가 PNG 영상을 반환하지 않았습니다.', response.status);
  }
  const blob = await response.blob();
  const signature = new Uint8Array(await blob.slice(0, 8).arrayBuffer());
  if (![137, 80, 78, 71, 13, 10, 26, 10].every((byte, index) => signature[index] === byte)) {
    throw new ApiError('invalid_image', '수신한 PNG 데이터가 올바르지 않습니다.', response.status);
  }
  return blob;
}

export class ApiClient implements ConsoleApi {
  readonly learning: LearningClient;
  constructor(
    private readonly token: TokenProvider,
    private readonly transport: FetchTransport = fetchTransport,
    private readonly onUnauthorized?: () => void,
  ) {
    this.learning = new LearningClient((path, schema, options = {}) => this.request(
      path, (response) => decode(response, schema), options.signal, options.body,
      options.method ?? 'GET', 'application/json', options.etag,
    ));
  }

  private async request<T>(path: string, consume: (response: Response) => Promise<T>, signal?: AbortSignal, body?: unknown, method = 'GET', accept = 'application/json', etag?: string): Promise<T> {
    signal?.throwIfAborted();
    const accessToken = await this.token();
    signal?.throwIfAborted();
    if (!accessToken) throw new AuthenticationRequiredError();
    return send(this.transport, path, {
      method, signal,
      headers: {
        Accept: accept,
        Authorization: `Bearer ${accessToken}`,
        ...(body === undefined ? {} : { 'Content-Type': 'application/json' }),
        ...(etag === undefined ? {} : { 'If-Match': etag }),
      },
      ...(body === undefined ? {} : { body: JSON.stringify(body) }),
    }, (response) => {
      if (response.status === 401) this.onUnauthorized?.();
      return consume(response);
    });
  }

  async getRuntime(signal?: AbortSignal) {
    return this.request('/api/runtime', (response) => decode(response, runtimeSchema), signal);
  }

  async getEnvironments(signal?: AbortSignal) {
    return this.request('/api/environments', (response) => decode(response, environmentsSchema), signal);
  }

  async getEnvironmentSchema(signal?: AbortSignal) {
    return this.request('/api/environment-schema', (response) => decode(response, jsonSchemaSchema), signal);
  }

  async getTemplates(signal?: AbortSignal) {
    return this.request('/api/environment-templates', (response) => decode(response, templatesSchema), signal);
  }

  async saveEnvironment(documentJson: string, expectedRevision: string | null, signal?: AbortSignal) {
    return this.request('/api/environments', (response) => decode(response, environmentSchema), signal, {
      document_json: documentJson, expected_revision: expectedRevision,
    }, 'POST');
  }

  async activateEnvironment(id: string, revision: string, signal?: AbortSignal) {
    return this.request(`/api/environments/${encodeURIComponent(id)}/activate`, (response) => decode(response, activationSchema), signal, { revision }, 'POST');
  }

  async getFrame(id: string, revision: string, camera: Camera, signal?: AbortSignal): Promise<FrameImage> {
    const query = new URLSearchParams({ revision, camera });
    return this.request(`/api/environments/${encodeURIComponent(id)}/frame?${query}`, async (response) => {
      const blob = await readPng(response);
      const frameId = response.headers.get('X-Frame-Id');
      const capturedAt = response.headers.get('X-Captured-At');
      const steps = response.headers.get('X-Physics-Steps');
      if (!frameId || !capturedAt || !Number.isFinite(Date.parse(capturedAt)) || !steps || !/^\d+$/.test(steps) || !Number.isSafeInteger(Number(steps))) {
        throw new ApiError('invalid_frame_metadata', '프레임의 촬영 시각 또는 물리 스텝 정보를 확인할 수 없습니다.', response.status);
      }
      return { blob, frameId, capturedAt, physicsSteps: Number(steps) };
    }, signal, undefined, 'GET', 'image/png');
  }

  async getRuns(signal?: AbortSignal) {
    return this.request('/api/runs', (response) => decode(response, runsSchema), signal);
  }

  async getRun(id: string, signal?: AbortSignal) {
    return this.request(`/api/runs/${encodeURIComponent(id)}`, (response) => decode(response, runSchema), signal);
  }

  async createRun(input: CreateRunInput, signal?: AbortSignal) {
    return this.request('/api/runs', (response) => decode(response, runSchema), signal, input, 'POST');
  }

  async approveRun(id: string, planResponseId: string, signal?: AbortSignal) {
    return this.request(`/api/runs/${encodeURIComponent(id)}/approve`, (response) => decode(response, runSchema), signal, {
      plan_response_id: planResponseId,
    }, 'POST');
  }

  async cancelRun(id: string, signal?: AbortSignal) {
    return this.request(`/api/runs/${encodeURIComponent(id)}/cancel`, (response) => decode(response, runSchema), signal, undefined, 'POST');
  }

  async getObservation(id: string, signal?: AbortSignal) {
    return this.request(`/api/runs/${encodeURIComponent(id)}/observation`, readPng, signal, undefined, 'GET', 'image/png');
  }
}
