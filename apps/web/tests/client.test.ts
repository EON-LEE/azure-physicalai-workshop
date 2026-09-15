// @vitest-environment node
import { describe, expect, it, vi } from 'vitest';
import { ApiClient, getPublicConfig, type FetchTransport } from '../src/api/client';
import { ApiError, AuthenticationRequiredError } from '../src/api/errors';
import { config, environment, pendingRun, pngBytes, runtime } from './fixtures/data';
import { deferred } from './helpers';

const json = (body: unknown, status = 200) => new Response(JSON.stringify(body), { status, headers: { 'Content-Type': 'application/json' } });
const transport = () => vi.fn<FetchTransport>();
const token = async () => 'test-only-access-token';

describe('same-origin API and authentication boundary', () => {
  it('fetches only the public configuration without anonymous API fallback', async () => {
    const fetch = transport().mockResolvedValue(json(config));
    await expect(getPublicConfig(undefined, fetch)).resolves.toEqual(config);
    expect(fetch).toHaveBeenCalledWith('/api/config', expect.objectContaining({ credentials: 'omit', cache: 'no-store', redirect: 'error' }));
    expect(fetch.mock.calls[0]?.[1]?.headers).toEqual({ Accept: 'application/json' });
  });

  it.each([
    { ...config, deployment: 'local' },
    { ...config, api_version: 'v2' },
    { ...config, auth: { ...config.auth, tenant_id: 'common' } },
    { ...config, auth: { ...config.auth, scope: '' } },
    { ...config, auth: { ...config.auth, client_id: undefined } },
  ])('fails closed on invalid bootstrap configuration: %j', async (bad) => {
    await expect(getPublicConfig(undefined, transport().mockResolvedValue(json(bad)))).rejects.toMatchObject({ code: 'invalid_response' });
  });

  it('acquires a token for each API request and sends no cookie credentials', async () => {
    const fetch = transport().mockResolvedValue(json(runtime));
    const acquireToken = vi.fn(token);
    const api = new ApiClient(acquireToken, fetch);
    await api.getRuntime();
    expect(acquireToken).toHaveBeenCalledTimes(1);
    expect(fetch).toHaveBeenCalledWith('/api/runtime', expect.objectContaining({
      headers: { Accept: 'application/json', Authorization: 'Bearer test-only-access-token' },
      credentials: 'omit', redirect: 'error', cache: 'no-store',
    }));
  });

  it('does not call any protected endpoint without an access token', async () => {
    const fetch = transport();
    await expect(new ApiClient(async () => '', fetch).getRuntime()).rejects.toBeInstanceOf(AuthenticationRequiredError);
    await expect(new ApiClient(async () => { throw new AuthenticationRequiredError(); }, fetch).getRuns()).rejects.toBeInstanceOf(AuthenticationRequiredError);
    expect(fetch).not.toHaveBeenCalled();
  });

  it.each([401, 403, 409, 422, 503])('surfaces HTTP %s and preserves structured details', async (status) => {
    const unauthorized = vi.fn();
    const fetch = transport().mockResolvedValue(json({ error: { code: 'action_failed', message: '구체적인 서버 오류', details: [{ path: '$.scene' }] } }, status));
    await expect(new ApiClient(token, fetch, unauthorized).getRuntime()).rejects.toMatchObject({
      status, code: 'action_failed', message: '구체적인 서버 오류', details: [{ path: '$.scene' }],
    });
    expect(unauthorized).toHaveBeenCalledTimes(status === 401 ? 1 : 0);
  });

  it('surfaces malformed error pages and never treats them as success', async () => {
    await expect(new ApiClient(token, transport().mockResolvedValue(new Response('<html>Unavailable</html>', { status: 503 }))).getRuns())
      .rejects.toMatchObject({ status: 503, code: 'http_error' });
    await expect(new ApiClient(token, transport().mockResolvedValue(json({ status: 'succeeded' }))).getRuntime())
      .rejects.toMatchObject({ code: 'invalid_response' });
  });

  it('does not hide network failures or fabricate a ready runtime', async () => {
    await expect(new ApiClient(token, transport().mockRejectedValue(new TypeError('offline'))).getRuntime())
      .rejects.toMatchObject({ code: 'network_error', status: 0 });
  });

  it('preserves raw JSON including duplicate keys and whitespace in the HTTP body', async () => {
    const fetch = transport().mockResolvedValue(json(environment));
    const raw = ' \r\n{ "environment_id": "a", "environment_id": "b", "seed": 0 }\n ';
    const api = new ApiClient(token, fetch);
    await api.saveEnvironment(raw, 'loaded-revision');
    const body = fetch.mock.calls[0]?.[1]?.body;
    expect(typeof body).toBe('string');
    expect(JSON.parse(String(body))).toEqual({ document_json: raw, expected_revision: 'loaded-revision' });
  });

  it('sends null only for a new environment and the exact response ID for approval', async () => {
    const fetch = transport().mockResolvedValueOnce(json(environment)).mockResolvedValueOnce(json(pendingRun)).mockResolvedValueOnce(json(pendingRun));
    const api = new ApiClient(token, fetch);
    await api.saveEnvironment('{}', null);
    await api.approveRun('run/with ?id', 'actual-response-id');
    await api.cancelRun('run/with ?id');
    expect(JSON.parse(String(fetch.mock.calls[0]?.[1]?.body))).toEqual({ document_json: '{}', expected_revision: null });
    expect(fetch.mock.calls[1]?.[0]).toBe('/api/runs/run%2Fwith%20%3Fid/approve');
    expect(JSON.parse(String(fetch.mock.calls[1]?.[1]?.body))).toEqual({ plan_response_id: 'actual-response-id' });
    expect(fetch.mock.calls[2]?.[1]).toMatchObject({ method: 'POST' });
    expect(fetch.mock.calls[2]?.[1]?.body).toBeUndefined();
  });

  it('aborts before a delayed token can dispatch an obsolete request', async () => {
    const pending = deferred<string>();
    const fetch = transport();
    const controller = new AbortController();
    const result = new ApiClient(() => pending.promise, fetch).getRuntime(controller.signal);
    controller.abort();
    pending.resolve('test-only-access-token');
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(fetch).not.toHaveBeenCalled();
  });

  it('bounds GET requests and reports timeout without retrying writes', async () => {
    vi.useFakeTimers();
    const fetch = transport().mockImplementation((_path, init) => new Promise((_resolve, reject) => {
      init?.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));
    const result = new ApiClient(token, fetch).getRuntime();
    const assertion = expect(result).rejects.toMatchObject({ code: 'request_timeout' });
    await vi.advanceTimersByTimeAsync(15_000);
    await assertion;
    expect(fetch).toHaveBeenCalledTimes(1);
  });

  it('keeps navigation cancellation wired after headers while a response body is still arriving', async () => {
    let responseSignal: AbortSignal | null = null;
    const fetch = transport().mockImplementation(async (_path, init) => {
      responseSignal = init?.signal ?? null;
      return new Response(new ReadableStream({
        start(controller) {
          controller.enqueue(new TextEncoder().encode('{'));
          init?.signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
        },
      }), { headers: { 'Content-Type': 'application/json' } });
    });
    const controller = new AbortController();
    const result = new ApiClient(token, fetch).getRuntime(controller.signal);
    for (let index = 0; index < 5; index++) await Promise.resolve();
    expect(fetch).toHaveBeenCalledTimes(1);
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
    expect(responseSignal).toMatchObject({ aborted: true });
  });
});

describe('authenticated PNG transport', () => {
  const headers = {
    'Content-Type': 'image/png',
    'X-Frame-Id': 'real-server-frame-id',
    'X-Captured-At': '2026-09-15T02:00:00+00:00',
    'X-Physics-Steps': '125',
  };
  it('uses bearer fetch and actual frame metadata rather than an unauthenticated img URL', async () => {
    const fetch = transport().mockResolvedValue(new Response(pngBytes, { headers }));
    const frame = await new ApiClient(token, fetch).getFrame('customer / a', 'revision+1', 'inspection');
    expect(fetch.mock.calls[0]?.[0]).toBe('/api/environments/customer%20%2F%20a/frame?revision=revision%2B1&camera=inspection');
    expect(fetch.mock.calls[0]?.[1]?.headers).toMatchObject({ Accept: 'image/png', Authorization: 'Bearer test-only-access-token' });
    expect(frame).toMatchObject({ frameId: 'real-server-frame-id', capturedAt: headers['X-Captured-At'], physicsSteps: 125 });
    expect(frame.blob.size).toBeGreaterThan(8);
  });

  it.each(['missing', 'not-a-timestamp'])('rejects missing or invalid frame metadata: %s', async (value) => {
    const frameHeaders: Record<string, string> = { ...headers };
    if (value === 'missing') delete frameHeaders['X-Frame-Id'];
    else frameHeaders['X-Captured-At'] = value;
    await expect(new ApiClient(token, transport().mockResolvedValue(new Response(pngBytes, { headers: frameHeaders })))
      .getFrame('cell', 'revision', 'overview')).rejects.toMatchObject({ code: 'invalid_frame_metadata' });
  });

  it('does not substitute invalid PNG data or accept an HTML login page', async () => {
    for (const response of [new Response('not png', { headers }), new Response('<html>login</html>', { headers: { 'Content-Type': 'text/html' } })]) {
      await expect(new ApiClient(token, transport().mockResolvedValue(response)).getFrame('cell', 'revision', 'overview')).rejects.toBeInstanceOf(ApiError);
    }
  });
});
