// @vitest-environment node
import { afterEach, describe, expect, it, vi } from 'vitest';
import { demoSchema, getDemo, getDemoEvidence, getDemoFrame } from '../src/public/api';
import { epoch, makePresentation, makeSnapshot, nextEpoch, observationId, referenceSnapshot } from './fixtures/public';
import { pngBytes } from './fixtures/data';

afterEach(() => vi.unstubAllGlobals());
const signal = () => new AbortController().signal;
function imageHeaders() {
  return {
    'Content-Type': 'image/png', 'X-Frame-Id': 'test-only-frame',
    'X-Captured-At': new Date().toISOString(), 'X-Physics-Steps': '125', 'X-Scene-Epoch': epoch,
    'Cache-Control': 'no-store',
  };
}

describe('frozen public presentation contract', () => {
  it('accepts absent/null presentation without inventing a run or result', () => {
    const absent = { ...referenceSnapshot(), presentation: undefined };
    expect(demoSchema.parse(absent).presentation).toBeUndefined();
    expect(demoSchema.parse(referenceSnapshot()).presentation).toBeNull();
    expect(demoSchema.parse(makeSnapshot()).presentation?.result).toBeNull();
  });
  it('rejects invalid phases, anonymous permissions and arbitrary evidence or camera URLs', () => {
    const base = makeSnapshot();
    const presentation = makePresentation();
    for (const invalid of [
      { ...base, capabilities: { ...base.capabilities, anonymous_control: true } },
      { ...base, simulation: { ...base.simulation, frame_url: 'https://untrusted.invalid/frame' } },
      { ...base, presentation: { ...presentation, decision: { ...presentation.decision, image_url: '/api/runs/private/observation' } } },
      { ...base, presentation: { ...presentation, motion: { ...presentation.motion, phase: 'fictional' } } },
      { ...base, presentation: { ...presentation, scene_epoch: 'not-an-epoch' } },
      { ...base, presentation: { ...presentation, cycle: 20 } },
    ]) expect(demoSchema.safeParse(invalid).success).toBe(false);
  });
  it('accepts unavailability alongside a last recorded result without requiring ready flags', () => {
    const base = referenceSnapshot();
    expect(demoSchema.safeParse({ ...base, presentation: makePresentation({ status: 'stopped' }) }).success).toBe(true);
  });
  it('rejects malformed JSON and invalid response shapes explicitly', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValueOnce(new Response('<html>broken</html>')).mockResolvedValueOnce(new Response('{}')));
    await expect(getDemo(signal())).rejects.toMatchObject({ code: 'invalid_snapshot' });
    await expect(getDemo(signal())).rejects.toMatchObject({ code: 'invalid_snapshot' });
  });
  it.each([401, 403, 409, 422, 503])('preserves structured HTTP %s without starting login or writes', async (status) => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({ error: { code: 'published_scene_changed', message: '현재 시연 확인 필요' } }), { status }));
    vi.stubGlobal('fetch', fetch);
    await expect(getDemo(signal())).rejects.toMatchObject({ status, code: 'published_scene_changed' });
    expect(fetch).toHaveBeenCalledWith('/api/demo', expect.objectContaining({ method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error', headers: { Accept: 'application/json' } }));
  });
});

describe('anonymous same-cycle public images', () => {
  it('fetches only the allowlisted live URL with the snapshot epoch and validates real metadata', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(pngBytes, { headers: imageHeaders() }));
    vi.stubGlobal('fetch', fetch);
    const result = await getDemoFrame(epoch, signal());
    expect(result).toMatchObject({ frameId: 'test-only-frame', sceneEpoch: epoch, physicsSteps: 125 });
    expect(fetch).toHaveBeenCalledWith(`/api/demo/frame?camera=overview&epoch=${epoch}`, expect.objectContaining({
      method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error', headers: { Accept: 'image/png' },
    }));
  });
  it.each([null, nextEpoch])('rejects an absent or mismatched scene epoch (%s) before displaying data', async (value) => {
    const headers: Record<string, string> = imageHeaders();
    if (value === null) delete headers['X-Scene-Epoch'];
    else headers['X-Scene-Epoch'] = value;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(pngBytes, { headers })));
    await expect(getDemoFrame(epoch, signal())).rejects.toMatchObject({ code: 'scene_changed', status: 409 });
  });
  it.each(['X-Frame-Id', 'X-Captured-At', 'X-Physics-Steps'])('rejects missing %s rather than using synthetic values', async (key) => {
    const headers: Record<string, string> = imageHeaders();
    delete headers[key];
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(pngBytes, { headers })));
    await expect(getDemoFrame(epoch, signal())).rejects.toThrow();
  });
  it('rejects stale live images, invalid type, corrupt PNGs and oversized data', async () => {
    const headers = imageHeaders();
    const fetch = vi.fn()
      .mockResolvedValueOnce(new Response(pngBytes, { headers: { ...headers, 'X-Captured-At': new Date(Date.now() - 10_000).toISOString() } }))
      .mockResolvedValueOnce(new Response('html', { headers: { ...headers, 'Content-Type': 'text/html' } }))
      .mockResolvedValueOnce(new Response('invalid PNG', { headers }))
      .mockResolvedValueOnce(new Response(pngBytes, { headers: { ...headers, 'Content-Length': String(6 * 1024 * 1024) } }));
    vi.stubGlobal('fetch', fetch);
    await expect(getDemoFrame(epoch, signal())).rejects.toMatchObject({ code: 'stale_frame' });
    await expect(getDemoFrame(epoch, signal())).rejects.toMatchObject({ code: 'invalid_image_metadata' });
    await expect(getDemoFrame(epoch, signal())).rejects.toMatchObject({ code: 'invalid_image' });
    await expect(getDemoFrame(epoch, signal())).rejects.toMatchObject({ code: 'invalid_image_size' });
  });
  it('accepts original evidence older than five seconds, only with matching observation and capture time', async () => {
    const decision = makePresentation().decision!;
    const headers = { 'Content-Type': 'image/png', 'X-Frame-Id': decision.observation_id, 'X-Captured-At': decision.captured_at };
    const fetch = vi.fn().mockResolvedValue(new Response(pngBytes, { headers }));
    vi.stubGlobal('fetch', fetch);
    await expect(getDemoEvidence(decision, signal())).resolves.toMatchObject({ frameId: observationId, capturedAt: decision.captured_at });
    expect(fetch).toHaveBeenCalledWith(`/api/demo/evidence?observation_id=${observationId}`, expect.objectContaining({ credentials: 'omit', cache: 'no-store' }));
  });
  it.each([
    ['2026-09-21T16:00:00+00:00', '2026-09-21T16:00:00Z'],
    ['2026-09-21T16:00:00Z', '2026-09-21T16:00:00+00:00'],
    ['2026-09-21T16:00:00.123456+00:00', '2026-09-21T16:00:00.123456Z'],
    ['2026-09-21T16:00:00.120000+00:00', '2026-09-21T16:00:00.12Z'],
    ['2026-09-22T01:00:00+09:00', '2026-09-21T16:00:00Z'],
  ])('matches evidence header %s to JSON %s by instant, not spelling', async (capturedAt, decisionTime) => {
    const decision = { ...makePresentation().decision!, captured_at: decisionTime };
    const snapshot = makeSnapshot({ presentation: makePresentation({ decision }) });
    expect(demoSchema.safeParse(snapshot).success).toBe(true);
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(pngBytes, { headers: {
      'Content-Type': 'image/png', 'X-Frame-Id': decision.observation_id, 'X-Captured-At': capturedAt,
    } })));
    await expect(getDemoEvidence(decision, signal())).resolves.toMatchObject({
      frameId: decision.observation_id, capturedAt,
    });
  });
  it('still rejects a one-millisecond evidence mismatch across UTC timestamp spellings', async () => {
    const decision = { ...makePresentation().decision!, captured_at: '2026-09-21T16:00:00.123Z' };
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(pngBytes, { headers: {
      'Content-Type': 'image/png', 'X-Frame-Id': decision.observation_id, 'X-Captured-At': '2026-09-21T16:00:00.124000+00:00',
    } })));
    await expect(getDemoEvidence(decision, signal())).rejects.toMatchObject({ code: 'evidence_changed', status: 409 });
  });
  it.each(['id', 'time'])('rejects the wrong current evidence %s and refreshes via conflict semantics', async (kind) => {
    const decision = makePresentation().decision!;
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response(pngBytes, { headers: {
      'Content-Type': 'image/png',
      'X-Frame-Id': kind === 'id' ? nextEpoch : decision.observation_id,
      'X-Captured-At': kind === 'time' ? new Date().toISOString() : decision.captured_at,
    } })));
    await expect(getDemoEvidence(decision, signal())).rejects.toMatchObject({ code: 'evidence_changed', status: 409 });
  });
  it('rejects malformed epoch IDs before making a request', async () => {
    const fetch = vi.fn();
    vi.stubGlobal('fetch', fetch);
    await expect(getDemoFrame('/api/private', signal())).rejects.toMatchObject({ code: 'invalid_epoch' });
    expect(fetch).not.toHaveBeenCalled();
  });
  it('keeps body reads abortable after receiving headers', async () => {
    const fetch = vi.fn().mockImplementation(async (_path, options: RequestInit) => new Response(new ReadableStream({
      start(controller) {
        controller.enqueue(pngBytes.slice(0, 8));
        options.signal?.addEventListener('abort', () => controller.error(new DOMException('Aborted', 'AbortError')));
      },
    }), { headers: imageHeaders() }));
    vi.stubGlobal('fetch', fetch);
    const controller = new AbortController();
    const result = getDemoFrame(epoch, controller.signal);
    for (let index = 0; index < 5; index++) await Promise.resolve();
    controller.abort();
    await expect(result).rejects.toMatchObject({ name: 'AbortError' });
  });
  it('times out a hung public GET without retrying it as a write', async () => {
    vi.useFakeTimers();
    const fetch = vi.fn().mockImplementation((_path, options: RequestInit) => new Promise((_resolve, reject) => {
      options.signal?.addEventListener('abort', () => reject(new DOMException('Aborted', 'AbortError')));
    }));
    vi.stubGlobal('fetch', fetch);
    const check = expect(getDemo(signal())).rejects.toMatchObject({ code: 'request_timeout' });
    await vi.advanceTimersByTimeAsync(15_000);
    await check;
    expect(fetch).toHaveBeenCalledTimes(1);
  });
});
