import { act, renderHook } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import { ApiError } from '../src/api/errors';
import { usePolling } from '../src/hooks/usePolling';
import { useProtectedImage } from '../src/hooks/useProtectedImage';
import { deferred, setVisibility } from './helpers';
import { fixturePng } from './fixtures/data';

async function flush() {
  await act(async () => { await Promise.resolve(); });
}

describe('bounded polling lifecycle', () => {
  it('does not overlap slow requests and waits at least one second after completion', async () => {
    vi.useFakeTimers();
    const slow = deferred<number>();
    const load = vi.fn<(signal: AbortSignal) => Promise<number>>().mockReturnValueOnce(slow.promise).mockResolvedValue(2);
    const { result, unmount } = renderHook(() => usePolling(load, { intervalMs: 1000 }));
    await act(async () => vi.advanceTimersByTimeAsync(9000));
    expect(load).toHaveBeenCalledTimes(1);
    await act(async () => slow.resolve(1));
    expect(result.current.data).toBe(1);
    await act(async () => vi.advanceTimersByTimeAsync(999));
    expect(load).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(load).toHaveBeenCalledTimes(2);
    unmount();
    await act(async () => vi.advanceTimersByTimeAsync(20_000));
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('aborts in-flight polling on hidden tabs and resumes without accumulating timers', async () => {
    vi.useFakeTimers();
    const slow = deferred<number>();
    const load = vi.fn<(signal: AbortSignal) => Promise<number>>().mockReturnValueOnce(slow.promise).mockResolvedValue(2);
    const { result, unmount } = renderHook(() => usePolling(load, { intervalMs: 1000 }));
    const signal = load.mock.calls[0]?.[0];
    act(() => setVisibility('hidden'));
    expect(signal?.aborted).toBe(true);
    await act(async () => { slow.resolve(999); await vi.advanceTimersByTimeAsync(20_000); });
    expect(result.current.data).toBeNull();
    expect(load).toHaveBeenCalledTimes(1);
    act(() => setVisibility('visible'));
    await flush();
    expect(result.current.data).toBe(2);
    expect(load).toHaveBeenCalledTimes(2);
    unmount();
    expect(load.mock.calls[1]?.[0].aborted).toBe(true);
  });

  it('stops at a terminal run instead of treating command acceptance as success', async () => {
    vi.useFakeTimers();
    const load = vi.fn().mockResolvedValueOnce({ status: 'running' }).mockResolvedValue({ status: 'succeeded' });
    const continuePolling = (run: { status: string }) => run.status !== 'succeeded';
    const { result } = renderHook(() => usePolling(load, { intervalMs: 2000, continuePolling }));
    await flush();
    expect(result.current.data).toEqual({ status: 'running' });
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    expect(result.current.data).toEqual({ status: 'succeeded' });
    await act(async () => vi.advanceTimersByTimeAsync(30_000));
    expect(load).toHaveBeenCalledTimes(2);
  });

  it('pauses authorization errors and backs off only retryable reads', async () => {
    vi.useFakeTimers();
    const forbidden = new ApiError('forbidden', '권한 없음', 403);
    const load = vi.fn().mockRejectedValue(forbidden);
    const { result, unmount } = renderHook(() => usePolling(load, { intervalMs: 1000 }));
    await flush();
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(load).toHaveBeenCalledTimes(1);
    expect(result.current.error).toBe(forbidden);
    unmount();
    const unavailable = vi.fn().mockRejectedValue(new ApiError('not_ready', '의존성 없음', 503));
    renderHook(() => usePolling(unavailable, { intervalMs: 1000 }));
    await flush();
    await act(async () => vi.advanceTimersByTimeAsync(1999));
    expect(unavailable).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(unavailable).toHaveBeenCalledTimes(2);
  });

  it('rate-limits manual refreshes as well as automatic frames', async () => {
    vi.useFakeTimers();
    const load = vi.fn().mockResolvedValue(1);
    const { result } = renderHook(() => usePolling(load, { intervalMs: 1000 }));
    await flush();
    act(() => { result.current.refresh(); result.current.refresh(); });
    await flush();
    expect(load).toHaveBeenCalledTimes(1);
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(load).toHaveBeenCalledTimes(2);
  });
});

describe('protected image object URL ownership', () => {
  it('revokes superseded URLs, then revokes the last URL on unmount', async () => {
    vi.useFakeTimers();
    const load = vi.fn().mockImplementation(async () => ({ blob: fixturePng() }));
    const { result, unmount } = renderHook(() => useProtectedImage(load, 1000));
    await flush();
    expect(result.current.url).toBe('blob:test-image-1');
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(result.current.url).toBe('blob:test-image-2');
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-image-1');
    unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-image-2');
  });

  it('revokes URLs on hidden tabs and never creates one for a late aborted response', async () => {
    vi.useFakeTimers();
    const late = deferred<{ blob: Blob }>();
    const load = vi.fn().mockResolvedValueOnce({ blob: fixturePng() }).mockReturnValue(late.promise);
    const { result, unmount } = renderHook(() => useProtectedImage(load, 1000));
    await flush();
    act(() => setVisibility('hidden'));
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-image-1');
    expect(result.current.url).toBeNull();
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    act(() => setVisibility('visible'));
    await flush();
    unmount();
    const calls = vi.mocked(URL.createObjectURL).mock.calls.length;
    await act(async () => late.resolve({ blob: fixturePng() }));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(calls);
  });
});
