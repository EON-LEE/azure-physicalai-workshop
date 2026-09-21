import { act, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import * as api from '../src/public/api';
import { PublicLiveCamera } from '../src/public/Media';
import { epoch, frame } from './fixtures/public';
import { deferred } from './helpers';

async function flush() {
  await act(async () => { await Promise.resolve(); });
}

describe('public live display uses a monotonic expiry', () => {
  it.each([-600_000, 600_000])('ignores a frozen wall clock skewed by %sms and expires before the generic clock tick', async (skew) => {
    vi.useFakeTimers();
    const capturedAt = '2026-09-22T00:00:00.123456+00:00';
    vi.spyOn(Date, 'now').mockReturnValue(Date.parse(capturedAt) + skew);
    const pending = deferred<api.PublicFrame>();
    const image = frame({ capturedAt, expiresAtMonotonicMs: performance.now() + 4500 });
    vi.spyOn(api, 'getDemoFrame').mockResolvedValueOnce(image).mockReturnValue(pending.promise);
    render(<PublicLiveCamera epoch={epoch} onSceneChanged={vi.fn()} />);
    await flush();
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
    expect(document.querySelector('time')).toHaveAttribute('dateTime', capturedAt);
    await act(async () => vi.advanceTimersByTimeAsync(4499));
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
  });

  it.each([-86_400_000, 86_400_000])('does not extend or shorten an accepted frame after a wall-clock jump of %sms', async (jump) => {
    vi.useFakeTimers();
    const clock = vi.spyOn(Date, 'now');
    const initialWallTime = Date.now();
    const pending = deferred<api.PublicFrame>();
    vi.spyOn(api, 'getDemoFrame')
      .mockResolvedValueOnce(frame({ expiresAtMonotonicMs: performance.now() + 400 }))
      .mockReturnValue(pending.promise);
    render(<PublicLiveCamera epoch={epoch} onSceneChanged={vi.fn()} />);
    await flush();
    clock.mockReturnValue(initialWallTime + jump);
    await act(async () => vi.advanceTimersByTimeAsync(399));
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(1));
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
  });

  it('rejects a frame whose monotonic budget expired before React displayed it', async () => {
    vi.useFakeTimers();
    vi.spyOn(api, 'getDemoFrame').mockResolvedValue(frame({ expiresAtMonotonicMs: performance.now() - 1 }));
    render(<PublicLiveCamera epoch={epoch} onSceneChanged={vi.fn()} />);
    await flush();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText(/멈춘 이미지를 LIVE로 표시하지 않습니다/)).toBeInTheDocument();
  });

  it('cleans up the deadline timer, request signal and image URL on unmount', async () => {
    vi.useFakeTimers();
    const fetchImage = vi.spyOn(api, 'getDemoFrame').mockResolvedValue(frame());
    const { unmount } = render(<PublicLiveCamera epoch={epoch} onSceneChanged={vi.fn()} />);
    await flush();
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
    expect(vi.getTimerCount()).toBeGreaterThan(0);
    unmount();
    expect(fetchImage.mock.calls[0]?.[1].aborted).toBe(true);
    expect(URL.revokeObjectURL).toHaveBeenCalledWith('blob:test-image-1');
    expect(vi.getTimerCount()).toBe(0);
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(fetchImage).toHaveBeenCalledTimes(1);
  });
});
