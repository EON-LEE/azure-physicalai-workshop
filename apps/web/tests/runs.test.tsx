import { act, fireEvent, render, screen, within } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import type { RunRecord } from '../src/api/contracts';
import { ApiError } from '../src/api/errors';
import { RunDetail } from '../src/views/RunDetail';
import { environment, pendingRun, runningRun, runtime, succeededRun } from './fixtures/data';
import { deferred, makeApi } from './helpers';

async function flush() {
  await act(async () => { await Promise.resolve(); });
}

describe('approval and confirmed run transitions', () => {
  it('exposes the configured goal and actual measured pose without adopting a different saved revision', async () => {
    const run = { ...succeededRun, execution: { ...succeededRun.execution!, final_position: [0.22, -0.38, 0.2] } };
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(run) });
    const { rerender } = render(<RunDetail api={api} initialRun={run} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    const comparison = within(screen.getByRole('region', { name: '고객 공정 실험 위치 비교' }));
    expect(comparison.getByText('시뮬레이터의 최종 측정 좌표')).toBeInTheDocument();
    expect(comparison.getAllByText('(0.22, -0.38, 0.2) m').length).toBeGreaterThan(0);
    rerender(<RunDetail api={api} initialRun={run} runtime={runtime} runtimeFresh environment={{ ...environment, revision: 'different-revision' }} />);
    expect(comparison.getByText('실행 시점의 저장 버전·목표 좌표 확인 필요')).toBeInTheDocument();
    expect(api.approveRun).not.toHaveBeenCalled();
  });

  it('shows a pending plan without auto-approving or fabricating completion', async () => {
    vi.useFakeTimers();
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(pendingRun) });
    render(<RunDetail api={api} initialRun={pendingRun} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(screen.getByText('승인 대기')).toBeInTheDocument();
    expect(screen.getByRole('checkbox')).not.toBeChecked();
    expect(screen.getByRole('button', { name: '계획 승인 및 실행' })).toBeDisabled();
    expect(api.approveRun).not.toHaveBeenCalled();
    expect(screen.queryByText('서버가 실행 성공을 확인했습니다')).not.toBeInTheDocument();
  });

  it('requires deliberate confirmation, sends the exact displayed model ID, and waits for terminal success', async () => {
    vi.useFakeTimers();
    let server = pendingRun;
    const api = makeApi({
      getRun: vi.fn().mockImplementation(async () => server),
      approveRun: vi.fn().mockImplementation(async () => { server = runningRun; return runningRun; }),
    });
    render(<RunDetail api={api} initialRun={pendingRun} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: '계획 승인 및 실행' }));
    await flush();
    expect(api.approveRun).toHaveBeenCalledWith(pendingRun.id, pendingRun.plan?.model_response_id, expect.any(AbortSignal));
    expect(api.approveRun).toHaveBeenCalledTimes(1);
    expect(screen.getByText('물리 실행 확인 중')).toBeInTheDocument();
    expect(screen.queryByText('서버가 실행 성공을 확인했습니다')).not.toBeInTheDocument();
    server = succeededRun;
    await act(async () => vi.advanceTimersByTimeAsync(3000));
    expect(screen.getByText('서버가 실행 성공을 확인했습니다')).toBeInTheDocument();
    const calls = api.getRun.mock.calls.length;
    await act(async () => vi.advanceTimersByTimeAsync(15_000));
    expect(api.getRun).toHaveBeenCalledTimes(calls);
  });

  it('does not show cancellation complete until the simulator confirms cancelled', async () => {
    vi.useFakeTimers();
    let server: RunRecord = runningRun;
    const api = makeApi({
      getRun: vi.fn().mockImplementation(async () => server),
      cancelRun: vi.fn().mockImplementation(async () => { server = { ...runningRun, status: 'cancelling' }; return server; }),
    });
    render(<RunDetail api={api} initialRun={runningRun} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    fireEvent.click(screen.getByRole('button', { name: '이 실행 취소' }));
    await flush();
    expect(screen.getByText('취소 요청 확인 중')).toBeInTheDocument();
    expect(screen.queryByText('서버가 취소를 확인했습니다')).not.toBeInTheDocument();
    server = { ...runningRun, status: 'cancelled' };
    await act(async () => vi.advanceTimersByTimeAsync(3000));
    expect(screen.getByText('서버가 취소를 확인했습니다')).toBeInTheDocument();
    expect(screen.queryByText('서버가 실행 성공을 확인했습니다')).not.toBeInTheDocument();
  });

  it('can cancel while an approval response is pending and ignores a late approval ACK', async () => {
    vi.useFakeTimers();
    const approval = deferred<RunRecord>();
    const cancelling: RunRecord = { ...runningRun, status: 'cancelling' };
    let server = pendingRun;
    const api = makeApi({
      getRun: vi.fn().mockImplementation(async () => server),
      approveRun: vi.fn().mockReturnValue(approval.promise),
      cancelRun: vi.fn().mockImplementation(async () => { server = cancelling; return cancelling; }),
    });
    render(<RunDetail api={api} initialRun={pendingRun} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: '계획 승인 및 실행' }));
    expect(screen.getByRole('button', { name: '이 실행 취소' })).toBeEnabled();
    fireEvent.click(screen.getByRole('button', { name: '이 실행 취소' }));
    await flush();
    expect(api.cancelRun).toHaveBeenCalledTimes(1);
    await act(async () => approval.resolve(runningRun));
    expect(screen.getByText('취소 요청 확인 중')).toBeInTheDocument();
    expect(screen.queryByText('물리 실행 확인 중')).not.toBeInTheDocument();
  });

  it('requires a new plan after an approval conflict instead of retrying the stale decision', async () => {
    vi.useFakeTimers();
    const api = makeApi({
      getRun: vi.fn().mockResolvedValue(pendingRun),
      approveRun: vi.fn().mockRejectedValue(new ApiError('scene_changed', '관측 후 부품 상태가 변경되었습니다.', 409)),
    });
    render(<RunDetail api={api} initialRun={pendingRun} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    fireEvent.click(screen.getByRole('checkbox'));
    fireEvent.click(screen.getByRole('button', { name: '계획 승인 및 실행' }));
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('관측 후 부품 상태가 변경되었습니다.');
    expect(screen.getByRole('button', { name: '계획 승인 및 실행' })).toBeDisabled();
    expect(screen.getByText(/서버가 이 계획의 승인을 거절했습니다/)).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(10_000));
    expect(api.approveRun).toHaveBeenCalledTimes(1);
  });

  it.each(['failed', 'timed_out'] as const)('does not convert %s into a successful run', async (status) => {
    const failed: RunRecord = { ...runningRun, status, error: { code: 'simulator_failed', message: '실제 물리 명령 실패', retryable: false } };
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(failed) });
    render(<RunDetail api={api} initialRun={failed} runtime={runtime} runtimeFresh environment={environment} />);
    await flush();
    expect(screen.getByRole('alert')).toHaveTextContent('실제 물리 명령 실패');
    expect(screen.queryByText('서버가 실행 성공을 확인했습니다')).not.toBeInTheDocument();
  });

  it('blocks approval after a scene epoch change and exposes actual trace identifiers', async () => {
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(pendingRun) });
    render(<RunDetail api={api} initialRun={pendingRun} runtime={{ ...runtime, simulation: { ...runtime.simulation, epoch: 'new-world' } }} runtimeFresh environment={environment} />);
    await flush();
    expect(screen.getByRole('checkbox')).toBeDisabled();
    expect(screen.getByText(/관측 이후 씬이 바뀌었습니다/)).toBeInTheDocument();
    fireEvent.click(screen.getByText('증거 및 실제 실행 ID'));
    const identifiers = within(screen.getByLabelText('계획 증거 ID'));
    expect(identifiers.getByText('test-only-foundry-response-id')).toBeVisible();
    expect(identifiers.getByText('test-only-observation-id')).toBeVisible();
  });

  it('aborts an in-flight reconciliation on unmount and stops subsequent polling', async () => {
    vi.useFakeTimers();
    const pending = deferred<RunRecord>();
    const api = makeApi({ getRun: vi.fn().mockReturnValue(pending.promise) });
    const { unmount } = render(<RunDetail api={api} initialRun={runningRun} runtime={runtime} runtimeFresh environment={environment} />);
    const signal = api.getRun.mock.calls[0]?.[1];
    unmount();
    expect(signal?.aborted).toBe(true);
    await act(async () => { pending.resolve(succeededRun); await vi.advanceTimersByTimeAsync(10_000); });
    expect(api.getRun).toHaveBeenCalledTimes(1);
  });
});
