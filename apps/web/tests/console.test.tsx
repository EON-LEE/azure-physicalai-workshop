import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import type { RuntimeInfo } from '../src/api/contracts';
import { ApiError } from '../src/api/errors';
import { ConsoleApp } from '../src/ConsoleApp';
import { account, environment, pendingRun, runtime } from './fixtures/data';
import { makeApi, setVisibility } from './helpers';

describe('console workflows with explicitly injected APIs', () => {
  it('does not fabricate images, metrics, history or a successful run when services are unavailable', async () => {
    const api = makeApi({
      getRuntime: vi.fn().mockRejectedValue(new ApiError('unavailable', 'Isaac Sim GPU 연결이 필요합니다.', 503)),
      getRuns: vi.fn().mockResolvedValue({ items: [] }),
    });
    render(<ConsoleApp api={api} account={account} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('Isaac Sim GPU 연결이 필요합니다.');
    expect(api.getFrame).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: '관측하고 계획 요청' })).toBeDisabled();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: /Run History/ }));
    expect(await screen.findByText('아직 실행 기록이 없습니다')).toBeInTheDocument();
    expect(screen.queryByRole('table')).not.toBeInTheDocument();
  });

  it('retains editor text across navigation and stops camera polling outside Factory Live', async () => {
    const api = makeApi();
    render(<ConsoleApp api={api} account={account} />);
    await waitFor(() => expect(api.getFrame).toHaveBeenCalled());
    await userEvent.click(screen.getByRole('button', { name: /Environment Studio/ }));
    expect(api.getFrame.mock.calls[0]?.[3]?.aborted).toBe(true);
    expect(URL.revokeObjectURL).toHaveBeenCalled();
    const raw = '{ "unsaved": "고객 편집 원문" }';
    fireEvent.change(await screen.findByLabelText('고객 환경 JSON 원본'), { target: { value: raw } });
    await userEvent.click(screen.getByRole('button', { name: /Run History/ }));
    await userEvent.click(screen.getByRole('button', { name: /Environment Studio/ }));
    expect(screen.getByLabelText('고객 환경 JSON 원본')).toHaveValue(raw);
  });

  it('reuses a request UUID for unchanged retries and generates a new one for changed input', async () => {
    const api = makeApi({ createRun: vi.fn().mockRejectedValue(new ApiError('network_error', '요청 결과 미확인')) });
    render(<ConsoleApp api={api} account={account} />);
    await waitFor(() => expect(api.getFrame).toHaveBeenCalled());
    fireEvent.change(screen.getByRole('textbox', { name: '검사·분류 작업을 설명하세요' }), { target: { value: '첫 번째 검사 지시' } });
    await userEvent.click(screen.getByRole('button', { name: '관측하고 계획 요청' }));
    await screen.findByRole('alert');
    await userEvent.click(screen.getByRole('button', { name: '동일 요청 다시 확인' }));
    await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(2));
    const first = api.createRun.mock.calls[0]?.[0];
    const retry = api.createRun.mock.calls[1]?.[0];
    expect(retry).toEqual(first);
    fireEvent.change(screen.getByRole('textbox', { name: '검사·분류 작업을 설명하세요' }), { target: { value: '변경된 검사 지시' } });
    await userEvent.click(screen.getByRole('button', { name: '동일 요청 다시 확인' }));
    await waitFor(() => expect(api.createRun).toHaveBeenCalledTimes(3));
    expect(api.createRun.mock.calls[2]?.[0].request_id).not.toBe(first?.request_id);
    expect(api.createRun.mock.calls[2]?.[0].instruction).toBe('변경된 검사 지시');
    expect(api.approveRun).not.toHaveBeenCalled();
  });

  it('keeps activation at loading after 202 until runtime reports the exact requested revision', async () => {
    vi.useFakeTimers();
    let reported: RuntimeInfo = { ...runtime, simulation: { ...runtime.simulation, status: 'unavailable', revision: null } };
    const api = makeApi({
      getRuntime: vi.fn().mockImplementation(async () => reported),
      activateEnvironment: vi.fn().mockResolvedValue({ activation_id: 'test-activation', environment_id: environment.environment_id, revision: environment.revision, status: 'loading' }),
    });
    render(<ConsoleApp api={api} account={account} />);
    await act(async () => { await Promise.resolve(); });
    fireEvent.click(screen.getByRole('button', { name: '저장된 씬 활성화' }));
    await act(async () => { await Promise.resolve(); });
    expect(screen.getByText('요청 접수됨 · 아직 준비 완료가 아닙니다')).toBeInTheDocument();
    expect(api.getFrame).not.toHaveBeenCalled();
    reported = { ...runtime, simulation: { ...runtime.simulation, revision: 'a-different-revision' } };
    await act(async () => vi.advanceTimersByTimeAsync(4000));
    expect(screen.queryByText('요청한 버전의 런타임 준비 확인됨')).not.toBeInTheDocument();
    reported = runtime;
    await act(async () => vi.advanceTimersByTimeAsync(3000));
    expect(screen.getByText('요청한 버전의 런타임 준비 확인됨')).toBeInTheDocument();
    expect(api.getFrame).toHaveBeenCalled();
  });

  it('labels old frames as stale and revokes object URLs when the tab is hidden', async () => {
    const api = makeApi();
    api.getFrame.mockImplementation(async () => ({
      blob: new Blob(['test-only'], { type: 'image/png' }), frameId: 'old-frame', capturedAt: new Date(Date.now() - 60_000).toISOString(), physicsSteps: 11,
    }));
    render(<ConsoleApp api={api} account={account} />);
    expect(await screen.findByText('오래된 프레임 / 시간 확인 필요')).toBeInTheDocument();
    expect(await screen.findByText('마지막 수신 이미지 · 현재 상태를 보장하지 않음')).toBeInTheDocument();
    act(() => setVisibility('hidden'));
    expect(URL.revokeObjectURL).toHaveBeenCalled();
    expect(api.getFrame.mock.calls[0]?.[3]?.aborted).toBe(true);
  });

  it('recovers an actual pending run from API history instead of auto-dispatching it', async () => {
    const api = makeApi({ getRuns: vi.fn().mockResolvedValue({ items: [pendingRun] }), getRun: vi.fn().mockResolvedValue(pendingRun) });
    render(<ConsoleApp api={api} account={account} />);
    expect(await screen.findByText('승인 대기')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '관측하고 계획 요청' })).toBeDisabled();
    expect(api.createRun).not.toHaveBeenCalled();
    expect(api.approveRun).not.toHaveBeenCalled();
  });
});
