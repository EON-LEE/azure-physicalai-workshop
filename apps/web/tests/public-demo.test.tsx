import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppEntry } from '../src/AppEntry';
import { DemoViewer } from '../src/public/DemoViewer';
import * as api from '../src/public/api';
import { ApiError } from '../src/api/errors';
import { taskSucceeded } from '../src/public/presentation';
import { deferred, setVisibility } from './helpers';
import { evidence, frame, makePresentation, makeSnapshot, nextEpoch, referenceSnapshot } from './fixtures/public';

beforeEach(() => window.history.replaceState(null, '', '/'));
afterEach(() => vi.unstubAllGlobals());

async function flush() {
  await act(async () => { await Promise.resolve(); });
}

function imageApis(presentation = makePresentation()) {
  return {
    frames: vi.spyOn(api, 'getDemoFrame').mockImplementation(async (epoch) => frame({ sceneEpoch: epoch })),
    evidence: vi.spyOn(api, 'getDemoEvidence').mockResolvedValue(evidence(presentation)),
  };
}

describe('no-login audience presentation', () => {
  it('loads only the public endpoint without config, authentication, private APIs or writes', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify(referenceSnapshot())));
    vi.stubGlobal('fetch', fetch);
    render(<AppEntry />);
    await screen.findByRole('heading', { name: '게시된 자동 시연이 없습니다' });
    expect(screen.queryByRole('button', { name: 'Microsoft로 로그인' })).not.toBeInTheDocument();
    expect(screen.getByRole('link', { name: '운영자' })).toHaveAttribute('href', '/operator');
    for (const [url, options] of fetch.mock.calls) {
      expect(url).toBe('/api/demo');
      expect(options).toMatchObject({ method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error' });
      expect(new Headers(options.headers).has('Authorization')).toBe(false);
    }
  });

  it.each([null, undefined])('does not infer a presentation from base ready flags when presentation is %s', async (presentation) => {
    const { frames } = imageApis();
    render(<DemoViewer loadSnapshot={async () => makeSnapshot({ presentation })} />);
    await screen.findByRole('heading', { name: '게시된 자동 시연이 없습니다' });
    expect(frames).not.toHaveBeenCalled();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.queryByText('종합 성공 확인')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: '양품 흐름' })).not.toBeInTheDocument();
    expect(screen.queryByRole('navigation', { name: '시나리오 단계' })).not.toBeInTheDocument();
    expect(screen.queryByText('작업 셀 구성도 · 실제 시뮬레이션 아님')).not.toBeInTheDocument();
  });

  it('shows actual current input, decision, measured phase and historical evidence without invented scores', async () => {
    const snapshot = makeSnapshot();
    const { frames, evidence: images } = imageApis(snapshot.presentation!);
    render(<DemoViewer loadSnapshot={async () => snapshot} />);
    await screen.findByText('LIVE · 실제 카메라 수신');
    expect(screen.getByText('서버 입력: 표면 흠집 부품')).toBeInTheDocument();
    expect(screen.getByText('불량으로 판단')).toBeInTheDocument();
    expect(screen.getByText('대상 트레이로 이동')).toBeInTheDocument();
    expect(await screen.findByRole('img', { name: /현재 Foundry 판단에 실제 사용된/ })).toBeInTheDocument();
    expect(screen.getByText('판단에 사용된 원본 입력 · LIVE 아님')).toBeInTheDocument();
    expect(frames).toHaveBeenCalledWith(snapshot.presentation?.scene_epoch, expect.any(AbortSignal));
    expect(images).toHaveBeenCalledWith(expect.objectContaining({ observation_id: snapshot.presentation?.decision?.observation_id }), expect.any(AbortSignal));
    expect(screen.queryByText('종합 성공 확인')).not.toBeInTheDocument();
    expect(screen.queryByText(/신뢰도|정확도 \d|%/)).not.toBeInTheDocument();
  });

  it('does not advance physical phases or complete a run just because time elapsed', async () => {
    vi.useFakeTimers();
    imageApis();
    const presentation = makePresentation({ motion: { status: 'running', phase: null, part_position_m: null, target_position_m: [0.22, -0.38, 0.2] } });
    const load = vi.fn().mockImplementation(async () => makeSnapshot({ presentation }));
    render(<DemoViewer loadSnapshot={load} />);
    await flush();
    await act(async () => vi.advanceTimersByTimeAsync(12_000));
    expect(screen.getByText('단계 정보 미수신')).toBeInTheDocument();
    expect(screen.getByText('최종 결과 대기')).toBeInTheDocument();
    expect(screen.queryByText('종합 성공 확인')).not.toBeInTheDocument();
    expect(load.mock.calls.length).toBeLessThanOrEqual(7);
  });

  it('offers keyboard pause/resume for viewing only, without stopping the server', async () => {
    const user = userEvent.setup();
    const { frames } = imageApis();
    const load = vi.fn().mockImplementation(async () => makeSnapshot());
    render(<DemoViewer loadSnapshot={load} />);
    await screen.findByText('LIVE · 실제 카메라 수신');
    const pause = screen.getByRole('button', { name: '관람 일시 정지' });
    pause.focus();
    await user.keyboard('{Enter}');
    expect(screen.getByRole('heading', { name: '영상 관람을 일시 정지했습니다' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '관람 재개' })).toHaveAttribute('aria-pressed', 'true');
    expect(window.location.search).toContain('viewing=paused');
    expect(frames.mock.calls[0]?.[1].aborted).toBe(true);
    expect(URL.revokeObjectURL).toHaveBeenCalled();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    await user.keyboard('{Enter}');
    await screen.findByText('LIVE · 실제 카메라 수신');
  });

  it('starts paused for reduced motion while still showing published metadata', async () => {
    vi.stubGlobal('matchMedia', vi.fn().mockReturnValue({ matches: true }));
    const { frames } = imageApis();
    render(<DemoViewer loadSnapshot={async () => makeSnapshot()} />);
    await screen.findByText('서버 입력: 표면 흠집 부품');
    expect(screen.getByRole('button', { name: '관람 재개' })).toBeInTheDocument();
    expect(frames).not.toHaveBeenCalled();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
  });

  it('clears previous-cycle images on epoch/cycle change and waits for matching new data', async () => {
    vi.useFakeTimers();
    const first = makePresentation();
    const { frames } = imageApis(first);
    const next = deferred<api.PublicFrame>();
    frames.mockImplementation((epoch) => epoch === nextEpoch ? next.promise : Promise.resolve(frame()));
    let presentation = first;
    render(<DemoViewer loadSnapshot={async () => makeSnapshot({ presentation })} />);
    await flush();
    expect(screen.getByText('불량으로 판단')).toBeInTheDocument();
    const created = vi.mocked(URL.createObjectURL).mock;
    const oldUrls = created.results.filter((_, index) => {
      const source = created.calls[index]?.[0];
      return source instanceof Blob && source.type === 'image/png';
    }).map(entry => entry.value);
    expect(oldUrls).toHaveLength(2);
    presentation = makePresentation({ cycle: 2, scene_epoch: nextEpoch, scenario: 'normal', status: 'inspecting', decision: null, motion: null });
    await act(async () => vi.advanceTimersByTimeAsync(2000));
    expect(screen.queryByText('불량으로 판단')).not.toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /현재 Foundry 판단에 실제 사용된/ })).not.toBeInTheDocument();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.getByText('서버 입력: 정상 부품')).toBeInTheDocument();
    for (const url of oldUrls) expect(URL.revokeObjectURL).toHaveBeenCalledWith(url);
    await act(async () => next.resolve(frame({ sceneEpoch: nextEpoch })));
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
  });

  it('refreshes the snapshot on a 409 camera mismatch without presenting the wrong image', async () => {
    vi.useFakeTimers();
    const { frames } = imageApis();
    frames.mockRejectedValue(new ApiError('scene_changed', '씬이 변경되었습니다.', 409));
    const load = vi.fn().mockImplementation(async () => makeSnapshot());
    render(<DemoViewer loadSnapshot={load} />);
    await flush();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.queryByText('불량으로 판단')).not.toBeInTheDocument();
    expect(screen.getByRole('heading', { name: '관측과 게시 회차가 일치하지 않습니다' })).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(1000));
    expect(load).toHaveBeenCalledTimes(2);
    expect(frames).toHaveBeenCalledTimes(1);
  });

  it('stops polling/aborts in-flight images on hidden tabs and unmount', async () => {
    vi.useFakeTimers();
    const pending = deferred<api.PublicFrame>();
    const { frames } = imageApis();
    frames.mockReturnValue(pending.promise);
    const load = vi.fn().mockImplementation(async () => makeSnapshot());
    const { unmount } = render(<DemoViewer loadSnapshot={load} />);
    await flush();
    act(() => setVisibility('hidden'));
    expect(frames.mock.calls[0]?.[1].aborted).toBe(true);
    await act(async () => vi.advanceTimersByTimeAsync(20_000));
    expect(load).toHaveBeenCalledTimes(1);
    expect(frames).toHaveBeenCalledTimes(1);
    unmount();
    const created = vi.mocked(URL.createObjectURL).mock.calls.length;
    await act(async () => pending.resolve(frame()));
    expect(URL.createObjectURL).toHaveBeenCalledTimes(created);
  });

  it('removes a PNG that the browser cannot decode and never calls it a successful task', async () => {
    imageApis();
    render(<DemoViewer loadSnapshot={async () => makeSnapshot()} />);
    const camera = await screen.findByRole('img', { name: /이번 시연 회차의 Isaac Sim/ });
    fireEvent.error(camera);
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.getByText('PNG를 표시할 수 없습니다. 다시 연결해 주세요.')).toBeInTheDocument();
    expect(screen.queryByText('종합 성공 확인')).not.toBeInTheDocument();
  });

  it('removes a frozen frame after the freshness deadline instead of leaving it LIVE', async () => {
    vi.useFakeTimers();
    const first = frame();
    const pending = deferred<api.PublicFrame>();
    const { frames } = imageApis();
    frames.mockResolvedValueOnce(first).mockReturnValue(pending.promise);
    render(<DemoViewer loadSnapshot={async () => makeSnapshot()} />);
    await flush();
    expect(screen.getByText('LIVE · 실제 카메라 수신')).toBeInTheDocument();
    await act(async () => vi.advanceTimersByTimeAsync(6000));
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.queryByRole('img', { name: /이번 시연 회차의 Isaac Sim/ })).not.toBeInTheDocument();
    expect(screen.getByText(/멈춘 이미지를 LIVE로 표시하지 않습니다/)).toBeInTheDocument();
  });

  it('shows explicit errors and no simulated replacement when the public snapshot fails', async () => {
    const { frames } = imageApis();
    render(<DemoViewer loadSnapshot={async () => { throw new ApiError('network_error', '공개 연결이 끊겼습니다.'); }} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('공개 연결이 끊겼습니다.');
    expect(screen.getByRole('heading', { name: '공개 시연 정보를 불러오지 못했습니다' })).toBeInTheDocument();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(frames).not.toHaveBeenCalled();
  });

  it('shows expired state without inferring motion from stale metadata', async () => {
    const { frames } = imageApis();
    render(<DemoViewer loadSnapshot={async () => makeSnapshot({ presentation: makePresentation({ expires_at: new Date(Date.now() - 1000).toISOString() }) })} />);
    await screen.findByRole('heading', { name: '게시된 시연 시간이 끝났습니다' });
    expect(frames).not.toHaveBeenCalled();
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(screen.getByText('마지막 수신 상태')).toBeInTheDocument();
  });

  it('does not equate a persisted result with camera availability or general policy quality', async () => {
    const presentation = makePresentation({
      status: 'completed',
      result: { status: 'succeeded', physical_success: true, inspection_correct: true, final_position_m: [0.22, -0.38, 0.2], completed_at: new Date().toISOString(), message: '테스트 전용 측정 결과' },
    });
    const { frames } = imageApis(presentation);
    render(<DemoViewer loadSnapshot={async () => ({ ...referenceSnapshot(), simulation: { status: 'unavailable', live_available: false, frame_url: null, message_code: 'unavailable' }, presentation })} />);
    await screen.findByText('종합 성공 확인');
    expect(screen.getByText(/마지막 게시 결과 · 현재 카메라 연결이나 실행을 의미하지 않음/)).toBeInTheDocument();
    expect(frames).not.toHaveBeenCalled();
    fireEvent.click(screen.getByText('무엇이 검증되었고, 무엇이 아닌가요?'));
    expect(screen.getByText(/학습된 VLA나 정책이 로봇을 제어하는 데모가 아닙니다/)).toBeVisible();
  });

  it.each([
    [true, true, true],
    [true, false, false],
    [false, true, false],
    [false, false, false],
    [true, null, false],
  ])('requires independent physical=%s AND inspection=%s for success=%s', async (physical, inspection, success) => {
    const result: NonNullable<api.Presentation['result']> = {
      status: 'succeeded', physical_success: physical as boolean, inspection_correct: inspection as boolean | null,
      final_position_m: [0.22, -0.38, 0.2], completed_at: new Date().toISOString(), message: '측정 결과',
    };
    expect(taskSucceeded(result)).toBe(success);
    const presentation = makePresentation({ status: 'completed', result });
    imageApis(presentation);
    render(<DemoViewer loadSnapshot={async () => makeSnapshot({ presentation })} />);
    const outcome = await screen.findByRole('region', { name: '판단과 물리 결과를 따로 확인합니다' });
    expect(within(outcome).getByText(success ? '종합 성공 확인' : '종합 성공 미확인')).toBeInTheDocument();
    expect(within(outcome).getByRole('heading', { name: '검사 결과' })).toBeInTheDocument();
    expect(within(outcome).getByRole('heading', { name: '물리 결과' })).toBeInTheDocument();
  });

  it('aborts a public snapshot on navigation', async () => {
    const pending = deferred<api.DemoSnapshot>();
    const load = vi.fn().mockReturnValue(pending.promise);
    const { unmount } = render(<DemoViewer loadSnapshot={load} />);
    await waitFor(() => expect(load).toHaveBeenCalledTimes(1));
    unmount();
    expect(load.mock.calls[0]?.[0].aborted).toBe(true);
  });
});
