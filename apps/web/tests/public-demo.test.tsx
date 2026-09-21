import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { AppEntry } from '../src/AppEntry';
import { DemoViewer } from '../src/public/DemoViewer';
import { demoSchema, getDemo, getDemoFrame, type DemoSnapshot } from '../src/public/api';

const data: DemoSnapshot = {
  api_version: 'public-demo-v1', access: 'public_read_only', deployment: 'azure',
  mode: 'reference', observed_at: '2026-09-21T00:00:00Z',
  scene: {
    id: 'inspection-cell-v1', name: 'Inspection and sorting cell', length_unit: 'm',
    robot: 'Franka reference arm', data_origin: 'synthetic_reference_configuration',
    stations: [
      { id: 'supply', role: 'source', position_m: [-0.5, 0, 0.2] },
      { id: 'inspection', role: 'inspection', position_m: [0, 0.4, 0.2] },
      { id: 'accepted', role: 'accepted', position_m: [0.5, 0.4, 0.2] },
      { id: 'rejected', role: 'rejected', position_m: [0.5, -0.4, 0.2] },
    ],
  },
  simulation: { status: 'not_published', live_available: false, message_code: 'live_not_published', frame_url: null },
  agent: { provider: 'microsoft_foundry', connectivity: 'verified', verified_at: '2026-09-20T15:34:04Z', verification_scope: 'connectivity_only' },
  learning: { status: 'cpu_smoke_verified', execution_location: 'azure_acr', data_kind: 'test_fixture', optimizer_steps: 1, quality_verified: false },
  capabilities: { anonymous_control: false, anonymous_editing: false, public_live_video: false },
};

beforeEach(() => window.history.replaceState(null, '', '/'));
afterEach(() => vi.unstubAllGlobals());

describe('login-free public viewer', () => {
  it('loads the real public endpoint without any authentication or private API requests', async () => {
    const requests: Array<{ url: string; options?: RequestInit }> = [];
    vi.stubGlobal('fetch', vi.fn(async (url: string, options?: RequestInit) => {
      requests.push({ url, options });
      if (url !== '/api/demo') throw new Error(`Unexpected private request: ${url}`);
      return new Response(JSON.stringify(data), { status: 200, headers: { 'Content-Type': 'application/json' } });
    }));
    render(<AppEntry />);
    expect(await screen.findByRole('heading', { name: '비전 검사 · 부품 분류' })).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Microsoft로 로그인' })).not.toBeInTheDocument();
    expect(screen.getByText('작업 셀 구성도 · 실제 시뮬레이션 아님')).toBeInTheDocument();
    expect(requests.length).toBeGreaterThan(0);
    for (const request of requests) {
      expect(request.url).toBe('/api/demo');
      expect(request.options?.method).toBe('GET');
      expect(request.options?.credentials).toBe('omit');
      expect(new Headers(request.options?.headers).has('Authorization')).toBe(false);
    }
    expect(screen.getByRole('link', { name: /운영자/ })).toHaveAttribute('href', '/operator');
  });

  it('explains normal/reject scenarios with keyboard-accessible controls and URL state, never a write', async () => {
    const load = vi.fn().mockResolvedValue(data);
    const user = userEvent.setup();
    render(<DemoViewer loadSnapshot={load} />);
    await screen.findByRole('heading', { name: '비전 검사 · 부품 분류' });
    const normal = screen.getByRole('button', { name: '양품 흐름' });
    normal.focus();
    await user.keyboard('{Enter}');
    expect(normal).toHaveAttribute('aria-pressed', 'true');
    await user.click(screen.getByRole('button', { name: /04.*결과를 다시/ }));
    expect(screen.getByText('설명 경로: 검사 → 다음 공정')).toBeInTheDocument();
    expect(window.location.search).toContain('path=accepted');
    expect(window.location.search).toContain('step=sort');
    expect(screen.queryByText('실제 카메라')).not.toBeInTheDocument();
    expect(load).toHaveBeenCalledTimes(1);
  });

  it('restores deep-linked explanation state and responds to browser history', async () => {
    window.history.replaceState(null, '', '/?path=accepted&step=approve');
    render(<DemoViewer loadSnapshot={async () => data} />);
    await screen.findByRole('button', { name: '양품 흐름' });
    expect(screen.getByRole('button', { name: /03.*실행 범위를/ })).toHaveAttribute('aria-pressed', 'true');
    window.history.replaceState(null, '', '/?path=rejected&step=observe');
    fireEvent.popState(window);
    expect(screen.getByRole('button', { name: '불량 격리' })).toHaveAttribute('aria-pressed', 'true');
  });

  it('shows a retryable public error instead of asking for login or fabricating demo data', async () => {
    render(<DemoViewer loadSnapshot={async () => { throw new Error('Unavailable'); }} />);
    expect(await screen.findByText('데모 정보를 불러오지 못했습니다')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '다시 불러오기' })).toBeInTheDocument();
    expect(screen.queryByRole('heading', { name: '비전 검사 · 부품 분류' })).not.toBeInTheDocument();
  });

  it('aborts its public fetch when navigating away', async () => {
    let seen: AbortSignal | undefined;
    const load = vi.fn((signal: AbortSignal) => {
      seen = signal;
      return new Promise<DemoSnapshot>(() => {});
    });
    const { unmount } = render(<DemoViewer loadSnapshot={load} />);
    await waitFor(() => expect(seen).toBeDefined());
    unmount();
    expect(seen?.aborted).toBe(true);
  });
});

describe('public response contract', () => {
  it('rejects an inconsistent live claim or anonymous-control capability', () => {
    expect(demoSchema.safeParse({ ...data, mode: 'live' }).success).toBe(false);
    expect(demoSchema.safeParse({ ...data, capabilities: { ...data.capabilities, anonymous_control: true } }).success).toBe(false);
  });
  it('rejects malformed snapshot responses instead of inventing content', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('{}', { status: 200 })));
    await expect(getDemo(new AbortController().signal)).rejects.toThrow('형식');
  });
  it('rejects camera responses without real frame metadata', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue(new Response('not a camera', { headers: { 'Content-Type': 'image/png' } })));
    await expect(getDemoFrame(new AbortController().signal)).rejects.toThrow('프레임');
  });
  it('accepts a real PNG envelope anonymously and rejects mislabeled image bytes', async () => {
    const imageHeaders = {
      'Content-Type': 'image/png', 'X-Frame-Id': 'frame-public',
      'X-Captured-At': new Date().toISOString(), 'X-Physics-Steps': '100',
    };
    const fetcher = vi.fn().mockResolvedValueOnce(new Response(
      new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10, 0]), { headers: imageHeaders },
    )).mockResolvedValueOnce(new Response('not-a-png', { headers: imageHeaders }));
    vi.stubGlobal('fetch', fetcher);
    const frame = await getDemoFrame(new AbortController().signal);
    expect(frame.frameId).toBe('frame-public');
    expect(frame.physicsSteps).toBe(100);
    expect(fetcher).toHaveBeenCalledWith('/api/demo/frame?camera=overview', expect.objectContaining({ credentials: 'omit' }));
    await expect(getDemoFrame(new AbortController().signal)).rejects.toThrow('PNG');
  });
  it('does not follow redirects or send credentials on the public request', async () => {
    const fetcher = vi.fn().mockResolvedValue(new Response(JSON.stringify(data)));
    vi.stubGlobal('fetch', fetcher);
    await getDemo(new AbortController().signal);
    expect(fetcher).toHaveBeenCalledWith('/api/demo', expect.objectContaining({ redirect: 'error', credentials: 'omit' }));
  });
});
