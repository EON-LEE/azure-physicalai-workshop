import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import * as api from '../src/public/api';
import { RecordedCases } from '../src/public/CustomerStory';
import { ApiError } from '../src/api/errors';
import { fixturePng } from './fixtures/data';
import { recordedCases } from './fixtures/public';

beforeEach(() => window.history.replaceState(null, '', '/'));

describe('recorded customer outcomes, not a substitute LIVE feed', () => {
  it('only reads the three curated cases on request and labels originals as historical', async () => {
    const data = recordedCases();
    const load = vi.spyOn(api, 'getDemoCases').mockResolvedValue(data);
    vi.spyOn(api, 'getDemoCaseEvidence').mockImplementation(async (_, item) => ({
      blob: fixturePng(), frameId: item.observation_id, capturedAt: item.captured_at,
    }));
    render(<RecordedCases presentationId={data.presentation_id} />);
    expect(load).not.toHaveBeenCalled();
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: '실제 실행 기록 3가지 비교' }));
    expect(await screen.findAllByRole('img')).toHaveLength(3);
    expect(screen.getByText('정상 트레이 도착 측정됨')).toBeInTheDocument();
    expect(screen.getByText('격리 트레이 도착 측정됨')).toBeInTheDocument();
    expect(screen.getByText('검사 불일치 · 이동 승인 및 실행 없음')).toBeInTheDocument();
    expect(screen.getAllByText(/실제 검사 입력 기록 · LIVE 아님/)).toHaveLength(3);
    expect(screen.queryByText('LIVE · 실제 카메라 수신')).not.toBeInTheDocument();
    expect(window.location.search).toBe('?evidence=recorded');
  });

  it('shows missing evidence and failures instead of inventing customer successes', async () => {
    const load = vi.spyOn(api, 'getDemoCases').mockRejectedValue(new ApiError('unavailable', '기록 보관소 연결 필요', 503));
    render(<RecordedCases presentationId={null} />);
    await userEvent.click(screen.getByRole('button', { name: '실제 실행 기록 3가지 비교' }));
    expect(await screen.findByRole('alert')).toHaveTextContent('기록 보관소 연결 필요');
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    load.mockResolvedValue({ api_version: 'public-demo-cases-v1', source: 'recorded_reference_runs', presentation_id: null, cases: [] });
    await userEvent.click(screen.getByRole('button', { name: '게시된 기록 다시 확인' }));
    expect(await screen.findByText('게시된 정상 분류·이동 완료 기록이 아직 없습니다.')).toBeInTheDocument();
    expect(screen.queryByText('정상 트레이 도착 측정됨')).not.toBeInTheDocument();
  });

  it('hides the previous publication when the operator changes the published presentation', async () => {
    const data = recordedCases();
    vi.spyOn(api, 'getDemoCases').mockResolvedValue(data);
    vi.spyOn(api, 'getDemoCaseEvidence').mockImplementation(async (_, item) => ({
      blob: fixturePng(), frameId: item.observation_id, capturedAt: item.captured_at,
    }));
    const { rerender } = render(<RecordedCases presentationId={data.presentation_id} />);
    await userEvent.click(screen.getByRole('button', { name: '실제 실행 기록 3가지 비교' }));
    expect(await screen.findAllByRole('img')).toHaveLength(3);
    rerender(<RecordedCases presentationId="new-publication" />);
    expect(screen.queryByRole('img')).not.toBeInTheDocument();
    expect(screen.getByText(/이전 기록은 숨겼습니다/)).toBeInTheDocument();
  });

  it('rejects inconsistent action, evidence URLs, duplicate cases and claimed duration', () => {
    const data = recordedCases();
    expect(api.demoCasesSchema.safeParse(data).success).toBe(true);
    const first = data.cases[0]!;
    const held = data.cases[2]!;
    for (const cases of [
      [first, first],
      [{ ...first, image_url: '/api/runs/private/observation' }],
      [{ ...held, motion_authorized: true }],
      [{ ...held, classification: 'accepted' }],
      [{ ...first, physical_duration_seconds: 31 }],
      [{ ...first, result: { ...first.result, physical_success: false } }],
    ]) expect(api.demoCasesSchema.safeParse({ ...data, cases }).success).toBe(false);
  });

});
