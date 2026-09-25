import { render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../src/api/client';
import { PolicyComparison } from '../src/learning/PolicyComparison';
import { simulationReportSchema } from '../src/learning/simulationReports';
import { publicLearningSchema } from '../src/public/learning-contract';
import { learningApi, learningFixture } from './fixtures/learning';
import { simulationReportFixture } from './fixtures/simulation-report';

describe('verified non-realtime report summaries', () => {
  it('keeps wall and simulation durations, all failures and native evidence hashes separate', () => {
    const report = simulationReportSchema.parse(simulationReportFixture());
    const fixture = learningFixture();
    const api = learningApi();
    render(<PolicyComparison api={api} evaluation={{
      ...fixture.evaluation, item: { ...fixture.evaluation.item, report },
    }} />);
    expect(screen.getByText('NON_REALTIME_SIMULATION')).toBeInTheDocument();
    expect(screen.getByText('56')).toBeInTheDocument();
    expect(screen.getByText('8')).toBeInTheDocument();
    const table = screen.getByRole('table', { name: '모든 40회 물리 시도 요약 · 원본 시계열 대체 아님' });
    expect(within(table).getAllByRole('row')).toHaveLength(41);
    expect(within(table).getAllByText('실패')).toHaveLength(5);
    expect(within(table).getAllByText('260')).toHaveLength(40);
    expect(screen.queryByRole('button', { name: '검토한 정책 게시' })).not.toBeInTheDocument();
    expect(api.release).not.toHaveBeenCalled();
  });

  it('downloads only an authenticated verified job report and revokes its temporary URL', async () => {
    const api = learningApi();
    api.reportDocument.mockResolvedValue(new Blob(['test-only native report'], { type: 'application/json' }));
    const fixture = learningFixture();
    const rendered = render(<PolicyComparison api={api} evaluation={{
      ...fixture.evaluation, item: { ...fixture.evaluation.item, report: simulationReportFixture() },
    }} />);
    await userEvent.click(screen.getByRole('button', { name: '전체 원본 보고서 불러오기' }));
    const link = await screen.findByRole('link', { name: '검증된 전체 JSON 저장' });
    const url = link.getAttribute('href');
    expect(api.reportDocument).toHaveBeenCalledWith(fixture.evaluation.item.id, expect.any(AbortSignal));
    rendered.unmount();
    expect(URL.revokeObjectURL).toHaveBeenCalledWith(url);
  });

  it('does not allow real-time relabeling or missing controller summaries', () => {
    const report = simulationReportFixture();
    expect(() => simulationReportSchema.parse({ ...report, real_time_admission: true })).toThrow();
    expect(() => simulationReportSchema.parse({ ...report, counts: { after: report.counts.after } })).toThrow();
  });

  it('keeps the public curated summary non-realtime without exposing a private artifact selector', () => {
    const report = simulationReportFixture();
    const { artifact_id: _privateArtifact, ...comparison } = report;
    const parsed = publicLearningSchema.parse({
      api_version: 'public-learning-v1', status: 'published',
      publication: {
        title: 'TEST ONLY', task: 'TEST ONLY recorded simulation', policy_type: 'smolvla',
        recorded_at: '2026-09-25T00:00:00Z', evaluation_status: 'failed',
        data_provenance: { human_teleop: 0, reference_controller: 20, learned: 0 },
        training: {
          optimizer_steps: 1, model_sha256: 'a'.repeat(64), parent_model_sha256: 'b'.repeat(64),
          dataset_sha256: 'c'.repeat(64), created_at: '2026-09-25T00:00:00Z', loss: null,
        },
        comparison, execution: 'recorded_evaluation_not_live',
      },
    });
    expect(parsed.publication?.comparison).toHaveProperty('real_time_admission', false);
    expect(parsed.publication?.comparison).not.toHaveProperty('artifact_id');
  });

  it('uses only the exact job report path and bearer transport', async () => {
    const transport = vi.fn().mockResolvedValue(new Response('test-only-json-bytes', {
      headers: { 'Content-Type': 'application/json', 'X-Report-SHA256': 'a'.repeat(64) },
    }));
    const api = new ApiClient(async () => 'test-only-access-token', transport);
    const blob = await api.learning.reportDocument('test-only-job');
    expect(blob.size).toBe(20);
    await waitFor(() => expect(transport).toHaveBeenCalledWith('/api/learning/jobs/test-only-job/report', expect.objectContaining({
      method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error',
      headers: expect.objectContaining({ Authorization: 'Bearer test-only-access-token' }),
    })));
  });
});
