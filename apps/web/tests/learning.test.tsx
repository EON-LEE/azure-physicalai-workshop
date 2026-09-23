import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../src/api/client';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import { LearningJobPanel } from '../src/learning/LearningJobPanel';
import { PolicyComparison } from '../src/learning/PolicyComparison';
import { TeachingControls } from '../src/learning/TeachingControls';
import { learningFixture, learningApi } from './fixtures/learning';
import { deferred } from './helpers';
import { LearningPublication } from '../src/public/LearningPublication';

afterEach(() => vi.unstubAllGlobals());

describe('learning API is private and typed', () => {
  it('uses bearer, unchanged ETag and original request ID for an explicit paid submission', async () => {
    const fixture = learningFixture();
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify(fixture.training), { headers: { 'Content-Type': 'application/json' } }));
    const client = new ApiClient(async () => 'test-only-access-token', transport);
    const body = {
      request_id: crypto.randomUUID(), dataset_id: fixture.dataset.item.id,
      parent_release_id: fixture.project.item.baseline_release_id, policy_type: fixture.project.item.policy_type,
      optimizer_steps: 100, paid_approved: true as const, maximum_cost_usd: '10.00',
    };
    await client.learning.train(fixture.project.item.id, body, fixture.project.etag);
    expect(transport).toHaveBeenCalledWith(`/api/learning/projects/${fixture.project.item.id}/train`, expect.objectContaining({
      method: 'POST', credentials: 'omit', redirect: 'error', cache: 'no-store',
      headers: expect.objectContaining({ Authorization: 'Bearer test-only-access-token', 'If-Match': fixture.project.etag }),
      body: JSON.stringify(body),
    }));
  });
});

describe('truthful Korean learning experience', () => {
  it('shows disabled capability without invented projects, jobs or learning success', async () => {
    const api = learningApi();
    api.capabilities.mockResolvedValue({ enabled: false, status: 'disabled', message: '검증된 통합이 필요합니다.', policy_types: ['gr00t_n1_5'], control_profiles: ['franka-position-hold-10hz-v1'], training_verified: false, coach_configured: false, bootstrap_allowed: false });
    render(<TeachingStudio api={api} environments={[]} />);
    expect(await screen.findByText('학습 기능이 아직 활성화되지 않았습니다')).toBeInTheDocument();
    expect(api.projects).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
    expect(screen.queryByText('학습 완료')).not.toBeInTheDocument();
  });

  it('shows actual job ID and unknown metrics, never treating submission ACK as a trained model', async () => {
    const fixture = learningFixture();
    const api = learningApi();
    render(<LearningJobPanel api={api} initial={fixture.training} />);
    expect(await screen.findByText(fixture.training.item.azure_job_id!)).toBeInTheDocument();
    expect(screen.getByText('optimizer step 미수신')).toBeInTheDocument();
    expect(screen.getByText('loss 미수신')).toBeInTheDocument();
    expect(screen.queryByText('학습 완료')).not.toBeInTheDocument();
    expect(api.release).not.toHaveBeenCalled();
  });

  it('retains failed paired trials and labels no improvement, even if the optimizer job finished', () => {
    const fixture = learningFixture();
    const api = learningApi();
    render(<PolicyComparison api={api} evaluation={fixture.evaluation} />);
    expect(screen.getByText('개선 미확인')).toBeInTheDocument();
    expect(screen.getAllByText('failed').length).toBeGreaterThan(0);
    expect(screen.getByText(/저장된 평가 기록/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '검토한 정책 게시' })).toBeDisabled();
    expect(api.release).not.toHaveBeenCalled();
  });

  it('does not jog after the operator releases while a server control grant is arriving', async () => {
    const fixture = learningFixture();
    const api = learningApi();
    const grant = deferred<typeof fixture.grant>();
    api.arm.mockReturnValue(grant.promise);
    render(<TeachingControls api={api} session={fixture.teaching} onChange={vi.fn()} />);
    const button = screen.getByRole('button', { name: 'X 양의 방향 5mm' });
    fireEvent(button, new MouseEvent('pointerdown', { bubbles: true, button: 0 }));
    await waitFor(() => expect(api.arm).toHaveBeenCalledTimes(1));
    fireEvent(button, new MouseEvent('pointerup', { bubbles: true, button: 0 }));
    grant.resolve(fixture.grant);
    await waitFor(() => expect(api.jog).toHaveBeenCalled());
    for (const call of api.jog.mock.calls) {
      expect(call[1].deadman).toBe(false);
      expect(call[1].delta_xyz_m).toEqual([0, 0, 0]);
    }
  });

  it('requires deliberate motion approval and keyboard-held controls use grants, not browser timestamps', async () => {
    const fixture = learningFixture();
    const api = learningApi();
    render(<TeachingControls api={api} session={fixture.teaching} onChange={vi.fn()} />);
    const button = screen.getByRole('button', { name: 'X 양의 방향 5mm' });
    button.focus();
    await userEvent.keyboard('[Space>]');
    await waitFor(() => expect(api.arm).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(api.jog).toHaveBeenCalled());
    const moving = api.jog.mock.calls.find((call) => call[1].deadman);
    expect(moving?.[1].grant_id).toBe(fixture.grant.item.id);
    expect(moving?.[1]).not.toHaveProperty('expires_at');
    await userEvent.keyboard('[/Space]');
  });

  it('public learning disclosure reads only curated GET data and never calls auth or private jobs', async () => {
    const fetch = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      api_version: 'public-learning-v1', status: 'not_published', publication: null,
    }), { headers: { 'Content-Type': 'application/json' } }));
    vi.stubGlobal('fetch', fetch);
    render(<LearningPublication />);
    expect(fetch).not.toHaveBeenCalled();
    await userEvent.click(screen.getByText('작업 시연·학습 전후의 검증된 기록'));
    await waitFor(() => expect(fetch).toHaveBeenCalledTimes(1));
    expect(fetch).toHaveBeenCalledWith('/api/demo/learning', expect.objectContaining({
      method: 'GET', credentials: 'omit', cache: 'no-store', redirect: 'error',
      headers: { Accept: 'application/json' },
    }));
    expect(await screen.findByText(/아직 공개된 학습 비교 기록이 없습니다/)).toBeInTheDocument();
  });

  it('does not convert an unconfirmed Azure submission into a ready candidate', async () => {
    const fixture = learningFixture();
    const api = learningApi();
    const unknown = { ...fixture.training, item: { ...fixture.training.item, status: 'submission_unknown' as const, azure_job_id: null } };
    api.job.mockResolvedValue(unknown);
    render(<LearningJobPanel api={api} initial={unknown} />);
    expect(await screen.findByText('제출 결과 미확인')).toBeInTheDocument();
    expect(screen.getByText('아직 실제 Azure 작업 ID를 확인하지 못했습니다')).toBeInTheDocument();
    expect(api.train).not.toHaveBeenCalled();
    expect(api.release).not.toHaveBeenCalled();
  });

  it('restores only a project returned in the authenticated owner list from a deep link', async () => {
    const fixture = learningFixture();
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${fixture.project.item.id}`);
    const api = learningApi();
    render(<TeachingStudio api={api} environments={[]} />);
    expect(await screen.findByRole('heading', { name: fixture.project.item.display_name })).toBeInTheDocument();
    expect(screen.getByLabelText('학습 프로젝트', { exact: true })).toHaveValue(fixture.project.item.id);
  });
});
