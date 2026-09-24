import { fireEvent, render, screen, waitFor, within } from '@testing-library/react';
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

  it('preserves actual provider state and durable cancellation metadata in a job response', async () => {
    const fixture = learningFixture();
    const pending = { ...fixture.training, item: {
      ...fixture.training.item, status: 'cancelling' as const,
      azure_status: 'Queued', backend_status: 'submitted' as const,
      job_deadline_utc: fixture.training.item.deadline,
      cancellation: {
        request_id: fixture.training.item.id, reason: 'deadline' as const,
        requested_at: fixture.training.item.created_at, state: 'uncertain' as const,
        error_code: 'job_operation_unconfirmed',
      },
    } };
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify(pending), { headers: { 'Content-Type': 'application/json' } }));
    const client = new ApiClient(async () => 'test-only-access-token', transport);
    const response = await client.learning.job(pending.item.id);
    expect(response.item).toMatchObject({
      azure_status: 'Queued', backend_status: 'submitted',
      job_deadline_utc: pending.item.deadline, cancellation: pending.item.cancellation,
    });
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

  it.each(['uncertain', 'forbidden'] as const)('shows %s deadline cancellation separately from an actually queued Azure job', async (state) => {
    const fixture = learningFixture();
    const api = learningApi();
    const pending = { ...fixture.training, item: {
      ...fixture.training.item, status: 'cancelling' as const, azure_status: 'Queued',
      backend_status: 'submitted' as const, job_deadline_utc: fixture.training.item.deadline,
      cancellation: {
        request_id: fixture.training.item.id, reason: 'deadline' as const,
        requested_at: fixture.training.item.created_at, state,
        error_code: 'test-only-cancel-unconfirmed',
      },
    } };
    api.job.mockResolvedValue(pending);
    render(<LearningJobPanel api={api} initial={pending} />);
    expect(await screen.findByText('Queued')).toBeInTheDocument();
    expect(screen.getByText('절대 기한에 따른 취소 요청')).toBeInTheDocument();
    expect(screen.getByText(/Azure의 종료는 아직 확인되지 않았습니다/)).toBeInTheDocument();
    expect(screen.queryByText('취소 확인됨')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: '실제 작업 취소 요청' })).toBeDisabled();
    expect(api.cancelJob).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
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

  it('selects an immutable approved validation case without sending a seed or split override', async () => {
    const fixture = learningFixture();
    const cases = [
      { case_id: 'train-10001', environment_id: 'train-10001', revision: 'a'.repeat(64), seed: 10001, split: 'train' },
      { case_id: 'validation-20001', environment_id: 'validation-20001', revision: 'b'.repeat(64), seed: 20001, split: 'validation' },
    ] as const;
    const project = { ...fixture.project, item: { ...fixture.project.item, teaching_cases: [...cases] } };
    const api = learningApi();
    api.projects.mockResolvedValue({ items: [project] });
    api.teach.mockRejectedValue(new Error('Test-only runtime not active; no fallback to anchor.'));
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${project.item.id}`);
    render(<TeachingStudio api={api} environments={[]} />);
    const select = await screen.findByLabelText('승인된 시연 배치', { exact: true });
    expect(within(select).getAllByRole('option')).toHaveLength(3);
    expect(within(select).queryByText(/test-held-out/)).not.toBeInTheDocument();
    await userEvent.selectOptions(select, 'validation-20001');
    await userEvent.click(screen.getByRole('checkbox', { name: '저속 시연 조작과 서버의 제한된 이동 권한을 승인합니다' }));
    await userEvent.click(screen.getByRole('button', { name: '새 직접 시연 세션 시작' }));
    await waitFor(() => expect(api.teach).toHaveBeenCalledTimes(1));
    expect(api.teach.mock.calls[0]?.[1]).toEqual({
      request_id: expect.any(String), source: 'human_teleop', motion_approved: true,
      case_id: 'validation-20001',
    });
    expect(api.teach.mock.calls[0]?.[2]).toBe(project.etag);
    expect(await screen.findByRole('alert')).toHaveTextContent('no fallback to anchor');
  });

  it('does not silently teach a legacy project anchor without explicit case authority', async () => {
    const fixture = learningFixture();
    const project = { ...fixture.project, item: { ...fixture.project.item, teaching_cases: [] } };
    const api = learningApi();
    api.projects.mockResolvedValue({ items: [project] });
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${project.item.id}`);
    render(<TeachingStudio api={api} environments={[]} />);
    await screen.findByRole('heading', { name: project.item.display_name });
    await userEvent.click(screen.getByRole('checkbox', { name: '저속 시연 조작과 서버의 제한된 이동 권한을 승인합니다' }));
    expect(screen.getByRole('button', { name: '새 직접 시연 세션 시작' })).toBeDisabled();
    expect(screen.getByText(/승인된 시연 배치가 없습니다/)).toBeInTheDocument();
    expect(api.teach).not.toHaveBeenCalled();
  });
});
