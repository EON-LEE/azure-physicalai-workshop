import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../src/api/client';
import { ConsoleApp } from '../src/ConsoleApp';
import { RunDetail } from '../src/views/RunDetail';
import { account, environment, runtime } from './fixtures/data';
import { skillInstruction, skillPlanId, skillReleaseId, skillRun } from './fixtures/released-skill';
import { makeApi } from './helpers';

describe('released task planning is not CV inspection', () => {
  it('displays the exact reviewed task and goal without a classification or Foundry response ID', async () => {
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(skillRun) });
    render(<RunDetail api={api} initialRun={skillRun} runtime={runtime} runtimeFresh environment={environment} />);
    expect(await screen.findByRole('heading', { name: '게시된 작업 계획과 실행' })).toBeInTheDocument();
    expect(screen.getByText('CV 검사 수행 안 함')).toBeInTheDocument();
    expect(screen.getAllByText(skillInstruction).length).toBeGreaterThan(0);
    expect(screen.queryByText('분류: 정상 후보')).not.toBeInTheDocument();
    expect(screen.queryByText('분류: 불량 후보')).not.toBeInTheDocument();
    expect(screen.queryByText('model_response_id')).not.toBeInTheDocument();
    expect(screen.getByRole('checkbox')).not.toBeChecked();
    expect(api.approveRun).not.toHaveBeenCalled();
  });

  it('requires human confirmation and sends only the skill approval ID', async () => {
    const api = makeApi({
      getRun: vi.fn().mockResolvedValue(skillRun),
      approveRun: vi.fn().mockResolvedValue({ ...skillRun, status: 'running' }),
    });
    render(<RunDetail api={api} initialRun={skillRun} runtime={runtime} runtimeFresh environment={environment} />);
    const confirm = await screen.findByRole('checkbox');
    expect(screen.getByRole('button', { name: '계획 승인 및 실행' })).toBeDisabled();
    await userEvent.click(confirm);
    await userEvent.click(screen.getByRole('button', { name: '계획 승인 및 실행' }));
    expect(api.approveRun).toHaveBeenCalledWith(skillRun.id, { skill_plan_id: skillPlanId }, expect.any(AbortSignal));
  });

  it('does not send an old freeform inspection instruction with a selected skill', async () => {
    const api = makeApi({
      getRuntime: vi.fn().mockResolvedValue({ ...runtime, agent: { ...runtime.agent, configured: false } }),
      createRun: vi.fn().mockResolvedValue(skillRun), getRun: vi.fn().mockResolvedValue(skillRun),
    });
    render(<ConsoleApp api={api} account={account} />);
    fireEvent.change(screen.getByRole('textbox', { name: '검사·분류 작업을 설명하세요' }), { target: { value: 'Old freeform classification request must not override this skill.' } });
    await userEvent.click(screen.getByText('검토된 학습 정책 선택 (선택 사항)'));
    fireEvent.change(screen.getByLabelText('정책 release ID · 비우면 기존 reference 제어'), { target: { value: skillReleaseId } });
    const submit = await screen.findByRole('button', { name: '게시된 작업 계획 확인' });
    await waitFor(() => expect(submit).toBeEnabled());
    await userEvent.click(submit);
    const input = api.createRun.mock.calls[0]?.[0];
    expect(input).toEqual({
      request_id: expect.any(String), environment_id: environment.environment_id,
      revision: environment.revision, execution_mode: 'released_skill', policy_release_id: skillReleaseId,
    });
    expect(input).not.toHaveProperty('instruction');
    expect(api.approveRun).not.toHaveBeenCalled();
  });

  it('serializes the new approval body without inventing a model response', async () => {
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify(skillRun), { headers: { 'Content-Type': 'application/json' } }));
    const client = new ApiClient(async () => 'test-only-access-token', transport);
    await client.approveRun(skillRun.id, { skill_plan_id: skillPlanId });
    expect(JSON.parse(String(transport.mock.calls[0]?.[1].body))).toEqual({ skill_plan_id: skillPlanId });
  });

  it.each([
    { target_station_id: 'accepted' }, { model_sha256: 'e'.repeat(64) },
    { task_id: 'another-task' }, { instruction: 'Caller-overridden task instruction.' },
  ])('blocks approval when the plan differs from the immutable policy: %j', async (override) => {
    const corrupted = { ...skillRun, plan: { ...skillRun.plan, ...override } };
    const api = makeApi({ getRun: vi.fn().mockResolvedValue(corrupted) });
    render(<RunDetail api={api} initialRun={corrupted} runtime={runtime} runtimeFresh environment={environment} />);
    await waitFor(() => expect(screen.getByRole('checkbox')).toBeDisabled());
    expect(api.approveRun).not.toHaveBeenCalled();
  });

  it.each([
    { execution_mode: 'inspection' },
    { policy: null },
    { plan: { ...skillRun.plan, classification: 'rejected' } },
    { plan: { ...skillRun.plan, model_response_id: 'invented-foundry-response' } },
  ])('rejects wire plans that mix skill and CV provenance: %j', async (override) => {
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify({ ...skillRun, ...override }), { headers: { 'Content-Type': 'application/json' } }));
    const client = new ApiClient(async () => 'test-only-access-token', transport);
    await expect(client.getRun(skillRun.id)).rejects.toMatchObject({ code: 'invalid_response' });
  });
});
