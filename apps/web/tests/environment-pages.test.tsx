import { fireEvent, render, renderHook, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { ApiClient } from '../src/api/client';
import { ConsoleApp } from '../src/ConsoleApp';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import { useEnvironmentPages } from '../src/hooks/useEnvironmentPages';
import { account } from './fixtures/data';
import { environmentCursor, seventyEnvironments } from './fixtures/environment-pages';
import { learningApi } from './fixtures/learning';
import { deferred, makeApi, setVisibility } from './helpers';

beforeEach(() => window.history.replaceState(null, '', '/operator?view=learning'));

describe('bounded owner environment pages', () => {
  it('serializes only the explicit bounded page and opaque cursor, retaining real records', async () => {
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify({
      items: seventyEnvironments.slice(50), next_cursor: null,
    }), { headers: { 'Content-Type': 'application/json' } }));
    const api = new ApiClient(async () => 'test-only-access-token', transport);
    const result = await api.getEnvironments(undefined, environmentCursor);
    expect(transport.mock.calls[0]?.[0]).toBe(`/api/environments?page_size=50&cursor=${environmentCursor}`);
    expect(result.items).toEqual(seventyEnvironments.slice(50));
    expect(result.next_cursor).toBeNull();
  });

  it('loads the missing P0 page and retains reviewed selections while rendering at most twenty case rows', async () => {
    const learning = learningApi();
    const api = makeApi({
      learning,
      getEnvironments: vi.fn().mockImplementation(async (_signal, cursor) => cursor
        ? { items: seventyEnvironments.slice(50), next_cursor: null }
        : { items: seventyEnvironments.slice(0, 50), next_cursor: environmentCursor }),
    });
    render(<ConsoleApp api={api} account={account} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    await screen.findByText('불러온 저장 환경 50개');
    expect(document.querySelectorAll('input[name="teaching-case"]').length).toBeLessThanOrEqual(20);
    const filter = screen.getByRole('searchbox', { name: '시연 배치 검색' });
    fireEvent.change(filter, { target: { value: 'case-040' } });
    await userEvent.click(screen.getByRole('checkbox', { name: /case-040/ }));
    await userEvent.click(screen.getByRole('button', { name: '저장 환경 더 불러오기' }));
    await screen.findByText('불러온 저장 환경 70개');
    expect(screen.getByRole('checkbox', { name: /case-040/ })).toBeChecked();
    fireEvent.change(filter, { target: { value: 'case-000' } });
    await userEvent.click(screen.getByRole('checkbox', { name: /case-000/ }));
    await userEvent.selectOptions(screen.getByLabelText('저장된 LIVE 환경'), 'case-000');
    fireEvent.change(screen.getByLabelText('프로젝트 이름'), { target: { value: 'TEST-ONLY owner project' } });
    fireEvent.change(screen.getByLabelText('운영자가 검토·등록한 P0 release ID'), { target: { value: '90000000-1111-4111-8111-111111111111' } });
    fireEvent.change(screen.getByLabelText('작업별 최대 승인 금액 (USD)'), { target: { value: '10' } });
    fireEvent.change(screen.getByLabelText('학습에서 제외할 seed 20~100개 (쉼표 구분)'), { target: { value: Array.from({ length: 20 }, (_, index) => 30001 + index).join(',') } });
    await userEvent.click(screen.getByRole('button', { name: '불변 작업 정의 저장' }));
    await waitFor(() => expect(learning.createProject).toHaveBeenCalledTimes(1));
    const body = learning.createProject.mock.calls[0]![0];
    expect(body.teaching_cases).toEqual(expect.arrayContaining([
      { case_id: 'case-040', environment_id: 'case-040', revision: '29'.padStart(64, '0'), seed: 20001, split: 'validation' },
      { case_id: 'case-000', environment_id: 'case-000', revision: '1'.padStart(64, '0'), seed: 10001, split: 'train' },
    ]));
    expect(body.evaluation_plan.cases).toHaveLength(20);
    expect(body.evaluation_plan.cases.every((entry) => seventyEnvironments.some((record) => record.environment_id === entry.environment_id && record.revision === entry.revision))).toBe(true);
  });

  it('flags a changed selected descriptor instead of approving its new revision after refresh', async () => {
    const learning = learningApi();
    const rendered = render(<TeachingStudio api={learning} environments={seventyEnvironments} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    const filter = screen.getByRole('searchbox', { name: '시연 배치 검색' });
    fireEvent.change(filter, { target: { value: 'case-000' } });
    await userEvent.click(screen.getByRole('checkbox', { name: /case-000/ }));
    const changed = seventyEnvironments.map((record) => record.environment_id === 'case-000'
      ? { ...record, revision: 'e'.repeat(64), document: { ...record.document, scene: { template_id: 'inspection-cell-learning-v1', seed: 10002 } } }
      : record);
    rendered.rerender(<TeachingStudio api={learning} environments={changed} />);
    expect(await screen.findByText(/선택한 배치의 저장 버전이 바뀌었습니다/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '불변 작업 정의 저장' })).toBeDisabled();
    expect(learning.createProject).not.toHaveBeenCalled();
    await userEvent.click(screen.getByRole('checkbox', { name: /case-000/ }));
    await userEvent.click(screen.getByRole('checkbox', { name: /case-000/ }));
    expect(screen.queryByText(/선택한 배치의 저장 버전이 바뀌었습니다/)).not.toBeInTheDocument();
  });

  it('reports conflicting duplicate page revisions instead of silently replacing selected records', async () => {
    const changed = { ...seventyEnvironments[0]!, revision: 'f'.repeat(64) };
    const api = makeApi({
      learning: learningApi(),
      getEnvironments: vi.fn().mockImplementation(async (_signal, cursor) => cursor
        ? { items: [changed, ...seventyEnvironments.slice(50)], next_cursor: null }
        : { items: seventyEnvironments.slice(0, 50), next_cursor: environmentCursor }),
    });
    render(<ConsoleApp api={api} account={account} />);
    await userEvent.click(await screen.findByRole('button', { name: '저장 환경 더 불러오기' }));
    expect(await screen.findByText(/같은 환경의 다른 저장 버전이 도착했습니다/)).toBeInTheDocument();
    expect(screen.getByText('불러온 저장 환경 70개')).toBeInTheDocument();
  });

  it('keeps the reviewed anchor revision until the operator explicitly reselects it', async () => {
    const learning = learningApi();
    const rendered = render(<TeachingStudio api={learning} environments={seventyEnvironments} />);
    await userEvent.click(await screen.findByRole('button', { name: '새 학습 작업 정의' }));
    const changed = seventyEnvironments.map((record, index) => index === 0 ? { ...record, revision: 'd'.repeat(64) } : record);
    rendered.rerender(<TeachingStudio api={learning} environments={changed} />);
    expect(await screen.findByText(/선택한 기준 환경의 저장 버전이 바뀌었습니다/)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: '불변 작업 정의 저장' })).toBeDisabled();
    await userEvent.click(screen.getByRole('button', { name: '현재 환경 버전으로 다시 선택' }));
    expect(screen.queryByText(/선택한 기준 환경의 저장 버전이 바뀌었습니다/)).not.toBeInTheDocument();
    expect(learning.createProject).not.toHaveBeenCalled();
  });

  it('aborts an in-flight page read in a hidden tab without accepting its later result', async () => {
    const next = deferred<{ items: typeof seventyEnvironments; next_cursor: null }>();
    let signal: AbortSignal | undefined;
    const api = makeApi({
      learning: learningApi(),
      getEnvironments: vi.fn().mockImplementation(async (requestSignal, cursor) => {
        if (!cursor) return { items: seventyEnvironments.slice(0, 50), next_cursor: environmentCursor };
        signal = requestSignal;
        return next.promise;
      }),
    });
    render(<ConsoleApp api={api} account={account} />);
    await userEvent.click(await screen.findByRole('button', { name: '저장 환경 더 불러오기' }));
    fireEvent(document, new Event('visibilitychange'));
    setVisibility('hidden');
    await waitFor(() => expect(signal?.aborted).toBe(true));
    next.resolve({ items: seventyEnvironments.slice(50), next_cursor: null });
    expect(screen.queryByText('불러온 저장 환경 70개')).not.toBeInTheDocument();
    setVisibility('visible');
  });

  it('does not carry accumulated environment records into a different authenticated account', async () => {
    const api = makeApi({ getEnvironments: vi.fn().mockResolvedValue({
      items: seventyEnvironments.slice(0, 50), next_cursor: environmentCursor,
    }) });
    const { result, rerender } = renderHook(({ scope }) => useEnvironmentPages(api, scope, true), { initialProps: { scope: 'test-owner-one' } });
    await waitFor(() => expect(result.current.items).toHaveLength(50));
    api.getEnvironments.mockResolvedValue({ items: [seventyEnvironments[69]!], next_cursor: null });
    rerender({ scope: 'test-owner-two' });
    await waitFor(() => expect(result.current.items).toHaveLength(1));
    expect(result.current.items[0]?.environment_id).toBe('case-000');
    expect(result.current.nextCursor).toBeNull();
  });
});
