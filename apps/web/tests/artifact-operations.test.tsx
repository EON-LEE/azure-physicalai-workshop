import { render, screen, waitFor } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import userEvent from '@testing-library/user-event';
import { ApiClient } from '../src/api/client';
import { ArtifactOperationPanel } from '../src/learning/ArtifactOperationPanel';
import { TeachingStudio } from '../src/learning/TeachingStudio';
import type { ArtifactOperation, Resource } from '../src/learning/contracts';
import { learningApi, learningFixture } from './fixtures/learning';

function operation(): Resource<ArtifactOperation> {
  const data = learningFixture();
  return { etag: 'test-only-operation', item: {
    kind: 'artifact_operation', id: data.dataset.item.id, actor_id: data.dataset.item.actor_id,
    project_id: data.project.item.id, target_id: data.dataset.item.id,
    created_at: data.dataset.item.created_at, updated_at: data.dataset.item.updated_at,
    operation: 'dataset', work_sha256: 'a'.repeat(64), deadline: new Date(Date.now() + 600000).toISOString(),
    max_bytes: 20 * 1024 ** 3, max_files: 100000, status: 'queued', phase: 'queued',
    result: null, error_code: null, message: 'TEST ONLY validation pending, no prepared dataset.',
  } };
}

describe('asynchronous data verification', () => {
  it('still opens an existing evaluation without changing artifact state', async () => {
    const warnings = vi.spyOn(console, 'error').mockImplementation(() => {});
    const fixture = learningFixture();
    const api = learningApi();
    api.records.mockImplementation(async (_id, kind) => ({
      items: kind === 'evaluation' ? [fixture.evaluation] : [],
    }));
    window.history.replaceState(null, '', `/operator?view=learning&learning_project=${fixture.project.item.id}`);
    render(<TeachingStudio api={api} environments={[]} />);
    await userEvent.click(await screen.findByRole('button', { name: '기록 보기' }));
    expect(await screen.findByText('개선 미확인')).toBeInTheDocument();
    expect(warnings).not.toHaveBeenCalled();
    warnings.mockRestore();
  });
  it('decodes HTTP202 as an operation, never as a prepared dataset', async () => {
    const pending = operation();
    const transport = vi.fn().mockResolvedValue(new Response(JSON.stringify(pending), {
      status: 202, headers: { 'Content-Type': 'application/json' },
    }));
    const api = new ApiClient(async () => 'test-only-token', transport);
    const value = await api.learning.seal(pending.item.project_id, {
      request_id: pending.item.id, teaching_session_ids: ['10000000-1111-4111-8111-111111111111'],
    }, 'test-only-etag');
    expect(value.item.kind).toBe('artifact_operation');
    expect(value.item.status).toBe('queued');
    expect(value.item).not.toHaveProperty('manifest_sha256');
  });

  it('waits for a verified result and matching persisted dataset without re-submitting work', async () => {
    const pending = operation();
    const fixture = learningFixture();
    const api = learningApi();
    const done = vi.fn();
    api.artifactOperation.mockResolvedValue({ ...pending, item: {
      ...pending.item, status: 'ready', phase: 'manifest_committed',
      result: { artifact_id: fixture.dataset.item.artifact_id, manifest_sha256: fixture.dataset.item.manifest_sha256, capture: null },
    } });
    render(<ArtifactOperationPanel api={api} initial={pending} onDataset={done} />);
    await waitFor(() => expect(done).toHaveBeenCalledWith(fixture.dataset));
    expect(api.seal).not.toHaveBeenCalled();
    expect(api.train).not.toHaveBeenCalled();
  });

  it('shows uncertain processing rather than ready or an automatic retry', async () => {
    const initial = operation();
    const uncertain = { ...initial, item: { ...initial.item, status: 'uncertain' as const, phase: 'stopped' as const, error_code: 'artifact_worker_lost' } };
    const api = learningApi();
    api.artifactOperation.mockResolvedValue(uncertain);
    const done = vi.fn();
    render(<ArtifactOperationPanel api={api} initial={uncertain} onDataset={done} />);
    expect(await screen.findByText('작업 결과 미확인')).toBeInTheDocument();
    expect(done).not.toHaveBeenCalled();
    expect(api.dataset).not.toHaveBeenCalled();
    expect(api.seal).not.toHaveBeenCalled();
  });
});
