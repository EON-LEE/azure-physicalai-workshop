import { useCallback } from 'react';
import { ApiError } from '../api/errors';
import { usePolling } from '../hooks/usePolling';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import { formatDate } from '../ui/format';
import type { ArtifactOperation, Dataset, LearningApi, Resource } from './contracts';

const labels = {
  queued: '검증 대기', running: '검증·게시 처리 중', ready: '검증된 manifest 게시 완료',
  failed: '검증 실패', timed_out: '검증 작업 시간 초과', uncertain: '작업 결과 미확인',
};
const keepPolling = (value: Resource<ArtifactOperation>) => ['queued', 'running'].includes(value.item.status);

export function ArtifactOperationPanel({ api, initial, onDataset }: {
  api: LearningApi; initial: Resource<ArtifactOperation>; onDataset(value: Resource<Dataset>): void;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    const value = await api.artifactOperation(initial.item.id, signal);
    if (value.item.status === 'ready' && value.item.operation === 'dataset' && value.item.result) {
      const dataset = await api.dataset(value.item.target_id, signal);
      if (dataset.item.artifact_id !== value.item.result.artifact_id ||
        dataset.item.manifest_sha256 !== value.item.result.manifest_sha256) {
        throw new ApiError('artifact_result_mismatch', '검증된 작업 결과와 데이터 manifest가 다릅니다.');
      }
      onDataset(dataset);
    }
    return value;
  }, [api, initial.item.id, onDataset]);
  const state = usePolling(load, { initialData: initial, intervalMs: 3000, continuePolling: keepPolling });
  const operation = (state.data ?? initial).item;
  return <section className="panel">
    <div className="panel-heading"><h3>대용량 데이터 검증 작업</h3><Badge>{labels[operation.status]}</Badge></div>
    <div className="learning-panel-body">
      <p>HTTP 접수는 데이터 준비 완료가 아닙니다. 별도 상주 worker가 원래 예산 안에서 검증하고 마지막 manifest를 게시합니다.</p>
      <dl className="learning-metadata">
        <FieldValue label="원래 작업 ID"><code>{operation.id}</code></FieldValue>
        <FieldValue label="원래 기한 · 대기 시간 포함">{formatDate(operation.deadline)}</FieldValue>
        <FieldValue label="검증 단계">{operation.phase}</FieldValue>
      </dl>
      {operation.message && <p role="status">{operation.message}</p>}
      {operation.error_code && <p role="alert">{operation.error_code} · 부분 결과를 준비된 데이터로 사용하지 않습니다.</p>}
      <p>작업을 자동으로 다시 제출하지 않습니다. 미확인 작업은 운영자가 원래 manifest와 worker 상태를 확인해야 합니다.</p>
      <ErrorNotice error={state.error} title="검증 작업 상태 조회 실패" retry={state.refresh} />
      <button type="button" className="button secondary" onClick={state.refresh}>검증 상태 다시 확인</button>
    </div>
  </section>;
}
