import { useCallback, useRef, useState } from 'react';
import { RefreshCw, Square } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import { formatDate } from '../ui/format';
import { terminalJob, type Job, type LearningApi, type Resource } from './contracts';

const labels: Record<Job['status'], string> = {
  awaiting_import: '완전한 평가 자료 가져오기 대기',
  submitting: 'Azure 제출 확인 중', submission_unknown: '제출 결과 미확인', submitted: 'Azure 작업 접수',
  running: '작업 실행 중', cancelling: '취소 확인 중', succeeded: '검증된 산출물 수신',
  failed: '작업 실패', cancelled: '취소 확인됨', timed_out: '시간 초과', blocked: '진행 차단',
};
const cancellationLabels: Record<NonNullable<Job['cancellation']>['state'], string> = {
  claimed: '취소 전송 여부 미확인', acknowledged: '취소 요청 접수 · 종료 대기',
  uncertain: '취소 결과 미확인', forbidden: '취소 권한 거부',
};
const continuePolling = (resource: Resource<Job>) => !terminalJob(resource.item);
export function LearningJobPanel({ api, initial, onUpdate }: {
  api: LearningApi; initial: Resource<Job>; onUpdate?(value: Resource<Job>): void;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    const result = await api.job(initial.item.id, signal);
    onUpdate?.(result);
    return result;
  }, [api, initial.item.id, onUpdate]);
  const state = usePolling(load, { intervalMs: 3000, initialData: initial, continuePolling });
  const [actionError, setActionError] = useState<unknown>(null);
  const [cancelling, setCancelling] = useState(false);
  const cancelId = useRef(crypto.randomUUID());
  const current = state.data ?? initial;
  const job = current.item;
  const managed = job.kind === 'evaluation' && job.provider === 'managed_batch';
  const cancel = async () => {
    setCancelling(true);
    try {
      const result = await api.cancelJob(job.id, cancelId.current, current.etag);
      state.setData(result);
      onUpdate?.(result);
      state.refresh();
    } catch (error) { setActionError(error); }
    finally { setCancelling(false); }
  };
  return <section className="panel learning-job" aria-labelledby={`job-${job.id}`}>
    <div className="panel-heading"><h3 id={`job-${job.id}`}>{job.kind === 'training' ? `실제 ${job.policy_type} 학습 작업` : job.comparison_kind === 'reference_bootstrap' ? '최초 정책 품질·안전 평가' : 'P0/P1 paired 평가 작업'}</h3><Badge tone={job.status === 'failed' || job.status === 'blocked' ? 'red' : 'blue'}>{labels[job.status]}</Badge></div>
    <div className="learning-panel-body">
      {managed && <div className="inline-note" role="status">
        <strong>managed_batch · 검증된 자료 가져오기</strong>
        <p>운영자가 미리 고정한 40개 물리 시도의 원본을 검증합니다. Azure ML 작업 접수·현재 LIVE 동작·품질 통과를 대신하는 상태가 아닙니다.</p>
      </div>}
      {job.execution_timing === 'paused_simulation' && <div className="inline-note warning" role="status">
        <strong>NON_REALTIME_SIMULATION · 실시간 제어 승인 아님</strong>
        <p>실제 벽시계 작업 기한과 모델·데이터의 시뮬레이션 시간은 다릅니다. 이 기록은 실시간 100ms/80ms 게이트 통과를 의미하지 않습니다.</p>
        <dl className="learning-metadata">
          <FieldValue label="새 control profile SHA"><code>{job.control_profile_sha256}</code></FieldValue>
          <FieldValue label="고정된 기준 / scene conditions"><code>{job.criteria_sha256}</code><code>{job.frozen_plan_sha256}</code></FieldValue>
        </dl>
      </div>}
      <p>제출 접수는 학습 완료가 아닙니다. 실제 optimizer·새 checkpoint·물리 평가 근거를 따로 확인합니다.</p>
      <dl className="learning-metadata">
        {!managed && <FieldValue label="Azure ML job ID"><code>{job.azure_job_id ?? '아직 실제 Azure 작업 ID를 확인하지 못했습니다'}</code></FieldValue>}
        <FieldValue label="원래 요청 / 작업 ID"><code>{job.id}</code></FieldValue>
        {!managed && <FieldValue label="서버 보고 optimizer steps">{job.metrics.optimizer_steps === null ? 'optimizer step 미수신' : new Intl.NumberFormat('ko-KR').format(job.metrics.optimizer_steps)}</FieldValue>}
        {!managed && <FieldValue label="서버 보고 loss">{job.metrics.loss === null ? 'loss 미수신' : job.metrics.loss}</FieldValue>}
        <FieldValue label="원래 작업 기한">{formatDate(job.deadline)}</FieldValue>
        {!managed && <FieldValue label="Azure 실제 상태"><code>{job.azure_status ?? job.backend_status ?? '아직 실제 상태를 확인하지 못했습니다'}</code></FieldValue>}
        {managed && job.import_operation_id && <FieldValue label="원본 검증 작업 ID"><code>{job.import_operation_id}</code></FieldValue>}
        {job.job_deadline_utc && <FieldValue label="원래 승인에 고정된 절대 기한"><time dateTime={job.job_deadline_utc}>{formatDate(job.job_deadline_utc)}</time></FieldValue>}
      </dl>
      {job.cancellation && !terminalJob(job) && <div className="inline-note warning" role={job.cancellation.state === 'forbidden' || job.cancellation.state === 'uncertain' ? 'alert' : 'status'}>
        <strong>{job.cancellation.reason === 'deadline' ? '절대 기한에 따른 취소 요청' : '운영자의 취소 요청'}</strong>
        <p>{cancellationLabels[job.cancellation.state]} · Azure의 종료는 아직 확인되지 않았습니다. 취소 요청이나 경과 시간만으로 완료 처리하지 않습니다.</p>
        {job.cancellation.error_code && <code>{job.cancellation.error_code}</code>}
        <p>서버는 원래 작업을 조회하며 유료 작업이나 불확실한 취소를 다시 제출하지 않습니다.</p>
      </div>}
      {job.message && <p role="status">{job.message}</p>}
      {job.kind === 'training' && job.status === 'succeeded' && <div className="inline-note"><strong>학습 산출물 검증됨 · 학습 효과는 별도 평가</strong><code>candidate_id: {job.candidate_id}</code></div>}
      {(job.status === 'submission_unknown' || job.status === 'submitting') && <p className="form-hint">새 유료 작업을 만들지 마세요. 원래 작업 ID로 조회하여 제출 여부를 확인합니다.</p>}
      <ErrorNotice error={state.error} title="Azure 작업 상태 갱신 실패" retry={state.refresh} />
      <ErrorNotice error={actionError} title="취소 결과 미확인 · 완료로 처리하지 않음" />
      <div className="button-row"><button type="button" className="button secondary" onClick={state.refresh}><RefreshCw size={15} aria-hidden="true" />작업 상태 확인</button>
        {!managed && !terminalJob(job) && <button type="button" className="button danger-quiet" disabled={cancelling || job.status === 'cancelling' || Boolean(job.cancellation)} onClick={() => void cancel()}><Square size={14} aria-hidden="true" />{cancelling ? '취소 요청 중…' : '실제 작업 취소 요청'}</button>}</div>
    </div>
  </section>;
}
