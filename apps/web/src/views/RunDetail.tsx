import { useCallback, useEffect, useRef, useState } from 'react';
import { Check, ChevronDown, CircleStop, FileImage, Fingerprint, Play, ShieldCheck } from 'lucide-react';
import { isTerminal, matchesRuntime, type ConsoleApi, type EnvironmentRecord, type RunRecord, type RuntimeInfo } from '../api/contracts';
import { ApiError, isAbort } from '../api/errors';
import { isPosition, stationPosition, workflowTarget } from '../environment/validation';
import { usePolling } from '../hooks/usePolling';
import { useProtectedImage } from '../hooks/useProtectedImage';
import { useRequestScope } from '../hooks/useRequestScope';
import { Badge, ErrorNotice, FieldValue, Loading, RunBadge, StepLabel } from '../ui/common';
import { formatDate } from '../ui/format';
import { formatPosition } from '../public/presentation';

const continueRunPolling = (run: RunRecord) => !isTerminal(run.status);

export function RunDetail({ api, initialRun, runtime, runtimeFresh, environment, onChange }: {
  api: ConsoleApi; initialRun: RunRecord; runtime: RuntimeInfo | null; runtimeFresh: boolean;
  environment: EnvironmentRecord | null; onChange?: (run: RunRecord) => void;
}) {
  const load = useCallback((signal: AbortSignal) => api.getRun(initialRun.id, signal), [api, initialRun.id]);
  const resource = usePolling(load, { intervalMs: 2000, initialData: initialRun, continuePolling: continueRunPolling });
  const run = resource.data ?? initialRun;
  const [confirmed, setConfirmed] = useState(false);
  const [approving, setApproving] = useState(false);
  const [cancelling, setCancelling] = useState(false);
  const [actionError, setActionError] = useState<unknown>(null);
  const [rejectedPlanId, setRejectedPlanId] = useState<string | null>(null);
  const actionSequence = useRef(0);
  const startRequest = useRequestScope();
  const targetPosition = environment?.revision === run.revision && run.plan
    ? stationPosition(environment, run.plan.target_station_id) : null;
  const finalPosition = run.execution?.final_position;

  useEffect(() => { onChange?.(run); }, [run, onChange]);
  useEffect(() => { setConfirmed(false); }, [run.plan?.model_response_id, run.status]);

  let approvalBlocked: string | null = null;
  if (!run.plan) approvalBlocked = '서버가 아직 승인할 계획을 반환하지 않았습니다.';
  else if (rejectedPlanId === run.plan.model_response_id) approvalBlocked = '서버가 이 계획의 승인을 거절했습니다. 기존 실행을 취소하고 새 관측으로 계획을 요청하세요.';
  else if (resource.error || resource.paused) approvalBlocked = '실행의 최신 상태를 다시 확인해야 합니다.';
  else if (!runtimeFresh || !matchesRuntime(runtime, run)) approvalBlocked = '이 실행의 환경과 버전이 준비된 런타임에서 확인되어야 합니다.';
  else if (run.plan.epoch !== runtime?.simulation.epoch) approvalBlocked = '관측 이후 씬이 바뀌었습니다. 기존 계획을 승인하지 말고 새 계획을 요청하세요.';
  else if (!environment || environment.revision !== run.revision) approvalBlocked = '저장 버전이 변경되었거나 확인되지 않았습니다. 새 계획을 요청하세요.';
  else if (workflowTarget(environment, run.plan.classification) !== run.plan.target_station_id) approvalBlocked = '계획의 대상 스테이션이 저장된 워크플로와 일치하지 않습니다.';

  const act = async (kind: 'approve' | 'cancel') => {
    if (kind === 'approve' && (approving || cancelling || !confirmed || approvalBlocked || run.status !== 'awaiting_approval' || !run.plan)) return;
    if (kind === 'cancel' && (cancelling || isTerminal(run.status) || run.status === 'cancelling')) return;
    const sequence = ++actionSequence.current;
    const request = startRequest();
    setActionError(null);
    if (kind === 'approve') setApproving(true);
    else setCancelling(true);
    try {
      const updated = kind === 'approve' && run.plan
        ? await api.approveRun(run.id, run.plan.model_response_id, request.signal)
        : await api.cancelRun(run.id, request.signal);
      if (!request.current() || sequence !== actionSequence.current) return;
      resource.setData(updated);
      resource.refresh();
      onChange?.(updated);
    } catch (error) {
      if (request.current() && sequence === actionSequence.current && !isAbort(error)) {
        if (kind === 'approve' && error instanceof ApiError && error.status === 409 && run.plan) setRejectedPlanId(run.plan.model_response_id);
        setActionError(error);
        resource.refresh();
      }
    } finally {
      if (request.current()) {
        if (kind === 'approve') setApproving(false);
        else setCancelling(false);
      }
      request.finish();
    }
  };

  return <section className="panel run-detail" aria-labelledby="run-detail-title">
    <div className="panel-heading">
      <div className="title-icon"><ShieldCheck size={19} aria-hidden="true" /><h2 id="run-detail-title">에이전트 계획과 실행</h2></div>
      <span role="status"><RunBadge status={run.status} /></span>
    </div>
    <div className="run-progress" aria-label="실행 진행 상태">
      <StepLabel number="01" title="관측·계획" done={run.plan !== null} active={run.status === 'planning'} />
      <StepLabel number="02" title="사람의 승인" done={run.execution !== null} active={run.status === 'awaiting_approval'} />
      <StepLabel number="03" title="실행 확인" done={run.status === 'succeeded'} active={run.status === 'running' || run.status === 'cancelling'} />
    </div>
    <div className="run-request"><span className="eyebrow">CUSTOMER INSTRUCTION</span><p>{run.instruction}</p>
      <small>{run.environment_id} · {formatDate(run.created_at)}</small>
    </div>
    {run.policy && <div className="plan-summary"><strong>검토된 학습 정책 · {run.policy.policy_type}</strong>
      <dl className="trace-list"><FieldValue label="policy release"><code>{run.policy.policy_release_id}</code></FieldValue>
        <FieldValue label="승인된 model SHA"><code>{run.policy.model_sha256}</code></FieldValue>
        <FieldValue label="고정된 운동 작업"><span>{run.policy.instruction}</span></FieldValue></dl>
      <p className="small-text muted">Foundry는 검사 계획을 제안하며 이 운동 정책을 학습시키거나 대신 제어하지 않습니다.</p>
    </div>}
    {run.execution?.policy_runtime && <dl className="execution-receipt">
      <FieldValue label="실제 적용 model SHA"><code>{run.execution.policy_runtime.applied_model_sha ?? '아직 실제 action 적용 없음'}</code></FieldValue>
      <FieldValue label="정책 예측 / 적용 action 수">{run.execution.policy_runtime.policy_predict_calls} / {run.execution.policy_runtime.applied_action_count}</FieldValue>
      <FieldValue label="Reference route 호출 수">{run.execution.policy_runtime.reference_route_calls}</FieldValue>
    </dl>}
    <ErrorNotice error={resource.error} title="실행 상태 갱신 실패 · 마지막 응답 표시 중" retry={resource.refresh} compact />
    {run.status === 'planning' && <div className="inline-note"><Loading>실제 관측을 바탕으로 Foundry 계획 생성 중…</Loading><p>계획 단계에서는 로봇을 움직이지 않습니다.</p></div>}
    {run.plan && <div className="plan-summary">
      <div className="plan-heading"><span className="eyebrow">PLAN SUMMARY</span><Badge tone={run.plan.classification === 'rejected' ? 'amber' : 'blue'}>분류: {run.plan.classification === 'rejected' ? '불량 후보' : '정상 후보'}</Badge></div>
      <p>{run.plan.summary}</p>
      <dl className="plan-targets">
        <FieldValue label="관측 부품"><code>{run.plan.object_id}</code></FieldValue>
        <FieldValue label="이동 대상"><code>{run.plan.target_station_id}</code></FieldValue>
      </dl>
      <dl className="plan-references" aria-label="계획 증거 ID">
        <FieldValue label="model_response_id"><code>{run.plan.model_response_id}</code></FieldValue>
        <FieldValue label="observation_id"><code>{run.plan.observation_id}</code></FieldValue>
      </dl>
      <span className="muted small-text">모델의 결과 요약입니다. 숨겨진 추론 과정은 표시하지 않습니다.</span>
    </div>}
    {run.status === 'awaiting_approval' && <div className="approval-box">
      <div className="title-icon"><ShieldCheck size={18} aria-hidden="true" /><strong>승인 전에는 이동하지 않습니다</strong></div>
      <p>부품, 분류 결과와 대상 스테이션을 확인하세요. 승인 시 서버가 씬 상태를 다시 검사한 뒤 물리 명령을 전송합니다.</p>
      {approvalBlocked && <p className="approval-blocked">{approvalBlocked}</p>}
      <label className="checkbox-label"><input type="checkbox" checked={confirmed} disabled={Boolean(approvalBlocked) || approving || cancelling}
        onChange={(event) => setConfirmed(event.target.checked)} />표시된 계획과 이동 대상을 확인했습니다</label>
      <button type="button" className="button full-width" disabled={!confirmed || Boolean(approvalBlocked) || approving || cancelling} onClick={() => void act('approve')}>
        <Play size={16} aria-hidden="true" />{approving ? '승인 요청 확인 중…' : '계획 승인 및 실행'}
      </button>
    </div>}
    {run.status === 'running' && <div className="inline-note"><strong>물리 실행 확인 중</strong><p>명령 접수는 완료가 아닙니다. 시뮬레이터가 반환하는 최종 상태를 기다립니다.</p></div>}
    {(run.status === 'cancelling' || cancelling) && <div className="inline-note warning" role="status"><strong>취소 요청 확인 중</strong><p>시뮬레이터의 확인 전까지 취소 완료로 표시하지 않습니다.</p></div>}
    {run.status === 'succeeded' && <div className="inline-note success" role="status"><strong><Check size={17} aria-hidden="true" />서버가 실행 성공을 확인했습니다</strong><p>최종 상태: succeeded · 아래 명령 ID 및 관측 증거로 결과를 검토하세요.</p></div>}
    {run.status === 'cancelled' && <div className="inline-note" role="status"><strong>서버가 취소를 확인했습니다</strong><p>실행 성공과는 별도의 결과입니다.</p></div>}
    {run.execution && <dl className="execution-receipt" aria-label="시뮬레이터 명령 응답">
      <FieldValue label="command_id"><code>{run.execution.command_id}</code></FieldValue>
      <FieldValue label="명령 상태 (서버 응답)"><code>{run.execution.status}</code></FieldValue>
    </dl>}
    {run.plan && <div className="inline-note" role="region" aria-label="고객 공정 실험 위치 비교">
      <strong>내 라인 실험 · 목표와 실제 도착 위치</strong>
      <dl className="plan-targets">
        <FieldValue label="이 실행 버전의 목표 좌표">{targetPosition ? formatPosition(targetPosition) : '실행 시점의 저장 버전·목표 좌표 확인 필요'}</FieldValue>
        <FieldValue label="시뮬레이터의 최종 측정 좌표">{finalPosition === undefined || finalPosition === null ? '최종 측정값 미수신'
          : isPosition(finalPosition) ? formatPosition(finalPosition) : '측정 좌표 형식 확인 필요'}</FieldValue>
      </dl>
      <p>환경의 위치를 바꿨다면 두 값을 비교하세요. 계획 좌표는 측정값이 아니며, 명령 접수는 실제 도착을 뜻하지 않습니다.</p>
    </div>}
    {['failed', 'timed_out'].includes(run.status) && !run.error && <div className="inline-note warning" role="alert"><strong>{run.status === 'timed_out' ? '실행 시간 초과' : '실행 실패'}</strong><p>서버가 상세 오류를 제공하지 않았습니다. 이벤트와 런타임 상태를 확인하세요.</p></div>}
    {run.error && <div className="error-notice" role="alert"><div className="error-content"><strong>{run.error.code}</strong><p>{run.error.message}</p><small>{run.error.retryable ? '원인을 해결한 후 새 계획을 요청할 수 있습니다.' : '이 실행을 자동으로 재시도하지 않습니다.'}</small></div></div>}
    <ErrorNotice error={actionError} title="작업 요청 결과 확인 필요" compact />
    {!isTerminal(run.status) && <div className="cancel-action">
      <button type="button" className="button danger-quiet" disabled={cancelling || run.status === 'cancelling'} onClick={() => void act('cancel')}><CircleStop size={16} aria-hidden="true" />{cancelling || run.status === 'cancelling' ? '취소 확인 대기' : '이 실행 취소'}</button>
      <small>취소 요청은 에이전트 응답을 기다리지 않습니다.</small>
    </div>}
    {run.plan && <Observation api={api} runId={run.id} />}
    <details className="trace-details">
      <summary><Fingerprint size={16} aria-hidden="true" />증거 및 실제 실행 ID<ChevronDown size={15} aria-hidden="true" /></summary>
      <dl className="trace-list">
        <FieldValue label="run_id"><code>{run.id}</code></FieldValue>
        <FieldValue label="environment revision"><code>{run.revision}</code></FieldValue>
        {run.plan && <>
          <FieldValue label="observation_id"><code>{run.plan.observation_id}</code></FieldValue>
          <FieldValue label="model_response_id"><code>{run.plan.model_response_id}</code></FieldValue>
          <FieldValue label="epoch"><code>{run.plan.epoch}</code></FieldValue>
          <FieldValue label="state_revision"><code>{run.plan.state_revision}</code></FieldValue>
        </>}
        {run.execution && <>
          <FieldValue label="command_id"><code>{run.execution.command_id}</code></FieldValue>
          <FieldValue label="command status"><code>{run.execution.status}</code></FieldValue>
          {run.execution.final_position !== undefined && <FieldValue label="final_position"><code>{JSON.stringify(run.execution.final_position)}</code></FieldValue>}
          {run.execution.completed_at && <FieldValue label="completed_at"><time dateTime={run.execution.completed_at}>{formatDate(run.execution.completed_at)}</time></FieldValue>}
        </>}
      </dl>
    </details>
    <details className="trace-details">
      <summary>서버 이벤트 <span className="count">{run.events.length}</span><ChevronDown size={15} aria-hidden="true" /></summary>
      {run.events.length === 0 ? <p className="muted">서버가 반환한 이벤트가 없습니다.</p> : <ol className="event-list">
        {run.events.map((event, index) => <li key={`${event.at}-${event.kind}-${index}`}><span className="event-dot" aria-hidden="true" /><div><strong>{event.kind}</strong><p>{event.message}</p><time dateTime={event.at}>{formatDate(event.at)}</time></div></li>)}
      </ol>}
    </details>
  </section>;
}

function Observation({ api, runId }: { api: ConsoleApi; runId: string }) {
  const [open, setOpen] = useState(false);
  return <div className="observation">
    <button type="button" className="text-button" aria-expanded={open} aria-controls={`observation-${runId}`} onClick={() => setOpen((value) => !value)}><FileImage size={16} aria-hidden="true" />{open ? '관측 증거 닫기' : '계획에 사용된 관측 증거 보기'}</button>
    {open && <ObservationImage key={runId} api={api} runId={runId} />}
  </div>;
}

function ObservationImage({ api, runId }: { api: ConsoleApi; runId: string }) {
  const load = useCallback(async (signal: AbortSignal) => ({ blob: await api.getObservation(runId, signal) }), [api, runId]);
  const image = useProtectedImage(load, null);
  return <figure id={`observation-${runId}`} className="evidence-image">
    {image.loading && !image.url && <Loading>인증된 관측 증거를 불러오는 중…</Loading>}
    {image.url && !image.decodeError && <img src={image.url} alt={`실행 ${runId}의 계획 생성에 사용된 Isaac Sim 합성 관측 증거`} onError={image.onDecodeError} />}
    <ErrorNotice error={image.error ?? (image.decodeError ? new Error('관측 증거 PNG를 표시할 수 없습니다.') : null)} title="관측 증거 로드 실패" retry={image.refresh} compact />
    <figcaption>저장된 관측 증거 · 현재 LIVE 영상이 아님 · Isaac Sim 합성 데이터</figcaption>
  </figure>;
}
