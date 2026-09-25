import { useCallback, useRef, useState } from 'react';
import { usePolling } from '../hooks/usePolling';
import { useRequestScope } from '../hooks/useRequestScope';
import { isAbort } from '../api/errors';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import { ArtifactOperationPanel } from './ArtifactOperationPanel';
import { LearningJobPanel } from './LearningJobPanel';
import { PolicyComparison } from './PolicyComparison';
import type { ArtifactOperation, Dataset, Evaluation, Job, LearningApi, Project, ReferenceCollection, Resource, SimulationLearningCapability } from './contracts';

const pollReference = (value: Resource<ReferenceCollection>) =>
  !['failed', 'cancelled', 'timed_out'].includes(value.item.status) && !['ready', 'invalid'].includes(value.item.capture_status);

export function ReferenceCollections({ api, project, stages }: {
  api: LearningApi; project: Resource<Project>; stages?: SimulationLearningCapability;
}) {
  const [caseId, setCaseId] = useState('');
  const [approved, setApproved] = useState(false);
  const [paid, setPaid] = useState(false);
  const [evaluationApproved, setEvaluationApproved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [current, setCurrent] = useState<Resource<ReferenceCollection> | null>(null);
  const [selected, setSelected] = useState<string[]>([]);
  const [operation, setOperation] = useState<Resource<ArtifactOperation> | null>(null);
  const [dataset, setDataset] = useState<Resource<Dataset> | null>(null);
  const [job, setJob] = useState<Resource<Job> | null>(null);
  const [evaluation, setEvaluation] = useState<Resource<Evaluation> | null>(null);
  const attempts = useRef(new Map<string, string>());
  const startRequest = useRequestScope();
  const requestId = (key: string) => {
    const previous = attempts.current.get(key);
    if (previous) return previous;
    const id = crypto.randomUUID(); attempts.current.set(key, id); return id;
  };
  const list = useCallback(async (signal: AbortSignal) => {
    const records = await api.records(project.item.id, 'reference_collection', signal);
    return records.items.flatMap((entry) => entry.item.kind === 'reference_collection' ? [{ item: entry.item, etag: entry.etag }] : []);
  }, [api, project.item.id]);
  const records = usePolling(list);
  const load = useCallback(async (signal: AbortSignal) => {
    if (!current) throw new Error('선택한 기준 시연이 없습니다.');
    const value = await api.reference(current.item.id, signal);
    records.setData((items) => [...(items ?? []).filter((entry) => entry.item.id !== value.item.id), value]);
    return value;
  }, [api, current?.item.id, records.setData]);
  const state = usePolling(load, { active: current !== null, initialData: current, intervalMs: 2000, continuePolling: pollReference });
  const actual = state.data ?? current;
  const active = actual && !['succeeded', 'failed', 'cancelled', 'timed_out'].includes(actual.item.status);
  const act = async (action: (signal: AbortSignal) => Promise<void>) => {
    if (busy) return;
    const request = startRequest();
    setBusy(true); setError(null);
    try { await action(request.signal); if (request.current()) records.refresh(); }
    catch (failure) { if (request.current() && !isAbort(failure)) setError(failure); }
    finally { if (request.current()) setBusy(false); request.finish(); }
  };
  const onDataset = useCallback((value: Resource<Dataset>) => {
    setDataset(value); setOperation(null); setPaid(false); setEvaluationApproved(false);
  }, []);
  const onJob = useCallback((value: Resource<Job>) => {
    setJob(value);
    if (value.item.kind === 'evaluation') setEvaluation({ item: value.item, etag: value.etag });
  }, []);
  const ready = records.data?.filter((entry) => entry.item.capture_status === 'ready' && entry.item.status === 'succeeded') ?? [];
  const candidateId = job?.item.kind === 'training' && job.item.status === 'succeeded' ? job.item.candidate_id : null;
  return <section className="panel">
    <div className="panel-heading"><h3>검토된 REFERENCE 시연 생성</h3><Badge>reference_controller · 수동 시연 아님</Badge></div>
    <div className="learning-panel-body">
      <p>운영자가 같은 원본 권한을 runtime에 설치하고 전체 기준 작업의 안전 검증을 승인한 경우에만 요청합니다. 직접 손으로 가르치는 NRT 조작은 아직 제공하지 않습니다.</p>
      <p>먼저 선택할 배치의 저장 버전을 Environment Studio에서 활성화하세요. 권한 JSON·모델 경로·만료 시각을 브라우저에서 바꾸지 않습니다.</p>
      {!stages?.reference_generation_enabled && <p role="status">기준 시연 생성 단계가 아직 활성화되지 않았습니다.</p>}
      <label>기준 시연 배치<select name="reference-case" value={caseId} onChange={(event) => { setCaseId(event.target.value); setApproved(false); }} disabled={busy || Boolean(active)}>
        <option value="">검토된 배치 선택</option>{project.item.teaching_cases.map((item) => <option key={item.case_id} value={item.case_id}>{item.case_id} · {item.split} · seed {item.seed}</option>)}
      </select></label>
      <label className="checkbox-label"><input name="reference-approved" type="checkbox" checked={approved} disabled={!stages?.reference_generation_enabled || busy || Boolean(active)} onChange={(event) => setApproved(event.target.checked)} />선택한 배치의 검토된 기준 제어기 이동을 승인합니다</label>
      <button type="button" className="button" disabled={!stages?.reference_generation_enabled || !caseId || !approved || busy || Boolean(active)} onClick={() => void act(async (signal) => {
        const value = await api.startReference(project.item.id, { request_id: requestId(`reference:${caseId}`), case_id: caseId, motion_approved: true }, project.etag, signal);
        signal.throwIfAborted();
        setCurrent(value); state.setData(value);
      })}>승인한 REFERENCE 시연 생성</button>
      <ErrorNotice error={error ?? state.error ?? records.error} title="기준 시연 결과 확인 필요 · 자동 재전송 없음" />
      {actual && <div className="inline-note" role="status">
        <strong>REFERENCE · {actual.item.status} · 캡처 {actual.item.capture_status}</strong>
        <p>{actual.item.message}</p>
        {actual.item.execution && <dl className="learning-metadata">
          <FieldValue label="실제 WALL 시간 (ms)">{actual.item.execution.simulation_runtime.wall_elapsed_ms}</FieldValue>
          <FieldValue label="실제 SIM 시간 (초)">{actual.item.execution.simulation_runtime.simulation_elapsed_seconds}</FieldValue>
          <FieldValue label="실제 기준 제어기 호출">{actual.item.execution.simulation_runtime.reference_route_calls}</FieldValue>
          <FieldValue label="runtime 권한 원본 SHA"><code>{actual.item.runtime_catalog_record_sha256}</code></FieldValue>
        </dl>}
        {!['succeeded', 'failed', 'cancelled', 'timed_out'].includes(actual.item.status) && <button type="button" className="button danger-quiet" disabled={busy || actual.item.status === 'cancelling'} onClick={() => void act(async (signal) => {
          const result = await api.cancelReference(actual.item.id, signal);
          signal.throwIfAborted();
          setCurrent(result); state.setData(result);
        })}>기준 시연 취소 요청</button>}
      </div>}
      {records.data && records.data.length > 0 && <details>
        <summary>기존 기준 시연 기록 · 원래 명령 조회</summary>
        {records.data.map((entry) => <div className="learning-record-row" key={entry.item.id}>
          <span>{entry.item.teaching_case.case_id} · {entry.item.status} · 캡처 {entry.item.capture_status}<code>{entry.item.id}</code></span>
          <button type="button" className="text-button" disabled={busy} onClick={() => { setCurrent(entry); state.setData(entry); state.refresh(); }}>기준 시연 기록 보기</button>
        </div>)}
      </details>}
      <h4>검증된 기준 시연 데이터</h4>
      {ready.map((entry) => <label className="checkbox-label" key={entry.item.id}><input type="checkbox" checked={selected.includes(entry.item.id)} onChange={(event) => setSelected((items) => event.target.checked ? [...items, entry.item.id] : items.filter((id) => id !== entry.item.id))} />
        {entry.item.teaching_case.case_id} · reference_controller · {entry.item.id}</label>)}
      <button type="button" className="button secondary" disabled={!stages?.reference_generation_enabled || !selected.length || busy || Boolean(operation)} onClick={() => void act(async (signal) => {
        const value = await api.seal(project.item.id, { request_id: requestId(`dataset:${selected.join(',')}`), teaching_session_ids: [], reference_collection_ids: selected }, project.etag, signal);
        signal.throwIfAborted();
        if (value.item.kind === 'artifact_operation') { setDataset(null); setOperation({ item: value.item, etag: value.etag }); }
        else onDataset({ item: value.item, etag: value.etag });
      })}>검증된 REFERENCE 데이터 확정</button>
      {operation && <ArtifactOperationPanel api={api} initial={operation} onDataset={onDataset} />}
      {dataset && <p>확정 데이터: <code>{dataset.item.id}</code> · 직접 {dataset.item.human_teleop_count} / 기준 제어기 {dataset.item.reference_controller_count}</p>}
      <label className="checkbox-label"><input name="reference-data-paid" type="checkbox" checked={paid} onChange={(event) => setPaid(event.target.checked)} />별도 유료 작업 예산 {project.item.budget.maximum_cost_usd} USD와 원래 시간 상한을 승인합니다</label>
      <button type="button" className="button" disabled={!stages?.training_enabled || !dataset || !paid || busy} onClick={() => dataset && void act(async (signal) => {
        const result = await api.train(project.item.id, {
          request_id: requestId(`train:${dataset.item.id}`), dataset_id: dataset.item.id,
          parent_release_id: project.item.baseline_release_id, pretrained_artifact_id: project.item.pretrained_artifact_id,
          policy_type: project.item.policy_type, optimizer_steps: project.item.budget.optimizer_steps,
          paid_approved: true, maximum_cost_usd: project.item.budget.maximum_cost_usd,
        }, project.etag, signal);
        signal.throwIfAborted(); onJob(result);
      })}>검증된 데이터로 제한된 학습 제출</button>
      <label className="checkbox-label"><input name="candidate-evaluation-motion" type="checkbox" checked={evaluationApproved} disabled={!stages?.evaluation_enabled || !candidateId || busy} onChange={(event) => setEvaluationApproved(event.target.checked)} />후보와 기준 정책의 전체 고정 평가 조건에서 제한된 이동을 별도로 승인합니다</label>
      <button type="button" className="button secondary" disabled={!stages?.evaluation_enabled || !candidateId || !paid || !evaluationApproved || busy} onClick={() => candidateId && void act(async (signal) => {
        const value = await api.evaluate(project.item.id, {
          request_id: requestId(`evaluate:${candidateId}`), candidate_id: candidateId,
          baseline_release_id: project.item.baseline_release_id, comparison_kind: project.item.project_kind === 'bootstrap' ? 'reference_bootstrap' : 'paired_policy',
          evaluation_plan_sha256: project.item.evaluation_plan_sha256, motion_approved: true, paid_approved: true, maximum_cost_usd: project.item.budget.maximum_cost_usd,
        }, project.etag, signal);
        signal.throwIfAborted(); onJob(value);
      })}>전체 고정 조건의 제한된 후보 평가</button>
      <p>학습 시작에 이미 게시된 P0의 품질을 요구하지 않습니다. 검증된 train-only 부모를 사용하는 bootstrap은 별도이며, 정책 게시와 고객 learned 실행에는 전체 평가·품질 승인이 필요합니다.</p>
      {job && <LearningJobPanel key={`job-${job.item.id}`} api={api} initial={job} onUpdate={onJob} />}
      {evaluation && <PolicyComparison key={`evaluation-${evaluation.item.id}`} api={api} evaluation={evaluation} releaseAllowed={stages?.release_enabled} />}
    </div>
  </section>;
}
