import { useEffect, useState } from 'react';
import { Download } from 'lucide-react';
import { isAbort } from '../api/errors';
import { useRequestScope } from '../hooks/useRequestScope';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import type { LearningApi } from './contracts';
import type { SimulationReport } from './simulationReports';

const labels = { before: 'P0', after: 'P1', reference: '기준 제어기', candidate: '최초 후보' };
const number = (value: number) => new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 3 }).format(value);

export function SimulationComparison({ api, jobId, jobStatus, report }: {
  api: LearningApi; jobId: string; jobStatus: string; report: SimulationReport;
}) {
  const [url, setUrl] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const requestScope = useRequestScope();
  useEffect(() => () => { if (url) URL.revokeObjectURL(url); }, [url]);
  const download = async () => {
    if (busy) return;
    const request = requestScope();
    setBusy(true); setError(null);
    try {
      const file = await api.reportDocument(jobId, request.signal);
      if (request.current()) setUrl(URL.createObjectURL(file));
    } catch (failure) {
      if (request.current() && !isAbort(failure)) setError(failure);
    } finally {
      if (request.current()) setBusy(false);
      request.finish();
    }
  };
  return <section className="panel policy-comparison" aria-labelledby="simulation-comparison-title">
    <div className="panel-heading"><h3 id="simulation-comparison-title">저장된 비실시간 물리 평가</h3><Badge tone="amber">NON_REALTIME_SIMULATION</Badge></div>
    <div className="learning-panel-body">
      <p>검증된 원본의 요약입니다. 현재 LIVE 동작이나 실시간 100ms/80ms 통과가 아닙니다. 전체 시도·시계열은 원본 보고서에 보존됩니다.</p>
      <div className="inline-note" role="status"><strong>{report.quality_gate_passed ? '시뮬레이션 품질 기준 통과' : '시뮬레이션 품질 기준 미달'}</strong>
        <p>Azure 작업 상태: {jobStatus} · {report.comparison_kind === 'reference_bootstrap' ? '기준 제어기는 학습된 P0가 아닙니다.' : `절대 성공률 변화: ${number(report.absolute_success_rate_improvement * 100)}%p`}</p>
      </div>
      <dl className="learning-metadata">
        <FieldValue label="전체 실제 WALL 시간 (초)">{number(report.total_wall_duration_ms / 1000)}</FieldValue>
        <FieldValue label="전체 실제 SIM 시간 (초)">{number(report.total_simulation_duration_ms / 1000)}</FieldValue>
        <FieldValue label="전체 시도 수">{report.total_trial_count}</FieldValue>
        <FieldValue label="안전 / 자원 위반">{report.safety_violation_count} / {report.resource_violation_count}</FieldValue>
        <FieldValue label="고정된 평가 기준 SHA"><code>{report.criteria_sha256}</code></FieldValue>
        <FieldValue label="모델 독립 scene conditions SHA"><code>{report.frozen_plan_sha256}</code></FieldValue>
        <FieldValue label="native plan / results SHA"><code>{report.native_plan_sha256}</code><code>{report.results_sha256}</code></FieldValue>
      </dl>
      <div className="table-scroll"><table><caption>각 역할의 전체 20개 조건 · 실패 포함</caption>
        <thead><tr><th>역할</th><th>성공 / 전체</th><th>정책 WALL p95 (ms)</th></tr></thead>
        <tbody>{Object.entries(report.counts).map(([role, counts]) => {
          if (!counts || !(role in labels)) return null;
          const key = role as keyof typeof labels;
          return <tr key={role}><td>{labels[key]}</td><td>{counts.success} / {counts.total}</td><td>{report.latency_wall_ms[key]?.p95 ?? '모델 미사용'}</td></tr>;
        })}</tbody>
      </table></div>
      <div className="table-scroll"><table><caption>모든 40회 물리 시도 요약 · 원본 시계열 대체 아님</caption>
        <thead><tr><th>역할 / seed</th><th>결과</th><th>WALL 초</th><th>SIM 초</th><th>파지 / 안정화</th><th>정책 WALL p95 ms</th><th>전체 구간 WALL 최대 ms</th></tr></thead>
        <tbody>{report.trials.map((trial) => <tr key={`${trial.policy}-${trial.episode_id}-${trial.attempt}`}>
          <td>{labels[trial.policy]} / {trial.seed}<code>{trial.environment_id}</code></td>
          <td>{trial.physical_success ? '성공' : '실패'}{trial.failure_reason && <small>{trial.failure_reason}</small>}</td>
          <td>{number(trial.wall_duration_ms / 1000)}</td><td>{number(trial.simulation_duration_ms / 1000)}</td>
          <td>{trial.task_evidence.grasp_verified ? '확인' : '미확인'} / {trial.task_evidence.settled ? '확인' : '미확인'}</td>
          <td>{trial.phase_wall_ms.policy.p95 ?? '모델 미사용'}</td><td>{trial.phase_wall_ms.interval.max ?? '측정 없음'}</td>
        </tr>)}</tbody>
      </table></div>
      <p className="form-hint">위치만으로 성공을 만들지 않습니다. 실제 틱의 이동·finger gap 기반 파지와 해제·안정화 증거이며, 접촉 센서를 측정했다고 주장하지 않습니다.</p>
      <p>별도 시뮬레이션 정책 게시·실행은 아직 승인되지 않았습니다. 이 보고서를 기존 실시간 release로 바꾸지 않습니다.</p>
      <button type="button" className="button secondary" disabled={busy} onClick={() => void download()}><Download size={16} aria-hidden="true" />{busy ? '검증된 원본을 불러오는 중…' : '전체 원본 보고서 불러오기'}</button>
      {url && <p role="status"><a href={url} download={`simulation-report-${jobId}.json`}>검증된 전체 JSON 저장</a></p>}
      <p className="small-text">원본 SHA256: <code>{report.report_sha256}</code></p>
      <ErrorNotice error={error} title="원본 보고서 다운로드 실패" />
    </div>
  </section>;
}
