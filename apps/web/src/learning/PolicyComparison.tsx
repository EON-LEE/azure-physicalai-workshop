import { useRef, useState } from 'react';
import { ShieldCheck } from 'lucide-react';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import type { Evaluation, LearningApi, PolicyRelease, Resource } from './contracts';
import { SimulationComparison } from './SimulationComparison';

const conclusion = { improved: '개선 확인', not_improved: '개선 미확인', inconclusive: '평가 결론 불충분' };
export function PolicyComparison({ api, evaluation, onReleased }: {
  api: LearningApi; evaluation: Resource<Evaluation>; onReleased?(release: Resource<PolicyRelease>): void;
}) {
  const [reviewed, setReviewed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [release, setRelease] = useState<Resource<PolicyRelease> | null>(null);
  const requestId = useRef(crypto.randomUUID());
  const report = evaluation.item.report;
  if (report && 'execution_timing' in report) {
    return <SimulationComparison api={api} jobId={evaluation.item.id} jobStatus={evaluation.item.status} report={report} />;
  }
  const bootstrap = report?.comparison_kind === 'reference_bootstrap';
  const eligible = evaluation.item.status === 'succeeded' && report?.quality_gate_passed && (report.comparison_kind === 'reference_bootstrap' || report.conclusion === 'improved');
  const publish = async () => {
    if (!eligible || !reviewed || busy) return;
    setBusy(true);
    try {
      const result = await api.release({
        request_id: requestId.current, candidate_id: evaluation.item.candidate_id,
        evaluation_run_id: evaluation.item.id, release_approved: true,
      }, evaluation.etag);
      setRelease(result);
      onReleased?.(result);
    } catch (failure) { setError(failure); }
    finally { setBusy(false); }
  };
  return <section className="panel policy-comparison" aria-labelledby="comparison-title">
    <div className="panel-heading"><h3 id="comparison-title">{bootstrap ? '첫 Franka 정책의 기준 제어기 대비 품질·안전 검증' : '같은 미사용 조건의 P0 / P1 비교'}</h3><Badge tone={eligible ? 'green' : 'amber'}>{report ? report.comparison_kind === 'reference_bootstrap' ? report.quality_gate_passed ? '초기 정책 품질 통과 · 개선 비교 아님' : '초기 정책 품질 미달' : conclusion[report.conclusion] : '결과 미수신'}</Badge></div>
    <div className="learning-panel-body">
      <p>저장된 평가 기록입니다. 현재 LIVE 동작이나 오늘 학습한 정책으로 바꾸어 표현하지 않습니다. {bootstrap ? '기준 제어기는 코드 기반이며 학습된 P0라고 부르지 않습니다.' : 'P0도 실제 학습된 정책이며 기존 스크립트 제어기가 아닙니다.'}</p>
      {report ? <>
        <dl className="learning-metadata">
          <FieldValue label="동일 held-out plan SHA"><code>{report.evaluation_plan_sha256}</code></FieldValue>
          <FieldValue label={bootstrap ? '기준 제어기 코드 SHA (모델 아님)' : 'P0 모델 SHA'}><code>{report.comparison_kind === 'reference_bootstrap' ? report.reference_controller_sha256 : report.before_model_sha256}</code></FieldValue>
          <FieldValue label={bootstrap ? '최초 후보 모델 SHA' : 'P1 모델 SHA'}><code>{report.comparison_kind === 'reference_bootstrap' ? report.candidate_model_sha256 : report.after_model_sha256}</code></FieldValue>
        </dl>
        <div className="table-scroll"><table><caption>모든 시험과 재시도 · 실패 포함 ({report.trials.length}개)</caption>
          <thead><tr><th>정책</th><th>seed / 시도</th><th>상태</th><th>물리 성공</th><th>위치 오차 (m)</th><th>시간 (s)</th><th>안전 위반</th></tr></thead>
          <tbody>{report.trials.map((trial) => <tr key={`${trial.seed}-${trial.policy}-${trial.attempt}`}>
            <td>{trial.policy === 'before' ? 'P0' : trial.policy === 'after' ? 'P1' : trial.policy === 'reference' ? '기준 제어기' : '최초 후보'}</td><td>{trial.seed} / {trial.attempt}<code>{trial.environment_id}</code></td>
            <td>{trial.status}</td><td>{trial.physical_success ? '확인' : '미확인'}</td>
            <td>{trial.axis_error_m?.join(', ') ?? '측정값 없음'}</td><td>{trial.duration_seconds}</td><td>{trial.safety_violations}</td>
          </tr>)}</tbody>
        </table></div>
        <p className="form-hint">loss 감소나 학습 종료만으로 게시하지 않습니다. 개선·물리 품질·안전 기준을 모두 통과하고 사람이 검토해야 합니다.</p>
      </> : <p className="form-hint">실제 paired report가 도착하기 전에는 성공률, 개선 수치 또는 영상 예시를 만들지 않습니다.</p>}
      <label className="checkbox-label"><input type="checkbox" checked={reviewed} disabled={!eligible || busy || Boolean(release)} onChange={(event) => setReviewed(event.target.checked)} />실패를 포함한 전체 시험과 정책 SHA를 검토했습니다</label>
      <button type="button" className="button" disabled={!eligible || !reviewed || busy || Boolean(release)} onClick={() => void publish()}><ShieldCheck size={15} aria-hidden="true" />검토한 정책 게시</button>
      {release && <div className="inline-note" role="status"><strong>검토된 정책 release가 생성되었습니다</strong><code>{release.item.id}</code><p>런타임 설치 상태는 아직 확인되지 않았습니다. 운영자의 실제 모델 설치·GPU 지연 검증·고정 카탈로그 활성화와 별도의 새 계획·승인이 필요합니다.</p><a className="text-button" href={`/operator?view=live&policy_release_id=${encodeURIComponent(release.item.id)}`}>검토된 정책으로 새 계획 준비</a></div>}
      <ErrorNotice error={error} title="정책 게시가 승인되지 않았습니다" />
    </div>
  </section>;
}
