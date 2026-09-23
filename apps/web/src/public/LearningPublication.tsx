import { useCallback, useState } from 'react';
import { BookOpen, ChevronDown } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import { getPublicLearning } from './api';
import { formatCount, formatTime } from './presentation';

export function LearningPublication() {
  const [open, setOpen] = useState(false);
  const load = useCallback((signal: AbortSignal) => getPublicLearning(signal), []);
  const response = usePolling(load, { active: open });
  const item = response.data?.publication;
  return <details className="public-verification" onToggle={(event) => setOpen(event.currentTarget.open)}>
    <summary><BookOpen size={18} aria-hidden="true" /><span>작업 시연·학습 전후의 검증된 기록</span><ChevronDown size={17} aria-hidden="true" /></summary>
    <div className="verification-content">
      {Boolean(response.error) && <p role="alert">게시된 학습 기록을 확인할 수 없습니다. 개인 데이터나 다른 모델 결과로 대신하지 않습니다. <button type="button" className="demo-text-button" onClick={response.refresh}>다시 확인</button></p>}
      {!item && !response.error && <p role="status">{response.loading ? '승인된 학습 비교 기록을 불러오는 중…' : '아직 공개된 학습 비교 기록이 없습니다. 현재 검사·분류 동작이나 좌표 변경 사례를 학습 결과로 부르지 않습니다.'}</p>}
      {item && <>
        <h3>{item.title}</h3><p>{item.task}</p><p><strong>저장된 평가 · 현재 LIVE 아님</strong> · {formatTime(item.recorded_at)}</p>
        <p>정책: <span translate="no">{item.policy_type}</span> · 실제 optimizer steps {formatCount(item.training.optimizer_steps)} · {item.training.loss === null ? 'loss 미게시' : `loss ${item.training.loss}`}</p>
        <p>직접 시연 {formatCount(item.data_provenance.human_teleop)} · 기준 제어기 {formatCount(item.data_provenance.reference_controller)} · 정책 생성 {formatCount(item.data_provenance.learned)}</p>
        <p>{item.comparison.comparison_kind === 'reference_bootstrap' ? '첫 정책의 기준 제어기 대비 품질·안전 평가입니다. 학습된 P0/P1 개선 비교가 아닙니다.' : item.comparison.conclusion === 'improved' ? '같은 held-out 조건에서 개선이 보고되었습니다.' : item.comparison.conclusion === 'not_improved' ? '학습 작업은 완료되었지만 개선은 확인되지 않았습니다.' : '학습 효과의 결론이 불충분합니다.'}</p>
        <p>모델 생성 기록: {formatTime(item.training.created_at)} · 이전 모델을 오늘의 학습 결과로 바꿔 표시하지 않습니다.</p>
        <div className="table-scroll"><table><caption>실패와 재시도를 포함한 전체 공개 평가 ({formatCount(item.comparison.trials.length)})</caption>
          <thead><tr><th>정책</th><th>seed / 시도</th><th>상태</th><th>물리 성공</th><th>시간</th></tr></thead><tbody>{item.comparison.trials.map((trial) => <tr key={`${trial.seed}:${trial.policy}:${trial.attempt}`}><td>{trial.policy}</td><td>{trial.seed} / {trial.attempt}</td><td>{trial.status}</td><td>{trial.physical_success ? '확인' : '실패 / 미확인'}</td><td>{trial.duration_seconds}s</td></tr>)}</tbody></table></div>
      </>}
    </div>
  </details>;
}
