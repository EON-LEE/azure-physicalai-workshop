import { useCallback, useState } from 'react';
import { ArrowDown, ArrowRight, Check, FileSearch, MoveRight, ShieldCheck } from 'lucide-react';
import { describeError } from '../api/errors';
import { usePolling } from '../hooks/usePolling';
import { useProtectedImage } from '../hooks/useProtectedImage';
import { getDemoCaseEvidence, getDemoCases, type RecordedCase } from './api';
import { formatCount, formatPosition, formatTime } from './presentation';

const outcomes = {
  normal_route: { title: '정상 부품 → 다음 공정', question: '검사만 하는 AI를 넘어, 선별 작업까지 연결할 수 있나?', empty: '게시된 정상 분류·이동 완료 기록이 아직 없습니다.' },
  defect_route: { title: '표면 결함 → 격리', question: '결함을 발견한 부품이 실제 격리 위치에 도착했나?', empty: '게시된 결함 격리·이동 완료 기록이 아직 없습니다.' },
  withheld: { title: '판단 불일치 → 이동 보류', question: 'AI가 잘못 판단했을 때 움직이지 않게 할 수 있나?', empty: '게시된 검사 불일치·이동 보류 기록이 아직 없습니다.' },
};

export function CustomerValue() {
  return <section className="customer-value" id="customer-value" aria-labelledby="customer-value-title">
    <div className="customer-request"><span>품질·생산 담당자의 요청</span><h2 id="customer-value-title">“불량을 알려주는 데서 끝내지 말고, 다음 공정에 들어가지 않게 분리해 주세요.”</h2>
      <p>이 데모는 <strong>제조 검사 후 선별·격리</strong> 한 공정을 다룹니다. 카메라를 본 AI의 판단을 제한된 로봇 스킬로 연결하고, 실제 도착 위치로 작업 결과를 확인합니다.</p></div>
    <ol className="customer-value-grid">
      <li><FileSearch size={21} aria-hidden="true" /><span>고객이 확인할 것</span><h3>알림이 아니라 작업 완료</h3><p>검사 원본, 판단 이유, 실제 이동 결과를 연결해 정상·격리 경로가 달라지는지 봅니다.</p><a href="#demo-stage">실제 검사·이동 보기<ArrowDown size={15} aria-hidden="true" /></a></li>
      <li><MoveRight size={21} aria-hidden="true" /><span>고객이 바꿔볼 것</span><h3>내 라인의 격리 위치</h3><p>트레이 위치와 합성 부품 조건을 바꾼 JSON을 만들고, 운영자에게 시뮬레이션 실험을 전달합니다.</p><a href="#customer-lab">실험 조건 정하기<ArrowDown size={15} aria-hidden="true" /></a></li>
      <li><ShieldCheck size={21} aria-hidden="true" /><span>도입 전에 검토할 것</span><h3>틀린 판단과 실행 권한</h3><p>모델이 항상 맞다고 가정하지 않습니다. 판단 불일치 기록을 보고, 고객 실험에서는 사람이 이동을 승인합니다.</p><a href="#demo-cases">실제 결과 비교하기<ArrowDown size={15} aria-hidden="true" /></a></li>
    </ol>
    <p className="customer-value-boundary">현재는 합성 부품과 Franka 한 대의 시뮬레이션입니다. 고객 제품의 검사 성능, 생산성 향상, 실물 설비 연결은 별도 PoC로 검증해야 합니다.</p>
  </section>;
}

function RecordedImage({ presentationId, item }: { presentationId: string; item: RecordedCase }) {
  const load = useCallback((signal: AbortSignal) => getDemoCaseEvidence(presentationId, item, signal), [presentationId, item]);
  const image = useProtectedImage(load, null);
  return <figure className="recorded-case-image">
    {image.url && !image.error && !image.decodeError ? <img src={image.url} width={960} height={540} loading="lazy"
      onError={image.onDecodeError} alt={`기록된 ${item.scenario === 'normal' ? '정상' : '표면 결함'} 시나리오의 실제 검사 원본. 현재 LIVE 영상이 아닙니다.`} />
      : <div role="status">{image.error ? describeError(image.error) : image.decodeError ? '검사 원본 PNG를 표시할 수 없습니다.' : '기록된 검사 원본을 읽는 중…'}</div>}
    {(Boolean(image.error) || image.decodeError) && <button type="button" className="demo-text-button" onClick={image.refresh}>이 기록의 이미지 다시 확인</button>}
    <figcaption>실제 검사 입력 기록 · LIVE 아님<br /><time dateTime={item.captured_at}>{formatTime(item.captured_at)}</time></figcaption>
  </figure>;
}

function RecordedOutcome({ presentationId, item }: { presentationId: string; item: RecordedCase }) {
  const blocked = item.kind === 'withheld';
  return <>
    <RecordedImage presentationId={presentationId} item={item} />
    <div className="recorded-case-details">
      <p className="recorded-case-classification">실제 모델 판단 <strong>{item.classification === 'accepted' ? '정상' : '불량'}</strong></p>
      <blockquote>{item.summary}</blockquote>
      <p className={blocked ? 'recorded-case-held' : 'recorded-case-passed'}><Check size={16} aria-hidden="true" />
        {blocked ? '검사 불일치 · 이동 승인 및 실행 없음' : item.kind === 'normal_route' ? '정상 트레이 도착 측정됨' : '격리 트레이 도착 측정됨'}</p>
      <dl><div><dt>{blocked ? '모델이 제안한 위치 · 실행 안 함' : '계획된 목표 위치'}</dt><dd>{formatPosition(item.target_position_m)}</dd></div>
        <div><dt>측정된 최종 위치</dt><dd>{formatPosition(item.result.final_position_m)}</dd></div>
        <div><dt>승인부터 물리 완료까지</dt><dd>{item.physical_duration_seconds === null ? '이동 실행 없음' : `${new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 2 }).format(item.physical_duration_seconds)}초`}</dd></div>
      </dl>
      <p className="recorded-case-source">{formatCount(item.cycle)}회차 · <time dateTime={item.result.completed_at}>{formatTime(item.result.completed_at)}</time></p>
    </div>
  </>;
}

export function RecordedCases({ presentationId }: { presentationId: string | null }) {
  const [expanded, setExpanded] = useState(() => new URLSearchParams(window.location.search).get('evidence') === 'recorded');
  const load = useCallback((signal: AbortSignal) => getDemoCases(signal), []);
  const records = usePolling(load, { active: expanded, intervalMs: null });
  const matching = records.data?.presentation_id === presentationId ? records.data : null;
  const kinds: RecordedCase['kind'][] = ['normal_route', 'defect_route', 'withheld'];
  const reveal = () => {
    setExpanded(true);
    const url = new URL(window.location.href);
    url.searchParams.set('evidence', 'recorded');
    window.history.replaceState(null, '', url);
    if (expanded) records.refresh();
  };
  return <section className="recorded-cases" id="demo-cases" aria-labelledby="recorded-cases-title">
    <div className="customer-section-heading"><div><p>업무 결과 비교 · 실제 저장 기록</p><h2 id="recorded-cases-title">같은 로봇, 다른 판단, 다른 조치</h2></div>
      <button type="button" className="demo-button" onClick={reveal} disabled={records.loading}>
        {records.loading ? '실행 기록 확인 중…' : expanded ? '게시된 기록 다시 확인' : '실제 실행 기록 3가지 비교'}<ArrowRight size={16} aria-hidden="true" /></button></div>
    <p className="customer-section-description">아래는 현재 카메라와 별개인 실행 기록입니다. 성공한 정상·격리 사례와, 존재하면 판단 불일치로 보류한 첫 사례를 비교합니다. 빈 사례를 가상의 성공 화면으로 채우지 않습니다.</p>
    {Boolean(records.error) && <p role="alert" className="public-network-error">{describeError(records.error)} 기록 다시 확인 버튼으로 재시도하세요.</p>}
    {expanded && records.data && !matching && <p role="status" className="public-network-error">게시 시연이 바뀌었습니다. 이전 기록은 숨겼습니다. 게시된 기록을 다시 확인하세요.</p>}
    <div className="recorded-case-grid">{kinds.map(kind => {
      const item = matching?.cases.find(record => record.kind === kind);
      return <article key={kind} className="recorded-case" aria-labelledby={`case-${kind}`}>
        <h3 id={`case-${kind}`}>{outcomes[kind].title}</h3><p className="recorded-case-question">{outcomes[kind].question}</p>
        {expanded && item && matching?.presentation_id ? <RecordedOutcome presentationId={matching.presentation_id} item={item} />
          : <p className="recorded-case-empty">{!expanded ? '위 버튼으로 실제 기록을 불러오세요. 이 카드는 실행 결과가 아닙니다.' : records.loading ? '게시된 기록 확인 중…' : records.error ? '기록을 확인하지 못했습니다.' : outcomes[kind].empty}</p>}
      </article>;
    })}</div>
    <p className="customer-value-boundary">보류 사례는 정답을 아는 합성 시험에서 비교한 결과입니다. 생산 라인의 모든 오판을 자동 검출하거나 안전을 보장한다는 의미가 아닙니다. 이 기록만으로 검사 정확도나 비용 절감률을 일반화하지 않습니다.</p>
  </section>;
}
