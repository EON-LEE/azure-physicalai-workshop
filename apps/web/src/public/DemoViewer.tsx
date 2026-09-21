import { useCallback, useEffect, useState } from 'react';
import { ArrowUpRight, Camera, ChevronDown, Cloud, Code2, Cpu, Eye, Factory, Pause, Play, ShieldCheck } from 'lucide-react';
import { describeError } from '../api/errors';
import { usePageVisible } from '../hooks/usePageVisible';
import { usePolling } from '../hooks/usePolling';
import { canShowCamera, getDemo, SNAPSHOT_MAX_AGE_MS, type DemoSnapshot } from './api';
import { CameraUnavailable, PublicLiveCamera } from './Media';
import { DecisionPanel, PhysicalOutcome, PresentationProgress, PublishedCounts } from './PresentationPanels';
import { formatCount, formatTime, presentationKey, presentationLabels } from './presentation';
import './demo.css';

function initiallyPaused() {
  return new URLSearchParams(window.location.search).get('viewing') === 'paused' ||
    (typeof window.matchMedia === 'function' && window.matchMedia('(prefers-reduced-motion: reduce)').matches);
}

export function DemoViewer({ loadSnapshot = getDemo }: {
  loadSnapshot?: (signal: AbortSignal) => Promise<DemoSnapshot>;
}) {
  const [paused, setPaused] = useState(initiallyPaused);
  const [clock, setClock] = useState(Date.now);
  const [invalidatedKey, setInvalidatedKey] = useState<string | null>(null);
  const visible = usePageVisible();
  const load = useCallback((signal: AbortSignal) => loadSnapshot(signal), [loadSnapshot]);
  // Initial metadata still loads for reduced-motion visitors; only ongoing viewing pauses.
  const [initialReceived, setInitialReceived] = useState(false);
  const snapshot = usePolling(load, { intervalMs: 2000, active: !paused || !initialReceived });
  useEffect(() => { if (snapshot.data || snapshot.error) setInitialReceived(true); }, [snapshot.data, snapshot.error]);
  useEffect(() => {
    if (!visible) return;
    setClock(Date.now());
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [visible]);
  useEffect(() => {
    const update = () => setPaused(new URLSearchParams(window.location.search).get('viewing') === 'paused');
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);

  const toggleViewing = () => {
    const next = !paused;
    setPaused(next);
    const url = new URL(window.location.href);
    if (next) url.searchParams.set('viewing', 'paused');
    else url.searchParams.delete('viewing');
    window.history.replaceState(null, '', url);
  };
  const retrySnapshot = () => { setInitialReceived(false); setInvalidatedKey(null); snapshot.refresh(); };
  const data = snapshot.data;
  const presentation = data?.presentation ?? null;
  const cycleKey = presentation ? presentationKey(presentation) : null;
  const integrityPending = cycleKey !== null && invalidatedKey === cycleKey;
  const resyncPresentation = useCallback(() => {
    setInvalidatedKey(cycleKey);
    snapshot.refresh();
  }, [cycleKey, snapshot.refresh]);
  const expired = Boolean(presentation && Date.parse(presentation.expires_at) <= clock);
  const observedAge = data ? clock - Date.parse(data.observed_at) : Number.POSITIVE_INFINITY;
  const snapshotFresh = Boolean(data && !snapshot.error && visible && !paused &&
    snapshot.lastReceivedAt !== null && clock - snapshot.lastReceivedAt <= SNAPSHOT_MAX_AGE_MS &&
    observedAge >= -5000 && observedAge <= SNAPSHOT_MAX_AGE_MS);
  const current = Boolean(presentation && snapshotFresh && !expired && !integrityPending);
  const liveReady = Boolean(data && presentation?.scene_epoch && current && canShowCamera(data) &&
    presentation.status !== 'stopped' && presentation.status !== 'failed');
  const viewing = !paused && visible && !snapshot.error && !expired;
  const running = current && presentation !== null && ['inspecting', 'awaiting_motion', 'moving'].includes(presentation.status);
  const stateLabel = paused ? '관람 일시 정지' : integrityPending ? '회차 일치 확인 중' : snapshot.error ? '연결 확인 필요'
    : expired ? '게시 시연 만료' : !presentation ? '게시된 자동 시연 없음'
      : !snapshotFresh ? '현재 상태 확인 중' : presentationLabels[presentation.status];

  let unavailableTitle = '게시된 자동 시연이 없습니다';
  let unavailableDescription = '운영자가 시연을 게시하면 실제 카메라와 검사·이동 결과가 자동으로 나타납니다.';
  if (!data) {
    unavailableTitle = snapshot.error ? '공개 시연 정보를 불러오지 못했습니다' : '공개 시연 확인 중…';
    unavailableDescription = snapshot.error ? '로그인은 필요하지 않습니다. 연결을 다시 확인해 주세요.' : 'Azure에 게시된 시연 상태를 읽고 있습니다.';
  } else if (paused || !visible) {
    unavailableTitle = '영상 관람을 일시 정지했습니다';
    unavailableDescription = '관람 재개를 누르면 최신 회차를 다시 확인합니다. 로봇이나 서버 시연을 멈추는 기능은 아닙니다.';
  } else if (integrityPending) {
    unavailableTitle = '관측과 게시 회차가 일치하지 않습니다';
    unavailableDescription = '다른 회차의 이미지와 판단을 숨겼습니다. 새 회차의 게시 정보를 기다리거나 상태를 다시 확인해 주세요.';
  } else if (snapshot.error || !snapshotFresh) {
    unavailableTitle = '현재 시연 연결을 확인할 수 없습니다';
    unavailableDescription = '마지막 수신 내용을 현재 실행처럼 표시하지 않습니다. 최신 상태를 다시 확인해 주세요.';
  } else if (expired) {
    unavailableTitle = '게시된 시연 시간이 끝났습니다';
    unavailableDescription = '마지막 결과는 아래에 남아 있지만 현재 실행 중이라는 뜻은 아닙니다.';
  } else if (presentation?.status === 'stopped' || presentation?.status === 'failed') {
    unavailableTitle = presentation.status === 'failed' ? '시연이 실패로 중단되었습니다' : '시연이 중지되었습니다';
    unavailableDescription = '실제 서버 상태입니다. 관람 화면은 시연을 재시작하지 않습니다. 게시 상태만 다시 확인할 수 있습니다.';
  } else if (presentation && (data.simulation.status === 'loading' || presentation.status === 'preparing')) {
    unavailableTitle = '실제 시연을 준비하고 있습니다';
    unavailableDescription = '현재 회차의 씬과 카메라를 기다립니다. 준비 상태를 완료나 실제 동작으로 표시하지 않습니다.';
  } else if (presentation) {
    unavailableTitle = '실시간 카메라가 준비되지 않았습니다';
    unavailableDescription = '게시된 판단이나 결과가 있어도 카메라 연결의 증거는 아닙니다. 현재 씬의 카메라만 표시합니다.';
  }

  return <div className="demo-page">
    <a href="#demo-main" className="demo-skip">시연 본문으로 이동</a>
    <header className="demo-header">
      <a href="/" className="demo-brand" aria-label="Azure Physical AI 공개 시연 홈"><span><Factory size={23} aria-hidden="true" /></span>
        <span translate="no">Azure <b>Physical AI</b><small>INSPECTION TO ACTION</small></span></a>
      <nav aria-label="시연 탐색"><a href="#demo-stage">실제 시연</a><a href="#demo-verification">검증 범위</a></nav>
      <a className="demo-operator" href="/operator"><Code2 size={15} aria-hidden="true" />운영자<ArrowUpRight size={14} aria-hidden="true" /></a>
    </header>
    <main id="demo-main">
      <section className="demo-intro" aria-labelledby="demo-title">
        <div><p className="demo-eyebrow">PUBLIC PRESENTATION · 읽기 전용</p>
          <h1 id="demo-title">흠집 있는 부품을,<br className="demo-mobile-break" /> <span>후공정 대신 격리합니다.</span></h1>
          <p className="demo-lead">{running && liveReady ? '접속하면 실제 자동 시연을 볼 수 있습니다. 관람을 위해 로그인하거나 실행 버튼을 누를 필요가 없습니다.' : '실제 카메라 → Foundry의 이미지 판단 → 제한된 로봇 이동 → 측정된 결과를 한 회차로 확인합니다.'}</p>
        </div>
        <span className="demo-audience"><Eye size={18} aria-hidden="true" /><span>로그인 없는 공개 관람<small>시연 제어·편집 권한은 없음</small></span></span>
      </section>

      <section className="demo-public-status" aria-label="공개 시연 상태">
        <div className="public-status-label" role="status"><span className={`state-dot ${running && liveReady ? 'active' : ''}`} aria-hidden="true" /><strong>{stateLabel}</strong></div>
        {presentation && <div className="public-cycle"><span>회차 <b>{formatCount(presentation.cycle)}</b> / {formatCount(presentation.total_cycles)}</span>
          <span className={`scenario-label ${presentation.scenario === 'surface_defect' ? 'defect' : 'normal'}`}>서버 입력: {presentation.scenario === 'surface_defect' ? '표면 흠집 부품' : '정상 부품'}</span></div>}
        <div className="viewing-control"><button type="button" className="demo-button" onClick={toggleViewing} aria-pressed={paused}>
          {paused ? <Play size={15} aria-hidden="true" /> : <Pause size={15} aria-hidden="true" />}{paused ? '관람 재개' : '관람 일시 정지'}
        </button><span>영상 관람만 멈춥니다. 로봇 정지가 아닙니다.</span></div>
      </section>
      {Boolean(snapshot.error) && <div className="public-network-error" role="alert"><span>{describeError(snapshot.error)}</span><button type="button" onClick={retrySnapshot} className="demo-text-button">게시 상태 다시 확인</button></div>}
      <section className="public-stage-layout" id="demo-stage" aria-label="실제 카메라와 같은 회차의 검사 판단">
        <div className="public-stage panel">
          <div className="public-panel-heading"><span className="title-icon"><Camera size={19} aria-hidden="true" /><h2>실제 작업 셀 카메라</h2></span><span className="small-label" translate="no">ISAAC SIM / PHYSX</span></div>
          {liveReady && presentation?.scene_epoch ? <PublicLiveCamera key={presentationKey(presentation)} epoch={presentation.scene_epoch} onSceneChanged={resyncPresentation} />
            : <CameraUnavailable title={unavailableTitle} description={unavailableDescription} retry={retrySnapshot} loading={snapshot.loading} />}
          <div className="public-station-legend" aria-label="실제 카메라의 스테이션 색상"><span><i className="source-station" />공급대</span><span><i className="accepted-station" />정상 트레이</span><span><i className="rejected-station" />불량 격리 트레이</span><small>표시 대상: 합성 부품 · 실물 아님</small></div>
          {presentation && <div className="public-task"><span>운영자가 게시한 작업</span><p>{presentation.instruction}</p></div>}
        </div>
        <DecisionPanel presentation={integrityPending ? null : presentation} snapshot={data} current={current} viewing={viewing} onSceneChanged={resyncPresentation} />
      </section>

      {presentation && <>
        <PresentationProgress presentation={presentation} current={current} />
        <PhysicalOutcome presentation={presentation} current={current && Boolean(data && canShowCamera(data))} />
        <PublishedCounts presentation={presentation} />
      </>}

      <details className="public-verification" id="demo-verification">
        <summary><ShieldCheck size={18} aria-hidden="true" /><span>무엇이 검증되었고, 무엇이 아닌가요?</span><ChevronDown size={17} aria-hidden="true" /></summary>
        <div className="verification-content">
          <p><strong>이 시연은 학습된 VLA나 정책이 로봇을 제어하는 데모가 아닙니다.</strong> Foundry 이미지 검사와 제한된 기존 로봇 제어·PhysX를 연결합니다. 검사 정답 일치와 물리 목표 도달을 모두 확인해야 해당 작업이 성공입니다.</p>
          <div className="verification-grid">
            <div><Camera size={21} aria-hidden="true" /><h3>NVIDIA Isaac Sim</h3><p>실제 물리 시뮬레이터의 합성 카메라와 부품 위치입니다. 실물 로봇 또는 공장 센서의 검증을 뜻하지 않습니다.</p></div>
            <div><Cpu size={21} aria-hidden="true" /><h3>Microsoft Foundry</h3><p>{data?.agent.connectivity === 'verified' ? '별도 연결 호출이 검증되어 있습니다.' : data?.agent.connectivity === 'configured' ? '모델 연결은 구성되어 있지만 호출 검증은 확인되지 않았습니다.' : '모델 연결 검증을 확인할 수 없습니다.'} 연결 여부는 검사 정확도나 이번 회차 성공의 증거가 아닙니다.</p>
              {data?.agent.verified_at && <small>연결 확인 기록: {formatTime(data.agent.verified_at)}</small>}</div>
            <div><Cloud size={21} aria-hidden="true" /><h3>Azure 학습 파이프라인</h3><p>{data?.learning.status === 'cpu_smoke_verified' ? `Azure ACR의 CPU 스모크: 테스트 표본으로 ${formatCount(data.learning.optimizer_steps)}회 optimizer step이 보고되었습니다.` : '게시된 CPU 스모크 검증이 없습니다.'} 로봇 학습 데이터, 정책 품질 또는 물리 평가를 검증한 결과가 아닙니다.</p></div>
          </div>
          <p>공개 화면은 승인된 현재 시연만 읽습니다. 방문자가 부품 종류를 고르거나 모델·로봇 실행을 요청하지 않습니다. 카메라와 증거 이미지의 회차가 바뀌면 이전 이미지를 제거합니다.</p>
        </div>
      </details>
    </main>
    <footer className="demo-footer"><span translate="no">Azure Physical AI</span><span>현재 시연 관람만 공개 · 제어·편집·개인 기록은 보호됨</span>
      <span>{data ? <>공개 응답 <time dateTime={data.observed_at}>{formatTime(data.observed_at)}</time></> : '공개 응답 확인 전'} · 브라우저 현지 시간</span></footer>
  </div>;
}
