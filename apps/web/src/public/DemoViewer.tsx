import { useCallback, useEffect, useState } from 'react';
import { ArrowDown, ArrowRight, Box, Check, ChevronLeft, ChevronRight, CircleDot, Cloud, Code2, Cpu, ExternalLink, Eye, Factory, Focus, Layers3, RefreshCw, ScanLine, ShieldCheck } from 'lucide-react';
import { usePolling } from '../hooks/usePolling';
import { useProtectedImage } from '../hooks/useProtectedImage';
import { getDemo, getDemoFrame, type DemoSnapshot } from './api';
import { WorkcellDiagram } from './WorkcellDiagram';
import './demo.css';

const steps = [
  { id: 'observe', title: '부품을 봅니다', service: 'Isaac Sim · 카메라', Icon: ScanLine,
    body: '가상 공장의 카메라가 부품을 관측합니다. 같은 작업 셀에서 이미지와 로봇 상태를 함께 가져옵니다.' },
  { id: 'reason', title: '검사 근거를 확인합니다', service: 'Microsoft Foundry · 에이전트', Icon: Cpu,
    body: '에이전트가 관측 이미지를 바탕으로 검사안을 만듭니다. 모델의 설명과 실제 로봇 실행 결과는 구분합니다.' },
  { id: 'approve', title: '실행 범위를 검토합니다', service: '승인 · 제어 검증', Icon: ShieldCheck,
    body: '운영자가 검사안과 목적지를 확인합니다. 허용된 작업과 이동 범위는 모델과 별도의 제어 계층이 검증합니다.' },
  { id: 'sort', title: '결과를 다시 봅니다', service: 'Isaac Sim · 로봇 제어', Icon: Box,
    body: '승인된 스킬이 부품을 옮깁니다. 완료 여부는 에이전트의 말이 아니라 부품의 실제 위치로 확인합니다.' },
] as const;

function readSelection() {
  const query = new URLSearchParams(window.location.search);
  const step = steps.findIndex((item) => item.id === query.get('step'));
  return { path: query.get('path') === 'accepted' ? 'accepted' as const : 'rejected' as const, step: Math.max(0, step) };
}

function LiveView() {
  const image = useProtectedImage(getDemoFrame, 1000);
  const stale = !image.data || image.clock - Date.parse(image.data.capturedAt) > 5000;
  const valid = image.url && !image.error && !image.decodeError && !stale;
  return <div className="demo-live" aria-live="polite">
    {valid ? <>
      <img src={image.url!} width={1280} height={720} alt="공개된 Isaac Sim 작업 셀의 실제 카메라 관측" onError={image.onDecodeError} />
      <span className="demo-live-stamp">실제 카메라 · {new Intl.DateTimeFormat('ko-KR', { timeStyle: 'medium' }).format(new Date(image.data!.capturedAt))}</span>
    </> : <div className="demo-live-message">
      <Focus size={36} aria-hidden="true" />
      <h3>{image.loading ? '실제 카메라 연결 중…' : '실시간 연결을 확인할 수 없습니다'}</h3>
      <p>이전 이미지나 대체 영상을 실제 동작처럼 보여주지 않습니다.</p>
      <button type="button" onClick={image.refresh}><RefreshCw size={15} aria-hidden="true" /> 다시 연결</button>
    </div>}
  </div>;
}

function Evidence({ data }: { data: DemoSnapshot }) {
  const connected = data.agent.connectivity === 'verified';
  return <section className="demo-evidence" aria-label="Azure 구성과 검증 범위">
    <div><Cloud aria-hidden="true" /><span><strong>Azure에서 실행</strong><small>웹 · API · 데이터 저장소</small></span><span className="demo-proof">배포됨</span></div>
    <div><Cpu aria-hidden="true" /><span><strong>Foundry 에이전트</strong><small>{connected ? '실제 연결 호출 검증 · 검사 정확도와 별개' : '모델 연결 구성 · 호출 검증 정보 없음'}</small></span><span className={connected ? 'demo-proof' : 'demo-proof muted'}>{connected ? '연결 확인' : '구성됨'}</span></div>
    <div><Layers3 aria-hidden="true" /><span><strong>정책 학습 파이프라인</strong><small>Azure CPU 테스트 표본 · 물리 평가 아님</small></span><span className="demo-proof muted">{data.learning.status === 'cpu_smoke_verified' ? '기반 검증' : '미게시'}</span></div>
  </section>;
}

export function DemoViewer({ loadSnapshot = getDemo }: {
  loadSnapshot?: (signal: AbortSignal) => Promise<DemoSnapshot>;
}) {
  const load = useCallback((signal: AbortSignal) => loadSnapshot(signal), [loadSnapshot]);
  const snapshot = usePolling(load, { intervalMs: 15000 });
  const [selection, setSelection] = useState(readSelection);
  useEffect(() => {
    const update = () => setSelection(readSelection());
    window.addEventListener('popstate', update);
    return () => window.removeEventListener('popstate', update);
  }, []);
  const choose = (path: 'accepted' | 'rejected', step: number) => {
    const selectedStep = steps[step];
    if (!selectedStep) throw new RangeError('Unknown demonstration explanation step.');
    setSelection({ path, step });
    const url = new URL(window.location.href);
    url.searchParams.set('path', path);
    url.searchParams.set('step', selectedStep.id);
    window.history.pushState(null, '', url);
  };
  const active = steps[selection.step];
  if (!active) throw new RangeError('Unknown demonstration explanation step.');
  const data = snapshot.data;
  const live = data?.mode === 'live' && !snapshot.error;
  return <div className="demo-page">
    <a href="#demo-main" className="demo-skip">데모 본문으로 이동</a>
    <header className="demo-header">
      <a href="/" className="demo-brand" aria-label="Azure Physical AI 데모 홈"><span><Factory size={22} aria-hidden="true" /></span><span>Azure <b>Physical AI</b><small>FACTORY EXPERIENCE</small></span></a>
      <nav aria-label="데모 탐색"><a href="#demo-story">데모 살펴보기</a><a href="#demo-architecture">Azure 구성</a></nav>
      <a className="demo-operator" href="/operator"><Code2 size={15} aria-hidden="true" /> 운영자 <ExternalLink size={13} aria-hidden="true" /></a>
    </header>
    <main id="demo-main">
      <section className="demo-intro">
        <div><p className="demo-eyebrow"><span /> SMART FACTORY / INSPECTION & SORTING</p>
          <h1>보는 AI에서,<br className="demo-mobile-break" /> <span>행동하는 공장으로.</span></h1>
          <p className="demo-lead">부품을 보고, 검사 근거를 확인하고, 알맞은 곳으로 옮기는 흐름.<br />하나의 작업 셀에서 Azure와 NVIDIA의 역할을 살펴보세요.</p>
        </div>
        <div className="demo-audience"><Eye size={17} aria-hidden="true" /><span>바로 보는 공개 데모<small>로그인 없이 시나리오 탐색</small></span></div>
      </section>

      {!data ? <section className="demo-load" aria-live="polite">
        <Factory size={40} aria-hidden="true" />
        <h2>{snapshot.error ? '데모 정보를 불러오지 못했습니다' : '작업 셀을 불러오는 중…'}</h2>
        <p>{snapshot.error ? '로그인은 필요하지 않습니다. 잠시 후 다시 연결해 주세요.' : 'Azure에서 공개된 시나리오 정보를 확인합니다.'}</p>
        {Boolean(snapshot.error) && <button type="button" onClick={snapshot.refresh}><RefreshCw size={16} aria-hidden="true" /> 다시 불러오기</button>}
      </section> : <>
        <section className="demo-stage-layout" id="demo-story">
          <div className="demo-stage">
            <div className="demo-stage-heading"><div><span className="demo-cell-number">CELL 01</span><h2>비전 검사 · 부품 분류</h2></div>
              <span className={live ? 'demo-state live' : 'demo-state'}><span />{live ? '실제 카메라' : '시나리오 가이드'}</span>
            </div>
            {live ? <LiveView /> : <WorkcellDiagram stations={data.scene.stations} path={selection.path} step={selection.step} />}
            <div className="demo-stage-caption"><span><CircleDot size={14} aria-hidden="true" />{live ? '게시된 Isaac Sim 관측 · 읽기 전용' : '작업 셀 구성도 · 실제 시뮬레이션 아님'}</span><span>합성 참조 환경</span></div>
            {!live && <p className="demo-availability" role="status">실시간 Isaac Sim 영상은 아직 연결되지 않았습니다. 아래 선택은 동작 실행이 아니라 시나리오 설명입니다.</p>}
            {Boolean(snapshot.error) && <p className="demo-network-error" role="alert">서버 연결이 끊겼습니다. 표시된 구성 정보가 최신이 아닐 수 있습니다. <button type="button" onClick={snapshot.refresh}>새로고침</button></p>}
            <div className="demo-paths"><span>어떤 경로를 볼까요?</span>
              <div role="group" aria-label="분류 경로 선택">
                <button type="button" aria-pressed={selection.path === 'accepted'} onClick={() => choose('accepted', selection.step)} className={selection.path === 'accepted' ? 'selected normal' : ''}><Check size={15} aria-hidden="true" /> 양품 흐름</button>
                <button type="button" aria-pressed={selection.path === 'rejected'} onClick={() => choose('rejected', selection.step)} className={selection.path === 'rejected' ? 'selected reject' : ''}><Focus size={15} aria-hidden="true" /> 불량 격리</button>
              </div>
            </div>
          </div>
          <aside className="demo-narrative" aria-label="검사·분류 단계 설명">
            <p className="demo-eyebrow">HOW IT WORKS</p><h2>“검사하고,<br />{selection.path === 'rejected' ? '불량이면 격리해.”' : '양품이면 다음 공정으로.”'}</h2>
            <p className="demo-narrative-note">에이전트의 판단과 로봇의 실행을 연결하는 4단계</p>
            <nav className="demo-steps" aria-label="시나리오 단계">
              {steps.map((step, index) => <button type="button" key={step.id} aria-pressed={selection.step === index} onClick={() => choose(selection.path, index)} className={selection.step === index ? 'active' : ''}>
                <span className="demo-step-number">{String(index + 1).padStart(2, '0')}</span><span><strong>{step.title}</strong><small>{step.service}</small></span><ChevronRight size={15} aria-hidden="true" />
              </button>)}
            </nav>
            <div className="demo-step-detail" aria-live="polite"><active.Icon size={22} aria-hidden="true" /><p>{active.body}</p>
              {selection.step === 3 && <strong className={selection.path === 'rejected' ? 'demo-reject-text' : 'demo-normal-text'}>{selection.path === 'rejected' ? '설명 경로: 검사 → 불량 격리 트레이' : '설명 경로: 검사 → 다음 공정'}</strong>}
              <div className="demo-step-controls"><button type="button" disabled={selection.step === 0} onClick={() => choose(selection.path, selection.step - 1)} aria-label="이전 설명 단계"><ChevronLeft size={16} aria-hidden="true" /></button><span>{selection.step + 1} / {steps.length}</span><button type="button" disabled={selection.step === steps.length - 1} onClick={() => choose(selection.path, selection.step + 1)} aria-label="다음 설명 단계"><ChevronRight size={16} aria-hidden="true" /></button></div>
            </div>
          </aside>
        </section>
        <Evidence data={data} />
        <section className="demo-architecture" id="demo-architecture">
          <div><p className="demo-eyebrow">ONE WORKCELL. CONNECTED ON AZURE.</p><h2>판단과 실행 사이에도,<br />확인할 수 있는 근거가 있습니다.</h2><p>에이전트의 “완료”라는 답만으로 성공 처리하지 않습니다.<br />관측, 승인된 명령, 물리 결과를 같은 실행에 연결합니다.</p></div>
          <div className="demo-stack">
            <div><ScanLine aria-hidden="true" /><strong>Isaac Sim</strong><span>가상 공장 · 카메라 · 물리 상태</span></div>
            <ArrowDown aria-hidden="true" />
            <div><Cpu aria-hidden="true" /><strong>Microsoft Foundry</strong><span>관측에 근거한 작업 계획</span></div>
            <ArrowDown aria-hidden="true" />
            <div><ShieldCheck aria-hidden="true" /><strong>승인된 로봇 스킬</strong><span>제한된 동작 · 결과 재확인</span></div>
          </div>
        </section>
      </>}
    </main>
    <footer className="demo-footer"><span>Azure Physical AI <span className="demo-footer-divider">/</span> 제조 레퍼런스 데모</span><span>공개 관람은 읽기 전용 · 환경 편집과 실행은 운영자 영역에서</span><a href="#demo-main">맨 위로 <ArrowRight size={13} aria-hidden="true" /></a></footer>
  </div>;
}
