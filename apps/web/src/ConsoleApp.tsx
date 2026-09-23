import { useCallback, useEffect, useRef, useState } from 'react';
import { ArrowUpRight, Blocks, BookOpen, ChevronRight, CircleHelp, Clock3, FileJson2, LayoutDashboard, LogOut, Radio, ShieldCheck } from 'lucide-react';
import { isTerminal, matchesRuntime, type ConsoleApi, type EnvironmentRecord, type RunRecord } from './api/contracts';
import { isAbort } from './api/errors';
import type { SignedInAccount } from './auth/types';
import { useSession } from './auth/context';
import { isDraftDirty, type StudioDraft } from './environment/validation';
import { buildCustomerExperiment, inspectionTask, readCustomerExperiment } from './environment/customerExperiment';
import { usePageVisible } from './hooks/usePageVisible';
import { usePolling } from './hooks/usePolling';
import { useRequestScope } from './hooks/useRequestScope';
import { Badge, Brand, ErrorNotice } from './ui/common';
import { formatDate } from './ui/format';
import { EnvironmentStudio } from './views/EnvironmentStudio';
import { FactoryLive, type RunDraft } from './views/FactoryLive';
import { History } from './views/History';
import type { ActivationState } from './views/RuntimePanel';
import { TeachingStudio } from './learning/TeachingStudio';

type View = 'live' | 'studio' | 'history' | 'learning';

const pages = {
  live: { title: 'Factory Live', subtitle: '물리 시뮬레이터의 관측부터 승인 기반 실행까지.', label: '라이브 운영', icon: LayoutDashboard, number: '01' },
  studio: { title: 'Environment Studio', subtitle: '고객 환경을 JSON으로 구성하고 저장 버전을 관리합니다.', label: '환경 스튜디오', icon: FileJson2, number: '02' },
  history: { title: 'Run History', subtitle: '실제 요청, 에이전트 계획과 물리 실행의 증거를 확인합니다.', label: '실행 기록', icon: Clock3, number: '03' },
  learning: { title: 'Teaching Studio', subtitle: '직접 시연, 정확한 모델 버전의 실제 학습과 같은 조건의 정책 비교를 연결합니다.', label: '작업 가르치기', icon: BookOpen, number: '04' },
};

export function ConsoleApp({ api, account, sessionExpired = false }: {
  api: ConsoleApi; account: SignedInAccount; sessionExpired?: boolean;
}) {
  const [view, setView] = useState<View>(() => {
    const requested = new URLSearchParams(window.location.search).get('view');
    return requested === 'studio' || requested === 'history' || requested === 'learning' ? requested : 'live';
  });
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [studioDraft, setStudioDraft] = useState<StudioDraft | null>(() => {
    if (!new URLSearchParams(window.location.search).has('experiment')) return null;
    const selection = readCustomerExperiment(window.location.search);
    return selection.error ? null : {
      text: JSON.stringify(buildCustomerExperiment(selection.value), null, 2),
      savedText: '', base: null, source: '공개 페이지에서 전달한 고객 공정 실험 초안',
    };
  });
  const [runDraft, setRunDraft] = useState<RunDraft>(() => ({
    instruction: studioDraft ? inspectionTask : '', attempt: null,
    policyReleaseId: new URLSearchParams(window.location.search).get('policy_release_id') ?? '',
  }));
  const [focusedRun, setFocusedRun] = useState<RunRecord | null>(null);
  const [activation, setActivation] = useState<ActivationState>({ submitting: false, receipt: null, error: null, timedOut: false });
  const [clock, setClock] = useState(Date.now);
  const [helpOpen, setHelpOpen] = useState(false);
  const titleRef = useRef<HTMLHeadingElement>(null);
  const session = useSession();
  const visible = usePageVisible();
  const startRequest = useRequestScope();
  const runtimeLoad = useCallback((signal: AbortSignal) => api.getRuntime(signal), [api]);
  const environmentsLoad = useCallback((signal: AbortSignal) => api.getEnvironments(signal), [api]);
  const runsLoad = useCallback((signal: AbortSignal) => api.getRuns(signal), [api]);
  const runtime = usePolling(runtimeLoad, { intervalMs: 3000, active: !sessionExpired });
  const environments = usePolling(environmentsLoad, { active: !sessionExpired });
  const runs = usePolling(runsLoad, { intervalMs: view === 'history' ? 10_000 : null, active: !sessionExpired });
  const records = environments.data?.items ?? [];
  const selected = records.find((item) => item.environment_id === selectedId) ?? null;
  const runtimeFresh = !runtime.paused && !runtime.error && runtime.lastReceivedAt !== null && clock - runtime.lastReceivedAt < 10_000 && !sessionExpired;
  const currentRun = focusedRun ?? (runs.data?.items.filter((run) => run.environment_id === selected?.environment_id)
    .sort((left, right) => Number(isTerminal(left.status)) - Number(isTerminal(right.status)) || Date.parse(right.created_at) - Date.parse(left.created_at))[0] ?? null);
  const activationReady = runtimeFresh && activation.receipt !== null && matchesRuntime(runtime.data, activation.receipt);
  const dirty = isDraftDirty(studioDraft);
  const page = pages[view];

  useEffect(() => {
    if (!visible) return;
    setClock(Date.now());
    const timer = setInterval(() => setClock(Date.now()), 1000);
    return () => clearInterval(timer);
  }, [visible]);

  useEffect(() => {
    if (selectedId !== null || !environments.data?.items.length) return;
    const active = environments.data.items.find((item) => item.environment_id === runtime.data?.simulation.environment_id);
    setSelectedId(active?.environment_id ?? environments.data.items[0]?.environment_id ?? null);
  }, [selectedId, environments.data, runtime.data?.simulation.environment_id]);

  useEffect(() => {
    if (!dirty) return;
    const warn = (event: BeforeUnloadEvent) => { event.preventDefault(); event.returnValue = ''; };
    window.addEventListener('beforeunload', warn);
    return () => window.removeEventListener('beforeunload', warn);
  }, [dirty]);

  useEffect(() => {
    if (!activation.receipt || activationReady || activation.timedOut) return;
    const timer = setTimeout(() => setActivation((value) => ({ ...value, timedOut: true })), 120_000);
    return () => clearTimeout(timer);
  }, [activation.receipt, activationReady, activation.timedOut]);

  const navigate = (next: View) => {
    if (next === view) return;
    setView(next);
    const url = new URL(window.location.href);
    url.searchParams.set('view', next);
    window.history.replaceState(null, '', url);
    requestAnimationFrame(() => titleRef.current?.focus());
  };

  const onSaved = useCallback((record: EnvironmentRecord) => {
    environments.setData((value) => ({ items: [...(value?.items ?? []).filter((item) => item.environment_id !== record.environment_id), record] }));
    setSelectedId(record.environment_id);
    runtime.refresh();
  }, [environments.setData, runtime.refresh]);

  const onRunChange = useCallback((run: RunRecord) => {
    setFocusedRun(run);
    runs.setData((value) => ({ items: [...(value?.items ?? []).filter((item) => item.id !== run.id), run] }));
  }, [runs.setData]);

  const activate = async (environment: EnvironmentRecord) => {
    if (sessionExpired || activation.submitting || (activation.receipt && !activationReady && !activation.timedOut)) return;
    setActivation({ submitting: true, receipt: null, error: null, timedOut: false });
    const request = startRequest();
    try {
      const receipt = await api.activateEnvironment(environment.environment_id, environment.revision, request.signal);
      if (!request.current()) return;
      setActivation({ submitting: false, receipt, error: null, timedOut: false });
      runtime.refresh();
    } catch (error) {
      if (request.current() && !isAbort(error)) {
        setActivation({ submitting: false, receipt: null, error, timedOut: false });
        runtime.refresh();
      }
    } finally {
      request.finish();
    }
  };

  const runtimeControls = {
    runtime: runtime.data,
    runtimeError: runtime.error,
    runtimeFresh,
    activation,
    onActivate: (environment: EnvironmentRecord) => void activate(environment),
    onRefresh: runtime.refresh,
    disabled: sessionExpired,
  };

  return <div className="app-shell">
    <a className="skip-link" href="#main-content">본문으로 건너뛰기</a>
    <aside className="sidebar" aria-label="주 메뉴">
      <Brand compact />
      <div className="workspace-label"><span className="workspace-avatar">P</span><div><strong>Physical AI Workspace</strong><small>고객 운영 콘솔</small></div><ChevronRight size={15} aria-hidden="true" /></div>
      <span className="nav-label">WORKSPACE</span>
      <nav aria-label="콘솔 화면">{(Object.keys(pages) as View[]).map((key) => {
        const item = pages[key];
        const Icon = item.icon;
        return <button type="button" key={key} className={`nav-item ${view === key ? 'active' : ''}`} aria-current={view === key ? 'page' : undefined} onClick={() => navigate(key)}>
          <Icon size={19} aria-hidden="true" /><span>{item.title}<small>{item.label}</small></span>
          {key === 'studio' && dirty ? <span className="draft-dot" aria-label="저장하지 않은 변경 있음" /> : <span className="nav-number">{item.number}</span>}
        </button>;
      })}</nav>
      <div className="sidebar-bottom">
        <div className="safety-card"><ShieldCheck size={21} aria-hidden="true" /><strong>Human in the loop</strong><p>판단은 에이전트와 함께.<br />실행 권한은 사람에게.</p><span>자동 승인 없음</span></div>
        <button type="button" className="sidebar-help" aria-expanded={helpOpen} aria-controls="console-help" onClick={() => setHelpOpen((value) => !value)}><CircleHelp size={17} aria-hidden="true" />콘솔 사용 안내<ArrowUpRight size={14} aria-hidden="true" /></button>
        <div className="sidebar-version"><span>AZURE NATIVE</span><span>API v1</span></div>
      </div>
    </aside>
    <div className="main-shell">
      <header className="topbar">
        <div className="breadcrumbs"><Blocks size={17} aria-hidden="true" /><span>Physical AI</span><ChevronRight size={13} aria-hidden="true" /><strong>{page.title}</strong></div>
        <div className="topbar-right"><Badge tone={runtimeFresh ? 'blue' : 'neutral'} dot>{runtimeFresh ? 'Azure API 응답 수신' : 'Azure 대상 · 연결 확인 필요'}</Badge>
          <div className="account-info"><span className="account-avatar" aria-hidden="true">{(account.name || account.username).slice(0, 1).toUpperCase()}</span><span title={account.username}>{account.name || account.username}<small>조직 계정</small></span></div>
          {session && <button type="button" className="icon-button" onClick={session.signOut} disabled={session.busy} aria-label="로그아웃"><LogOut size={17} aria-hidden="true" /></button>}
        </div>
      </header>
      <main id="main-content" className="main-content">
        <div className="page-heading"><div><span className="eyebrow">{page.label}</span><h1 ref={titleRef} tabIndex={-1}>{page.title}</h1><p>{page.subtitle}</p></div>
          <div className="live-scope"><Radio size={16} aria-hidden="true" /><span>ISAAC SIM<small>물리 시뮬레이션 · 합성 데이터</small></span></div>
        </div>
        {sessionExpired && <div className="session-expired" role="alert"><ShieldCheck size={19} aria-hidden="true" /><div><strong>로그인이 만료되었거나 추가 인증이 필요합니다</strong><p>새 실행과 승인을 중단했습니다. 마지막 수신 데이터는 현재 상태를 보장하지 않습니다.</p></div>
          {session && <button type="button" className="button small" disabled={session.busy} onClick={session.signIn}>다시 로그인</button>}</div>}
        {helpOpen && <section className="console-help panel" id="console-help" aria-label="콘솔 사용 안내">
          <strong>환경 저장 → 씬 활성화 → 작업 계획 → 명시적 승인 → 실행 결과 확인</strong>
          <p>Environment Studio에서 참조 템플릿 또는 고객 JSON을 검토하고 저장하세요. LIVE 씬의 버전이 런타임과 일치하면 실제 관측을 요청할 수 있습니다. 승인 전에는 이동하지 않으며, 명령 접수와 취소 요청을 완료로 표시하지 않습니다.</p>
          <p>라이브 이미지는 Isaac Sim의 인증된 합성 카메라 출력입니다. 임의 Python 실행과 재생 대체 모드는 제공하지 않습니다. Teaching Studio는 별도 통합 검증 후 활성화됩니다. 미저장 JSON은 새로고침이나 로그아웃 전에 보관하세요.</p>
        </section>}
        <ErrorNotice error={environments.error} title="저장된 환경 목록을 확인할 수 없습니다" retry={environments.refresh} />
        {view === 'live' && <ErrorNotice error={runs.error} title="기존 실행 기록을 확인할 수 없습니다" retry={runs.refresh} />}
        {view === 'live' && <FactoryLive {...runtimeControls} api={api} environments={records} environment={selected}
          onSelect={(id) => { setSelectedId(id); if (!focusedRun || isTerminal(focusedRun.status)) setFocusedRun(null); }}
          onOpenStudio={() => navigate('studio')} currentRun={currentRun} onRunChange={onRunChange} draft={runDraft} setDraft={setRunDraft}
          requestsKnown={runs.data !== null && !runs.error} />}
        {view === 'studio' && <EnvironmentStudio {...runtimeControls} api={api} environments={records} initialEnvironment={selected}
          draft={studioDraft} setDraft={setStudioDraft} onSaved={onSaved} />}
        {view === 'history' && <History api={api} runs={runs.data} loading={runs.loading} error={runs.error} refresh={runs.refresh} environments={records}
          runtime={runtime.data} runtimeFresh={runtimeFresh} onRunChange={onRunChange} />}
        {view === 'learning' && (api.learning && !sessionExpired
          ? <TeachingStudio api={api.learning} environments={records} consoleApi={api} />
          : <section className="panel"><p className="inline-note">인증된 학습 API가 연결되지 않았습니다. 익명 학습이나 대체 실행을 사용하지 않습니다.</p></section>)}
        <footer className="page-footer"><span><ShieldCheck size={13} aria-hidden="true" />구성 저장 ≠ 시뮬레이터 검증 ≠ Azure 배포 검증</span>
          <span>런타임 응답 수신: {formatDate(runtime.lastReceivedAt)}{runtime.paused ? ' · 폴링 일시 중지' : ''}</span></footer>
      </main>
    </div>
  </div>;
}
