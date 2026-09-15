import { useState, type Dispatch, type SetStateAction } from 'react';
import { ArrowRight, Bot, Camera, FileJson2, ScanLine, Send, ShieldCheck } from 'lucide-react';
import {
  environmentMode, isTerminal, matchesRuntime,
  type Camera as CameraType, type ConsoleApi, type CreateRunInput, type EnvironmentRecord, type RunRecord,
} from '../api/contracts';
import { isAbort } from '../api/errors';
import { useRequestScope } from '../hooks/useRequestScope';
import { Badge, EmptyState, ErrorNotice } from '../ui/common';
import { shortId } from '../ui/format';
import { LiveCamera } from './LiveCamera';
import { RunDetail } from './RunDetail';
import { RuntimePanel, type RuntimeControlsProps } from './RuntimePanel';

export interface RunDraft {
  instruction: string;
  attempt: CreateRunInput | null;
}

export interface FactoryLiveProps extends RuntimeControlsProps {
  api: ConsoleApi;
  environments: EnvironmentRecord[];
  onSelect(id: string): void;
  onOpenStudio(): void;
  currentRun: RunRecord | null;
  onRunChange(run: RunRecord): void;
  draft: RunDraft;
  setDraft: Dispatch<SetStateAction<RunDraft>>;
  requestsKnown: boolean;
}

export function FactoryLive(props: FactoryLiveProps) {
  const { api, environment, environments, runtime, runtimeFresh, currentRun, onRunChange, disabled } = props;
  const [camera, setCamera] = useState<CameraType>('overview');
  const live = environment && environmentMode(environment) === 'live';
  const ready = Boolean(live && runtimeFresh && matchesRuntime(runtime, environment) && !disabled);
  const reason = !environment ? 'Environment Studio에서 구성을 저장하고 씬을 활성화하세요.'
    : !live ? 'REPLAY 또는 확인되지 않은 구성에서는 LIVE 카메라를 연결하지 않습니다.'
      : '선택한 저장 버전이 Isaac Sim에 로드된 뒤 실제 카메라 영상을 연결합니다.';

  return <>
    <div className="workspace-toolbar">
      <div className="environment-picker"><label htmlFor="live-environment">작업 환경</label>
        <select id="live-environment" value={environment?.environment_id ?? ''} disabled={!environments.length} onChange={(event) => props.onSelect(event.target.value)}>
          {!environments.length && <option value="">저장된 환경 없음</option>}
          {environments.map((item) => <option key={item.environment_id} value={item.environment_id}>{item.display_name}</option>)}
        </select>
        {environment && <Badge tone={live ? 'blue' : 'amber'}>{environmentMode(environment)?.toUpperCase() ?? '모드 미확인'}</Badge>}
      </div>
      <button type="button" className="text-button" onClick={props.onOpenStudio}><FileJson2 size={15} aria-hidden="true" />환경 편집<ArrowRight size={14} aria-hidden="true" /></button>
    </div>
    <div className="factory-grid">
      <div className="factory-primary">
        <section className="panel live-panel" aria-labelledby="live-view-title">
          <div className="panel-heading"><div className="title-icon"><ScanLine size={19} aria-hidden="true" /><h2 id="live-view-title">공장 라이브 뷰</h2></div>
            <div className="segmented-control" role="group" aria-label="카메라 선택">
              <button type="button" aria-pressed={camera === 'overview'} onClick={() => setCamera('overview')}>전체 셀</button>
              <button type="button" aria-pressed={camera === 'inspection'} onClick={() => setCamera('inspection')}>검사 지점</button>
            </div>
          </div>
          {environment ? <LiveCamera key={`${environment.environment_id}-${environment.revision}-${camera}`} api={api} environment={environment} camera={camera} enabled={ready} unavailableReason={reason} />
            : <div className="camera-viewport"><div className="camera-placeholder"><Camera size={38} strokeWidth={1.2} aria-hidden="true" /><strong>첫 번째 환경을 연결하세요</strong><p>{reason}</p><button type="button" className="button" onClick={props.onOpenStudio}>Environment Studio 열기<ArrowRight size={15} aria-hidden="true" /></button></div></div>}
        </section>
        <section className="panel instruction-panel" aria-labelledby="instruction-title">
          <div className="panel-heading"><div className="title-icon"><Bot size={19} aria-hidden="true" /><h2 id="instruction-title">작업 지시</h2></div><Badge>사람의 승인 필요</Badge></div>
          <RunComposer api={api} environment={environment} ready={ready && Boolean(runtime?.agent.configured) && props.requestsKnown} currentRun={currentRun}
            onRunChange={onRunChange} draft={props.draft} setDraft={props.setDraft} />
        </section>
        <section className="workflow-guide" aria-label="안전한 실행 흐름">
          <span><b>01</b><strong>환경 구성</strong><small>고객 JSON 저장</small></span><ArrowRight size={15} aria-hidden="true" />
          <span><b>02</b><strong>실제 관측</strong><small>Isaac Sim + Foundry</small></span><ArrowRight size={15} aria-hidden="true" />
          <span><b>03</b><strong>검토와 승인</strong><small>명시적인 실행 허가</small></span><ArrowRight size={15} aria-hidden="true" />
          <span><b>04</b><strong>결과 확인</strong><small>명령 ID와 증거 검토</small></span>
        </section>
      </div>
      <aside className="factory-secondary" aria-label="런타임과 에이전트">
        {currentRun && <RunDetail key={currentRun.id} api={api} initialRun={currentRun} runtime={runtime} runtimeFresh={runtimeFresh && !disabled}
          environment={environments.find((item) => item.environment_id === currentRun.environment_id) ?? null} onChange={onRunChange} />}
        <RuntimePanel {...props} />
        {!currentRun && <section className="panel agent-idle"><div className="panel-heading"><div className="title-icon"><ShieldCheck size={19} aria-hidden="true" /><h2>에이전트 계획과 실행</h2></div></div>
            <EmptyState icon={<Bot size={28} />} title="검토할 계획이 없습니다">환경을 활성화한 뒤 작업을 요청하면 관측, 계획 요약과 실제 응답 ID가 이곳에 표시됩니다. 자동 승인하지 않습니다.</EmptyState>
          </section>}
      </aside>
    </div>
  </>;
}

function RunComposer({ api, environment, ready, currentRun, onRunChange, draft, setDraft }: {
  api: ConsoleApi; environment: EnvironmentRecord | null; ready: boolean; currentRun: RunRecord | null;
  onRunChange(run: RunRecord): void; draft: RunDraft; setDraft: Dispatch<SetStateAction<RunDraft>>;
}) {
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const startRequest = useRequestScope();
  const activeRun = currentRun && !isTerminal(currentRun.status);

  const submit = async () => {
    if (!environment || !ready || activeRun || submitting || !draft.instruction.trim()) return;
    const previous = draft.attempt;
    const input: CreateRunInput = previous &&
      previous.environment_id === environment.environment_id && previous.revision === environment.revision &&
      previous.instruction === draft.instruction ? previous : {
        request_id: crypto.randomUUID(),
        environment_id: environment.environment_id,
        revision: environment.revision,
        instruction: draft.instruction,
      };
    setDraft((value) => ({ ...value, attempt: input }));
    setSubmitting(true);
    setError(null);
    const request = startRequest();
    try {
      const run = await api.createRun(input, request.signal);
      if (!request.current()) return;
      onRunChange(run);
      setDraft((value) => ({ ...value, attempt: null }));
    } catch (failure) {
      if (request.current() && !isAbort(failure)) setError(failure);
    } finally {
      if (request.current()) setSubmitting(false);
      request.finish();
    }
  };

  return <form onSubmit={(event) => { event.preventDefault(); void submit(); }}>
    <label htmlFor="run-instruction">검사·분류 작업을 설명하세요</label>
    <textarea id="run-instruction" value={draft.instruction} rows={3} disabled={submitting || Boolean(activeRun)} required
      aria-describedby="run-instruction-help"
      placeholder="예: 이 부품을 검사하고 결함이 보이면 불량 분류 스테이션으로 이동하는 계획을 세워 주세요."
      onChange={(event) => { const instruction = event.target.value; setDraft((value) => ({ ...value, instruction })); }} />
    <div className="composer-footer"><p id="run-instruction-help"><ShieldCheck size={15} aria-hidden="true" />요청은 계획만 생성합니다. 로봇 이동은 승인 후 시작됩니다.</p>
      <button type="submit" className="button" disabled={!ready || !draft.instruction.trim() || submitting || Boolean(activeRun)}><Send size={15} aria-hidden="true" />{submitting ? '관측·계획 요청 중…' : error && draft.attempt ? '동일 요청 다시 확인' : '관측하고 계획 요청'}</button>
    </div>
    {!ready && <p className="form-hint">LIVE 씬의 저장 버전이 런타임과 일치하고 Foundry 설정이 확인되어야 요청할 수 있습니다.</p>}
    {activeRun && <p className="form-hint">진행 중인 실행이 있습니다. 기존 계획을 검토하거나 취소를 확인한 뒤 새 작업을 요청하세요.</p>}
    {submitting && <div className="inline-note" role="status"><strong>계획 응답을 기다리고 있습니다 · 로봇 이동 없음</strong><p>실행 ID가 수신되면 취소할 수 있습니다. 화면을 나가도 서버에 접수된 요청이 취소되는 것은 아닙니다.</p></div>}
    {draft.attempt && <small className="request-id" title={draft.attempt.request_id}>request_id: {shortId(draft.attempt.request_id)} · 같은 입력의 재시도에 동일 ID 사용</small>}
    <ErrorNotice error={error} title="계획 요청 결과를 확인할 수 없습니다" />
  </form>;
}
