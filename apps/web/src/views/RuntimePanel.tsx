import { ArrowUpRight, Box, Cpu, Database, RefreshCw, ShieldCheck } from 'lucide-react';
import { environmentMode, matchesRuntime, type Activation, type EnvironmentRecord, type RuntimeInfo } from '../api/contracts';
import { Badge, ErrorNotice, FieldValue } from '../ui/common';
import { shortId, runtimeLabels } from '../ui/format';

export interface ActivationState {
  submitting: boolean;
  receipt: Activation | null;
  error: unknown;
  timedOut: boolean;
}

export interface RuntimeControlsProps {
  runtime: RuntimeInfo | null;
  runtimeError: unknown;
  runtimeFresh: boolean;
  environment: EnvironmentRecord | null;
  activation: ActivationState;
  onActivate(environment: EnvironmentRecord): void;
  onRefresh(): void;
  disabled?: boolean;
}

export function RuntimePanel({ runtime, runtimeError, runtimeFresh, environment, activation, onActivate, onRefresh, disabled = false }: RuntimeControlsProps) {
  const matching = runtimeFresh && matchesRuntime(runtime, environment);
  const mode = environment ? environmentMode(environment) : null;
  const waiting = Boolean(activation.receipt && !(runtimeFresh && matchesRuntime(runtime, activation.receipt)) && !activation.timedOut);
  const currentActivation = activation.receipt?.environment_id === environment?.environment_id ? activation.receipt : null;
  const readyMessage = matching ? '선택한 저장 버전과 런타임이 일치합니다.' : 'JSON 저장만으로 물리 시뮬레이터가 준비되지는 않습니다.';

  return <section className="panel runtime-panel" aria-labelledby="runtime-title">
    <div className="panel-heading"><div className="title-icon"><Box size={19} aria-hidden="true" /><h2 id="runtime-title">실행 환경</h2></div>
      <button type="button" className="icon-button" onClick={onRefresh} aria-label="런타임 상태 새로고침"><RefreshCw size={16} aria-hidden="true" /></button>
    </div>
    <dl className="service-list">
      <div><dt><Box size={16} aria-hidden="true" />Isaac Sim</dt><dd><Badge tone={!runtimeFresh ? 'amber' : runtime?.simulation.status === 'ready' ? 'green' : 'amber'} dot>
        {!runtimeFresh ? '현재 상태 미확인' : runtime ? runtimeLabels[runtime.simulation.status] : '연결 확인 전'}
      </Badge></dd></div>
      <div><dt><Cpu size={16} aria-hidden="true" />Foundry Agent</dt><dd><Badge tone={runtime?.agent.configured ? 'blue' : 'neutral'}>{runtime?.agent.configured ? '설정됨 · 호출 미검증' : '설정 확인 전'}</Badge></dd></div>
      <div><dt><Database size={16} aria-hidden="true" />Evidence store</dt><dd>{runtime ? 'Azure Cosmos / Blob' : '확인 전'}</dd></div>
    </dl>
    {runtime?.simulation.message && <p className="runtime-message">{runtime.simulation.message}</p>}
    <ErrorNotice error={runtimeError} title="런타임 상태를 확인할 수 없습니다" compact />
    {environment && <dl className="runtime-revision">
      <FieldValue label="선택한 환경"><span>{environment.display_name}</span><code>{environment.environment_id}</code></FieldValue>
      <FieldValue label="저장 버전"><code title={environment.revision}>{shortId(environment.revision)}</code></FieldValue>
      <FieldValue label="런타임 버전"><code title={runtime?.simulation.revision ?? undefined}>{runtime?.simulation.revision ? shortId(runtime.simulation.revision) : '확인 전'}</code></FieldValue>
    </dl>}
    <p className="muted small-text">{readyMessage}</p>
    {mode === 'replay' ? <div className="inline-note warning"><strong>REPLAY 구성</strong><p>실시간 API는 재생 구성을 활성화하지 않습니다. LIVE 구성으로 명시적으로 변경하고 저장하세요.</p></div>
      : <button type="button" className="button secondary full-width" disabled={disabled || !environment || mode !== 'live' || activation.submitting || waiting || matching}
        onClick={() => environment && onActivate(environment)}>
        <ArrowUpRight size={16} aria-hidden="true" />{matching ? '저장 버전 활성 상태' : activation.submitting ? '활성화 요청 중…' : waiting ? '활성화 확인 중…' : '저장된 씬 활성화'}
      </button>}
    {activation.receipt && <div className="activation-state" role="status">
      <strong>{runtimeFresh && matchesRuntime(runtime, activation.receipt) ? '요청한 버전의 런타임 준비 확인됨'
        : activation.timedOut ? '활성화 응답 후 준비 확인이 지연되고 있습니다'
          : '요청 접수됨 · 아직 준비 완료가 아닙니다'}</strong>
      {!currentActivation && <span>대상 환경: {activation.receipt.environment_id}</span>}
      <code>activation_id: {activation.receipt.activation_id}</code>
      {activation.timedOut && <span>자동 성공 처리하지 않습니다. 런타임 상태와 서버 메시지를 확인하세요.</span>}
    </div>}
    <ErrorNotice error={activation.error} title="씬 활성화 실패" compact />
    <div className="verification-note"><ShieldCheck size={16} aria-hidden="true" /><span>{runtime?.release_ready ? '서버 release_ready 보고됨' : '릴리스 준비 상태 미검증'}<small>설정 여부는 Azure/GPU 동작 검증이 아닙니다.</small></span></div>
  </section>;
}
