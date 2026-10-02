import { AlertCircle, ArrowRight, Check, Factory, RefreshCw } from 'lucide-react';
import type { ReactNode } from 'react';
import { ApiError, describeError, errorGuidance, needsLogin } from '../api/errors';
import type { RunStatus } from '../api/contracts';
import { useSession } from '../auth/context';
import { runLabels } from './format';

export function Brand({ compact = false }: { compact?: boolean }) {
  return <div className={`brand ${compact ? 'compact' : ''}`}>
    <span className="brand-mark"><Factory size={23} strokeWidth={1.8} aria-hidden="true" /></span>
    <span><strong>Factory Console</strong><small>AZURE PHYSICAL AI</small></span>
  </div>;
}

export function Badge({ children, tone = 'neutral', dot = false }: {
  children: ReactNode; tone?: 'neutral' | 'blue' | 'green' | 'amber' | 'red'; dot?: boolean;
}) {
  return <span className={`badge badge-${tone}`}>{dot && <span className="status-dot" aria-hidden="true" />}{children}</span>;
}

export function RunBadge({ status }: { status: RunStatus }) {
  const tone = status === 'succeeded' ? 'green'
    : ['failed', 'timed_out'].includes(status) ? 'red'
      : ['awaiting_approval', 'cancelling'].includes(status) ? 'amber'
        : ['planning', 'running'].includes(status) ? 'blue' : 'neutral';
  return <Badge tone={tone} dot>{runLabels[status]}</Badge>;
}

export function ErrorNotice({ error, title = '작업을 완료하지 못했습니다', retry, compact = false }: {
  error: unknown; title?: string; retry?: () => void; compact?: boolean;
}) {
  const session = useSession();
  if (!error) return null;
  const guidance = errorGuidance(error);
  return <div className={`error-notice ${compact ? 'compact' : ''}`} role="alert">
    <AlertCircle size={18} aria-hidden="true" />
    <div className="error-content">
      <strong>{title}</strong>
      <p>{describeError(error)}</p>
      {guidance && <p>{guidance}</p>}
      {error instanceof ApiError && <code>{error.code}{error.status > 0 ? ` · HTTP ${error.status}` : ''}</code>}
      {error instanceof ApiError && error.details.length > 0 && <details>
        <summary>서버 검증 상세 ({error.details.length})</summary>
        <pre>{JSON.stringify(error.details, null, 2)}</pre>
      </details>}
      {(retry || (needsLogin(error) && session)) && <div className="button-row">
        {retry && <button type="button" className="button small secondary" onClick={retry}><RefreshCw size={14} aria-hidden="true" />다시 확인</button>}
        {needsLogin(error) && session && <button type="button" className="button small" disabled={session.busy} onClick={session.signIn}>다시 로그인<ArrowRight size={14} aria-hidden="true" /></button>}
      </div>}
    </div>
  </div>;
}

export function EmptyState({ icon, title, children, action }: {
  icon: ReactNode; title: string; children: ReactNode; action?: ReactNode;
}) {
  return <div className="empty-state">
    <div className="empty-icon" aria-hidden="true">{icon}</div>
    <h3>{title}</h3><p>{children}</p>{action}
  </div>;
}

export function StepLabel({ number, title, done, active }: {
  number: string; title: string; done: boolean; active?: boolean;
}) {
  return <div className={`step ${done ? 'done' : active ? 'active' : ''}`}>
    <span className="step-number">{done ? <Check size={15} aria-label="완료" /> : number}</span>
    <span>{title}</span>
  </div>;
}

export function Loading({ children = '서버에서 확인 중…' }: { children?: ReactNode }) {
  return <div className="loading" role="status"><RefreshCw size={16} className="spin" aria-hidden="true" />{children}</div>;
}

export function FieldValue({ label, children }: { label: string; children: ReactNode }) {
  return <div className="field-value"><dt>{label}</dt><dd>{children}</dd></div>;
}
