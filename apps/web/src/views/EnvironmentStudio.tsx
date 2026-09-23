import { useCallback, useEffect, useMemo, useRef, useState, type Dispatch, type SetStateAction } from 'react';
import { Braces, Check, FileJson2, FileUp, Info, RotateCcw, Save, ShieldCheck } from 'lucide-react';
import type { ConsoleApi, EnvironmentRecord } from '../api/contracts';
import { ApiError, describeError, isAbort } from '../api/errors';
import {
  getEnvironmentValidator, draftFromRecord, expectedRevision, isDraftDirty,
  MAX_DOCUMENT_BYTES, validateDocument, type StudioDraft, type ValidationIssue,
} from '../environment/validation';
import { usePolling } from '../hooks/usePolling';
import { useRequestScope } from '../hooks/useRequestScope';
import { Badge, ErrorNotice, Loading } from '../ui/common';
import { formatDate, shortId } from '../ui/format';
import { RuntimePanel, type RuntimeControlsProps } from './RuntimePanel';
import { CustomerScenarioPlanner } from '../environment/CustomerScenarioPlanner';

interface StudioProps extends Omit<RuntimeControlsProps, 'environment'> {
  api: ConsoleApi;
  environments: EnvironmentRecord[];
  initialEnvironment: EnvironmentRecord | null;
  draft: StudioDraft | null;
  setDraft: Dispatch<SetStateAction<StudioDraft | null>>;
  onSaved(record: EnvironmentRecord): void;
}

export function EnvironmentStudio(props: StudioProps) {
  const { api, environments, initialEnvironment, draft, setDraft, onSaved, disabled } = props;
  const schemaLoad = useCallback((signal: AbortSignal) => api.getEnvironmentSchema(signal), [api]);
  const templatesLoad = useCallback((signal: AbortSignal) => api.getTemplates(signal), [api]);
  const schema = usePolling(schemaLoad);
  const templates = usePolling(templatesLoad);
  const [selectedId, setSelectedId] = useState(initialEnvironment?.environment_id ?? '');
  const [templateIndex, setTemplateIndex] = useState('');
  const [issues, setIssues] = useState<ValidationIssue[] | null>(null);
  const [error, setError] = useState<unknown>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);
  const [importing, setImporting] = useState(false);
  const [reloading, setReloading] = useState(false);
  const editor = useRef<HTMLTextAreaElement>(null);
  const fileInput = useRef<HTMLInputElement>(null);
  const startRequest = useRequestScope();
  const dirty = isDraftDirty(draft);
  const busy = saving || importing || reloading;
  const compiled = useMemo(() => {
    if (!schema.data) return { validator: null, error: null };
    try {
      return { validator: getEnvironmentValidator(schema.data), error: null };
    } catch (failure) {
      return { validator: null, error: new Error(`API 스키마를 검사기로 준비하지 못했습니다: ${describeError(failure)}`) };
    }
  }, [schema.data]);

  useEffect(() => {
    if (!draft) setDraft(initialEnvironment ? draftFromRecord(initialEnvironment) : { text: '', savedText: '', base: null, source: '새 고객 구성' });
  }, [draft, setDraft, initialEnvironment]);

  const replaceDraft = (next: StudioDraft) => {
    if (dirty && !window.confirm('저장하지 않은 JSON 변경 내용이 있습니다. 현재 편집 내용을 바꾸시겠습니까?')) return false;
    setDraft(next);
    setIssues(null);
    setError(null);
    setNotice(null);
    return true;
  };

  const validate = () => {
    const result = validateDocument(draft?.text ?? '', compiled.validator);
    setIssues(result.issues);
    setNotice(result.issues.length === 0 ? '구조 검사를 통과했습니다. 서버의 의미 검증과 물리 시뮬레이터 검증은 아직 수행되지 않았습니다.' : null);
    return result;
  };

  const save = async () => {
    if (!draft || busy || disabled) return;
    const result = validate();
    if (!result.document || result.issues.length) {
      editor.current?.focus();
      return;
    }
    setSaving(true);
    setError(null);
    setNotice(null);
    const request = startRequest();
    try {
      const record = await api.saveEnvironment(draft.text, expectedRevision(draft, result.document), request.signal);
      if (!request.current()) return;
      setDraft({ text: draft.text, savedText: draft.text, base: record, source: '저장된 고객 구성' });
      setSelectedId(record.environment_id);
      onSaved(record);
      setNotice('원본 JSON을 저장했습니다. 이 저장은 시뮬레이터 검증이 아닙니다. 저장된 씬을 활성화하고 런타임을 확인하세요.');
    } catch (failure) {
      if (request.current() && !isAbort(failure)) setError(failure);
    } finally {
      if (request.current()) setSaving(false);
      request.finish();
    }
  };

  const importFile = async (file: File) => {
    if (busy) return;
    if (!file.name.toLowerCase().endsWith('.json')) {
      setError(new Error('JSON 파일만 가져올 수 있습니다. Python 또는 실행 파일은 업로드하지 않습니다.'));
      return;
    }
    if (file.size > MAX_DOCUMENT_BYTES) {
      setError(new Error('가져올 JSON 파일은 1 MiB 이하여야 합니다.'));
      return;
    }
    setImporting(true);
    const request = startRequest();
    try {
      const text = await file.text();
      if (request.current()) replaceDraft({ text, savedText: draft?.savedText ?? '', base: draft?.base ?? null, source: `가져온 JSON: ${file.name}` });
    } catch (failure) {
      if (request.current()) setError(new Error(`JSON 파일을 읽지 못했습니다: ${describeError(failure)}`));
    } finally {
      if (request.current()) setImporting(false);
      request.finish();
    }
  };

  const reloadLatest = async () => {
    if (!draft?.base || busy) return;
    if (dirty && !window.confirm('서버의 최신 구성을 불러오면 현재 편집 내용을 잃습니다. 원본 JSON을 별도로 보관한 뒤 계속하세요.')) return;
    setReloading(true);
    const id = draft.base.environment_id;
    const request = startRequest();
    try {
      const list = await api.getEnvironments(request.signal);
      if (!request.current()) return;
      const latest = list.items.find((item) => item.environment_id === id);
      if (!latest) throw new Error('이 환경이 더 이상 서버 목록에 없습니다. 편집 내용은 유지합니다.');
      setDraft(draftFromRecord(latest));
      onSaved(latest);
      setIssues(null);
      setError(null);
      setNotice('서버의 최신 저장 버전을 불러왔습니다. 필요한 변경 내용을 다시 적용하세요.');
    } catch (failure) {
      if (request.current() && !isAbort(failure)) setError(failure);
    } finally {
      if (request.current()) setReloading(false);
      request.finish();
    }
  };

  return <div className="studio-grid">
    <div className="studio-primary">
      <CustomerScenarioPlanner onApply={(document, source) => replaceDraft({
        text: JSON.stringify(document, null, 2), savedText: '', base: null, source,
      })} />
      <section className="panel studio-sources" aria-labelledby="studio-sources-title">
        <div className="panel-heading"><div className="title-icon"><FileJson2 size={19} aria-hidden="true" /><h2 id="studio-sources-title">환경 문서 선택</h2></div><Badge>JSON Schema 2020-12</Badge></div>
        <div className="source-options">
          <div><label htmlFor="saved-environment">저장된 환경</label><div className="joined-field">
            <select id="saved-environment" value={selectedId} disabled={busy} onChange={(event) => setSelectedId(event.target.value)}>
              <option value="">환경 선택</option>{environments.map((item) => <option key={item.environment_id} value={item.environment_id}>{item.display_name}</option>)}
            </select><button type="button" className="button secondary" disabled={!selectedId || busy} onClick={() => {
              const record = environments.find((item) => item.environment_id === selectedId);
              if (record) replaceDraft(draftFromRecord(record));
            }}>불러오기</button>
          </div></div>
          <div><label htmlFor="environment-template">API 참조 템플릿</label><div className="joined-field">
            <select id="environment-template" value={templateIndex} disabled={busy || !templates.data} onChange={(event) => setTemplateIndex(event.target.value)}>
              <option value="">템플릿 선택</option>{templates.data?.items.map((item, index) => <option key={`${item.name}-${index}`} value={index}>{item.name}</option>)}
            </select><button type="button" className="button secondary" disabled={templateIndex === '' || busy} onClick={() => {
              const template = templates.data?.items[Number(templateIndex)];
              if (template) replaceDraft({ text: JSON.stringify(template.document, null, 2), savedText: '', base: null, source: `참조 템플릿: ${template.name}` });
            }}>템플릿 적용</button>
          </div></div>
        </div>
        <p className="source-help"><Info size={14} aria-hidden="true" />참조 템플릿은 고객 실제 설비 데이터가 아닙니다. 환경 ID, 좌표와 제한값을 검토하세요.</p>
        <ErrorNotice error={templates.error} title="템플릿을 불러오지 못했습니다" retry={templates.refresh} compact />
      </section>
      <section className="panel json-panel" aria-labelledby="json-editor-title">
        <div className="panel-heading"><div className="title-icon"><Braces size={19} aria-hidden="true" /><h2 id="json-editor-title">환경 JSON 편집기</h2></div>
          <Badge tone={dirty ? 'amber' : 'neutral'}>{dirty ? '저장하지 않은 변경' : draft?.base ? '저장된 구성' : '새 문서'}</Badge>
        </div>
        <div className="editor-toolbar"><span>{draft?.source}</span><button type="button" className="text-button" disabled={busy} onClick={() => fileInput.current?.click()}><FileUp size={15} aria-hidden="true" />JSON 파일 가져오기</button>
          <input ref={fileInput} type="file" id="json-file" className="visually-hidden" accept=".json,application/json" aria-label="환경 JSON 파일" disabled={busy}
            onChange={(event) => { const file = event.target.files?.[0]; event.target.value = ''; if (file) void importFile(file); }} />
        </div>
        <label htmlFor="environment-json" className="visually-hidden">고객 환경 JSON 원본</label>
        <textarea ref={editor} id="environment-json" className="json-editor" spellCheck={false} autoCapitalize="off" autoCorrect="off" rows={24}
          aria-describedby="json-help json-revision" aria-invalid={Boolean(issues?.length)} value={draft?.text ?? ''} disabled={busy}
          placeholder="API의 참조 템플릿을 선택하거나 고객 JSON 파일을 가져오세요."
          onChange={(event) => { const text = event.target.value; setDraft((value) => value ? { ...value, text } : { text, savedText: '', base: null, source: '새 고객 구성' }); setIssues(null); setNotice(null); }} />
        <div className="editor-status"><span>JSON · UTF-8 · 좌표 단위 m</span><span id="json-revision" title={draft?.base?.revision}>불러온 버전 {draft?.base ? shortId(draft.base.revision) : '없음 (신규)'}</span></div>
        <p id="json-help" className="editor-help">편집한 원문을 <code>document_json</code>으로 그대로 전송합니다. 동일 환경 수정 시 불러온 revision을 함께 보내며, 충돌 시 자동 덮어쓰지 않습니다. ID를 바꾸면 새 환경으로 저장합니다.</p>
        {schema.loading && !schema.data && <Loading>서버 환경 스키마를 불러오는 중…</Loading>}
        <ErrorNotice error={schema.error ?? compiled.error} title="환경 스키마를 사용할 수 없습니다" retry={schema.refresh} compact />
        {issues && issues.length > 0 && <div className="validation-errors" role="alert"><strong>JSON 검사: {issues.length}개 항목을 확인하세요</strong>
          <ul>{issues.map((issue, index) => <li key={`${issue.path}-${index}`}><button type="button" onClick={() => { editor.current?.focus(); if (issue.offset !== undefined) editor.current?.setSelectionRange(issue.offset, issue.offset + 1); }}><code>{issue.path}</code> {issue.message}</button></li>)}</ul>
        </div>}
        {notice && <div className="inline-note" role="status"><Check size={16} aria-hidden="true" /><p>{notice}</p></div>}
        <ErrorNotice error={error} title={error instanceof ApiError && error.status === 409 ? '저장 버전 충돌 · 원본 JSON 유지됨' : '환경 작업 실패'} />
        {error instanceof ApiError && error.status === 409 && draft?.base && <button type="button" className="button secondary conflict-reload" disabled={busy} onClick={() => void reloadLatest()}><RotateCcw size={15} aria-hidden="true" />최신 버전 불러오기 (편집 내용 교체)</button>}
        <div className="editor-actions"><span className="muted small-text">{draft?.base ? `마지막 서버 저장: ${formatDate(draft.base.updated_at)}` : '아직 서버에 저장되지 않았습니다'}</span>
          <div className="button-row"><button type="button" className="button secondary" disabled={busy || !compiled.validator} onClick={validate}><Check size={15} aria-hidden="true" />JSON 검사</button>
            <button type="button" className="button" disabled={busy || disabled || !compiled.validator || !draft?.text.trim()} onClick={() => void save()}><Save size={15} aria-hidden="true" />{saving ? '저장 중…' : 'JSON 저장'}</button></div>
        </div>
      </section>
    </div>
    <aside className="studio-secondary" aria-label="스키마 도움말과 활성화">
      <RuntimePanel {...props} environment={draft?.base ?? null} disabled={disabled || dirty || busy} />
      {dirty && <p className="form-hint">활성화는 저장된 버전에만 적용됩니다. 편집 내용을 먼저 저장하세요.</p>}
      <section className="panel schema-help" aria-labelledby="schema-help-title">
        <div className="panel-heading"><div className="title-icon"><Info size={19} aria-hidden="true" /><h2 id="schema-help-title">환경 스키마 안내</h2></div></div>
        <dl>
          <div><dt>scene</dt><dd>서버에 등록된 template_id, robot_profile과 재현 가능한 seed를 지정합니다.</dd></div>
          <div><dt>workspace / stations</dt><dd>작업 영역과 source, inspection, accepted, rejected 스테이션의 3차원 좌표를 미터로 지정합니다.</dd></div>
          <div><dt>workflow</dt><dd>공급 → 검사 → 정상/불량 분류 경로를 실제 스테이션 ID에 연결합니다.</dd></div>
          <div><dt>execution</dt><dd><b>LIVE</b>는 실시간 시뮬레이터용입니다. <b>REPLAY</b>는 구성만 편집할 수 있고 이 API로 활성화할 수 없습니다.</dd></div>
          <div><dt>limits</dt><dd>요청 속도와 하중 제한입니다. 로봇 컨트롤러의 안전 제한을 덮어쓰지 않습니다.</dd></div>
        </dl>
        {schema.data && <details><summary>API가 제공한 전체 JSON Schema</summary><pre>{JSON.stringify(schema.data, null, 2)}</pre></details>}
      </section>
      <div className="extension-boundary"><ShieldCheck size={19} aria-hidden="true" /><div><strong>검토형 Python 씬 확장</strong><Badge>현재 미제공</Badge><p>이 버전은 선언형 JSON만 지원합니다. Python 업로드·실행 및 정책 학습 API는 연결되어 있지 않습니다.</p></div></div>
    </aside>
  </div>;
}
