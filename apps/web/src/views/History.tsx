import { useState } from 'react';
import { ArrowRight, Clock3, RefreshCw, Search } from 'lucide-react';
import type { ConsoleApi, EnvironmentRecord, RunRecord, Runs, RunStatus, RuntimeInfo } from '../api/contracts';
import { Badge, EmptyState, ErrorNotice, Loading, RunBadge } from '../ui/common';
import { formatDate, runLabels, shortId } from '../ui/format';
import { RunDetail } from './RunDetail';

export function History({ api, runs, loading, error, refresh, environments, runtime, runtimeFresh, onRunChange }: {
  api: ConsoleApi; runs: Runs | null; loading: boolean; error: unknown; refresh(): void;
  environments: EnvironmentRecord[]; runtime: RuntimeInfo | null; runtimeFresh: boolean; onRunChange(run: RunRecord): void;
}) {
  const [status, setStatus] = useState<RunStatus | ''>('');
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<RunRecord | null>(null);
  const list = (runs?.items ?? [])
    .filter((run) => (!status || run.status === status) && `${run.id} ${run.environment_id} ${run.instruction}`.toLocaleLowerCase('ko').includes(search.toLocaleLowerCase('ko')))
    .sort((left, right) => Date.parse(right.created_at) - Date.parse(left.created_at));

  return <div className={`history-layout ${selected ? 'with-detail' : ''}`}>
    <section className="panel history-panel" aria-labelledby="history-table-title">
      <div className="panel-heading"><div className="title-icon"><Clock3 size={19} aria-hidden="true" /><h2 id="history-table-title">내 실행 기록</h2><Badge>{runs ? `${runs.items.length}건` : '확인 전'}</Badge></div>
        <button type="button" className="button small secondary" disabled={loading} onClick={refresh}><RefreshCw size={14} aria-hidden="true" />새로고침</button>
      </div>
      <div className="history-filters"><div className="search-field"><Search size={16} aria-hidden="true" /><label htmlFor="run-search" className="visually-hidden">실행 ID, 환경 또는 작업 검색</label><input type="search" id="run-search" placeholder="실행 ID, 환경 또는 작업 검색" value={search} onChange={(event) => setSearch(event.target.value)} /></div>
        <label htmlFor="run-status" className="visually-hidden">실행 상태 필터</label><select id="run-status" value={status} onChange={(event) => {
          const value = event.target.value;
          if (value === '' || Object.hasOwn(runLabels, value)) setStatus(value as RunStatus | '');
        }}><option value="">모든 상태</option>{Object.entries(runLabels).map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select>
      </div>
      <ErrorNotice error={error} title="실행 목록을 갱신하지 못했습니다" retry={refresh} />
      {Boolean(error) && runs && <p className="form-hint">마지막으로 수신한 기록입니다. 현재 서버 상태를 보장하지 않습니다.</p>}
      {loading && !runs && <Loading>현재 계정의 실행 기록을 불러오는 중…</Loading>}
      {runs && list.length === 0 ? <EmptyState icon={<Clock3 size={30} />} title={runs.items.length ? '조건에 맞는 실행이 없습니다' : '아직 실행 기록이 없습니다'}>{runs.items.length ? '검색어나 상태 필터를 변경하세요.' : 'Factory Live에서 요청한 실제 작업만 표시합니다. 예제 실행 기록은 생성하지 않습니다.'}</EmptyState> : list.length > 0 && <div className="table-scroll"><table>
        <caption className="visually-hidden">현재 소유자의 API 실행 기록</caption>
        <thead><tr><th scope="col">상태</th><th scope="col">작업 / 환경</th><th scope="col">요청 시각</th><th scope="col"><span className="visually-hidden">실행 상세</span></th></tr></thead>
        <tbody>{list.map((run) => <tr key={run.id} className={selected?.id === run.id ? 'selected-row' : ''}>
          <td><RunBadge status={run.status} /></td>
          <td><strong className="instruction-cell">{run.instruction}</strong><small>{run.environment_id} <code title={run.id}>{shortId(run.id)}</code></small></td>
          <td><time dateTime={run.created_at}>{formatDate(run.created_at)}</time></td>
          <td><button type="button" className="icon-button" aria-label={`실행 ${run.id} 상세 보기`} aria-pressed={selected?.id === run.id} onClick={() => setSelected(run)}><ArrowRight size={17} aria-hidden="true" /></button></td>
        </tr>)}</tbody>
      </table></div>}
      <div className="history-footnote">API가 반환한 기록만 표시 · 브라우저 현지 시간 · 성공은 최종 <code>succeeded</code> 상태 기준</div>
    </section>
    {selected && <aside className="history-detail" aria-label="선택한 실행 상세">
      <button type="button" className="text-button detail-close" onClick={() => setSelected(null)}>상세 닫기</button>
      <RunDetail key={selected.id} api={api} initialRun={selected} runtime={runtime} runtimeFresh={runtimeFresh}
        environment={environments.find((item) => item.environment_id === selected.environment_id) ?? null} onChange={onRunChange} />
    </aside>}
  </div>;
}
