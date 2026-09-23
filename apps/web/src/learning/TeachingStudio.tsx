import { useCallback, useEffect, useRef, useState } from 'react';
import { BookOpen, Bot, Plus, RefreshCw, ShieldCheck, Upload } from 'lucide-react';
import type { ConsoleApi, EnvironmentRecord } from '../api/contracts';
import { usePolling } from '../hooks/usePolling';
import { Badge, EmptyState, ErrorNotice, FieldValue, Loading } from '../ui/common';
import { formatDate } from '../ui/format';
import { LiveCamera } from '../views/LiveCamera';
import { LearningJobPanel } from './LearningJobPanel';
import { PolicyComparison } from './PolicyComparison';
import { TeachingControls } from './TeachingControls';
import {
  sourceLabel, type CoachResponse, type CreateProjectBody, type Evaluation, type Job,
  type LearningApi, type LearningRecord, type Project, type Resource, type Teaching,
} from './contracts';
import './learning.css';
import { defaultTeachingCase, savedTeachingCases, splitLabel } from './teachingCases';

export function TeachingStudio({ api, environments, consoleApi }: {
  api: LearningApi; environments: EnvironmentRecord[]; consoleApi?: ConsoleApi;
}) {
  const load = useCallback((signal: AbortSignal) => api.capabilities(signal), [api]);
  const capability = usePolling(load);
  const list = useCallback((signal: AbortSignal) => api.projects(signal), [api]);
  const projects = usePolling(list, { active: capability.data?.enabled === true });
  const [selected, setSelected] = useState<Resource<Project> | null>(null);
  const [create, setCreate] = useState(false);
  useEffect(() => {
    if (selected || !projects.data) return;
    const id = new URLSearchParams(window.location.search).get('learning_project');
    const project = projects.data.items.find((entry) => entry.item.id === id);
    if (project) setSelected(project);
  }, [selected, projects.data]);
  return <div className="learning-studio">
    <div className="learning-boundary"><BookOpen size={21} aria-hidden="true" /><div><strong>작업 시연 → 고정 데이터 → 실제 정책 학습 → 같은 시험 → 검토된 정책</strong>
      <p>현재 상용 후보는 SmolVLA입니다. 라이선스가 미확인된 GR00T 요청을 다른 모델로 자동 대체하지 않습니다. 좌표 JSON 변경·Foundry 대화를 학습으로 부르지 않으며 ACT는 보조 경로입니다.</p></div><Badge>{capability.data?.enabled ? '승인된 API 연결' : '기본 비활성'}</Badge></div>
    <ErrorNotice error={capability.error} title="학습 기능 상태를 확인하지 못했습니다" retry={capability.refresh} />
    {!capability.data && !capability.error && <Loading>학습 API 통합 상태를 확인하는 중…</Loading>}
    {capability.data && !capability.data.enabled && <section className="panel"><EmptyState icon={<ShieldCheck size={27} />} title="학습 기능이 아직 활성화되지 않았습니다">
      {capability.data.message} 실제 Azure 학습·정책 적용·미사용 조건 평가를 검증한 뒤 운영자가 활성화해야 합니다. 유료 작업, 시연, checkpoint 예시를 대신 만들지 않습니다.
    </EmptyState><button type="button" className="button secondary" onClick={capability.refresh}><RefreshCw size={15} aria-hidden="true" />통합 상태 다시 확인</button></section>}
    {capability.data?.enabled && <>
      <div className="workspace-toolbar"><div><label htmlFor="learning-project">학습 프로젝트</label><select id="learning-project" name="learning-project" value={selected?.item.id ?? ''} onChange={(event) => {
        const item = projects.data?.items.find((entry) => entry.item.id === event.target.value);
        setSelected(item ?? null);
        const url = new URL(window.location.href);
        if (item) url.searchParams.set('learning_project', item.item.id);
        else url.searchParams.delete('learning_project');
        window.history.replaceState(null, '', url);
      }}><option value="">프로젝트를 선택하세요</option>{projects.data?.items.map((entry) => <option key={entry.item.id} value={entry.item.id}>{entry.item.display_name}</option>)}</select></div>
        <button type="button" className="button secondary" onClick={() => setCreate((value) => !value)}><Plus size={15} aria-hidden="true" />새 학습 작업 정의</button></div>
      <ErrorNotice error={projects.error} title="내 학습 프로젝트를 불러오지 못했습니다" retry={projects.refresh} />
      {create && <ProjectForm api={api} environments={environments} policyTypes={capability.data.policy_types} bootstrapAllowed={capability.data.bootstrap_allowed} onCreated={(value) => { setSelected(value); projects.refresh(); setCreate(false); }} />}
      {selected && <ProjectWorkspace key={selected.item.id} api={api} project={selected} consoleApi={consoleApi} environments={environments} coachConfigured={capability.data.coach_configured} />}
      {!selected && !create && <EmptyState icon={<BookOpen size={28} />} title="학습할 작업을 선택하세요">이미 검토된 P0와 실제 시연 데이터를 연결합니다. 아직 정책이 없는 환경의 초기 P0는 별도의 승인된 부트스트랩 절차가 필요합니다.</EmptyState>}
    </>}
  </div>;
}

function ProjectForm({ api, environments, onCreated, bootstrapAllowed, policyTypes }: {
  api: LearningApi; environments: EnvironmentRecord[]; onCreated(value: Resource<Project>): void; bootstrapAllowed: boolean; policyTypes: Project['policy_type'][];
}) {
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const requestId = useRef(crypto.randomUUID());
  const planId = useRef(crypto.randomUUID());
  const [environmentId, setEnvironmentId] = useState(environments[0]?.environment_id ?? '');
  const [kind, setKind] = useState<'adaptation' | 'bootstrap'>('adaptation');
  const [policyType, setPolicyType] = useState<Project['policy_type'] | ''>(policyTypes[0] ?? '');
  const availableCases = savedTeachingCases(environments);
  const [approvedCaseIds, setApprovedCaseIds] = useState<string[]>([]);
  const environment = environments.find((item) => item.environment_id === environmentId);
  const submit = async (form: HTMLFormElement) => {
    const fields = new FormData(form);
    const teachingCases = availableCases.filter((item) => approvedCaseIds.includes(item.case_id));
    const seeds = String(fields.get('seeds')).split(',').map((value) => Number(value.trim()));
    if (!environment || !policyType || seeds.length < 20 || seeds.length > 100 || new Set(seeds).size !== seeds.length || seeds.some((value) => !Number.isSafeInteger(value) || value < 0)) {
      setError(new Error('저장 환경과 중복 없는 held-out seed 20~100개를 확인하세요.')); return;
    }
    const cases = seeds.flatMap((seed) => {
      const found = environments.find((item) => {
        const scene = item.document.scene;
        return Boolean(scene && typeof scene === 'object' && 'seed' in scene && scene.seed === seed &&
          'template_id' in scene && scene.template_id === 'inspection-cell-learning-v1');
      });
      return found ? [{ seed, environment_id: found.environment_id, revision: found.revision }] : [];
    });
    if (cases.length !== seeds.length) {
      setError(new Error('각 held-out seed에 대해 저장된 inspection-cell-learning-v1 배치와 revision이 필요합니다. 번호만 정한 시험을 실제 배치 변화로 간주하지 않습니다.')); return;
    }
    if (!teachingCases.length || new Set(teachingCases.map((item) => item.seed)).size !== teachingCases.length ||
      teachingCases.some((item) => seeds.includes(item.seed)) || seeds.includes(900002)) {
      setError(new Error('승인할 학습·검증 배치를 선택하세요. split 간 seed 중복, held-out 시험 seed와 G0 900002는 허용되지 않습니다.')); return;
    }
    const body: CreateProjectBody = {
      request_id: requestId.current, display_name: String(fields.get('name')), task_id: String(fields.get('task')),
      instruction: String(fields.get('instruction')), goal_station_id: String(fields.get('goal')),
      environment_id: environment.environment_id, revision: environment.revision,
      project_kind: kind, baseline_release_id: kind === 'adaptation' ? String(fields.get('baseline')) : null,
      policy_type: policyType,
      pretrained_artifact_id: kind === 'bootstrap' ? String(fields.get('baseline')) : null,
      control_profile_id: 'franka-position-hold-10hz-v1',
      teaching_cases: teachingCases,
      evaluation_plan: {
        id: planId.current, seeds, cases, held_out_episode_ids: [], minimum_success_rate: .9,
        maximum_axis_error_m: .04, maximum_inference_p95_ms: 80, max_step_seconds: 30,
        max_cartesian_speed_m_s: .2,
      },
      budget: {
        teaching_seconds: 120, training_seconds: 3600, evaluation_seconds: 1800,
        optimizer_steps: Number(fields.get('steps')), maximum_cost_usd: String(fields.get('cost')),
      },
    };
    setBusy(true); setError(null);
    try { onCreated(await api.createProject(body)); }
    catch (failure) { setError(failure); }
    finally { setBusy(false); }
  };
  const stations = Array.isArray(environment?.document.stations) ? environment.document.stations : [];
  return <form className="panel learning-project-form" onSubmit={(event) => { event.preventDefault(); void submit(event.currentTarget); }}>
    <div className="panel-heading"><h2>불변 작업·데이터·평가 기준 정의</h2></div>
    <div className="learning-form-grid">
      <label className="wide-field">라이선스·하드웨어 검토 후 허용된 정확한 모델 버전<select name="policy-type" required value={policyType} onChange={(event) => { if (event.target.value === 'gr00t_n1_5' || event.target.value === 'gr00t_n1_7' || event.target.value === 'smolvla') setPolicyType(event.target.value); }}><option value="">승인된 버전 선택</option>{policyTypes.map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
      {bootstrapAllowed && <label className="wide-field">작업 유형<select value={kind} name="project-kind" onChange={(event) => setKind(event.target.value === 'bootstrap' ? 'bootstrap' : 'adaptation')}><option value="adaptation">실제 P0에서 고객 P1 학습</option><option value="bootstrap">승인 운영자: 첫 Franka P0 부트스트랩</option></select></label>}
      <label>프로젝트 이름<input name="name" required maxLength={120} autoComplete="off" /></label>
      <label>등록할 task ID<input name="task" required pattern="[a-z][a-z0-9-]*" defaultValue="manufacturing-part-placement-v1" autoComplete="off" spellCheck={false} /></label>
      <label>저장된 LIVE 환경<select name="environment" value={environmentId} onChange={(event) => setEnvironmentId(event.target.value)}>{environments.map((item) => <option key={item.environment_id} value={item.environment_id}>{item.display_name}</option>)}</select></label>
      <label>목표 스테이션<select name="goal" required defaultValue="rejected">{stations.map((item) => {
        if (!item || typeof item !== 'object' || !('id' in item) || typeof item.id !== 'string') return null;
        return <option key={item.id} value={item.id}>{item.id}</option>;
      })}</select></label>
      <label className="wide-field">실제 작업 지시<input name="instruction" required maxLength={512} autoComplete="off" defaultValue="Pick up the synthetic part from the source platform and place it in the quarantine tray." /></label>
      <label className="wide-field">{kind === 'bootstrap' ? '검증·등록된 train-only 부모 artifact ID (실행 정책 아님)' : '운영자가 검토·등록한 P0 release ID'}<input name="baseline" required autoComplete="off" spellCheck={false} /></label>
      <label>optimizer step 상한<input name="steps" type="number" required min={1} max={100000} defaultValue={100} autoComplete="off" /></label>
      <label>작업별 최대 승인 금액 (USD)<input name="cost" type="number" required min=".01" max={10000} step=".01" autoComplete="off" /></label>
      <label className="wide-field">학습에서 제외할 seed 20~100개 (쉼표 구분)<input name="seeds" required autoComplete="off" placeholder="예: 200, 201, 202, …" /></label>
      <fieldset className="wide-field teaching-case-approval"><legend>명시적으로 승인할 학습·검증 배치</legend>
        <p className="small-text muted">저장된 capture split과 revision을 고정합니다. held-out test, G0 통합 전용 배치와 캡처가 비활성인 환경은 시연 목록에 넣지 않습니다.</p>
        {availableCases.length ? availableCases.map((item) => <label className="checkbox-label" key={item.case_id}>
          <input type="checkbox" name="teaching-case" value={item.case_id} checked={approvedCaseIds.includes(item.case_id)} onChange={(event) => setApprovedCaseIds((ids) => event.target.checked ? [...ids, item.case_id] : ids.filter((id) => id !== item.case_id))} />
          <span>{item.environment_id} · {splitLabel(item.split)} · seed {item.seed}<code>{item.revision}</code></span>
        </label>) : <p className="form-hint">명시적인 train/validation 캡처 설정이 있는 저장 배치를 먼저 준비하세요.</p>}
      </fieldset>
    </div>
    <p className="form-hint">이 화면은 가격이나 용량을 추정하지 않습니다. 실제 유료 제출 전 서버가 승인된 compute·가격·시간 한도를 검증합니다. P0/P1는 동일한 미사용 조건에서 비교하며 성공 장면만 남기지 않습니다.</p>
    <ErrorNotice error={error} title="작업 정의를 저장하지 못했습니다" />
    <button type="submit" className="button" disabled={busy}>{busy ? '저장 중…' : '불변 작업 정의 저장'}</button>
  </form>;
}

function ProjectWorkspace({ api, project, consoleApi, environments, coachConfigured }: {
  api: LearningApi; project: Resource<Project>; consoleApi?: ConsoleApi; environments: EnvironmentRecord[]; coachConfigured: boolean;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    const groups = await Promise.all((['teaching', 'dataset', 'training', 'evaluation', 'candidate', 'release'] as const)
      .map((kind) => api.records(project.item.id, kind, signal)));
    return groups.flatMap((group) => group.items);
  }, [api, project.item.id]);
  const records = usePolling(load);
  const [selectedSessions, setSelectedSessions] = useState<string[]>([]);
  const [teaching, setTeaching] = useState<Resource<Teaching> | null>(null);
  const [job, setJob] = useState<Resource<Job> | null>(null);
  const [evaluation, setEvaluation] = useState<Resource<Evaluation> | null>(null);
  const [datasetId, setDatasetId] = useState('');
  const [candidateId, setCandidateId] = useState('');
  const [motionApproved, setMotionApproved] = useState(false);
  const [teachingCaseId, setTeachingCaseId] = useState(() => defaultTeachingCase(project.item.teaching_cases, project.item.environment_id, project.item.revision));
  const selectedCase = project.item.teaching_cases.find((item) => item.case_id === teachingCaseId);
  const [paidApproved, setPaidApproved] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const [coach, setCoach] = useState<CoachResponse | null>(null);
  const requests = useRef(new Map<string, string>());
  const operate = async (key: string, action: (id: string) => Promise<void>) => {
    if (busy) return;
    const id = requests.current.get(key) ?? crypto.randomUUID();
    requests.current.set(key, id);
    setBusy(true); setError(null);
    try { await action(id); requests.current.delete(key); records.refresh(); }
    catch (failure) { setError(failure); }
    finally { setBusy(false); }
  };
  const updateTeaching = useCallback((value: Resource<Teaching>) => setTeaching((old) => old && value.item.last_sequence < old.item.last_sequence ? old : value), []);
  const updateJob = useCallback((value: Resource<Job>) => {
    if (value.item.kind === 'evaluation') setEvaluation({ item: value.item, etag: value.etag });
  }, []);
  const environment = environments.find((item) =>
    teaching?.item.teaching_case?.environment_id === item.environment_id &&
    teaching.item.teaching_case.revision === item.revision);
  const data = records.data ?? [];
  const datasets = data.filter((entry) => entry.item.kind === 'dataset');
  const candidates = data.filter((entry) => entry.item.kind === 'candidate');
  const sessionList = data.filter((entry) => entry.item.kind === 'teaching');
  return <>
    <section className="panel learning-project-summary"><div className="panel-heading"><h2>{project.item.display_name}</h2><Badge>{project.item.policy_type} · 검증 전 자동 게시 없음</Badge></div>
      <div className="learning-panel-body"><p>{project.item.instruction}</p><dl className="learning-metadata"><FieldValue label="환경 revision"><code>{project.item.revision}</code></FieldValue><FieldValue label="기준 P0 release"><code>{project.item.baseline_release_id}</code></FieldValue><FieldValue label="held-out plan SHA"><code>{project.item.evaluation_plan_sha256}</code></FieldValue><FieldValue label="조작 프로파일"><code>{project.item.control_profile_id}</code></FieldValue></dl></div></section>
    <div className="learning-workflow-grid">
      <section className="panel"><div className="panel-heading"><h2>1 · 직접 시연과 데이터</h2><button type="button" className="icon-button" aria-label="학습 자료 새로고침" onClick={records.refresh}><RefreshCw size={16} aria-hidden="true" /></button></div>
        <div className="learning-panel-body">
          <label htmlFor="teaching-case">승인된 시연 배치</label>
          <select id="teaching-case" name="teaching-case" value={teachingCaseId} disabled={busy} onChange={(event) => { setTeachingCaseId(event.target.value); setMotionApproved(false); }}>
            <option value="">승인된 배치를 선택하세요</option>
            {project.item.teaching_cases.map((item) => <option key={item.case_id} value={item.case_id}>{item.case_id} · {splitLabel(item.split)} · seed {item.seed}</option>)}
          </select>
          {selectedCase ? <div className="inline-note"><strong>{splitLabel(selectedCase.split)} · {selectedCase.environment_id}</strong><code>{selectedCase.revision}</code><p>이 저장 버전이 런타임에 활성화되어 있어야 시연을 시작할 수 있습니다. 검증 데이터는 optimizer 입력으로 바뀌지 않습니다.</p></div>
            : <p className="form-hint">승인된 시연 배치가 없습니다. anchor를 임의로 train으로 취급하지 않으며, 목록이 없다면 승인 배치를 포함한 새 프로젝트가 필요합니다.</p>}
          <label className="checkbox-label"><input type="checkbox" checked={motionApproved} onChange={(event) => setMotionApproved(event.target.checked)} />저속 시연 조작과 서버의 제한된 이동 권한을 승인합니다</label>
          {consoleApi && selectedCase && <button type="button" className="button secondary" disabled={!motionApproved || busy || Boolean(teaching && ['starting', 'recording', 'finishing', 'cancelling'].includes(teaching.item.status))} onClick={() => void operate(`activate:${selectedCase.case_id}`, async () => {
            await consoleApi.activateEnvironment(selectedCase.environment_id, selectedCase.revision);
          })}>선택한 승인 배치 활성화 요청</button>}
          <button type="button" className="button" disabled={!selectedCase || !motionApproved || busy || Boolean(teaching && ['starting', 'recording', 'finishing', 'cancelling'].includes(teaching.item.status))} onClick={() => selectedCase && void operate(`teach:${selectedCase.case_id}`, async (id) => {
            setTeaching(await api.teach(project.item.id, { request_id: id, source: 'human_teleop', motion_approved: true, case_id: selectedCase.case_id }, project.etag));
          })}>새 직접 시연 세션 시작</button>
          <p className="small-text muted">자동화된 teacher 데이터는 별도의 기준 제어기 출처로 기록합니다. 업로드 중은 학습 데이터 준비 완료가 아닙니다.</p>
          <div className="teaching-list">{sessionList.map((entry) => {
            if (entry.item.kind !== 'teaching') return null;
            const item = entry.item;
            return <div key={item.id}><label className="checkbox-label"><input type="checkbox" disabled={item.status !== 'ready'} checked={selectedSessions.includes(item.id)} onChange={(event) => setSelectedSessions((ids) => event.target.checked ? [...ids, item.id] : ids.filter((id) => id !== item.id))} />{sourceLabel(item.source)} · {item.status}</label>
              <code>{item.id}</code><p className="small-text">{item.teaching_case ? `${item.teaching_case.case_id} · ${splitLabel(item.teaching_case.split)} · seed ${item.teaching_case.seed}` : '기존 기록 · case/split 승인 미확인'}</p><button type="button" className="text-button" onClick={() => setTeaching({ item, etag: entry.etag })}>세션 상태 보기</button></div>;
          })}</div>
          <button type="button" className="button secondary" disabled={!selectedSessions.length || busy} onClick={() => void operate(`seal:${selectedSessions.join(',')}`, async (id) => {
            const sealed = await api.seal(project.item.id, { request_id: id, teaching_session_ids: selectedSessions }, project.etag);
            setDatasetId(sealed.item.id);
          })}><Upload size={15} aria-hidden="true" />검증된 시연으로 데이터 버전 확정</button>
        </div>
      </section>
      <section className="panel"><div className="panel-heading"><h2>2 · 실제 학습과 paired 평가</h2></div><div className="learning-panel-body">
        <label htmlFor="learning-dataset">고정 데이터 버전</label><select id="learning-dataset" name="learning-dataset" value={datasetId} onChange={(event) => setDatasetId(event.target.value)}><option value="">데이터 선택</option>{datasets.map((entry) => <option value={entry.item.id} key={entry.item.id}>{entry.item.id}</option>)}</select>
        {datasets.map((entry) => entry.item.kind === 'dataset' && entry.item.id === datasetId && <div className="inline-note" key={entry.item.id}><code>{entry.item.manifest_sha256}</code><p>직접 {entry.item.human_teleop_count} · 기준 제어기 {entry.item.reference_controller_count} · 정책 생성 {entry.item.learned_policy_count}</p><p>학습(train) {entry.item.captures.filter((capture) => capture.split === 'train').length} · 검증(validation) {entry.item.captures.filter((capture) => capture.split === 'validation').length} · 검증은 optimizer/statistics에서 제외</p></div>)}
        <label className="checkbox-label"><input type="checkbox" checked={paidApproved} onChange={(event) => setPaidApproved(event.target.checked)} />실제 Azure 작업 비용 상한 {project.item.budget.maximum_cost_usd} USD와 시간 제한을 승인합니다</label>
        <button type="button" className="button" disabled={!datasetId || !paidApproved || busy} onClick={() => void operate(`train:${datasetId}`, async (id) => {
          setJob(await api.train(project.item.id, { request_id: id, dataset_id: datasetId, parent_release_id: project.item.baseline_release_id, pretrained_artifact_id: project.item.pretrained_artifact_id, policy_type: project.item.policy_type, optimizer_steps: project.item.budget.optimizer_steps, paid_approved: true, maximum_cost_usd: project.item.budget.maximum_cost_usd }, project.etag));
        })}>승인한 {project.item.policy_type} 학습 제출</button>
        <label htmlFor="learning-candidate">실제 학습에서 나온 후보 P1</label><select id="learning-candidate" name="learning-candidate" value={candidateId} onChange={(event) => setCandidateId(event.target.value)}><option value="">후보 선택</option>{candidates.map((entry) => <option key={entry.item.id} value={entry.item.id}>{entry.item.id}</option>)}</select>
        <button type="button" className="button secondary" disabled={!candidateId || !motionApproved || !paidApproved || busy} onClick={() => void operate(`evaluate:${candidateId}`, async (id) => {
          const value = await api.evaluate(project.item.id, { request_id: id, candidate_id: candidateId, baseline_release_id: project.item.baseline_release_id, comparison_kind: project.item.project_kind === 'bootstrap' ? 'reference_bootstrap' : 'paired_policy', evaluation_plan_sha256: project.item.evaluation_plan_sha256, motion_approved: true, paid_approved: true, maximum_cost_usd: project.item.budget.maximum_cost_usd }, project.etag);
          setJob(value); setEvaluation(value);
        })}>{project.item.project_kind === 'bootstrap' ? '동일 조건에서 기준 제어기 / 최초 후보 평가' : '동일 held-out 조건에서 P0 / P1 평가'}</button>
        <p className="form-hint">학습은 수 시간 걸릴 수 있습니다. 사전 checkpoint를 이번 작업의 산출물로 바꿔 끼우지 않습니다.</p>
      </div></section>
    </div>
    <ErrorNotice error={records.error} title="학습 기록 갱신 실패" retry={records.refresh} />
    <ErrorNotice error={error} title="학습 작업 결과 확인 필요 · 자동 재시도 없음" />
    {teaching && <TeachingSessionPanel key={teaching.item.id} api={api} session={teaching} onChange={updateTeaching} consoleApi={consoleApi} environment={environment} />}
    {job && <LearningJobPanel key={job.item.id} api={api} initial={job} onUpdate={updateJob} />}
    {evaluation && <PolicyComparison key={evaluation.item.id} api={api} evaluation={evaluation} onReleased={() => records.refresh()} />}
    <section className="panel learning-records"><div className="panel-heading"><h2>작업·후보·게시 기록</h2></div><div className="learning-panel-body">{data.filter((entry) => ['training', 'evaluation', 'release'].includes(entry.item.kind)).map((entry) => <RecordRow key={`${entry.item.kind}:${entry.item.id}`} entry={entry} select={(value) => {
      if (value.item.kind === 'training' || value.item.kind === 'evaluation') {
        setJob({ item: value.item, etag: value.etag });
        if (value.item.kind === 'evaluation') setEvaluation({ item: value.item, etag: value.etag });
      }
    }} />)}</div></section>
    <section className="panel learning-coach"><div className="panel-heading"><span className="title-icon"><Bot size={18} aria-hidden="true" /><h2>Foundry 학습 코치</h2></span><Badge>제안만 · 조작 정책 아님</Badge></div><div className="learning-panel-body">
      <form onSubmit={(event) => { event.preventDefault(); const prompt = String(new FormData(event.currentTarget).get('prompt')); void operate(`coach:${prompt}`, async (id) => {
        setCoach(await api.coach(project.item.id, { request_id: id, instruction: prompt, dataset_id: datasetId || null, evaluation_run_id: evaluation?.item.id ?? null }));
      }); }}><label htmlFor="learning-coach-prompt">검토하거나 설명할 내용을 입력하세요</label><textarea id="learning-coach-prompt" name="prompt" required maxLength={2000} autoComplete="off" placeholder="예: 직접 시연과 기준 제어기 데이터를 구분하고 다음 검증 단계를 설명해 주세요…" /><button type="submit" className="button secondary" disabled={!coachConfigured || busy}>실제 코치에 검토 제안 요청</button></form>
      {!coachConfigured && <p className="form-hint">별도의 학습 코치가 구성되지 않았습니다. 검사 에이전트를 대신 사용하지 않습니다.</p>}
      {coach && <div className="inline-note" role="status"><p>{coach.proposal.summary}</p><code>{coach.model_response_id}</code><p>이 제안은 비용·이동·정책 게시 권한을 부여하지 않습니다.</p></div>}
    </div></section>
  </>;
}

const keepTeachingPolling = (value: Resource<Teaching>) => !['ready', 'invalid', 'cancelled', 'blocked'].includes(value.item.status);

function TeachingSessionPanel({ api, session, onChange, consoleApi, environment }: {
  api: LearningApi; session: Resource<Teaching>; onChange(value: Resource<Teaching>): void;
  consoleApi?: ConsoleApi; environment?: EnvironmentRecord;
}) {
  const load = useCallback(async (signal: AbortSignal) => {
    const value = await api.teaching(session.item.id, signal);
    onChange(value);
    return value;
  }, [api, session.item.id, onChange]);
  const state = usePolling(load, { intervalMs: 2000, initialData: session, continuePolling: keepTeachingPolling });
  const current = state.data && state.data.item.last_sequence >= session.item.last_sequence ? state.data : session;
  const [error, setError] = useState<unknown>(null);
  const [busy, setBusy] = useState(false);
  const control = async (action: 'finish' | 'cancel') => {
    setBusy(true);
    try {
      const value = await api.teachingControl(current.item.id, action, { request_id: crypto.randomUUID(), lease_id: current.item.lease_id, epoch: current.item.epoch }, current.etag);
      state.setData(value); onChange(value); state.refresh();
    } catch (failure) { setError(failure); }
    finally { setBusy(false); }
  };
  return <section className="panel teaching-session"><div className="panel-heading"><h2>현재 선택한 시연</h2><Badge>{current.item.status}</Badge></div>
    <div className="learning-panel-body"><p>{sourceLabel(current.item.source)} · 물리 상태: {current.item.physical_status ?? '확인 전'} · 캡처 상태: {current.item.status}</p><code>{current.item.id}</code>
      {current.item.teaching_case && <dl className="learning-metadata"><FieldValue label="승인된 시연 배치">{current.item.teaching_case.case_id} · {splitLabel(current.item.teaching_case.split)} · seed {current.item.teaching_case.seed}</FieldValue><FieldValue label="실제 캡처 환경 / revision"><code>{current.item.teaching_case.environment_id}</code><code>{current.item.teaching_case.revision}</code></FieldValue></dl>}
      <p className="small-text muted">권한 만료: {formatDate(current.item.expires_at)} · 업로드는 물리 실행과 별도로 완료됩니다.</p>
      {consoleApi && environment && <div className="teaching-cameras">{(['overview', 'inspection'] as const).map((camera) => <LiveCamera key={`${current.item.id}:${camera}`} api={consoleApi} environment={environment} camera={camera} enabled={current.item.status === 'recording'} unavailableReason="활성 시연의 실제 카메라만 연결합니다." />)}</div>}
      <TeachingControls api={api} session={current} onChange={onChange} />
      <div className="button-row"><button type="button" className="button secondary" disabled={busy || current.item.status !== 'recording'} onClick={() => void control('finish')}>시연 마감 및 캡처 검증 요청</button>
        <button type="button" className="button danger-quiet" disabled={busy || ['ready', 'cancelled', 'invalid', 'blocked'].includes(current.item.status)} onClick={() => void control('cancel')}>시연 취소 요청</button></div>
      <ErrorNotice error={state.error ?? error} title="시연 상태를 확인하지 못했습니다" retry={state.refresh} />
    </div></section>;
}

function RecordRow({ entry, select }: { entry: Resource<LearningRecord>; select(value: Resource<LearningRecord>): void }) {
  return <div className="learning-record-row"><span><strong>{entry.item.kind}</strong><code>{entry.item.id}</code><small>{formatDate(entry.item.created_at)}</small></span>
    {'status' in entry.item && <Badge>{entry.item.status}</Badge>}<button type="button" className="text-button" onClick={() => select(entry)}>기록 보기</button></div>;
}
