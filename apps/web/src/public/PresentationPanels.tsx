import { ArrowRight, Bot, Check, CircleHelp, Crosshair, ScanLine, ShieldCheck, X } from 'lucide-react';
import type { DemoSnapshot, Presentation } from './api';
import { DecisionEvidence } from './Media';
import { formatCount, formatPosition, formatTime, motionLabels, phaseLabels, progressFor, taskSucceeded } from './presentation';

export function DecisionPanel({ presentation, snapshot, current, viewing, onSceneChanged }: {
  presentation: Presentation | null; snapshot: DemoSnapshot | null; current: boolean; viewing: boolean; onSceneChanged(): void;
}) {
  const decision = presentation?.decision;
  const target = decision && snapshot?.scene.stations.find((station) => station.id === decision.target_station_id);
  const targetName = target?.role === 'accepted' ? '정상 트레이' : target?.role === 'rejected' ? '불량 격리 트레이' : '서버 지정 스테이션';
  return <aside className="public-decision panel" aria-labelledby="decision-heading">
    <div className="public-panel-heading"><span className="title-icon"><ScanLine size={19} aria-hidden="true" /><h2 id="decision-heading">같은 회차의 검사 근거</h2></span>
      <span className="small-label" translate="no">FOUNDRY</span></div>
    {decision ? <>
      <DecisionEvidence key={`${presentation.id}:${presentation.cycle}:${presentation.scene_epoch}:${decision.observation_id}`}
        decision={decision} active={viewing} onSceneChanged={onSceneChanged} />
      {!current && <p className="last-known-note">마지막 게시 판단입니다. 현재 실행 상태를 보장하지 않습니다.</p>}
      <div className="public-decision-copy">
        <div className="decision-classification"><span>모델의 실제 분류</span>
          <strong className={decision.classification === 'accepted' ? 'accepted-text' : 'rejected-text'}>{decision.classification === 'accepted' ? '정상으로 판단' : '불량으로 판단'}</strong>
        </div>
        <blockquote>{decision.summary || '서버가 판단 이유를 제공하지 않았습니다.'}</blockquote>
        <div className="decision-destination"><ArrowRight size={17} aria-hidden="true" /><span>판단의 이동 대상<strong>{targetName}</strong><code translate="no">{decision.target_station_id}</code></span></div>
        <p className="decision-limits">이 분류만으로 성공 처리하지 않습니다. 아래에서 실제 검사 정답과 물리 이동 결과를 각각 확인합니다.</p>
      </div>
    </> : <div className="decision-waiting" role="status"><Bot size={30} strokeWidth={1.5} aria-hidden="true" />
      <h3>{presentation?.status === 'inspecting' && current ? '실제 이미지 판단을 기다리고 있습니다' : '아직 게시된 판단이 없습니다'}</h3>
      <p>Foundry의 원본 입력과 판단 이유를 수신하면 여기에 표시합니다. 연결 검증을 부품 검사 결과로 대신하지 않습니다.</p>
    </div>}
    <div className="public-control-boundary"><ShieldCheck size={16} aria-hidden="true" /><p>운영자가 허용한 유한한 시연만 관람합니다.<br />이 화면에서 모델 호출이나 이동 명령을 보내지 않습니다.</p></div>
  </aside>;
}

function Measurement({ title, value, positive, description }: {
  title: string; value: string; positive: boolean | null; description: string;
}) {
  const Icon = positive === true ? Check : positive === false ? X : CircleHelp;
  return <div className={`result-measure ${positive === true ? 'passed' : positive === false ? 'failed' : 'pending'}`}>
    <span className="measurement-icon"><Icon size={18} aria-hidden="true" /></span>
    <div><h3>{title}</h3><strong>{value}</strong><p>{description}</p></div>
  </div>;
}

export function PhysicalOutcome({ presentation, current }: { presentation: Presentation; current: boolean }) {
  const { motion, result } = presentation;
  const success = taskSucceeded(result);
  return <section className="public-outcome panel" aria-labelledby="outcome-heading">
    <div className="public-panel-heading"><span className="title-icon"><Crosshair size={19} aria-hidden="true" /><h2 id="outcome-heading">판단과 물리 결과를 따로 확인합니다</h2></span>
      <span className={`result-badge ${success ? 'positive' : result ? 'negative' : ''}`}>
        {result ? success ? '종합 성공 확인' : '종합 성공 미확인' : '최종 결과 대기'}
      </span>
    </div>
    <p className="public-outcome-context">{current ? '이번 게시 회차의 서버 측정 결과' : '마지막 게시 결과 · 현재 카메라 연결이나 실행을 의미하지 않음'}</p>
    <div className="public-result-grid">
      <Measurement title="검사 결과" value={result?.inspection_correct === true ? '입력 정답과 일치' : result?.inspection_correct === false ? '입력 정답과 불일치' : '정확성 미확인'}
        positive={result?.inspection_correct ?? null} description="현재 합성 부품의 알려진 종류와 모델 분류를 비교합니다. 일반적인 검사 정확도 수치가 아닙니다." />
      <Measurement title="물리 결과" value={result ? result.physical_success ? '물리 목표 도달 확인' : '물리 목표 미달성' : '물리 결과 미수신'}
        positive={result?.physical_success ?? null} description="명령 접수가 아니라 시뮬레이터가 측정한 부품 이동 결과입니다. 검사 판단의 정오와는 별개입니다." />
    </div>
    <div className="public-motion-data">
      <dl>
        <div><dt>명령 상태</dt><dd>{motion ? motionLabels[motion.status] : '명령 미수신'}</dd></div>
        <div><dt>실제 동작 단계</dt><dd>{motion?.phase ? phaseLabels[motion.phase] : '단계 정보 미수신'}</dd></div>
        <div><dt>{result ? '최종 부품 위치' : '관측된 부품 위치'}</dt><dd>{formatPosition(result ? result.final_position_m : motion?.part_position_m ?? null)}</dd></div>
        <div><dt>명령 목표 위치</dt><dd>{formatPosition(motion?.target_position_m ?? null)}</dd></div>
      </dl>
      {result && <div className="result-message" role="status"><strong>{result.message || '추가 결과 설명이 없습니다.'}</strong><span>결과 기록 <time dateTime={result.completed_at}>{formatTime(result.completed_at)}</time></span></div>}
    </div>
  </section>;
}

export function PresentationProgress({ presentation, current }: { presentation: Presentation; current: boolean }) {
  return <section className="public-progress panel" aria-labelledby="progress-heading">
    <div className="public-panel-heading"><h2 id="progress-heading">이번 회차의 실제 진행</h2><span className="small-label">{current ? '서버 상태 기준' : '마지막 수신 상태'}</span></div>
    <ol aria-label="관측부터 물리 결과까지의 서버 진행 상태">
      {progressFor(presentation).map((step, index) => <li key={step.title} className={step.confirmed ? 'confirmed' : step.active && current ? 'current' : ''} aria-current={step.active && current ? 'step' : undefined}>
        <span className="progress-marker" aria-hidden="true">{step.confirmed ? <Check size={15} /> : formatCount(index + 1)}</span>
        <div><h3>{step.title}</h3><p>{step.detail}</p></div>
      </li>)}
    </ol>
    <p className="progress-boundary">화면의 시간 경과로 단계를 진행시키지 않습니다. 이동에는 모델과 별도의 제한된 제어 계층이 적용됩니다.</p>
  </section>;
}

export function PublishedCounts({ presentation }: { presentation: Presentation }) {
  const { counts } = presentation;
  return <section className="public-counts" aria-label="현재 게시 시연의 누적 서버 집계">
    <span>현재 시연 집계<small>서버가 반환한 횟수 · 전체 시스템 성능이 아님</small></span>
    <dl>
      <div><dt>시도</dt><dd>{formatCount(counts.attempted)}</dd></div>
      <div><dt>검사 정답 일치</dt><dd>{formatCount(counts.inspected_correctly)}</dd></div>
      <div><dt>물리 목표 도달</dt><dd>{formatCount(counts.physically_completed)}</dd></div>
      <div><dt>서버 성공 집계</dt><dd>{formatCount(counts.succeeded)}</dd></div>
      <div><dt>실패</dt><dd>{formatCount(counts.failed)}</dd></div>
    </dl>
  </section>;
}
