import { useEffect, useRef, useState } from 'react';
import { CircleStop, Hand } from 'lucide-react';
import { isAbort } from '../api/errors';
import { ErrorNotice } from '../ui/common';
import type { JogBody, LearningApi, Resource, Teaching } from './contracts';

const pageHidden = () => document.visibilityState === 'hidden';

export function TeachingControls({ api, session, onChange }: {
  api: LearningApi; session: Resource<Teaching>; onChange(value: Resource<Teaching>): void;
}) {
  const latest = useRef(session);
  latest.current = session;
  const sequence = useRef(session.item.last_sequence);
  sequence.current = Math.max(sequence.current, session.item.last_sequence);
  const held = useRef(false);
  const generation = useRef(0);
  const armRequest = useRef<AbortController | null>(null);
  const mounted = useRef(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const releaseRef = useRef<() => void>(() => {});
  const enabled = session.item.status === 'recording';

  const release = () => {
    if (!held.current) return;
    held.current = false;
    generation.current += 1;
    armRequest.current?.abort();
    if (mounted.current) setBusy(false);
    const current = latest.current;
    const body: JogBody = {
      request_id: crypto.randomUUID(), lease_id: current.item.lease_id, epoch: current.item.epoch,
      sequence: ++sequence.current, deadman: false, delta_xyz_m: [0, 0, 0], gripper: 'hold',
    };
    void api.jog(current.item.id, body, current.etag).then((result) => {
      latest.current = result;
      if (mounted.current) onChange(result);
    }).catch((failure: unknown) => {
      if (mounted.current) setError(failure);
      else console.error('Teaching hold release could not be confirmed.', failure);
    });
  };
  releaseRef.current = release;

  useEffect(() => {
    mounted.current = true;
    const release = () => releaseRef.current();
    const visibility = () => { if (document.visibilityState === 'hidden') release(); };
    const keyup = (event: KeyboardEvent) => { if (event.key === ' ' || event.key === 'Enter' || event.key === 'Escape') release(); };
    window.addEventListener('pointerup', release);
    window.addEventListener('pointercancel', release);
    window.addEventListener('blur', release);
    window.addEventListener('keyup', keyup);
    document.addEventListener('visibilitychange', visibility);
    return () => {
      mounted.current = false;
      release();
      window.removeEventListener('pointerup', release);
      window.removeEventListener('pointercancel', release);
      window.removeEventListener('blur', release);
      window.removeEventListener('keyup', keyup);
      document.removeEventListener('visibilitychange', visibility);
      armRequest.current?.abort();
    };
  }, []);

  const press = async (delta: [number, number, number], gripper: JogBody['gripper']) => {
    if (!enabled || held.current || busy || pageHidden()) return;
    held.current = true;
    setBusy(true);
    setError(null);
    const turn = ++generation.current;
    const current = latest.current;
    const controller = new AbortController();
    armRequest.current = controller;
    const input: JogBody = {
      request_id: crypto.randomUUID(), lease_id: current.item.lease_id, epoch: current.item.epoch,
      sequence: ++sequence.current, deadman: true, delta_xyz_m: delta, gripper,
    };
    try {
      const grant = await api.arm(current.item.id, input, current.etag, controller.signal);
      if (!mounted.current || !held.current || turn !== generation.current || pageHidden()) return;
      const result = await api.jog(current.item.id, { ...input, request_id: crypto.randomUUID(), grant_id: grant.item.id }, current.etag, controller.signal);
      if (mounted.current && held.current && turn === generation.current) {
        latest.current = result;
        onChange(result);
      }
    } catch (failure) {
      if (mounted.current && !isAbort(failure)) setError(failure);
    } finally {
      if (mounted.current && turn === generation.current) setBusy(false);
    }
  };

  const controls: Array<{ label: string; text: string; delta: [number, number, number]; gripper: JogBody['gripper'] }> = [
    { label: 'X 양의 방향 5mm', text: 'X +', delta: [.005, 0, 0], gripper: 'hold' },
    { label: 'X 음의 방향 5mm', text: 'X −', delta: [-.005, 0, 0], gripper: 'hold' },
    { label: 'Y 양의 방향 5mm', text: 'Y +', delta: [0, .005, 0], gripper: 'hold' },
    { label: 'Y 음의 방향 5mm', text: 'Y −', delta: [0, -.005, 0], gripper: 'hold' },
    { label: 'Z 양의 방향 5mm', text: 'Z +', delta: [0, 0, .005], gripper: 'hold' },
    { label: 'Z 음의 방향 5mm', text: 'Z −', delta: [0, 0, -.005], gripper: 'hold' },
    { label: '그리퍼 열기', text: '그리퍼 열기', delta: [0, 0, 0], gripper: 'open' },
    { label: '그리퍼 닫기', text: '그리퍼 닫기', delta: [0, 0, 0], gripper: 'close' },
  ];
  return <section className="teaching-controls" aria-label="승인된 시연 조작">
    <div className="title-icon"><Hand size={18} aria-hidden="true" /><h3>누르고 유지하는 동안 한 번의 제한된 입력</h3></div>
    <p>Space·Enter 또는 포인터를 유지하세요. 한 번에 최대 5mm, 권한은 서버가 짧게 발급합니다. 해제·화면 이탈 시 hold를 요청합니다. 화면이 60Hz 로봇 제어기를 대신하지 않습니다.</p>
    <div className="jog-grid">{controls.map((control) => <button type="button" className="button secondary" key={control.label} aria-label={control.label} disabled={!enabled}
      onPointerDown={(event) => { if (event.button === 0) { event.preventDefault(); void press(control.delta, control.gripper); } }}
      onPointerUp={release} onPointerCancel={release} onBlur={release}
      onKeyDown={(event) => { if ((event.key === ' ' || event.key === 'Enter') && !event.repeat) { event.preventDefault(); void press(control.delta, control.gripper); } }}
      onKeyUp={(event) => { if (event.key === ' ' || event.key === 'Enter') { event.preventDefault(); release(); } }}>
      {control.text}
    </button>)}</div>
    <button type="button" className="button danger-quiet" onClick={release}><CircleStop size={15} aria-hidden="true" />입력 해제 / hold 요청</button>
    <span className="small-text muted" role="status">{busy ? '서버 권한 확인 및 입력 응답 대기…' : '다음 명시적 조작을 기다립니다.'}</span>
    <ErrorNotice error={error} title="시연 입력을 확인하지 못했습니다 · 자동 재전송하지 않음" />
  </section>;
}
