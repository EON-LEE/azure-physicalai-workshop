import { useCallback, useEffect } from 'react';
import { Camera, Clock3, FileImage, Radio, RefreshCw, Unplug } from 'lucide-react';
import { describeError } from '../api/errors';
import { useProtectedImage } from '../hooks/useProtectedImage';
import { FRAME_MAX_AGE_MS, getDemoEvidence, getDemoFrame, needsSnapshotRefresh, type PublicDecision } from './api';
import { formatCount, formatTime } from './presentation';

export function CameraUnavailable({ title, description, retry, loading = false }: {
  title: string; description: string; retry(): void; loading?: boolean;
}) {
  return <div className="presentation-camera unavailable" role="status">
    <div className="camera-unavailable-copy">
      <Camera size={35} strokeWidth={1.5} aria-hidden="true" />
      <h3>{title}</h3>
      <p>{description}</p>
      <span>실시간 관측 대신 구성도·재생 영상·가상 동작을 보여주지 않습니다.</span>
      <button type="button" className="demo-button secondary" onClick={retry} disabled={loading}>
        <RefreshCw size={15} aria-hidden="true" />{loading ? '상태 확인 중…' : '시연 상태 다시 확인'}
      </button>
    </div>
  </div>;
}

export function PublicLiveCamera({ epoch, onSceneChanged }: {
  epoch: string; onSceneChanged(): void;
}) {
  const load = useCallback((signal: AbortSignal) => getDemoFrame(epoch, signal), [epoch]);
  const image = useProtectedImage(load, 1000);
  useEffect(() => {
    if (needsSnapshotRefresh(image.error)) onSceneChanged();
  }, [image.error, onSceneChanged]);
  const age = image.data ? image.clock - Date.parse(image.data.capturedAt) : null;
  const stale = age !== null && (age > FRAME_MAX_AGE_MS || age < -FRAME_MAX_AGE_MS);
  const connected = Boolean(image.url && image.data && !image.error && !image.decodeError && !image.paused && !stale);
  const retry = () => { onSceneChanged(); image.refresh(); };

  return <div className="public-camera-feed">
    <div className="presentation-camera">
      {connected && image.url ? <img src={image.url} width={1280} height={720}
        alt="이번 시연 회차의 Isaac Sim 실제 물리 시뮬레이터 카메라. 공급대는 파랑, 정상 트레이는 초록, 격리 트레이는 빨강입니다."
        fetchPriority="high" onError={image.onDecodeError} /> : <div className="camera-unavailable-copy" role="status">
        <Unplug size={32} strokeWidth={1.5} aria-hidden="true" />
        <h3>{image.paused ? '화면이 숨겨져 영상 수신을 멈췄습니다' : image.error || image.decodeError || stale ? '현재 카메라 연결을 확인할 수 없습니다' : '현재 회차의 카메라 연결 중…'}</h3>
        <p>{image.error ? describeError(image.error) : image.decodeError ? 'PNG를 표시할 수 없습니다. 다시 연결해 주세요.' : stale ? '촬영 시각이 오래되었거나 맞지 않습니다. 멈춘 이미지를 LIVE로 표시하지 않습니다.' : '같은 씬의 촬영 시각과 실제 물리 스텝을 확인합니다.'}</p>
        {(Boolean(image.error) || image.decodeError || stale) && <button type="button" className="demo-button secondary" onClick={retry}><RefreshCw size={15} aria-hidden="true" />카메라 다시 확인</button>}
      </div>}
      {connected && <span className="public-live-indicator"><Radio size={13} aria-hidden="true" />LIVE · 실제 카메라 수신</span>}
      <span className="camera-source">ISAAC SIM · 합성 카메라 / 실물 센서 아님</span>
    </div>
    <div className="public-frame-metadata" aria-label="카메라 수신 정보">
      <span><Clock3 size={14} aria-hidden="true" />{connected && image.data ? <time dateTime={image.data.capturedAt}>{formatTime(image.data.capturedAt)}</time> : '촬영 시각 확인 전'}</span>
      <span>물리 스텝 <strong>{connected && image.data ? formatCount(image.data.physicsSteps) : '—'}</strong></span>
      <span className="public-frame-id" title={connected ? image.data?.frameId : undefined} translate="no">{connected && image.data ? image.data.frameId : '프레임 미확인'}</span>
    </div>
  </div>;
}

export function DecisionEvidence({ decision, active, onSceneChanged }: {
  decision: PublicDecision; active: boolean; onSceneChanged(): void;
}) {
  const { observation_id, captured_at, image_url } = decision;
  const load = useCallback((signal: AbortSignal) => getDemoEvidence({
    observation_id, captured_at, image_url, classification: decision.classification,
    summary: decision.summary, target_station_id: decision.target_station_id,
  }, signal), [observation_id, captured_at, image_url, decision.classification, decision.summary, decision.target_station_id]);
  const image = useProtectedImage(load, null, active);
  useEffect(() => {
    if (needsSnapshotRefresh(image.error)) onSceneChanged();
  }, [image.error, onSceneChanged]);
  const valid = Boolean(active && image.url && image.data && !image.error && !image.decodeError && !image.paused);
  return <figure className="public-decision-evidence">
    <div className="decision-image">
      {valid && image.url ? <img src={image.url} width={640} height={360} loading="lazy"
        alt="현재 Foundry 판단에 실제 사용된 같은 회차의 원본 검사 입력. 과거 촬영 이미지이며 현재 LIVE 화면이 아닙니다."
        onError={image.onDecodeError} /> : <div className="evidence-placeholder" role="status">
        <FileImage size={23} aria-hidden="true" />
        <span>{!active || image.paused ? '관측 증거 수신 일시 중지' : image.error || image.decodeError ? '판단 입력 이미지를 확인할 수 없습니다' : '같은 회차의 검사 입력 확인 중…'}</span>
        {(Boolean(image.error) || image.decodeError) && active && <button type="button" className="demo-text-button" onClick={() => { image.refresh(); onSceneChanged(); }}>검사 입력 다시 확인</button>}
      </div>}
    </div>
    <figcaption><strong>판단에 사용된 원본 입력 · LIVE 아님</strong>
      <span>촬영 <time dateTime={decision.captured_at}>{formatTime(decision.captured_at)}</time></span>
    </figcaption>
    {Boolean(image.error) && <p className="public-inline-error" role="alert">{describeError(image.error)}</p>}
  </figure>;
}
