import { useCallback } from 'react';
import { Camera as CameraIcon, Clock3, Maximize2, Radio, RefreshCw, WifiOff } from 'lucide-react';
import type { Camera, ConsoleApi, EnvironmentRecord } from '../api/contracts';
import { useProtectedImage } from '../hooks/useProtectedImage';
import { Badge, ErrorNotice } from '../ui/common';
import { formatDate, shortId } from '../ui/format';

export function LiveCamera({ api, environment, camera, enabled, unavailableReason }: {
  api: ConsoleApi; environment: EnvironmentRecord; camera: Camera; enabled: boolean; unavailableReason: string;
}) {
  const load = useCallback((signal: AbortSignal) => api.getFrame(environment.environment_id, environment.revision, camera, signal),
    [api, environment.environment_id, environment.revision, camera]);
  const stream = useProtectedImage(load, 1000, enabled);
  const captured = stream.data ? Date.parse(stream.data.capturedAt) : null;
  const age = captured === null ? null : stream.clock - captured;
  const stale = age !== null && (age > 5000 || age < -5000);
  const connected = Boolean(stream.url && !stream.error && !stream.paused && !stale && !stream.decodeError);
  const status = stream.paused ? '수신 일시 중지' : stream.decodeError ? '영상 표시 오류'
    : stream.error ? '연결 끊김' : stale ? '오래된 프레임 / 시간 확인 필요' : connected ? 'LIVE · 프레임 수신 중' : '프레임 대기';

  return <div className="camera-container">
    <div className="camera-viewport">
      {stream.url && !stream.decodeError ? <img src={stream.url} alt={`${environment.display_name} ${camera === 'overview' ? '전체 셀' : '검사 지점'}의 Isaac Sim 합성 카메라 관측`}
        onError={stream.onDecodeError} /> : <div className="camera-placeholder">
        {stream.error || stream.decodeError ? <WifiOff size={36} strokeWidth={1.2} aria-hidden="true" /> : <CameraIcon size={36} strokeWidth={1.2} aria-hidden="true" />}
        <strong>{enabled ? stream.paused ? '탭이 숨겨져 영상 수신을 멈췄습니다' : stream.error ? '실시간 카메라에 연결할 수 없습니다' : stream.decodeError ? 'PNG 영상을 표시할 수 없습니다' : '실시간 관측을 기다리고 있습니다' : '활성화된 씬이 필요합니다'}</strong>
        <p>{enabled ? 'API의 인증된 PNG만 표시합니다. 대체 이미지나 브라우저 애니메이션은 사용하지 않습니다.' : unavailableReason}</p>
      </div>}
      <div className="camera-topline">
        <span className={`camera-status ${connected ? 'connected' : 'disconnected'}`} role="status"><Radio size={13} aria-hidden="true" />{status}</span>
        <span className="camera-name">{camera === 'overview' ? 'OVERVIEW' : 'INSPECTION'}<Maximize2 size={12} aria-hidden="true" /></span>
      </div>
      {stream.url && !connected && <div className="stale-overlay">마지막 수신 이미지 · 현재 상태를 보장하지 않음</div>}
      <div className="camera-bottomline"><span>ISAAC SIM / PHYSICS CAMERA</span><span>합성 영상 · 실물 센서 아님</span></div>
    </div>
    <div className="frame-metadata">
      <span><Clock3 size={14} aria-hidden="true" />촬영 <time dateTime={stream.data?.capturedAt}>{stream.data ? formatDate(stream.data.capturedAt) : '확인 전'}</time></span>
      <span>물리 스텝 <strong>{stream.data?.physicsSteps ?? '—'}</strong></span>
      <span className="frame-id" title={stream.data?.frameId}>FRAME {stream.data ? shortId(stream.data.frameId) : '—'}</span>
    </div>
    {(stream.error || stream.decodeError) && <div className="camera-error">
      <ErrorNotice error={stream.error ?? new Error('수신한 PNG를 브라우저에서 해석할 수 없습니다. 서버 카메라 출력을 확인하세요.')} title="영상 확인 필요" compact retry={stream.refresh} />
    </div>}
    <div className="camera-caption"><Badge tone="blue">LIVE 전용</Badge><span>보이는 동안 최대 1 fps · 촬영 시각은 서버 응답 기준 · 브라우저 현지 시간</span>
      <button type="button" className="text-button" onClick={stream.refresh} disabled={!enabled || stream.loading || stream.paused}><RefreshCw size={13} aria-hidden="true" />재연결</button>
    </div>
  </div>;
}
