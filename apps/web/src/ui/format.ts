import type { RunStatus, RuntimeInfo } from '../api/contracts';

const dateTime = new Intl.DateTimeFormat('ko-KR', {
  year: 'numeric', month: '2-digit', day: '2-digit',
  hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});

export function formatDate(value: string | number | null): string {
  if (value === null) return '확인 전';
  const parsed = new Date(value);
  return Number.isFinite(parsed.getTime()) ? dateTime.format(parsed) : value.toString();
}

export function shortId(value: string): string {
  return value.length > 18 ? `${value.slice(0, 10)}…${value.slice(-6)}` : value;
}

export const runLabels: Record<RunStatus, string> = {
  planning: '계획 생성 중',
  awaiting_approval: '승인 대기',
  running: '실행 중',
  cancelling: '취소 확인 중',
  succeeded: '실행 성공',
  failed: '실행 실패',
  cancelled: '취소 확인됨',
  timed_out: '시간 초과',
};

export const runtimeLabels: Record<RuntimeInfo['simulation']['status'], string> = {
  ready: '시뮬레이터 준비',
  loading: '씬 로딩 중',
  unavailable: '연결 불가',
  occupied: '시뮬레이터 사용 중',
};
