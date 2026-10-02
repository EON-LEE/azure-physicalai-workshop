import type { Presentation } from './api';

export const presentationLabels: Record<Presentation['status'], string> = {
  preparing: '시연 준비 중',
  inspecting: '이미지 검사 중',
  awaiting_motion: '허용된 이동 대기',
  moving: '로봇 이동 중',
  completed: '회차 종료',
  stopped: '시연 중지됨',
  failed: '시연 실패',
};
export const motionLabels: Record<NonNullable<Presentation['motion']>['status'], string> = {
  queued: '이동 명령 대기',
  running: '물리 동작 중',
  succeeded: '명령 종료 확인',
  failed: '이동 실패',
  cancelled: '이동 취소 확인',
  timed_out: '이동 시간 초과',
  cancelling: '취소 확인 중',
};
export const phaseLabels: Record<NonNullable<NonNullable<Presentation['motion']>['phase']>, string> = {
  idle: '대기',
  approaching: '부품 접근',
  grasping: '부품 잡기',
  lifting: '들어 올리기',
  inspection_station: '검사 위치',
  transporting: '대상 트레이로 이동',
  releasing: '부품 내려놓기',
  returning: '복귀',
  complete: '동작 종료',
  stopped: '동작 중지',
};
const countFormatter = new Intl.NumberFormat('ko-KR');
const coordinateFormatter = new Intl.NumberFormat('ko-KR', { maximumFractionDigits: 3 });
const timeFormatter = new Intl.DateTimeFormat('ko-KR', {
  month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
});
export const formatCount = (value: number) => countFormatter.format(value);
export const formatTime = (value: string) => timeFormatter.format(new Date(value));
export const formatPosition = (value: [number, number, number] | null) =>
  value ? `(${value.map((coordinate) => coordinateFormatter.format(coordinate)).join(', ')}) m` : '측정값 미수신';

export function taskSucceeded(result: Presentation['result']): boolean {
  return Boolean(result && result.status === 'succeeded' && result.physical_success && result.inspection_correct === true);
}

export function presentationKey(presentation: Presentation): string {
  return `${presentation.id}:${presentation.cycle}:${presentation.scene_epoch ?? 'no-scene'}:${presentation.run_id ?? 'no-run'}`;
}

export function progressFor(presentation: Presentation) {
  const { decision, motion, result, status } = presentation;
  return [
    { title: '카메라 관측', detail: decision ? '검사 입력 수신' : status === 'inspecting' ? '관측·검사 중' : '입력 대기', confirmed: Boolean(decision), active: status === 'preparing' },
    { title: 'Foundry 판단', detail: decision ? '실제 판단 수신' : status === 'inspecting' ? '이미지 검사 중' : '판단 미수신', confirmed: Boolean(decision), active: status === 'inspecting' },
    { title: '로봇 이동', detail: motion ? motionLabels[motion.status] : '명령 미수신', confirmed: motion?.status === 'succeeded', active: status === 'awaiting_motion' || status === 'moving' },
    { title: '결과 측정', detail: result ? taskSucceeded(result) ? '검사·이동 모두 확인' : '성공 조건 미충족' : '최종 결과 미수신', confirmed: taskSucceeded(result), active: false },
  ];
}
