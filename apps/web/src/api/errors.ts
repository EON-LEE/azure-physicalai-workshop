export class ApiError extends Error {
  constructor(
    public readonly code: string,
    message: string,
    public readonly status: number = 0,
    public readonly details: unknown[] = [],
  ) {
    super(message);
    this.name = 'ApiError';
  }
}

export class AuthenticationRequiredError extends Error {
  constructor(message = '로그인 세션을 다시 확인해야 합니다. Microsoft Entra로 다시 로그인하세요.') {
    super(message);
    this.name = 'AuthenticationRequiredError';
  }
}

export function isAbort(error: unknown): boolean {
  return error instanceof Error && error.name === 'AbortError';
}

export function describeError(error: unknown): string {
  return error instanceof Error ? error.message : '알 수 없는 오류가 발생했습니다. 작업 결과를 확인한 뒤 다시 시도하세요.';
}

export function canRetryRead(error: unknown): boolean {
  return error instanceof ApiError &&
    (error.code === 'network_error' || error.code === 'request_timeout' || error.status === 429 || error.status >= 500);
}

export function needsLogin(error: unknown): boolean {
  return error instanceof AuthenticationRequiredError || (error instanceof ApiError && error.status === 401);
}

export function errorGuidance(error: unknown): string | null {
  if (needsLogin(error)) return '인증이 필요합니다. 익명 접근으로 전환하지 않습니다.';
  if (!(error instanceof ApiError)) return null;
  if (error.status === 403) return '현재 계정에 작업 권한이 없습니다. 테넌트와 API 권한을 관리자에게 확인하세요.';
  if (error.status === 409) return '서버 상태 또는 저장 버전이 변경되었습니다. 최신 상태를 확인하세요. 자동 덮어쓰기나 자동 재승인을 하지 않습니다.';
  if (error.status === 422) return '입력값과 아래 검증 내용을 확인하세요. 원본 JSON은 유지됩니다.';
  if (error.status === 503) return '필수 Azure 또는 시뮬레이터 의존성이 준비되지 않았습니다. 대체 실행 결과를 만들지 않습니다.';
  return null;
}
