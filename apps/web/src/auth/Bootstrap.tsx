import { useCallback, useEffect, useMemo, useState } from 'react';
import { ArrowRight, Box, LockKeyhole, ShieldCheck } from 'lucide-react';
import type { PublicConfig } from '../api/contracts';
import { ApiClient } from '../api/client';
import { AuthenticationRequiredError, isAbort } from '../api/errors';
import { Brand, ErrorNotice, Loading } from '../ui/common';
import { SessionContext } from './context';
import type { AuthAdapter, SignedInAccount } from './types';
import { ConsoleApp } from '../ConsoleApp';

interface BootstrapProps {
  loadConfig(signal: AbortSignal): Promise<PublicConfig>;
  createAuth(config: PublicConfig): AuthAdapter | Promise<AuthAdapter>;
}

type BootstrapState =
  | { phase: 'loading' }
  | { phase: 'error'; error: unknown }
  | { phase: 'signedOut'; adapter: AuthAdapter }
  | { phase: 'authenticated'; adapter: AuthAdapter; account: SignedInAccount };

export function Bootstrap({ loadConfig, createAuth }: BootstrapProps) {
  const [state, setState] = useState<BootstrapState>({ phase: 'loading' });
  const [attempt, setAttempt] = useState(0);
  const [loginError, setLoginError] = useState<unknown>(null);
  const [redirecting, setRedirecting] = useState(false);

  useEffect(() => {
    let current = true;
    const controller = new AbortController();
    setState({ phase: 'loading' });
    setLoginError(null);
    setRedirecting(false);
    const start = async () => {
      try {
        const config = await loadConfig(controller.signal);
        if (!current) return;
        const adapter = await createAuth(config);
        if (!current) return;
        const account = await adapter.initialize();
        if (current) setState(account ? { phase: 'authenticated', adapter, account } : { phase: 'signedOut', adapter });
      } catch (error) {
        if (current && !isAbort(error)) setState({ phase: 'error', error });
      }
    };
    void start();
    return () => { current = false; controller.abort(); };
  }, [loadConfig, createAuth, attempt]);

  if (state.phase === 'authenticated') return <AuthenticatedSession adapter={state.adapter} account={state.account} />;

  const signIn = async () => {
    if (state.phase !== 'signedOut' || redirecting) return;
    setRedirecting(true);
    setLoginError(null);
    try {
      await state.adapter.signIn();
    } catch (error) {
      setRedirecting(false);
      setLoginError(error);
    }
  };

  return <main className="login-page">
    <div className="login-story">
      <Brand />
      <div className="login-story-content">
        <span className="eyebrow light">FROM OBSERVATION TO ACTION</span>
        <h1>관측하고, 판단하고.<br /><span>승인 후 실행합니다.</span></h1>
        <p>고객의 공장 환경을 구성하고 Isaac Sim의 실제 관측을 바탕으로 검사·분류 작업을 관리하세요.</p>
        <div className="login-principles">
          <span><Box size={20} aria-hidden="true" />Isaac Sim 물리 시뮬레이션</span>
          <span><ShieldCheck size={20} aria-hidden="true" />사람의 승인을 거치는 Foundry 계획</span>
          <span><LockKeyhole size={20} aria-hidden="true" />Microsoft Entra 인증</span>
        </div>
      </div>
      <p className="login-disclaimer">Azure 대상 콘솔 · 연결 전 상태<br />로그인은 Azure 배포 또는 GPU 동작 검증을 의미하지 않습니다.</p>
    </div>
    <div className="login-form-area">
      <div className="login-card">
        <div className="login-symbol"><LockKeyhole size={26} aria-hidden="true" /></div>
        <span className="eyebrow">SECURE WORKSPACE</span>
        <h2>작업 공간에 로그인</h2>
        <p>조직 계정으로 인증한 뒤 실제 환경, 런타임 상태와 실행 기록을 불러옵니다.</p>
        {state.phase === 'loading' && <Loading>API 설정과 인증 세션 확인 중…</Loading>}
        {state.phase === 'error' && <ErrorNotice error={state.error} title="콘솔을 시작할 수 없습니다" retry={() => setAttempt((value) => value + 1)} />}
        {state.phase === 'signedOut' && <button type="button" className="button login-button" onClick={() => void signIn()} disabled={redirecting}>
          <span className="microsoft-mark" aria-hidden="true"><i /><i /><i /><i /></span>
          {redirecting ? 'Microsoft 로그인으로 이동 중…' : 'Microsoft로 로그인'}<ArrowRight size={17} aria-hidden="true" />
        </button>}
        <ErrorNotice error={loginError} title="Microsoft 로그인 실패" />
        <div className="login-boundary"><ShieldCheck size={17} aria-hidden="true" /><p>설정이나 인증이 실패하면 접근을 중단합니다. 익명 접속, 데모 런타임 및 재생 대체 모드는 제공하지 않습니다.</p></div>
      </div>
      <span className="login-footer">FACTORY CONSOLE <span>HTTP API v1</span></span>
    </div>
  </main>;
}

function AuthenticatedSession({ adapter, account }: { adapter: AuthAdapter; account: SignedInAccount }) {
  const [expired, setExpired] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const unauthorized = useCallback(() => setExpired(true), []);
  const api = useMemo(() => new ApiClient(async () => {
    try {
      return await adapter.acquireToken();
    } catch (failure) {
      if (failure instanceof AuthenticationRequiredError) setExpired(true);
      throw failure;
    }
  }, undefined, unauthorized), [adapter, unauthorized]);

  const redirect = (action: 'signIn' | 'signOut') => {
    if (busy) return;
    setBusy(true);
    setError(null);
    void adapter[action]().catch((failure: unknown) => { setError(failure); setBusy(false); });
  };

  return <SessionContext.Provider value={{ account, expired, busy, signIn: () => redirect('signIn'), signOut: () => redirect('signOut') }}>
    {Boolean(error) && <div className="session-error"><ErrorNotice error={error} title="인증 작업 실패" /></div>}
    <ConsoleApp api={api} account={account} sessionExpired={expired} />
  </SessionContext.Provider>;
}
