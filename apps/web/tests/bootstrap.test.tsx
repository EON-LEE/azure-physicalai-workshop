import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { describe, expect, it, vi } from 'vitest';
import { Bootstrap } from '../src/auth/Bootstrap';
import type { AuthAdapter } from '../src/auth/types';
import { ApiError, AuthenticationRequiredError } from '../src/api/errors';
import { account, config } from './fixtures/data';
import { deferred } from './helpers';

function auth(overrides: Partial<AuthAdapter> = {}): AuthAdapter {
  return {
    initialize: vi.fn().mockResolvedValue(null),
    signIn: vi.fn().mockResolvedValue(undefined),
    signOut: vi.fn().mockResolvedValue(undefined),
    acquireToken: vi.fn().mockRejectedValue(new AuthenticationRequiredError()),
    ...overrides,
  };
}

describe('production bootstrap failure states', () => {
  it('stops on a configuration failure and does not initialize auth or a fixture console', async () => {
    const createAuth = vi.fn(() => auth());
    const loadConfig = vi.fn().mockRejectedValue(new ApiError('not_configured', 'SPA 클라이언트 설정이 필요합니다.', 503));
    render(<Bootstrap loadConfig={loadConfig} createAuth={createAuth} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('콘솔을 시작할 수 없습니다');
    expect(createAuth).not.toHaveBeenCalled();
    expect(screen.queryByRole('button', { name: /Factory Live/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Microsoft로 로그인' })).not.toBeInTheDocument();
  });

  it('allows an explicit config retry without silently falling back', async () => {
    const loadConfig = vi.fn().mockRejectedValueOnce(new Error('설정 읽기 실패')).mockResolvedValue(config);
    const adapter = auth();
    render(<Bootstrap loadConfig={loadConfig} createAuth={() => adapter} />);
    await userEvent.click(await screen.findByRole('button', { name: '다시 확인' }));
    expect(await screen.findByRole('button', { name: 'Microsoft로 로그인' })).toBeEnabled();
    expect(loadConfig).toHaveBeenCalledTimes(2);
    expect(adapter.signIn).not.toHaveBeenCalled();
  });

  it('displays redirect initialization errors instead of treating them as signed out success', async () => {
    render(<Bootstrap loadConfig={async () => config} createAuth={() => auth({ initialize: vi.fn().mockRejectedValue(new Error('redirect_state_mismatch')) })} />);
    expect(await screen.findByRole('alert')).toHaveTextContent('redirect_state_mismatch');
    expect(screen.queryByRole('button', { name: 'Microsoft로 로그인' })).not.toBeInTheDocument();
  });

  it('requires an explicit login gesture and announces redirect failure', async () => {
    const adapter = auth({ signIn: vi.fn().mockRejectedValue(new Error('리디렉션 로그인 실패')) });
    render(<Bootstrap loadConfig={async () => config} createAuth={() => adapter} />);
    const button = await screen.findByRole('button', { name: 'Microsoft로 로그인' });
    expect(adapter.signIn).not.toHaveBeenCalled();
    await userEvent.click(button);
    expect(await screen.findByRole('alert')).toHaveTextContent('리디렉션 로그인 실패');
    expect(adapter.signIn).toHaveBeenCalledTimes(1);
  });

  it('makes an expired silent-token session visible and never makes unauthenticated API calls', async () => {
    const fetch = vi.spyOn(globalThis, 'fetch').mockRejectedValue(new Error('No network should be used without a token'));
    const adapter = auth({ initialize: vi.fn().mockResolvedValue(account) });
    render(<Bootstrap loadConfig={async () => config} createAuth={() => adapter} />);
    await waitFor(() => expect(screen.getByText('로그인이 만료되었거나 추가 인증이 필요합니다')).toBeInTheDocument());
    expect(fetch).not.toHaveBeenCalled();
    expect(screen.getByRole('button', { name: '관측하고 계획 요청' })).toBeDisabled();
  });

  it('aborts bootstrap on unmount and ignores late configuration', async () => {
    const pending = deferred<typeof config>();
    const loadConfig = vi.fn().mockReturnValue(pending.promise);
    const createAuth = vi.fn(() => auth());
    const { unmount } = render(<Bootstrap loadConfig={loadConfig} createAuth={createAuth} />);
    unmount();
    expect(loadConfig.mock.calls[0]?.[0].aborted).toBe(true);
    pending.resolve(config);
    await Promise.resolve();
    expect(createAuth).not.toHaveBeenCalled();
  });
});
