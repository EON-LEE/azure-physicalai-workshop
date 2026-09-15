import { beforeEach, describe, expect, it, vi } from 'vitest';
import { InteractionRequiredAuthError } from '@azure/msal-browser';
import { AuthenticationRequiredError } from '../src/api/errors';
import { createMsalAuth } from '../src/auth/msal';
import { config } from './fixtures/data';

const sdk = vi.hoisted(() => ({
  construct: vi.fn(),
  initialize: vi.fn(),
  handleRedirectPromise: vi.fn(),
  getAllAccounts: vi.fn(),
  getActiveAccount: vi.fn(),
  setActiveAccount: vi.fn(),
  loginRedirect: vi.fn(),
  logoutRedirect: vi.fn(),
  acquireTokenSilent: vi.fn(),
}));

vi.mock('@azure/msal-browser', () => ({
  PublicClientApplication: class {
    constructor(options: unknown) { sdk.construct(options); return sdk; }
  },
  InteractionRequiredAuthError: class extends Error {},
}));

const account = {
  homeAccountId: 'test-only-account',
  username: 'fixture-operator@example.invalid',
  name: '테스트 운영자',
  tenantId: config.auth.tenant_id,
};

beforeEach(() => {
  sdk.initialize.mockResolvedValue(undefined);
  sdk.handleRedirectPromise.mockResolvedValue(null);
  sdk.getAllAccounts.mockReturnValue([account]);
  sdk.getActiveAccount.mockReturnValue(account);
  sdk.setActiveAccount.mockReturnValue(undefined);
  sdk.loginRedirect.mockResolvedValue(undefined);
  sdk.logoutRedirect.mockResolvedValue(undefined);
  sdk.acquireTokenSilent.mockResolvedValue({ accessToken: 'test-only-access-token' });
});

describe('small MSAL redirect adapter', () => {
  it('uses only tenant/client/scope from bootstrap and a same-origin redirect URI', async () => {
    const adapter = createMsalAuth(config);
    await adapter.initialize();
    await adapter.signIn();
    expect(sdk.construct).toHaveBeenCalledWith({
      auth: {
        clientId: config.auth.client_id,
        authority: `https://login.microsoftonline.com/${config.auth.tenant_id}`,
        redirectUri: `${window.location.origin}/`,
        postLogoutRedirectUri: `${window.location.origin}/`,
        navigateToLoginRequestUrl: false,
      },
      cache: { cacheLocation: 'sessionStorage' },
    });
    expect(sdk.loginRedirect).toHaveBeenCalledWith({ scopes: [config.auth.scope], prompt: 'select_account' });
  });

  it('initializes once even when React StrictMode repeats initialization', async () => {
    const adapter = createMsalAuth(config);
    const [one, two] = await Promise.all([adapter.initialize(), adapter.initialize()]);
    expect(one).toEqual(two);
    expect(sdk.initialize).toHaveBeenCalledTimes(1);
    expect(sdk.handleRedirectPromise).toHaveBeenCalledTimes(1);
  });

  it('acquires delegated access tokens silently for the active tenant account', async () => {
    const adapter = createMsalAuth(config);
    await adapter.initialize();
    await expect(adapter.acquireToken()).resolves.toBe('test-only-access-token');
    expect(sdk.acquireTokenSilent).toHaveBeenCalledWith({ scopes: [config.auth.scope], account });
    expect(sdk.loginRedirect).not.toHaveBeenCalled();
  });

  it('requires a user gesture for interaction-required errors instead of an anonymous retry', async () => {
    sdk.acquireTokenSilent.mockRejectedValue(new InteractionRequiredAuthError('interaction_required'));
    const adapter = createMsalAuth(config);
    await adapter.initialize();
    await expect(adapter.acquireToken()).rejects.toBeInstanceOf(AuthenticationRequiredError);
    expect(sdk.loginRedirect).not.toHaveBeenCalled();
  });

  it('does not silently select another tenant or arbitrarily choose among multiple accounts', async () => {
    sdk.getActiveAccount.mockReturnValue({ ...account, tenantId: 'different-tenant' });
    sdk.getAllAccounts.mockReturnValue([{ ...account, tenantId: 'different-tenant' }]);
    await expect(createMsalAuth(config).initialize()).resolves.toBeNull();
    sdk.getActiveAccount.mockReturnValue(null);
    sdk.getAllAccounts.mockReturnValue([account, { ...account, homeAccountId: 'another-account' }]);
    await expect(createMsalAuth(config).initialize()).resolves.toBeNull();
  });

  it('compares UUID tenants case-insensitively but rejects a foreign active token account', async () => {
    const tenant = 'ABCDEF12-3456-4789-8ABC-DEF123456789';
    sdk.getActiveAccount.mockReturnValue({ ...account, tenantId: tenant.toLowerCase() });
    sdk.getAllAccounts.mockReturnValue([]);
    const adapter = createMsalAuth({ ...config, auth: { ...config.auth, tenant_id: tenant } });
    await expect(adapter.initialize()).resolves.toMatchObject({ id: account.homeAccountId });
    sdk.getActiveAccount.mockReturnValue({ ...account, tenantId: 'different-tenant' });
    await expect(adapter.acquireToken()).rejects.toBeInstanceOf(AuthenticationRequiredError);
    expect(sdk.acquireTokenSilent).not.toHaveBeenCalled();
  });
});
