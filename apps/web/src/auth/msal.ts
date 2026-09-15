import { InteractionRequiredAuthError, PublicClientApplication, type AccountInfo } from '@azure/msal-browser';
import type { PublicConfig } from '../api/contracts';
import { AuthenticationRequiredError } from '../api/errors';
import type { AuthAdapter, SignedInAccount } from './types';

function toAccount(account: AccountInfo): SignedInAccount {
  return { id: account.homeAccountId, name: account.name || account.username, username: account.username };
}

export function createMsalAuth(config: PublicConfig): AuthAdapter {
  const scopes = [config.auth.scope];
  const tenantId = config.auth.tenant_id.toLowerCase();
  const belongsToTenant = (account: AccountInfo) => account.tenantId.toLowerCase() === tenantId;
  const redirectUri = `${window.location.origin}/`;
  const client = new PublicClientApplication({
    auth: {
      clientId: config.auth.client_id,
      authority: `https://login.microsoftonline.com/${tenantId}`,
      redirectUri,
      postLogoutRedirectUri: redirectUri,
      navigateToLoginRequestUrl: false,
    },
    cache: { cacheLocation: 'sessionStorage' },
  });
  let initialization: Promise<SignedInAccount | null> | null = null;

  return {
    initialize() {
      initialization ??= (async () => {
        await client.initialize();
        const redirect = await client.handleRedirectPromise();
        const accounts = client.getAllAccounts().filter(belongsToTenant);
        const active = redirect?.account ?? client.getActiveAccount();
        const account = active && belongsToTenant(active) ? active : accounts.length === 1 ? accounts[0] : null;
        if (!account) return null;
        client.setActiveAccount(account);
        return toAccount(account);
      })();
      return initialization;
    },
    async signIn() {
      await client.loginRedirect({ scopes, prompt: 'select_account' });
    },
    async signOut() {
      await client.logoutRedirect({ account: client.getActiveAccount() });
    },
    async acquireToken() {
      const account = client.getActiveAccount();
      if (!account || !belongsToTenant(account)) throw new AuthenticationRequiredError();
      try {
        const result = await client.acquireTokenSilent({ scopes, account });
        if (!result.accessToken) throw new AuthenticationRequiredError();
        return result.accessToken;
      } catch (error) {
        if (error instanceof InteractionRequiredAuthError) throw new AuthenticationRequiredError();
        throw error;
      }
    },
  };
}
