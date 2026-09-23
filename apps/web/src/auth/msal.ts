import { InteractionRequiredAuthError, PublicClientApplication, type AccountInfo } from '@azure/msal-browser';
import type { PublicConfig } from '../api/contracts';
import { AuthenticationRequiredError } from '../api/errors';
import type { AuthAdapter, SignedInAccount } from './types';
import { customerExperimentLink, readCustomerExperiment } from '../environment/customerExperiment';

const experimentStatePrefix = 'physicalai-experiment:';

function toAccount(account: AccountInfo): SignedInAccount {
  return { id: account.homeAccountId, name: account.name || account.username, username: account.username };
}

export function createMsalAuth(config: PublicConfig): AuthAdapter {
  const scopes = [config.auth.scope];
  const tenantId = config.auth.tenant_id.toLowerCase();
  const belongsToTenant = (account: AccountInfo) => account.tenantId.toLowerCase() === tenantId;
  const redirectUri = `${window.location.origin}/operator`;
  const client = new PublicClientApplication({
    auth: {
      clientId: config.auth.client_id,
      authority: `https://login.microsoftonline.com/${tenantId}`,
      redirectUri,
      postLogoutRedirectUri: `${window.location.origin}/`,
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
        if (redirect?.state?.startsWith(experimentStatePrefix)) {
          const experiment = readCustomerExperiment(redirect.state.slice(experimentStatePrefix.length));
          if (experiment.error) throw new Error(experiment.error);
          window.history.replaceState(null, '', customerExperimentLink(experiment.value));
        }
        client.setActiveAccount(account);
        return toAccount(account);
      })();
      return initialization;
    },
    async signIn() {
      const hasExperiment = new URLSearchParams(window.location.search).has('experiment');
      const experiment = readCustomerExperiment(window.location.search);
      if (hasExperiment && experiment.error) throw new Error(experiment.error);
      await client.loginRedirect({
        scopes, prompt: 'select_account',
        ...(hasExperiment ? {
          state: experimentStatePrefix + new URLSearchParams({
            experiment: experiment.value.experiment, sample: experiment.value.sample,
          }),
        } : {}),
      });
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
