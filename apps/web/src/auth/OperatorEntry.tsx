import { getPublicConfig } from '../api/client';
import type { PublicConfig } from '../api/contracts';
import { Bootstrap } from './Bootstrap';

async function createAuthentication(config: PublicConfig) {
  const { createMsalAuth } = await import('./msal');
  return createMsalAuth(config);
}

export default function OperatorEntry() {
  return <Bootstrap loadConfig={getPublicConfig} createAuth={createAuthentication} />;
}
