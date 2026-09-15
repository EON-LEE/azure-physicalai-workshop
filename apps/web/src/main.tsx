import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { getPublicConfig } from './api/client';
import type { PublicConfig } from './api/contracts';
import { Bootstrap } from './auth/Bootstrap';
import './styles.css';

async function createAuthentication(config: PublicConfig) {
  const { createMsalAuth } = await import('./auth/msal');
  return createMsalAuth(config);
}

const root = document.getElementById('root');
if (!root) throw new Error('Factory Console root element is missing.');

createRoot(root).render(
  <StrictMode>
    <Bootstrap loadConfig={getPublicConfig} createAuth={createAuthentication} />
  </StrictMode>,
);
