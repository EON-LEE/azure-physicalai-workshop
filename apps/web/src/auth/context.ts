import { createContext, useContext } from 'react';
import type { SignedInAccount } from './types';

export interface SessionControls {
  account: SignedInAccount;
  expired: boolean;
  busy: boolean;
  signIn(): void;
  signOut(): void;
}

export const SessionContext = createContext<SessionControls | null>(null);
export const useSession = () => useContext(SessionContext);
