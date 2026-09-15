export interface SignedInAccount {
  id: string;
  name: string;
  username: string;
}

export interface AuthAdapter {
  initialize(): Promise<SignedInAccount | null>;
  signIn(): Promise<void>;
  signOut(): Promise<void>;
  acquireToken(): Promise<string>;
}
