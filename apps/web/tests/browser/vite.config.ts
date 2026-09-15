import { fileURLToPath } from 'node:url';
import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  root: fileURLToPath(new URL('.', import.meta.url)),
  plugins: [react()],
  build: { outDir: '../../.fixture-dist', emptyOutDir: true },
  preview: {
    headers: {
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' blob: data:; frame-src 'none'; object-src 'none'; base-uri 'self'; form-action 'self'",
      'X-Content-Type-Options': 'nosniff',
    },
  },
});
