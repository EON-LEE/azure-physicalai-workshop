import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react(), {
    name: 'exclude-test-runtime',
    generateBundle(_options, bundle) {
      for (const output of Object.values(bundle)) {
        if (output.type !== 'chunk') continue;
        for (const module of Object.keys(output.modules)) {
          if (/[/\\](tests|fixtures)[/\\]/i.test(module)) {
            this.error(`Test-only module reached the production bundle: ${module}`);
          }
          const visited = new Set<string>();
          const inspectPublicImports = (name: string) => {
            if (visited.has(name)) return;
            visited.add(name);
            const chunk = bundle[name];
            if (!chunk || chunk.type !== 'chunk') return;
            for (const module of Object.keys(chunk.modules)) {
              if (/[/\\]src[/\\]auth[/\\]|[/\\]src[/\\]ConsoleApp\.tsx|[/\\]src[/\\]api[/\\]client\.ts|[/\\]@azure[/\\]msal-/i.test(module)) {
                this.error(`Protected operator code reached the public first-load graph: ${module}`);
              }
            }
            for (const dependency of chunk.imports) inspectPublicImports(dependency);
          };
          for (const chunk of Object.values(bundle)) {
            if (chunk.type === 'chunk' && chunk.isEntry) inspectPublicImports(chunk.fileName);
          }
        }
      }
    },
  }],
  build: {
    sourcemap: false,
    manifest: true,
  },
  preview: {
    headers: {
      'Content-Security-Policy': "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; connect-src 'self' https://login.microsoftonline.com; frame-src 'self' https://login.microsoftonline.com; frame-ancestors 'none'; object-src 'none'; base-uri 'self'",
      'X-Content-Type-Options': 'nosniff',
      'Referrer-Policy': 'no-referrer',
    },
  },
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: false,
      },
    },
  },
});
