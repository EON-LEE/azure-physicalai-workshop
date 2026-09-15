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
        }
      }
    },
  }],
  build: {
    sourcemap: false,
    manifest: true,
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
