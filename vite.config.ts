import { createLogger, defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

const logger = createLogger();
const originalLoggerError = logger.error;

logger.error = (message, options) => {
  const proxyError = options?.error as
    | { code?: string; address?: string; port?: number }
    | undefined;

  const expectedWebApiConnectionRefused =
    message.includes('http proxy error:') &&
    proxyError?.code === 'ECONNREFUSED' &&
    proxyError?.address === '127.0.0.1' &&
    proxyError?.port === 1421;

  if (expectedWebApiConnectionRefused) return;

  originalLoggerError(message, options);
};

export default defineConfig({
  customLogger: logger,
  plugins: [react()],
  clearScreen: false,
  server: {
    host: '0.0.0.0',
    port: 1420,
    strictPort: true,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:1421',
        changeOrigin: true,
      },
    },
  },
  envPrefix: ['VITE_', 'TAURI_']
});
