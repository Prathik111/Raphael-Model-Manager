import React from 'react';
import ReactDOM from 'react-dom/client';
import { isTauri } from '@tauri-apps/api/core';
import './styles.css';

const WEB_TOKEN_STORAGE_KEY = 'raphael.webToken';
const TAURI_INIT_TIMEOUT_MS = 1500;
const TAURI_INIT_POLL_MS = 20;

function hasExplicitWebToken(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    if (window.sessionStorage.getItem(WEB_TOKEN_STORAGE_KEY)) return true;
    const rawHash = window.location.hash.startsWith('#')
      ? window.location.hash.slice(1)
      : window.location.hash;
    return Boolean(new URLSearchParams(rawHash).get('access_token'));
  } catch {
    return false;
  }
}

async function waitForTauriRuntime(): Promise<void> {
  if (isTauri() || hasExplicitWebToken()) return;

  const deadline = Date.now() + TAURI_INIT_TIMEOUT_MS;
  while (Date.now() < deadline) {
    await new Promise<void>(resolve => window.setTimeout(resolve, TAURI_INIT_POLL_MS));
    if (isTauri()) return;
  }
}

async function bootstrap() {
  await waitForTauriRuntime();
  const { default: App } = await import('./App');

  ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode><App /></React.StrictMode>
  );
}

void bootstrap();
