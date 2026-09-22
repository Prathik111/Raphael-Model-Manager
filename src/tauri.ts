import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import type {
  AppState,
  CivitaiImportPreview,
  DownloadProgress,
  ModelRecord,
  StorageStats,
  LibraryCounts,
  TagRecord,
  WebAppStatus,
  ExamplesRefreshProgress,
  ModelTagsRefreshProgress,
  ModelImagesResponse,
  CacheStats,
  CacheOperationResult,
} from './types';

const WEB_REQUEST_TIMEOUT_MS = 10_000;
const WEB_STATUS_TIMEOUT_MS = 4_000;
const WEB_TOKEN_STORAGE_KEY = 'raphael.webToken';

function isDesktopOrigin(): boolean {
  if (typeof window === 'undefined') return false;

  const { protocol, hostname } = window.location;
  return (
    protocol === 'tauri:' ||
    protocol === 'asset:' ||
    hostname === 'localhost' ||
    hostname === '127.0.0.1' ||
    hostname === '[::1]'
  );
}

function hasWebAccessToken(): boolean {
  if (typeof window === 'undefined') return false;
  try {
    if (isDesktopOrigin()) {
      window.sessionStorage.removeItem(WEB_TOKEN_STORAGE_KEY);
      return false;
    }

    if (window.sessionStorage.getItem(WEB_TOKEN_STORAGE_KEY)) return true;

    const rawHash = window.location.hash.startsWith('#')
      ? window.location.hash.slice(1)
      : window.location.hash;
    return Boolean(new URLSearchParams(rawHash).get('access_token'));
  } catch {
    return false;
  }
}

export const isWebApp = !isDesktopOrigin() && hasWebAccessToken();

function useWebTransport(): boolean {
  return !isDesktopOrigin() && isWebApp;
}

function getWebAccessToken(): string | null {
  if (!isWebApp) return null;
  try {
    const stored = window.sessionStorage.getItem(WEB_TOKEN_STORAGE_KEY);
    if (stored) return stored;

    const rawHash = window.location.hash.startsWith('#')
      ? window.location.hash.slice(1)
      : window.location.hash;
    const params = new URLSearchParams(rawHash);
    const token = params.get('access_token');
    if (!token) return null;

    window.sessionStorage.setItem(WEB_TOKEN_STORAGE_KEY, token);
    const cleanUrl = window.location.pathname + window.location.search;
    window.history.replaceState(null, document.title, cleanUrl);
    return token;
  } catch {
    return null;
  }
}

async function webFetch(
  url: string,
  init: RequestInit = {},
  timeoutMs = WEB_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const headers = new Headers(init.headers);
    const token = getWebAccessToken();
    if (token && !headers.has('Authorization')) {
      headers.set('Authorization', `Bearer ${token}`);
    }
    return await fetch(url, {
      ...init,
      headers,
      credentials: 'same-origin',
      cache: 'no-store',
      signal: controller.signal,
    });
  } catch (error) {
    if (error instanceof DOMException && error.name === 'AbortError') {
      throw new Error(`Web API request timed out after ${Math.round(timeoutMs / 1000)}s`);
    }
    throw new Error(`Web API connection failed: ${error instanceof Error ? error.message : String(error)}`);
  } finally {
    window.clearTimeout(timeout);
  }
}

async function webCommand<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  const response = await webFetch('/api/command/' + encodeURIComponent(command), {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(args),
  });

  const payload = await response.json().catch(() => null);
  if (!response.ok) {
    throw new Error(payload?.error || `Web API request failed: ${response.status}`);
  }
  return payload as T;
}

type WebTaskStart = { task_id: string; state: 'queued' };
type WebTaskStatus = {
  task_id: string;
  state: 'queued' | 'running' | 'completed' | 'failed';
  result: unknown | null;
  error: string | null;
};

const sleep = (ms: number) => new Promise<void>(resolve => window.setTimeout(resolve, ms));

class WebTaskFailedError extends Error {
  readonly webTaskFailure = true;
}

async function webTaskCommand<T>(commandName: string, args: Record<string, unknown> = {}): Promise<T> {
  if (!isWebApp) {
    return invoke<T>(commandName, args);
  }

  const startResponse = await webFetch('/api/task/start', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ command: commandName, args }),
  }, 5000);

  const startPayload = await startResponse.json().catch(() => null);
  if (!startResponse.ok) {
    throw new Error(startPayload?.error || `Web task start failed: ${startResponse.status}`);
  }

  const { task_id } = startPayload as WebTaskStart;
  if (!task_id) throw new Error('Web task start returned no task ID');

  const deadline = Date.now() + 15 * 60 * 1000;
  let connectionFailures = 0;
  let delayMs = 400;

  while (Date.now() < deadline) {
    try {
      const response = await webFetch(
        '/api/tasks/' + encodeURIComponent(task_id),
        {},
        5000,
      );
      const payload = await response.json().catch(() => null);

      if (!response.ok) {
        throw new Error(payload?.error || `Web task status failed: ${response.status}`);
      }

      connectionFailures = 0;
      delayMs = 400;

      const task = payload as WebTaskStatus;
      if (task.state === 'completed') {
        return task.result as T;
      }
      if (task.state === 'failed') {
        throw new WebTaskFailedError(task.error || 'Web task failed');
      }

      await sleep(delayMs);
      delayMs = Math.min(1200, delayMs + 100);
    } catch (error) {
      if (error instanceof WebTaskFailedError) throw error;
      connectionFailures += 1;
      if (connectionFailures >= 20) {
        throw error instanceof Error ? error : new Error(String(error));
      }
      await sleep(Math.min(5000, 500 * connectionFailures));
    }
  }

  throw new Error(`Web task timed out after 15 minutes (task ${task_id})`);
}

const command = <T,>(name: string, args: Record<string, unknown> = {}) =>
  useWebTransport() ? webCommand<T>(name, args) : invoke<T>(name, args);

export const fileUrl = (path: string) =>
  useWebTransport()
    ? '/api/file?path=' + encodeURIComponent(path)
    : convertFileSrc(path);

export const api = {
  isWebApp,
  getState: () => command<AppState>('get_app_state'),
  checkRegistryHealth: () => command<boolean>('check_registry_health'),
  chooseModelsFolder: async () => {
    if (isWebApp) throw new Error('Choose the models folder from the Raphael desktop app on the host PC.');
    const result = await open({ directory: true, multiple: false, title: 'Select your ComfyUI models folder' });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  chooseDirectory: async (defaultPath?: string) => {
    if (isWebApp) throw new Error('Folder browsing is only available in the Raphael desktop app.');
    const result = await open({ directory: true, multiple: false, title: 'Choose model download folder', defaultPath });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  chooseCacheDirectory: async (defaultPath?: string) => {
    if (isWebApp) throw new Error('Folder browsing is only available in the Raphael desktop app.');
    const result = await open({ directory: true, multiple: false, title: 'Select Raphael cache folder', defaultPath });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  chooseImageFile: async () => {
    if (isWebApp) throw new Error('Custom cover selection is only available in the Raphael desktop app.');
    const result = await open({
      directory: false,
      multiple: false,
      title: 'Choose custom cover image',
      filters: [{ name: 'Images', extensions: ['png', 'jpg', 'jpeg', 'webp'] }],
    });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  setModelsRoot: (path: string) => command<AppState>('set_models_root', { path }),
  listModels: (params?: { type?: string; query?: string; tags?: string[] }) =>
    command<ModelRecord[]>('list_models', { ...params }),
  getLibraryCounts: () => command<LibraryCounts>('get_library_counts'),
  getTags: () => command<TagRecord[]>('get_tags'),
  addSubfolderTags: () => useWebTransport() ? webTaskCommand<number>('add_subfolder_tags') : command<number>('add_subfolder_tags'),
  setModelTags: (id: number, tags: string[]) =>
    command<ModelRecord>('set_model_tags', { id, tags }),
  setModelType: (id: number, modelType: string) =>
    command<ModelRecord>('set_model_type', { id, modelType }),
  setModelCoverPosition: (id: number, x: number, y: number) =>
    command<ModelRecord>('set_model_cover_position', { id, x, y }),
  setModelCoverFromImage: (id: number, imageId: number) =>
    command<ModelRecord>('set_model_cover_from_image', { id, imageId }),
  setModelCustomCover: (id: number, sourcePath: string) =>
    command<ModelRecord>('set_model_custom_cover', { id, sourcePath }),
  resetModelCover: (id: number) =>
    command<ModelRecord>('reset_model_cover', { id }),
  deleteModel: (id: number) =>
    useWebTransport() ? webTaskCommand<void>('delete_model', { id }) : command<void>('delete_model', { id }),
  getImages: (id: number, limit = 20) =>
    command<ModelImagesResponse>('get_model_images', { id, limit }),
  syncModelGallery: (id: number, targetCount = 20) =>
    useWebTransport() ? webTaskCommand<boolean>('sync_model_gallery', { id, targetCount }) : command<boolean>('sync_model_gallery', { id, targetCount }),
  loadMoreModelExamples: (id: number, amount?: number) =>
    useWebTransport() ? webTaskCommand<boolean>('load_more_model_examples', { id, amount }) : command<boolean>('load_more_model_examples', { id, amount }),
  getExampleLoadAmount: () =>
    command<number>('get_example_load_amount'),
  setExampleLoadAmount: (amount: number) =>
    command<number>('set_example_load_amount', { amount }),
  refreshAllExamples: () =>
    command<ExamplesRefreshProgress>('refresh_all_examples'),
  getExamplesRefreshStatus: () =>
    command<ExamplesRefreshProgress | null>('get_examples_refresh_status'),
  refreshAllModelTags: () =>
    command<ModelTagsRefreshProgress>('refresh_all_model_tags'),
  getModelTagsRefreshStatus: () =>
    command<ModelTagsRefreshProgress | null>('get_model_tags_refresh_status'),
  importCivitai: (url: string) =>
    useWebTransport() ? webTaskCommand<CivitaiImportPreview>('preview_civitai_import', { url }) : command<CivitaiImportPreview>('preview_civitai_import', { url }),
  installCivitai: (
    url: string,
    targetDirectory?: string,
    selectedType?: ModelRecord['model_type']
  ) =>
    command<DownloadProgress>('install_civitai_model', {
      url,
      targetDirectory,
      selectedType,
    }),
  getDownloadProgress: () =>
    command<DownloadProgress[]>('get_download_progress'),
  getParallelDownloads: () =>
    command<number>('get_parallel_downloads'),
  setParallelDownloads: (value: number) =>
    command<number>('set_parallel_downloads', { value }),
  clearDownloadProgress: (taskId: string) =>
    command<void>('clear_download_progress', { taskId }),
  refreshModel: (id: number) =>
    useWebTransport() ? webTaskCommand<ModelRecord>('refresh_model_civitai', { id }) : command<ModelRecord>('refresh_model_civitai', { id }),
  linkModelCivitai: (id: number, url: string) =>
    useWebTransport() ? webTaskCommand<ModelRecord>('link_model_civitai', { id, url }) : command<ModelRecord>('link_model_civitai', { id, url }),
  openFolder: (path: string) =>
    command<void>('open_in_file_manager', { path }),
  setCivitaiToken: (token: string) =>
    command<void>('set_civitai_token', { token }),
  getCivitaiTokenSet: () =>
    command<boolean>('is_civitai_token_set'),
  getStorage: () =>
    command<StorageStats>('get_storage_stats'),
  getCacheStats: () =>
    command<CacheStats>('get_cache_stats'),
  setCacheMaxBytes: (maxBytes: number) =>
    useWebTransport() ? webTaskCommand<CacheStats>('set_cache_max_bytes', { maxBytes }) : command<CacheStats>('set_cache_max_bytes', { maxBytes }),
  setCacheLocation: (path: string) =>
    useWebTransport() ? webTaskCommand<CacheStats>('set_cache_location', { path }) : command<CacheStats>('set_cache_location', { path }),
  clearCacheImages: () =>
    useWebTransport() ? webTaskCommand<CacheOperationResult>('clear_cache_images') : command<CacheOperationResult>('clear_cache_images'),
  clearCompleteCache: () =>
    useWebTransport() ? webTaskCommand<CacheOperationResult>('clear_complete_cache') : command<CacheOperationResult>('clear_complete_cache'),
  pruneCacheImages: (keepPerModel: number) =>
    useWebTransport() ? webTaskCommand<CacheOperationResult>('prune_cache_images', { keepPerModel }) : command<CacheOperationResult>('prune_cache_images', { keepPerModel }),
  cleanCacheOrphans: () =>
    useWebTransport() ? webTaskCommand<CacheOperationResult>('clean_cache_orphans') : command<CacheOperationResult>('clean_cache_orphans'),
  getWebAppStatus: async () => {
    if (!useWebTransport()) return invoke<WebAppStatus>('get_web_app_status');
    const response = await webFetch('/api/status', {}, WEB_STATUS_TIMEOUT_MS);
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(payload?.error || `Web API status failed: ${response.status}`);
    return payload as WebAppStatus;
  },
  checkWebHealth: async () => {
    if (!useWebTransport()) return { latencyMs: 0 };
    const started = performance.now();
    const response = await webFetch('/api/health', {}, 2500);
    if (!response.ok) throw new Error(`Web API health check failed: ${response.status}`);
    return { latencyMs: Math.round(performance.now() - started) };
  },
  setWebAppEnabled: (enabled: boolean) =>
    useWebTransport()
      ? Promise.reject(new Error('Web app controls are available from the desktop host only.'))
      : invoke<WebAppStatus>('toggle_web_app', { enabled }),
};

export async function subscribeToModelChanges(cb: () => void) {
  if (useWebTransport()) {
    let disposed = false;
    let revision = 0;

    const waitForChanges = async () => {
      while (!disposed) {
        try {
          const response = await webFetch(
            '/api/changes?since=' + encodeURIComponent(String(revision)),
            {},
            32_000,
          );
          if (!response.ok) {
            throw new Error(`Web change monitor failed: ${response.status}`);
          }

          const payload = await response.json().catch(() => null) as { revision?: number } | null;
          const nextRevision = Number(payload?.revision ?? revision);
          if (nextRevision !== revision) {
            revision = nextRevision;
            if (!disposed) cb();
          } else if (!disposed) {
            // The long-poll timed out without a change; immediately wait again.
          }
        } catch {
          if (disposed) return;
          await sleep(1500);
        }
      }
    };

    void waitForChanges();
    return () => {
      disposed = true;
    };
  }
  return listen('models-changed', cb);
}

export async function subscribeToExamplesRefresh(cb: (progress: ExamplesRefreshProgress) => void) {
  if (useWebTransport()) {
    let disposed = false;
    let inFlight = false;
    const poll = async () => {
      if (disposed || inFlight) return;
      inFlight = true;
      try {
        const response = await webFetch('/api/command/get_examples_refresh_status', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: '{}',
        });
        if (!response.ok) return;
        const payload = await response.json().catch(() => null);
        if (!disposed && payload) cb(payload as ExamplesRefreshProgress);
      } catch {
        // Connection state is handled by the dedicated heartbeat.
      } finally {
        inFlight = false;
      }
    };
    await poll();
    const timer = window.setInterval(() => void poll(), 1200);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }
  return listen<ExamplesRefreshProgress>('examples-refresh-progress', event => cb(event.payload));
}

export async function subscribeToModelTagsRefresh(cb: (progress: ModelTagsRefreshProgress) => void) {
  if (useWebTransport()) {
    let disposed = false;
    let inFlight = false;
    const poll = async () => {
      if (disposed || inFlight) return;
      inFlight = true;
      try {
        const response = await webFetch('/api/command/get_model_tags_refresh_status', {
          method: 'POST',
          headers: { 'content-type': 'application/json' },
          body: '{}',
        });
        if (!response.ok) return;
        const payload = await response.json().catch(() => null);
        if (!disposed && payload) cb(payload as ModelTagsRefreshProgress);
      } catch {
        // Connection state is handled by the dedicated heartbeat.
      } finally {
        inFlight = false;
      }
    };
    await poll();
    const timer = window.setInterval(() => void poll(), 1200);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }
  return listen<ModelTagsRefreshProgress>('model-tags-refresh-progress', event => cb(event.payload));
}
