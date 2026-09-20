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
  ModelImagesResponse,
  CacheStats,
  CacheOperationResult,
} from './types';

export const isWebApp = typeof window !== 'undefined' && !(window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;

const WEB_REQUEST_TIMEOUT_MS = 10_000;
const WEB_STATUS_TIMEOUT_MS = 4_000;

async function webFetch(
  url: string,
  init: RequestInit = {},
  timeoutMs = WEB_REQUEST_TIMEOUT_MS,
): Promise<Response> {
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, {
      ...init,
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

const command = <T,>(name: string, args: Record<string, unknown> = {}) =>
  isWebApp ? webCommand<T>(name, args) : invoke<T>(name, args);

export const fileUrl = (path: string) =>
  isWebApp
    ? '/api/file?path=' + encodeURIComponent(path)
    : convertFileSrc(path);

export const api = {
  isWebApp,
  getState: () => command<AppState>('get_app_state'),
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
  addSubfolderTags: () => command<number>('add_subfolder_tags'),
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
    command<void>('delete_model', { id }),
  getImages: (id: number, limit = 20) =>
    command<ModelImagesResponse>('get_model_images', { id, limit }),
  syncModelGallery: (id: number, targetCount = 20) =>
    command<boolean>('sync_model_gallery', { id, targetCount }),
  loadMoreModelExamples: (id: number, amount?: number) =>
    command<boolean>('load_more_model_examples', { id, targetCount: amount }),
  getExampleLoadAmount: () =>
    command<number>('get_example_load_amount'),
  setExampleLoadAmount: (amount: number) =>
    command<number>('set_example_load_amount', { amount }),
  refreshAllExamples: () =>
    command<ExamplesRefreshProgress>('refresh_all_examples'),
  getExamplesRefreshStatus: () =>
    command<ExamplesRefreshProgress | null>('get_examples_refresh_status'),
  importCivitai: (url: string) =>
    command<CivitaiImportPreview>('preview_civitai_import', { url }),
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
    command<ModelRecord>('refresh_model_civitai', { id }),
  linkModelCivitai: (id: number, url: string) =>
    command<ModelRecord>('link_model_civitai', { id, url }),
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
    command<CacheStats>('set_cache_max_bytes', { maxBytes }),
  setCacheLocation: (path: string) =>
    command<CacheStats>('set_cache_location', { path }),
  clearCacheImages: () =>
    command<CacheOperationResult>('clear_cache_images'),
  clearCompleteCache: () =>
    command<CacheOperationResult>('clear_complete_cache'),
  pruneCacheImages: (keepPerModel: number) =>
    command<CacheOperationResult>('prune_cache_images', { keepPerModel }),
  cleanCacheOrphans: () =>
    command<CacheOperationResult>('clean_cache_orphans'),
  getWebAppStatus: async () => {
    if (!isWebApp) return invoke<WebAppStatus>('get_web_app_status');
    const response = await webFetch('/api/status', {}, WEB_STATUS_TIMEOUT_MS);
    const payload = await response.json().catch(() => null);
    if (!response.ok) throw new Error(payload?.error || `Web API status failed: ${response.status}`);
    return payload as WebAppStatus;
  },
  setWebAppEnabled: (enabled: boolean) =>
    isWebApp
      ? Promise.reject(new Error('Web app controls are available from the desktop host only.'))
      : invoke<WebAppStatus>('toggle_web_app', { enabled }),
};

export async function subscribeToModelChanges(cb: () => void) {
  if (isWebApp) {
    let disposed = false;
    let inFlight = false;
    const poll = async () => {
      if (disposed || inFlight) return;
      inFlight = true;
      try {
        await api.getWebAppStatus();
        if (!disposed) cb();
      } catch {
        // The dedicated web-connection monitor reports the offline state.
      } finally {
        inFlight = false;
      }
    };
    void poll();
    const timer = window.setInterval(() => void poll(), 3000);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }
  return listen('models-changed', cb);
}

export async function subscribeToExamplesRefresh(cb: (progress: ExamplesRefreshProgress) => void) {
  if (isWebApp) {
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
    const timer = window.setInterval(() => void poll(), 750);
    return () => {
      disposed = true;
      window.clearInterval(timer);
    };
  }
  return listen<ExamplesRefreshProgress>('examples-refresh-progress', event => cb(event.payload));
}
