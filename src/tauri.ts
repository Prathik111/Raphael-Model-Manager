import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import type {
  AppState,
  CivitaiImportPreview,
  ModelImage,
  ModelRecord,
  StorageStats,
  LibraryCounts,
  TagRecord,
  WebAppStatus,
} from './types';

export const isWebApp = typeof window !== 'undefined' && !(window as Window & { __TAURI_INTERNALS__?: unknown }).__TAURI_INTERNALS__;

async function webCommand<T>(command: string, args: Record<string, unknown> = {}): Promise<T> {
  const response = await fetch('/api/command/' + encodeURIComponent(command), {
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
  getImages: (id: number) =>
    command<ModelImage[]>('get_model_images', { id }),
  syncModelGallery: (id: number) =>
    command<void>('sync_model_gallery', { id }),
  importCivitai: (url: string) =>
    command<CivitaiImportPreview>('preview_civitai_import', { url }),
  installCivitai: (
    url: string,
    targetDirectory?: string,
    selectedType?: ModelRecord['model_type']
  ) =>
    command<ModelRecord>('install_civitai_model', {
      url,
      targetDirectory,
      selectedType,
    }),
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
  getWebAppStatus: async () => {
    if (!isWebApp) return invoke<WebAppStatus>('get_web_app_status');
    const response = await fetch('/api/status');
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
    const timer = window.setInterval(cb, 3000);
    return () => window.clearInterval(timer);
  }
  return listen('models-changed', cb);
}
