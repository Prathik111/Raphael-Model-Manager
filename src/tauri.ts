import { invoke, convertFileSrc } from '@tauri-apps/api/core';
import { listen } from '@tauri-apps/api/event';
import { open } from '@tauri-apps/plugin-dialog';
import type { AppState, CivitaiImportPreview, ModelImage, ModelRecord, StorageStats, LibraryCounts, TagRecord } from './types';

export const fileUrl = (path: string) => convertFileSrc(path);

export const api = {
  getState: () => invoke<AppState>('get_app_state'),
  chooseModelsFolder: async () => {
    const result = await open({ directory: true, multiple: false, title: 'Select your ComfyUI models folder' });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  chooseDirectory: async (defaultPath?: string) => {
    const result = await open({ directory: true, multiple: false, title: 'Choose model download folder', defaultPath });
    return Array.isArray(result) ? result[0] ?? null : result;
  },
  setModelsRoot: (path: string) => invoke<AppState>('set_models_root', { path }),
  listModels: (params?: { type?: string; query?: string; tags?: string[] }) => invoke<ModelRecord[]>('list_models', { ...params }),
  getLibraryCounts: () => invoke<LibraryCounts>('get_library_counts'),
  getTags: () => invoke<TagRecord[]>('get_tags'),
  setModelTags: (id: number, tags: string[]) => invoke<ModelRecord>('set_model_tags', { id, tags }),
  setModelType: (id: number, modelType: string) => invoke<ModelRecord>('set_model_type', { id, modelType }),
  getImages: (id: number) => invoke<ModelImage[]>('get_model_images', { id }),
  syncModelGallery: (id: number) => invoke<void>('sync_model_gallery', { id }),
  importCivitai: (url: string) => invoke<CivitaiImportPreview>('preview_civitai_import', { url }),
  installCivitai: (url: string, targetDirectory?: string, selectedType?: ModelRecord['model_type']) => invoke<ModelRecord>('install_civitai_model', { url, targetDirectory, selectedType }),
  refreshModel: (id: number) => invoke<ModelRecord>('refresh_model_civitai', { id }),
  linkModelCivitai: (id: number, url: string) => invoke<ModelRecord>('link_model_civitai', { id, url }),
  openFolder: (path: string) => invoke<void>('open_in_file_manager', { path }),
  setCivitaiToken: (token: string) => invoke<void>('set_civitai_token', { token }),
  getCivitaiTokenSet: () => invoke<boolean>('is_civitai_token_set'),
  getStorage: () => invoke<StorageStats>('get_storage_stats')
};

export async function subscribeToModelChanges(cb: () => void) {
  return listen('models-changed', cb);
}
