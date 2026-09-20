export type ModelType = 'Checkpoint' | 'LoRA' | 'VAE' | 'ControlNet' | 'Embedding' | 'Upscaler' | 'Text Encoder' | 'CLIP Vision' | 'IP-Adapter' | 'Other';
export interface ModelRecord { id:number; path:string; relative_path:string; filename:string; model_type:ModelType; size_bytes:number; modified_at:number; civitai_model_id:number|null; civitai_version_id:number|null; civitai_url:string|null; civitai_name:string|null; version_name:string|null; base_model:string|null; creator:string|null; description:string|null; tags:string[]; activation_prompts:string[]; source_hash:string|null; thumbnail_path:string|null; cover_path:string|null; cover_source_image_id:number|null; cover_position_x:number; cover_position_y:number; downloaded_at:number; updated_at:number; }
export interface ModelImage { id:number; civitai_image_id:number; local_path:string|null; thumbnail_path:string|null; width:number|null; height:number|null; prompt:string|null; negative_prompt:string|null; steps:number|null; cfg:number|null; sampler:string|null; seed:number|null; meta_json:string|null; }
export interface CategoryStats { type:ModelType; count:number; bytes:number; }
export interface StorageStats { total_model_bytes:number; cached_bytes:number; categories:CategoryStats[]; }
export interface CacheStats {
  location: string;
  used_bytes: number;
  max_bytes: number;
  over_limit: boolean;
  files: number;
  image_files: number;
  image_bytes: number;
  featured_files: number;
  featured_bytes: number;
  gallery_files: number;
  gallery_bytes: number;
  thumbnail_files: number;
  thumbnail_bytes: number;
  cover_files: number;
  cover_bytes: number;
  other_files: number;
  other_bytes: number;
}

export interface CacheOperationResult {
  deleted_files: number;
  freed_bytes: number;
  remaining_bytes: number;
  over_limit: boolean;
}

export interface LibraryCounts { all:number; by_type:Record<string,number>; }
export interface TagRecord { name:string; count:number; }
export interface AppState { models_root:string|null; storage:StorageStats; }
export interface CivitaiImportPreview { model:{id?:number;name?:string;type?:string;description?:string|null;tags?:string[];creator?:string|null;thumbnail_path?:string|null}; version:{id:number;name:string;base_model:string|null;download_url:string;filename:string|null;size_bytes:number|null;activation_prompts:string[]}; target_directory:string; thumbnail_path?:string|null; images_count_hint?:number; }

export interface DownloadProgress {
  visible: boolean;
  task_id: string | null;
  filename: string;
  phase: string;
  downloaded_bytes: number;
  total_bytes: number | null;
  percent: number | null;
  error: string | null;
}

export interface ExamplesRefreshProgress {
  current: number;
  total: number;
  model_id: number | null;
  model_name: string | null;
  version_current: number;
  version_total: number;
  images_saved: number;
  status: string;
  done: boolean;
  error: string | null;
}

export interface ModelImagesResponse {
  images: ModelImage[];
  has_more: boolean;
}

export interface WebAppStatus { enabled: boolean; url: string | null; port: number; }
