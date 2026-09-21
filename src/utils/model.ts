import type { ModelType } from '../types';

export function civitaiTypeToModelType(type?: string | null): ModelType {
  switch ((type || '').toLowerCase().replace(/[-_\s]/g, '')) {
    case 'checkpoint': return 'Checkpoint';
    case 'lora':
    case 'locon':
    case 'lycoris': return 'LoRA';
    case 'vae': return 'VAE';
    case 'controlnet': return 'ControlNet';
    case 'textualinversion':
    case 'embedding': return 'Embedding';
    case 'upscaler': return 'Upscaler';
    case 'clip':
    case 'clipvision':
    case 'ipadapter': return 'Other';
    default: return 'Other';
  }
}

export function folderForModelType(type: ModelType): string {
  switch (type) {
    case 'Checkpoint': return 'checkpoints';
    case 'LoRA': return 'loras';
    case 'VAE': return 'vae';
    case 'ControlNet': return 'controlnet';
    case 'Embedding': return 'embeddings';
    case 'Upscaler': return 'upscale_models';
    case 'Text Encoder': return 'text_encoders';
    case 'CLIP Vision': return 'clip_vision';
    case 'IP-Adapter': return 'ipadapter';
    default: return 'other';
  }
}

export function defaultImportDirectory(root: string | null, type: ModelType): string {
  if (!root) return '';
  const cleanRoot = root.replace(/[\\/]+$/, '');
  return cleanRoot + '\\' + folderForModelType(type);
}

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  const units = ['KB', 'MB', 'GB', 'TB'];
  let index = -1;
  let value = n;
  do {
    value /= 1024;
    index++;
  } while (value >= 1024 && index < units.length - 1);
  return `${value.toFixed(value >= 100 ? 0 : value >= 10 ? 1 : 2)} ${units[index]}`;
}

export function fmtCount(n: number): string {
  return new Intl.NumberFormat().format(Math.max(0, n));
}

export function fmtDateTime(seconds: number): string {
  if (!seconds) return '—';
  return new Date(seconds * 1000).toLocaleString();
}

export function initials(value: string): string {
  return value
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0])
    .join('')
    .toUpperCase();
}
