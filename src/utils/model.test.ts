import { describe, expect, it } from 'vitest';
import {
  civitaiTypeToModelType,
  defaultImportDirectory,
  folderForModelType,
  fmtBytes,
  fmtCount,
  initials,
} from './model';

describe('model utilities', () => {
  it('maps Civitai types consistently', () => {
    expect(civitaiTypeToModelType('LoCon')).toBe('LoRA');
    expect(civitaiTypeToModelType('textual-inversion')).toBe('Embedding');
    expect(civitaiTypeToModelType('CLIP Vision')).toBe('Other');
    expect(civitaiTypeToModelType(undefined)).toBe('Other');
  });

  it('uses stable ComfyUI folders', () => {
    expect(folderForModelType('Checkpoint')).toBe('checkpoints');
    expect(folderForModelType('LoRA')).toBe('loras');
    expect(folderForModelType('IP-Adapter')).toBe('ipadapter');
  });

  it('normalizes the selected models root for imports', () => {
    expect(defaultImportDirectory('C:\\ComfyUI\\models\\', 'LoRA')).toBe('C:\\ComfyUI\\models\\loras');
    expect(defaultImportDirectory(null, 'LoRA')).toBe('');
  });

  it('formats bytes and counts predictably', () => {
    expect(fmtBytes(500)).toBe('500 B');
    expect(fmtBytes(10240)).toBe('10.0 KB');
    expect(fmtCount(1234567)).toBe('1,234,567');
    expect(fmtCount(-5)).toBe('0');
  });

  it('builds initials from the first two words', () => {
    expect(initials('Raphael Model Manager')).toBe('RM');
    expect(initials('  single  ')).toBe('S');
  });
});
