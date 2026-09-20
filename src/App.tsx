import { useEffect, useRef, useState } from 'react';
import { api, fileUrl, subscribeToModelChanges } from './tauri';
import type { AppState, CivitaiImportPreview, DownloadProgress, ModelImage, ModelRecord, ModelType, LibraryCounts, TagRecord } from './types';

const TYPES: Array<{ key: ModelType | 'All'; label: string }> = [
  { key: 'All', label: 'ALL' }, { key: 'Checkpoint', label: 'CHECKPOINTS' }, { key: 'LoRA', label: 'LORAS' },
  { key: 'VAE', label: 'VAE' }, { key: 'ControlNet', label: 'CONTROLNET' }, { key: 'Embedding', label: 'EMBEDDINGS' },
  { key: 'Upscaler', label: 'UPSCALERS' }, { key: 'Other', label: 'OTHER' }
];

const IMPORT_TYPES: ModelType[] = [
  'Checkpoint', 'LoRA', 'VAE', 'ControlNet', 'Embedding', 'Upscaler', 'Other'
];

const MODEL_TYPES: Array<{value:string; label:string}> = [
  { value: 'Auto', label: 'AUTO · FOLDER' },
  { value: 'Checkpoint', label: 'CHECKPOINT' },
  { value: 'LoRA', label: 'LORA' },
  { value: 'VAE', label: 'VAE' },
  { value: 'ControlNet', label: 'CONTROLNET' },
  { value: 'Embedding', label: 'EMBEDDING' },
  { value: 'Upscaler', label: 'UPSCALER' },
  { value: 'Text Encoder', label: 'TEXT ENCODER' },
  { value: 'CLIP Vision', label: 'CLIP VISION' },
  { value: 'IP-Adapter', label: 'IP-ADAPTER' },
  { value: 'Other', label: 'OTHER' }
];

function civitaiTypeToModelType(type?: string | null): ModelType {
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

function folderForModelType(type: ModelType): string {
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

function defaultImportDirectory(root: string | null, type: ModelType): string {
  if (!root) return '';
  const cleanRoot = root.replace(/[\\/]+$/, '');
  return cleanRoot + '\\' + folderForModelType(type);
}

type ThumbnailFit = 'cover' | 'contain' | 'fill';
const THUMBNAIL_FIT_KEY = 'raphael.thumbnailFit';

function initialThumbnailFit(): ThumbnailFit {
  if (typeof window === 'undefined') return 'cover';
  const saved = window.localStorage.getItem(THUMBNAIL_FIT_KEY);
  return saved === 'contain' || saved === 'fill' || saved === 'cover' ? saved : 'cover';
}

function fmtBytes(n: number) { if (n < 1024) return `${n} B`; const u=['KB','MB','GB','TB']; let i=-1,v=n; do { v/=1024; i++; } while(v>=1024 && i<u.length-1); return `${v.toFixed(v>=100?0:v>=10?1:2)} ${u[i]}`; }
function fmtDateTime(seconds: number) { if (!seconds) return '—'; return new Date(seconds * 1000).toLocaleString(); }
function initials(s: string) { return s.split(/\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase(); }

function PulseMark() {
  return <div className="raphael-core" aria-label="Raphael">
    <span className="core-dot"/>
    <i className="core-orbit orbit-a"/>
    <i className="core-orbit orbit-b"/>
    <i className="core-orbit orbit-c"/>
  </div>;
}

function Setup({ onReady }: { onReady: (state: AppState)=>void }) {
  const [busy, setBusy] = useState(false);
  const webMode = api.isWebApp;
  const choose = async () => { setBusy(true); try { const p = await api.chooseModelsFolder(); if (p) onReady(await api.setModelsRoot(p)); } finally { setBusy(false); } };
  return <div className="setup-shell">
    <div className="scan-corners"/>
    <PulseMark/>
    <div className="setup-card hud-panel">
      <div className="eyebrow">RAPHAEL MODEL MANAGER</div>
      <h1>{webMode ? 'Web app is waiting for the host' : 'Select your ComfyUI models folder'}</h1>
      <p>{webMode
        ? 'Select the models folder from Raphael on the Windows host. The LAN web app uses that same library.'
        : 'The manager watches this folder and everything beneath it. Your model files stay where they are.'}</p>
      {!webMode ? <><button className="primary-btn" onClick={choose} disabled={busy}>{busy ? 'OPENING…' : 'BROWSE MODELS FOLDER'}</button><div className="tiny">Example: C:\ComfyUI\models</div></> : <div className="tiny">Return to the Raphael desktop window and choose the host models folder.</div>}
    </div>
  </div>;
}

function SettingsOverlay({
  thumbnailFit,
  onThumbnailFitChange,
  onClose,
}: {
  thumbnailFit: ThumbnailFit;
  onThumbnailFitChange: (fit: ThumbnailFit) => void;
  onClose: () => void;
}) {
  const [folderBusy, setFolderBusy] = useState(false);
  const [closing, setClosing] = useState(false);
  const [tagBusy, setTagBusy] = useState(false);
  const [tagResult, setTagResult] = useState<string | null>(null);
  const [settingsError, setSettingsError] = useState<string | null>(null);
  const [civitaiToken, setCivitaiToken] = useState('');
  const [tokenSet, setTokenSet] = useState(false);
  const [tokenBusy, setTokenBusy] = useState(false);
  const [tokenMessage, setTokenMessage] = useState<string | null>(null);
  const [parallelDownloads, setParallelDownloads] = useState(3);
  const [parallelBusy, setParallelBusy] = useState(false);
  const [parallelMessage, setParallelMessage] = useState<string | null>(null);

  const dismiss = () => {
    if (closing) return;
    setClosing(true);
    window.setTimeout(onClose, 180);
  };

  useEffect(() => {
    api.getCivitaiTokenSet().then(setTokenSet).catch(() => setTokenSet(false));
    api.getParallelDownloads().then(value => setParallelDownloads(Math.max(1, Math.min(8, value)))).catch(() => setParallelDownloads(3));
  }, []);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        dismiss();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [closing]);

  const saveCivitaiToken = async () => {
    const value = civitaiToken.trim();
    if (!value || tokenBusy) return;
    setTokenBusy(true);
    setTokenMessage(null);
    setSettingsError(null);
    try {
      await api.setCivitaiToken(value);
      setCivitaiToken('');
      setTokenSet(true);
      setTokenMessage('CIVITAI TOKEN SAVED TO RAPHAEL');
    } catch (error) {
      setSettingsError(String(error));
    } finally {
      setTokenBusy(false);
    }
  };

  const clearCivitaiToken = async () => {
    if (tokenBusy) return;
    setTokenBusy(true);
    setTokenMessage(null);
    setSettingsError(null);
    try {
      await api.setCivitaiToken('');
      setCivitaiToken('');
      setTokenSet(false);
      setTokenMessage('CIVITAI TOKEN CLEARED');
    } catch (error) {
      setSettingsError(String(error));
    } finally {
      setTokenBusy(false);
    }
  };

  const changeParallelDownloads = async (value: number) => {
    if (parallelBusy) return;
    setParallelBusy(true);
    setParallelMessage(null);
    try {
      const saved = await api.setParallelDownloads(value);
      setParallelDownloads(saved);
      setParallelMessage(`PARALLEL DOWNLOADS · ${saved}`);
    } catch (error) {
      setSettingsError(String(error));
    } finally {
      setParallelBusy(false);
    }
  };

  const changeFolder = async () => {
    if (api.isWebApp || folderBusy) return;
    setFolderBusy(true);
    try {
      const path = await api.chooseModelsFolder();
      if (path) {
        await api.setModelsRoot(path);
        window.location.reload();
      }
    } catch (error) {
      setSettingsError(String(error));
    } finally {
      setFolderBusy(false);
    }
  };

  const addSubfolderTags = async () => {
    if (tagBusy) return;
    setTagBusy(true);
    setTagResult(null);
    setSettingsError(null);
    try {
      const count = await api.addSubfolderTags();
      setTagResult(count === 1 ? 'ADDED SUBFOLDER TAGS TO 1 MODEL' : `ADDED SUBFOLDER TAGS TO ${count} MODELS`);
    } catch (error) {
      setSettingsError(String(error));
    } finally {
      setTagBusy(false);
    }
  };

  return <div className={"settings-backdrop" + (closing ? " overlay-leaving" : "")} onClick={()=>dismiss()}>
    <section className="settings-panel hud-panel" onClick={e => e.stopPropagation()}>
      <header className="settings-header">
        <div>
          <div className="eyebrow">RAPHAEL CORE</div>
          <h2>SETTINGS</h2>
        </div>
        <button className="settings-close" aria-label="Close settings" onClick={dismiss}>×</button>
      </header>
      {settingsError ? <div className="settings-global-error"><div className="error-box">{settingsError}</div></div> : null}

      <div className="settings-scroll">
        <section className="settings-section">
          <div className="section-head">LIBRARY LOCATION</div>
          <div className="settings-path" title={window.location.href}>
            {api.isWebApp ? 'HOST FOLDER IS CONTROLLED BY THE DESKTOP APP' : 'CHANGE THE COMFYUI MODELS ROOT FOLDER'}
          </div>
          <button className="primary-btn" onClick={changeFolder} disabled={api.isWebApp || folderBusy}>
            {api.isWebApp ? 'DESKTOP ONLY' : folderBusy ? 'OPENING…' : 'CHANGE FOLDER'}
          </button>
        </section>

        <section className="settings-section">
          <div className="section-head">CIVITAI ACCESS</div>
          <p className="settings-copy">Paste an optional Civitai API token here. Raphael stores it in the host credential store and uses it for authenticated model downloads and API requests.</p>
          <div className="settings-token-status">
            <span className={tokenSet ? 'online' : ''}>{tokenSet ? 'TOKEN CONFIGURED' : 'NO TOKEN CONFIGURED'}</span>
            <span>{api.isWebApp ? 'HOST TOKEN' : 'LOCAL TOKEN'}</span>
          </div>
          <div className="settings-token-row">
            <input
              className="settings-token-input"
              type="password"
              value={civitaiToken}
              onChange={e => setCivitaiToken(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter') { e.preventDefault(); void saveCivitaiToken(); } }}
              placeholder={tokenSet ? 'Paste a new token to replace the current one' : 'Paste Civitai API token'}
              autoComplete="off"
              disabled={tokenBusy}
            />
            <button className="primary-btn small" onClick={saveCivitaiToken} disabled={tokenBusy || !civitaiToken.trim()}>
              {tokenBusy ? 'SAVING…' : 'SAVE TOKEN'}
            </button>
            {tokenSet ? <button className="text-btn settings-clear-btn" onClick={clearCivitaiToken} disabled={tokenBusy}>CLEAR</button> : null}
          </div>
          {tokenMessage ? <div className="settings-success">{tokenMessage}</div> : null}
        </section>

        <section className="settings-section">
          <div className="section-head">DOWNLOAD CONCURRENCY</div>
          <p className="settings-download-copy">Controls how many model files Raphael downloads at the same time. Additional installs stay queued and start automatically as slots open.</p>
          <div className="settings-parallel-row">
            <select className="type-select settings-parallel-select" value={parallelDownloads} onChange={e => void changeParallelDownloads(Number(e.target.value))} disabled={parallelBusy} aria-label="Parallel downloads">
              {[1,2,3,4,5,6,7,8].map(value => <option key={value} value={value}>{value} {value === 1 ? 'DOWNLOAD' : 'DOWNLOADS'}</option>)}
            </select>
            <span className="tag-save-state">{parallelBusy ? 'SAVING…' : parallelMessage || 'DEFAULT · 3'}</span>
          </div>
        </section>

        <section className="settings-section">
          <div className="section-head">THUMBNAIL SCALING</div>
          <p className="settings-copy">Controls how model thumbnails are scaled inside the fixed Raphael card and import placeholder.</p>
          <div className="settings-options">
            <button className={thumbnailFit === 'cover' ? 'active' : ''} onClick={() => onThumbnailFitChange('cover')}>
              <b>COVER</b><span>Fill the frame and crop overflow.</span>
            </button>
            <button className={thumbnailFit === 'contain' ? 'active' : ''} onClick={() => onThumbnailFitChange('contain')}>
              <b>FIT</b><span>Show the whole image with letterboxing.</span>
            </button>
            <button className={thumbnailFit === 'fill' ? 'active' : ''} onClick={() => onThumbnailFitChange('fill')}>
              <b>STRETCH</b><span>Resize completely to the placeholder.</span>
            </button>
          </div>
        </section>

        <section className="settings-section">
          <div className="section-head">FOLDER → TAGS</div>
          <p className="settings-copy">Adds every model subfolder below its ComfyUI type folder as a tag without removing existing tags.</p>
          <div className="folder-tag-example"><span>checkpoints/Illustrus/model.safetensors</span><b>→</b><em>Illustrus</em></div>
          <div className="folder-tag-example"><span>loras/Illustrus/Character/model.safetensors</span><b>→</b><em>Illustrus · Character</em></div>
          <button className="primary-btn" onClick={addSubfolderTags} disabled={tagBusy}>{tagBusy ? 'SCANNING…' : 'ADD SUBFOLDERS AS TAGS'}</button>
          {tagResult ? <div className="settings-success">{tagResult}</div> : null}
        </section>
      </div>
    </section>
  </div>;
}

function TypePlaceholder({ type }: { type: ModelType }) {
  const label = type.toUpperCase();
  const mark = type === 'Checkpoint' ? 'CKPT'
    : type === 'LoRA' ? 'LORA'
    : type === 'ControlNet' ? 'CN'
    : type === 'Embedding' ? 'EMB'
    : type === 'Upscaler' ? 'UP'
    : type === 'Text Encoder' ? 'TXT'
    : type === 'CLIP Vision' ? 'CV'
    : type === 'IP-Adapter' ? 'IP'
    : type === 'VAE' ? 'VAE'
    : 'OTH';
  return <div className="type-placeholder" aria-label={label}>
    <div className="type-placeholder-orbit"/>
    <div className="type-placeholder-mark">{mark}</div>
    <div className="type-placeholder-label">{label}</div>
    <div className="type-placeholder-scan"/>
  </div>;
}

function ModelCard({ model, selected, onClick }: { model: ModelRecord; selected: boolean; onClick: ()=>void }) {
  return <button className={`model-card ${selected ? 'selected' : ''}`} onClick={onClick}>
    <div className="thumb model-thumb">
      <TypePlaceholder type={model.model_type as ModelType}/>
      {(model.cover_path || model.thumbnail_path) ? <img src={fileUrl(model.cover_path || model.thumbnail_path || '')} alt="" style={{objectPosition: `${model.cover_position_x}% ${model.cover_position_y}%`}} onError={(e)=>{e.currentTarget.style.display="none";}}/> : null}
    </div>
    <div className="card-body">
      <div className="card-title">{model.civitai_name || model.filename.replace(/\.[^.]+$/, '')}</div>
      <div className="card-sub">{model.model_type} · {fmtBytes(model.size_bytes)}</div>
      <div className="card-path">{model.relative_path}</div>
    </div>
    {model.civitai_model_id ? <span className="civitai-dot" title="Linked to Civitai"/> : null}
  </button>;
}

function Gallery({ model, images, hasMore, fetchBusy, onChooseThumbnail, onFetchMore }: { model: ModelRecord; images: ModelImage[]; hasMore: boolean; fetchBusy: boolean; onChooseThumbnail: (imageId: number) => Promise<void>; onFetchMore: () => Promise<void> }) {
  const [busyImage, setBusyImage] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);

  if (!images.length) return <div className="empty-inline">No cached Civitai images yet.</div>;

  const choose = async (imageId: number) => {
    if (busyImage !== null) return;
    setBusyImage(imageId);
    setError(null);
    try {
      await onChooseThumbnail(imageId);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusyImage(null);
    }
  };

  return <div className="gallery-shell">
    {error ? <div className="error-box gallery-error">{error}</div> : null}
    <div className="gallery-grid">
      {images.map((img) => {
        const imagePath = img.local_path || img.thumbnail_path;
        const active = model.cover_path === img.thumbnail_path || model.cover_path === img.local_path;
        const ready = Boolean(img.thumbnail_path || img.local_path);
        return <div className={`gallery-item ${active ? 'active-thumbnail' : ''}`} key={img.id}>
          <div className="gallery-image-wrap">
            {imagePath ? <img src={fileUrl(imagePath)} alt="Civitai example"/> : <div className="thumb placeholder">IMAGE</div>}
            {active ? <span className="gallery-active-badge">ACTIVE THUMBNAIL</span> : null}
          </div>
          {img.prompt ? <div className="gallery-prompt">{img.prompt}</div> : null}
          <div className="gallery-meta">
            <span>{img.steps || img.cfg || img.sampler ? `${img.sampler || 'sampler'}${img.steps ? ` · ${img.steps} steps` : ''}` : 'CIVITAI EXAMPLE'}</span>
            {img.prompt ? <button onClick={()=>navigator.clipboard?.writeText(img.prompt!)}>COPY PROMPT</button> : null}
          </div>
          <button
            className={`gallery-thumbnail-btn ${active ? 'active' : ''}`}
            onClick={() => void choose(img.id)}
            disabled={!ready || busyImage !== null}
          >
            {busyImage === img.id ? 'APPLYING…' : active ? 'ACTIVE THUMBNAIL' : 'USE AS THUMBNAIL'}
          </button>
        </div>;
      })}
    </div>
    {hasMore ? <button className="gallery-more-btn" onClick={() => void onFetchMore()} disabled={fetchBusy}>{fetchBusy ? 'FETCHING 10 MORE…' : 'FETCH 10 MORE EXAMPLES'}</button> : <div className="gallery-end-note">END OF CIVITAI EXAMPLES</div>}
  </div>;
}


function TagEditor({ model, allTags, onSave, onFilter }: { model: ModelRecord; allTags: TagRecord[]; onSave: (tags: string[])=>Promise<void>; onFilter: (tag: string)=>void }) {
  const [draft, setDraft] = useState<string[]>(model.tags);
  const [input, setInput] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const needle = input.trim().toLowerCase();

  useEffect(() => {
    setDraft(model.tags);
  }, [model.id, model.tags.join('\u0001')]);

  const suggestions = allTags
    .filter(tag => !draft.some(existing => existing.toLowerCase() === tag.name.toLowerCase()))
    .filter(tag => !needle || tag.name.toLowerCase().includes(needle))
    .slice(0, 8);

  const persist = async (next: string[]) => {
    setSaving(true);
    setSaveError(null);
    try { await onSave(next); setDraft(next); }
    catch (error) { setSaveError(String(error)); }
    finally { setSaving(false); }
  };

  const add = (raw: string) => {
    const value = raw.trim();
    if (!value || draft.some(tag => tag.toLowerCase() === value.toLowerCase()) || saving) return;
    const next = [...draft, value];
    setDraft(next); setInput('');
    if (api.isWebApp) void persist(next);
  };
  const remove = (tag: string) => {
    if (saving) return;
    const next = draft.filter(existing => existing !== tag);
    setDraft(next);
    if (api.isWebApp) void persist(next);
  };
  const save = async () => persist(draft);

  return <div className="tag-editor">
    <div className="tag-editor-chips">
      {draft.map(tag => <span className="editable-tag" key={tag}>
        <button className="tag-value" title="Filter by this tag" onClick={()=>onFilter(tag)}>{tag}</button>
        <button className="tag-remove" aria-label={'Remove ' + tag} onClick={()=>remove(tag)} disabled={saving}>×</button>
      </span>)}
      {!draft.length ? <span className="empty-inline">No tags assigned.</span> : null}
    </div>
    <div className="tag-input-row">
      <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();add(input);}}} placeholder="Type a tag…" disabled={saving}/>
      <button className="primary-btn small" onClick={()=>add(input)} disabled={saving || !input.trim()}>ADD</button>
    </div>
    {input.trim() ? <div className="tag-suggestions">
      {suggestions.length ? suggestions.map(tag => <button key={tag.name} onClick={()=>add(tag.name)} disabled={saving}><span>{tag.name}</span><b>{tag.count}</b></button>) : <div className="tag-create-hint">Press ENTER to create “{input.trim()}”.</div>}
    </div> : null}
    {saveError ? <div className="error-box tag-save-error">{saveError}</div> : null}
    <div className="tag-editor-footer">
      <span>{api.isWebApp ? 'AUTO-SAVES' : `${draft.length} TAG${draft.length === 1 ? '' : 'S'}`}</span>
      {!api.isWebApp ? <button className="text-btn" onClick={save} disabled={saving}>{saving ? 'SAVING…' : 'SAVE TAGS'}</button> : <span className="tag-save-state">{saving ? 'SYNCING…' : 'SYNCED'}</span>}
    </div>
  </div>;
}

function TagFilterPanel({ tags, activeTags, onToggle, onClear }: { tags: TagRecord[]; activeTags: string[]; onToggle: (tag: string)=>void; onClear: ()=>void }) {
  const [filter, setFilter] = useState('');
  const visible = tags.filter(tag => !filter.trim() || tag.name.toLowerCase().includes(filter.trim().toLowerCase()));
  return <div className="tag-filter-panel" onClick={e=>e.stopPropagation()}>
    <div className="tag-filter-head"><span>TAG FILTERS</span><b>{activeTags.length ? activeTags.length + ' ACTIVE' : 'ALL TAGS'}</b></div>
    <input className="tag-filter-input" value={filter} onChange={e=>setFilter(e.target.value)} placeholder="Find a tag…"/>
    <div className="tag-filter-list">
      {visible.map(tag => <button key={tag.name} className={activeTags.some(x=>x.toLowerCase()===tag.name.toLowerCase())?'active':''} onClick={()=>onToggle(tag.name)}><span>{tag.name}</span><b>{tag.count}</b></button>)}
      {!visible.length ? <div className="empty-inline">No tags found.</div> : null}
    </div>
    <div className="tag-filter-foot">
      <span>AND logic · combine multiple tags</span>
      {activeTags.length ? <button className="text-btn" onClick={onClear}>CLEAR</button> : null}
    </div>
  </div>;
}

function CoverEditorOverlay({
  model,
  onClose,
  onUpdated,
}: {
  model: ModelRecord;
  onClose: () => void;
  onUpdated: (model: ModelRecord) => void;
}) {
  const sourcePath = model.cover_path || model.thumbnail_path;
  const [coverPath, setCoverPath] = useState(sourcePath);
  const [position, setPosition] = useState({
    x: Number.isFinite(model.cover_position_x) ? model.cover_position_x : 50,
    y: Number.isFinite(model.cover_position_y) ? model.cover_position_y : 50,
  });
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [closing, setClosing] = useState(false);
  const dragRef = useRef<{ pointerId: number; x: number; y: number; startX: number; startY: number } | null>(null);

  const dismiss = (force = false) => {
    if ((!force && busy) || closing) return;
    setClosing(true);
    window.setTimeout(onClose, 180);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        dismiss();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [busy, closing]);

  const clamp = (value: number) => Math.max(0, Math.min(100, value));

  const startDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!coverPath || busy) return;
    const rect = event.currentTarget.getBoundingClientRect();
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      pointerId: event.pointerId,
      x: position.x,
      y: position.y,
      startX: event.clientX,
      startY: event.clientY,
    };
    void rect;
  };

  const moveDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    const drag = dragRef.current;
    if (!drag || drag.pointerId !== event.pointerId) return;
    const rect = event.currentTarget.getBoundingClientRect();
    setPosition({
      x: clamp(drag.x - ((event.clientX - drag.startX) / rect.width) * 100),
      y: clamp(drag.y - ((event.clientY - drag.startY) / rect.height) * 100),
    });
  };

  const stopDrag = (event: React.PointerEvent<HTMLDivElement>) => {
    if (dragRef.current?.pointerId === event.pointerId) {
      dragRef.current = null;
      try { event.currentTarget.releasePointerCapture(event.pointerId); } catch {}
    }
  };

  const chooseCustom = async () => {
    if (api.isWebApp || busy) return;
    setBusy(true);
    setError(null);
    try {
      const source = await api.chooseImageFile();
      if (!source) return;
      const updated = await api.setModelCustomCover(model.id, source);
      setCoverPath(updated.cover_path || updated.thumbnail_path);
      setPosition({
        x: Number.isFinite(updated.cover_position_x) ? updated.cover_position_x : 50,
        y: Number.isFinite(updated.cover_position_y) ? updated.cover_position_y : 50,
      });
      onUpdated(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const resetCover = async () => {
    if (busy || !model.cover_path) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await api.resetModelCover(model.id);
      setCoverPath(updated.thumbnail_path);
      setPosition({ x: 50, y: 50 });
      onUpdated(updated);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const apply = async () => {
    if (busy) return;
    setBusy(true);
    setError(null);
    try {
      const updated = await api.setModelCoverPosition(model.id, position.x, position.y);
      onUpdated(updated);
      dismiss(true);
    } catch (e) {
      setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  return <div className={"modal-backdrop cover-editor-backdrop" + (closing ? " overlay-leaving" : "")} onClick={()=>dismiss()}>
    <div className="cover-editor hud-panel" onClick={e => e.stopPropagation()}>
      <header className="cover-editor-header">
        <div>
          <div className="eyebrow">COVER EDITOR</div>
          <h2>{model.civitai_name || model.filename}</h2>
        </div>
        <button className="settings-close" aria-label="Close cover editor" onClick={()=>dismiss()} disabled={busy}>×</button>
      </header>

      <div className="cover-editor-body">
        <div
          className={`cover-editor-frame ${coverPath ? 'draggable' : 'empty'}`}
          onPointerDown={startDrag}
          onPointerMove={moveDrag}
          onPointerUp={stopDrag}
          onPointerCancel={stopDrag}
        >
          {coverPath
            ? <img
                src={fileUrl(coverPath)}
                alt=""
                draggable={false}
                style={{ objectPosition: `${position.x}% ${position.y}%` }}
                onError={() => setError('The selected cover image could not be displayed.')}
              />
            : <TypePlaceholder type={model.model_type as ModelType}/>}
          <div className="cover-editor-grid"/>
          <div className="cover-editor-corners"/>
          {coverPath ? <div className="cover-editor-hint">DRAG TO REPOSITION</div> : null}
        </div>

        <div className="cover-editor-tools">
          <div className="cover-editor-tool-row">
            <button className="cover-upload-btn" onClick={chooseCustom} disabled={api.isWebApp || busy} title={api.isWebApp ? 'Custom cover selection is available in the desktop app' : 'Choose a custom cover image'}>
              <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M3.5 6.5h6l1.7 2H20.5v9.8a1.7 1.7 0 0 1-1.7 1.7H5.2a1.7 1.7 0 0 1-1.7-1.7Z"/><path d="M3.5 7.5V5.4a1.4 1.4 0 0 1 1.4-1.4h4.1l1.6 2H19a1.5 1.5 0 0 1 1.5 1.5"/></svg>
              <span>{busy ? 'WORKING…' : 'CUSTOM COVER'}</span>
            </button>
            <button className="text-btn" onClick={resetCover} disabled={busy || !model.cover_path}>USE THUMBNAIL</button>
          </div>

          <div className="cover-position-readout">
            <span>X {position.x.toFixed(0)}%</span>
            <span>Y {position.y.toFixed(0)}%</span>
          </div>
          <p className="settings-copy">The selected image is cropped to Raphael’s cover frame. Drag it inside the frame to choose which part is visible.</p>
          {api.isWebApp ? <div className="cover-web-note">POSITION EDITING WORKS IN WEB MODE. CUSTOM FILE PICKING IS DESKTOP ONLY.</div> : null}
          {error ? <div className="error-box settings-error">{error}</div> : null}
        </div>
      </div>

      <div className="modal-actions cover-editor-actions">
        <button className="text-btn" onClick={()=>dismiss()} disabled={busy}>CANCEL</button>
        <button className="primary-btn" onClick={apply} disabled={busy}>{busy ? 'SAVING…' : 'APPLY COVER'}</button>
      </div>
    </div>
  </div>;
}

function Inspector({ model, images, allTags, galleryHasMore, galleryFetchBusy, onRefresh, onLinkCivitai, onSaveTags, onSaveType, onDelete, onFilterTag, onChangeCover, onChooseThumbnail, onFetchMore }: { model: ModelRecord; images: ModelImage[]; allTags: TagRecord[]; galleryHasMore: boolean; galleryFetchBusy: boolean; onRefresh: ()=>void; onLinkCivitai: (url: string)=>Promise<void>; onSaveTags: (tags: string[])=>Promise<void>; onSaveType: (type: string)=>Promise<void>; onDelete: ()=>Promise<void>; onFilterTag: (tag: string)=>void; onChangeCover: ()=>void; onChooseThumbnail: (imageId: number)=>Promise<void>; onFetchMore: ()=>Promise<void> }) {
  const [tab, setTab] = useState<'overview'|'examples'|'files'>('overview');
  const [civitaiUrl, setCivitaiUrl] = useState(model.civitai_url || '');
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);
  const [deleteOpen, setDeleteOpen] = useState(false);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState<string | null>(null);
  const [deleteClosing, setDeleteClosing] = useState(false);
  const promptText = model.activation_prompts.join(', ');

  const dismissDelete = (force = false) => {
    if ((!force && deleteBusy) || deleteClosing) return;
    setDeleteClosing(true);
    window.setTimeout(() => {
      setDeleteOpen(false);
      setDeleteClosing(false);
    }, 180);
  };

  useEffect(() => {
    if (!deleteOpen) return;
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        event.preventDefault();
        dismissDelete();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [deleteOpen, deleteBusy, deleteClosing]);
  return <aside className="inspector hud-panel">
    <div className="inspector-header">
      <div className="inspector-model-heading"><div className="inspector-model-title"><div className="eyebrow">MODEL</div><h2>{model.civitai_name || model.filename}</h2></div></div>
      <select className="type-select" value={model.model_type} onChange={async e=>{try{await onSaveType(e.target.value);}catch{e.currentTarget.value=model.model_type;}}} aria-label="Model type">{MODEL_TYPES.map(x=><option key={x.value} value={x.value}>{x.label}</option>)}</select>
      <button className="cover-change-btn" onClick={onChangeCover} title="Change this model's cover">CHANGE COVER</button>
    </div>
    <div className="inspector-tabs">{(['overview','examples','files'] as const).map(t=><button className={tab===t?'active':''} onClick={()=>setTab(t)} key={t}>{t.toUpperCase()}</button>)}</div>
    {tab==='overview' && <div className="inspector-scroll">
      <section><div className="section-head">DESCRIPTION</div><p className="description">{model.description || 'No description cached from Civitai.'}</p></section>
      <section><div className="section-head">ACTIVATION PROMPTS</div>{promptText ? <><div className="prompt-box">{promptText}</div><button className="text-btn" onClick={()=>navigator.clipboard?.writeText(promptText)}>COPY ALL</button></> : <div className="empty-inline">No activation prompts were published for this version.</div>}</section>
      <section><div className="section-head section-head-row"><span>TAGS</span><span className="section-action">EDITABLE</span></div><TagEditor model={model} allTags={allTags} onSave={onSaveTags} onFilter={onFilterTag}/></section>
      <section><div className="section-head">LOCATION</div><div className="mono-box">{model.path}</div><button className="text-btn" disabled={api.isWebApp} title={api.isWebApp ? 'Opening the Windows file manager is available only in the desktop app' : undefined} onClick={()=>api.openFolder(model.path)}>{api.isWebApp ? 'OPEN FOLDER · DESKTOP' : 'OPEN FOLDER'}</button></section>
      <section><div className="section-head">CIVITAI SOURCE</div>{model.civitai_url ? <div className="civitai-source-panel"><div className="source-line"><span className="source-dot"/><span className="source-label">LINKED SOURCE</span><span className="source-domain">{new URL(model.civitai_url).hostname.replace(/^www\./,'').toUpperCase()}</span></div><div className="mono-box source-url">{model.civitai_url}</div><button className="text-btn" onClick={onRefresh}>REFRESH SOURCE DATA</button></div> : <div className="civitai-link-panel"><div className="source-line"><span className="source-dot"/><span className="source-label">LINK LOCAL MODEL</span><span className="source-domain">CIVITAI</span></div><p className="empty-inline">Paste a Civitai model page to pull its metadata, gallery and thumbnail into Raphael.</p><div className="link-row"><input value={civitaiUrl} onChange={e=>{setCivitaiUrl(e.target.value);setLinkError(null);}} placeholder="civitai.com/models/... or civitai.red/models/..."/><button className="primary-btn small" disabled={linkBusy} onClick={async()=>{if(!civitaiUrl.trim()) return; setLinkBusy(true); setLinkError(null); try { await onLinkCivitai(civitaiUrl.trim()); } catch (e) { setLinkError(String(e)); } finally { setLinkBusy(false); }}}>{linkBusy?'FETCHING…':'FETCH DETAILS'}</button></div>{linkError ? <div className="error-box">{linkError}</div> : null}</div>}</section>
    </div>}
    {tab==='examples' && <div className="inspector-scroll"><section><div className="section-head section-head-row"><span>CACHED CIVITAI GALLERY · {images.length}</span><span className="section-action">PICK A THUMBNAIL</span></div><Gallery model={model} images={images} hasMore={galleryHasMore} fetchBusy={galleryFetchBusy} onChooseThumbnail={onChooseThumbnail} onFetchMore={onFetchMore}/></section></div>}
    {tab==='files' && <div className="inspector-scroll"><section><div className="section-head">LOCAL FILE</div><div className="kv"><span>SIZE</span><b>{fmtBytes(model.size_bytes)}</b></div><div className="kv"><span>TYPE</span><b>{model.model_type}</b></div><div className="kv"><span>BASE</span><b>{model.base_model || '—'}</b></div><div className="kv"><span>VERSION</span><b>{model.version_name || '—'}</b></div><div className="kv"><span>CREATOR</span><b>{model.creator || '—'}</b></div><div className="kv"><span>DOWNLOADED</span><b>{fmtDateTime(model.downloaded_at)}</b></div><div className="kv"><span>SHA256</span><b className="wrap">{model.source_hash || 'Not computed'}</b></div></section><section className="danger-section"><div className="section-head">DANGER ZONE</div><p className="danger-copy">Permanently delete this model file from disk and remove its Raphael metadata and cached gallery entries.</p><button className="danger-btn" onClick={()=>{setDeleteError(null);setDeleteOpen(true);}} disabled={deleteBusy}>DELETE MODEL</button></section></div>}
    {deleteOpen && <div className={"modal-backdrop inspector-delete-backdrop" + (deleteClosing ? " overlay-leaving" : "")} onClick={()=>dismissDelete()}><div className="delete-modal hud-panel" onClick={e=>e.stopPropagation()}><div className="eyebrow">DESTRUCTIVE ACTION</div><h3>DELETE MODEL?</h3><p>This will permanently remove <b>{model.filename}</b> from your ComfyUI models folder. Raphael metadata and cached gallery files for this model will also be removed.</p>{deleteError ? <div className="error-box modal-error">{deleteError}</div> : null}<div className="modal-actions"><button className="text-btn" onClick={()=>dismissDelete()} disabled={deleteBusy}>CANCEL</button><button className="danger-btn confirm" disabled={deleteBusy} onClick={async()=>{setDeleteBusy(true);setDeleteError(null);try{await onDelete();dismissDelete(true);}catch(e){setDeleteError(String(e));}finally{setDeleteBusy(false);}}}>{deleteBusy?'DELETING…':'DELETE PERMANENTLY'}</button></div></div></div>}
  </aside>;
}

function DownloadProgressWidget({ progress, onClear }: { progress: DownloadProgress; onClear: (taskId: string) => void }) {
  const percent = progress.percent !== null ? Math.max(0, Math.min(100, progress.percent)) : null;
  const active = progress.phase !== 'COMPLETED' && progress.phase !== 'FAILED' && progress.phase !== 'ALREADY INSTALLED';
  const width = percent !== null ? percent : 8;
  const finished = !active;
  return <div className={'download-widget ' + (progress.phase === 'FAILED' ? 'failed' : finished ? 'done' : '')}>
    <div className="download-widget-head">
      <span>{active ? 'INSTALLATION' : progress.phase}</span>
      <span className="download-widget-percent">{percent !== null ? `${percent.toFixed(0)}%` : '…'}</span>
    </div>
    <div className="download-widget-name" title={progress.filename}>{progress.filename || 'CIVITAI MODEL'}</div>
    <div className="download-bar"><div className="download-bar-fill" style={{width: width + '%'}}/></div>
    <div className="download-widget-meta">
      <span>{progress.phase}</span>
      {progress.error ? <span title={progress.error}>ERROR</span> : progress.total_bytes ? <span>{fmtBytes(progress.downloaded_bytes)} / {fmtBytes(progress.total_bytes)}</span> : <span>{fmtBytes(progress.downloaded_bytes)}</span>}
    </div>
    {finished && progress.task_id ? <button className="download-widget-clear" onClick={()=>onClear(progress.task_id!)} aria-label="Dismiss download status">×</button> : null}
  </div>;
}

function App() {
  const [state,setState]=useState<AppState|null>(null); const [models,setModels]=useState<ModelRecord[]>([]); const [selectedId,setSelectedId]=useState<number|null>(null);
  const refreshGeneration = useRef(0);
  const [type,setType]=useState<ModelType|'All'>('All'); const [query,setQuery]=useState(''); const [activeTags,setActiveTags]=useState<string[]>([]); const [tagPanelOpen,setTagPanelOpen]=useState(false); const [allTags,setAllTags]=useState<TagRecord[]>([]); const [images,setImages]=useState<ModelImage[]>([]); const [galleryHasMore,setGalleryHasMore]=useState(true); const [galleryFetchBusy,setGalleryFetchBusy]=useState(false); const bulkFileInputRef=useRef<HTMLInputElement>(null); const [bulkBusy,setBulkBusy]=useState(false); const [bulkMessage,setBulkMessage]=useState<string|null>(null); const [importUrl,setImportUrl]=useState(''); const [preview,setPreview]=useState<CivitaiImportPreview|null>(null); const [busy,setBusy]=useState(false); const [sort,setSort]=useState('name'); const [counts,setCounts]=useState<LibraryCounts>({all:0,by_type:{}}); const [importError,setImportError]=useState<string|null>(null); const [downloadPath,setDownloadPath]=useState(''); const [importType,setImportType]=useState<ModelType>('Other'); const [customDownloadPath,setCustomDownloadPath]=useState(false); const [webStatus,setWebStatus]=useState<{enabled:boolean;url:string|null;port:number}>({enabled:false,url:null,port:1421}); const [webBusy,setWebBusy]=useState(false); const [webError,setWebError]=useState<string|null>(null);
  const [settingsOpen,setSettingsOpen]=useState(false);
  const [coverEditorOpen,setCoverEditorOpen]=useState(false);
  const [thumbnailFit,setThumbnailFit]=useState<ThumbnailFit>(initialThumbnailFit);
  const [downloadProgress,setDownloadProgress]=useState<DownloadProgress[]>([]);
  const [importClosing,setImportClosing]=useState(false);
  useEffect(()=>{window.localStorage.setItem(THUMBNAIL_FIT_KEY,thumbnailFit);},[thumbnailFit]);

  useEffect(() => {
    let disposed = false;
    const pull = async () => {
      try {
        const progress = await api.getDownloadProgress();
        if (!disposed) setDownloadProgress(progress.filter(item => item.visible));
      } catch {}
    };
    void pull();
    const timer = window.setInterval(() => void pull(), 450);
    return () => { disposed = true; window.clearInterval(timer); };
  }, []);

  const selected = models.find(m=>m.id===selectedId) || null;
  const clearDownload = async (taskId: string) => {
    try {
      await api.clearDownloadProgress(taskId);
      setDownloadProgress(current => current.filter(item => item.task_id !== taskId));
    } catch {}
  };

  const closeImport = () => {
    if (importClosing) return;
    setImportClosing(true);
    window.setTimeout(() => {
      setPreview(null);
      setImportUrl('');
      setDownloadPath('');
      setCustomDownloadPath(false);
      setImportError(null);
      setImportClosing(false);
    }, 180);
  };

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      if (settingsOpen || coverEditorOpen) return;
      if (preview || importUrl) {
        event.preventDefault();
        closeImport();
      }
    };
    window.addEventListener('keydown', onKeyDown);
    return () => window.removeEventListener('keydown', onKeyDown);
  }, [settingsOpen, coverEditorOpen, preview, importUrl, importClosing]);
  async function refresh(){
    const generation=++refreshGeneration.current;
    try {
      const s=await api.getState();
      if(generation!==refreshGeneration.current) return;
      setState(s);
      if(!s.models_root){
        setModels([]);
        setCounts({all:0,by_type:{}});
        setAllTags([]);
        setSelectedId(null);
        return;
      }
      const [list,allCounts,tags]=await Promise.all([
        api.listModels({type:type==='All'?undefined:type,query:query||undefined,tags:activeTags}),
        api.getLibraryCounts(),
        api.getTags()
      ]);
      if(generation!==refreshGeneration.current) return;
      list.sort((a,b)=>sort==='size'?b.size_bytes-a.size_bytes:sort==='path'?a.relative_path.localeCompare(b.relative_path):(a.civitai_name||a.filename).localeCompare(b.civitai_name||b.filename));
      setModels(list);
      setCounts(allCounts);
      setAllTags(tags);
      setSelectedId(previous=>{
        if(previous!==null && list.some(m=>m.id===previous)) return previous;
        return list[0]?.id ?? null;
      });
    } catch (error) {
      if(generation===refreshGeneration.current) console.error('Raphael refresh failed',error);
    }
  }
  useEffect(()=>{refresh(); api.getWebAppStatus().then(setWebStatus).catch(()=>{});},[]);
  const toggleWebApp = async () => { if (api.isWebApp || webBusy) return; setWebBusy(true); setWebError(null); try { const next=await api.setWebAppEnabled(!webStatus.enabled); setWebStatus(next); } catch (e) { setWebError(String(e)); } finally { setWebBusy(false); } };
  useEffect(()=>{
    let disposed=false;
    let stop:undefined|(()=>void);
    subscribeToModelChanges(()=>{void refresh()}).then(unlisten=>{
      if(disposed){
        unlisten();
      }else{
        stop=unlisten;
      }
    });
    return ()=>{
      disposed=true;
      stop?.();
    };
  },[type,query,sort,activeTags]);
  useEffect(()=>{
    if(!selected){setImages([]);setGalleryHasMore(false);setGalleryFetchBusy(false);return;}
    const modelId=selected.id;
    setGalleryHasMore(true);
    setGalleryFetchBusy(false);
    api.getImages(modelId,20).then(setImages).catch(()=>setImages([]));
    void (async()=>{try{const more=await api.syncModelGallery(modelId,20);if(selectedId===modelId)setGalleryHasMore(more);}catch{}})();
    const timer=window.setInterval(()=>api.getImages(modelId,20).then(setImages).catch(()=>{}),2000);
    return ()=>window.clearInterval(timer);
  },[selectedId, selected?.civitai_model_id]);
  useEffect(()=>{const t=setTimeout(()=>refresh(),180); return ()=>clearTimeout(t);},[query,type,sort,activeTags]);
  if(!state) return <div className="loading-shell"><PulseMark/></div>;
  if(!state.models_root) return <><Background/><Setup onReady={s=>{setState(s); refresh();}}/></>;
  const doImport = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try { const result=await api.importCivitai(importUrl.trim()); const nextType=civitaiTypeToModelType(result.model.type); setImportType(nextType); setCustomDownloadPath(false); setPreview(result); setDownloadPath(defaultImportDirectory(state.models_root,nextType) || result.target_directory); } catch (e) { setImportError(String(e)); } finally {setBusy(false);} };
  const install = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try { await api.installCivitai(importUrl.trim(),customDownloadPath ? downloadPath : undefined,importType); setBusy(false); closeImport(); await refresh(); } catch (e) { setImportError(String(e)); setBusy(false); } };
  const onFetchMore = async()=>{ if(!selected || galleryFetchBusy || !galleryHasMore) return; const modelId=selected.id; const target=images.length+10; setGalleryFetchBusy(true); try { const more=await api.syncModelGallery(modelId,target); const next=await api.getImages(modelId,target); setImages(next); setGalleryHasMore(more); } catch { } finally { setGalleryFetchBusy(false); } };
  const handleBulkLinkFile = async(file: File)=>{
    setBulkBusy(true);
    setBulkMessage(null);
    setImportError(null);
    try {
      const urls=Array.from(new Set((await file.text()).split(/\\r?\\n/).map(line=>line.trim()).filter(Boolean)));
      if(!urls.length) throw new Error('The selected file does not contain any links.');
      const results=await Promise.allSettled(urls.map(url=>api.installCivitai(url)));
      const started=results.filter(result=>result.status==='fulfilled').length;
      const failed=results.length-started;
      if(!started) throw new Error(`No models could be queued from the selected file. ${failed} link(s) failed validation.`);
      setBulkMessage(failed ? `QUEUED ${started} MODELS · ${failed} LINK(S) FAILED TO START` : `QUEUED ${started} MODELS`);
      await refresh();
      closeImport();
    } catch(error) {
      setImportError(String(error));
    } finally {
      setBulkBusy(false);
      if(bulkFileInputRef.current) bulkFileInputRef.current.value='';
    }
  };
  return <div className={`app-shell thumb-fit-${thumbnailFit}`}><Background/><div className="noise"/>
    <header className="topbar"><div className="brand"><PulseMark/><span>RAPHAEL MODEL MANAGER</span></div><div className="top-stats"><span>CACHED <b>{fmtBytes(state.storage.cached_bytes)}</b></span><span>TOTAL <b>{fmtBytes(state.storage.total_model_bytes)}</b></span></div><div className="top-actions"><button className={`web-app-btn ${webStatus.enabled ? 'active' : ''}`} disabled={api.isWebApp || webBusy} title={api.isWebApp ? 'LAN web app is controlled from the host desktop' : 'Expose Raphael to other devices on your private LAN'} onClick={toggleWebApp}>{webBusy ? 'STARTING…' : api.isWebApp ? 'WEB APP · CONNECTED' : webStatus.enabled ? 'WEB APP · ON' : 'ENABLE WEB APP'}</button>{webStatus.enabled && webStatus.url ? <a className="web-app-url" href={webStatus.url} target="_blank" rel="noreferrer">{webStatus.url}</a> : null}{webError ? <span className="web-app-error" title={webError}>WEB ERROR</span> : null}<div className="root-path" title={state.models_root}>{state.models_root}</div></div></header>
    <div className="workspace">
      <aside className="sidebar hud-panel"><div className="side-title">LIBRARY</div><nav>{TYPES.map(t=><button key={t.key} className={type===t.key?'active':''} onClick={()=>setType(t.key)}><span>{t.label}</span><b>{t.key==='All' ? counts.all : (counts.by_type[t.key] ?? 0)}</b></button>)}</nav><div className="sidebar-foot">
        {downloadProgress.length ? <div className="download-list">{downloadProgress.map(progress => <DownloadProgressWidget key={progress.task_id || progress.filename} progress={progress} onClear={clearDownload}/>)}</div> : null}
        <button className="settings-trigger" aria-label="Open settings" title="SETTINGS" onClick={()=>setSettingsOpen(true)}>
          <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M12 8.2a3.8 3.8 0 1 0 0 7.6 3.8 3.8 0 0 0 0-7.6Zm0-5.2 1 .3.7 2.1c.4.1.8.3 1.2.5l2-.9.9.7-.2 2.2c.3.3.6.6.9.9l2.2-.2.7.9-.9 2c.2.4.4.8.5 1.2l2.1.7.3 1-.3 1-2.1.7a7.4 7.4 0 0 1-.5 1.2l.9 2-.7.9-2.2-.2c-.3.3-.6.6-.9.9l.2 2.2-.9.7-2-.9c-.4.2-.8.4-1.2.5l-.7 2.1-1 .3-1-.3-.7-2.1a7.4 7.4 0 0 1-1.2-.5l-2 .9-.9-.7.2-2.2a7.2 7.2 0 0 1-.9-.9l-2.2.2-.7-.9.9-2c-.2-.4-.4-.8-.5-1.2l-2.1-.7-.3-1 .3-1 2.1-.7c.1-.4.3-.8.5-1.2l-.9-2 .7-.9 2.2.2c.3-.3.6-.6.9-.9l-.2-2.2.9-.7 2 .9c.4-.2.8-.4 1.2-.5l.7-2.1 1-.3Z"/></svg>
        </button>
      </div></aside>
      <main className="library"><div className="library-head"><div><div className="eyebrow">{type.toUpperCase()}</div><h1>{type==='All'?'MODEL LIBRARY':type.toUpperCase()}</h1></div><div className="library-tools"><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search models, tags, tag:…"/><button className={`tag-filter-button ${activeTags.length?'active':''}`} onClick={()=>setTagPanelOpen(v=>!v)}>TAGS{activeTags.length ? ` · ${activeTags.length}` : ''}</button><button className="import-btn" onClick={()=>{setImportClosing(false);setImportError(null);setImportUrl('');setImportType('Other');setCustomDownloadPath(false);setDownloadPath('');setPreview({model:{},version:{id:0,name:'',base_model:null,download_url:'',filename:null,size_bytes:null,activation_prompts:[]},target_directory:'',thumbnail_path:null});}}>IMPORT CIVITAI</button><select value={sort} onChange={e=>setSort(e.target.value)}><option value="name">NAME</option><option value="size">SIZE</option><option value="path">PATH</option></select></div>{tagPanelOpen && <TagFilterPanel tags={allTags} activeTags={activeTags} onToggle={tag=>setActiveTags(current=>current.some(x=>x.toLowerCase()===tag.toLowerCase())?current.filter(x=>x.toLowerCase()!==tag.toLowerCase()):[...current,tag])} onClear={()=>setActiveTags([])}/>}</div>{activeTags.length ? <div className="active-tag-bar">{activeTags.map(tag=><button key={tag} onClick={()=>setActiveTags(current=>current.filter(x=>x.toLowerCase()!==tag.toLowerCase()))}>{tag}<span>×</span></button>)}<span className="active-tag-help">TAG FILTERS</span></div> : null}<div className="grid">{models.map(m=><ModelCard key={m.id} model={m} selected={m.id===selectedId} onClick={()=>setSelectedId(m.id)}/>)}{!models.length&&<div className="empty-state">No models match the current view.</div>}</div></main>
      {selected && <Inspector key={selected.id} model={selected} images={images} allTags={allTags} galleryHasMore={galleryHasMore} galleryFetchBusy={galleryFetchBusy} onFetchMore={onFetchMore} onRefresh={async()=>{await api.refreshModel(selected.id); await refresh();}} onLinkCivitai={async(url)=>{await api.linkModelCivitai(selected.id,url); await refresh();}} onSaveTags={async(tags)=>{await api.setModelTags(selected.id,tags); await refresh();}} onSaveType={async(nextType)=>{await api.setModelType(selected.id,nextType); await refresh();}} onDelete={async()=>{await api.deleteModel(selected.id); setSelectedId(null); await refresh();}} onFilterTag={tag=>{setActiveTags(current=>current.some(x=>x.toLowerCase()===tag.toLowerCase())?current:[...current,tag]);}} onChangeCover={()=>setCoverEditorOpen(true)} onChooseThumbnail={async imageId=>{const updated=await api.setModelCoverFromImage(selected.id,imageId); setModels(current=>current.map(item=>item.id===updated.id?updated:item));}}/>}
      {selected && coverEditorOpen && <CoverEditorOverlay model={selected} onClose={()=>setCoverEditorOpen(false)} onUpdated={updated=>{setModels(current=>current.map(item=>item.id===updated.id?updated:item));}}/>}
    </div>
    {settingsOpen && <SettingsOverlay thumbnailFit={thumbnailFit} onThumbnailFitChange={setThumbnailFit} onClose={()=>setSettingsOpen(false)}/>}
    {(preview || importUrl) && <div className={"modal-backdrop" + (importClosing ? " overlay-leaving" : "")} onClick={()=>{if(!busy) closeImport();}}><div className="import-modal hud-panel" onClick={e=>e.stopPropagation()}><div className="eyebrow">CIVITAI IMPORT</div><h2>INSTALL A MODEL</h2>{!preview || preview.version.id===0 ? <>{importError ? <div className="error-box modal-error">{importError}</div> : null}<div className="import-row"><input value={importUrl} onChange={e=>setImportUrl(e.target.value)} onKeyDown={e=>e.key==='Enter'&&doImport()} placeholder="civitai.com/models/... or civitai.red/models/..." autoFocus/><button className="primary-btn" onClick={doImport} disabled={busy || bulkBusy}>{busy?'ANALYZING…':'ANALYZE'}</button></div><div className="bulk-link-row"><input ref={bulkFileInputRef} className="bulk-link-input" type="file" accept=".txt,text/plain" onChange={e=>{const file=e.target.files?.[0];if(file)void handleBulkLinkFile(file);}}/><button className="text-btn" onClick={()=>bulkFileInputRef.current?.click()} disabled={busy || bulkBusy}>{bulkBusy?'QUEUING LINKS…':'UPLOAD LINK FILE'}</button><span className="bulk-link-status">ONE CIVITAI LINK PER LINE · DOWNLOADS FOLLOW YOUR PARALLEL SETTING</span>{bulkMessage ? <span className="bulk-link-status">{bulkMessage}</span> : null}</div></> : <><div className="import-preview-grid"><div className="preview-image">{preview.thumbnail_path ? <><TypePlaceholder type={importType}/><img src={fileUrl(preview.thumbnail_path)} alt="" onError={(e)=>{e.currentTarget.style.display="none";}}/></> : <TypePlaceholder type={importType}/>}</div><div className="preview-panel"><div className="preview-topline"><span className="type-chip">{importType}</span><span className="source-domain">CIVITAI</span></div><h3>{preview.model.name || preview.version.filename || 'Model'}</h3><div className="preview-meta">{preview.version.base_model || 'Base model unavailable'} · {preview.version.filename || 'Filename automatic'}</div><div className="kv"><span>MODEL FILE</span><b>{preview.version.filename || 'Automatic filename'}</b></div><div className="kv"><span>SIZE</span><b>{preview.version.size_bytes ? fmtBytes(preview.version.size_bytes) : 'Unknown'}</b></div><div className="section-head">LIBRARY TAG</div><div className="import-type-row"><label htmlFor="import-type">CLASSIFY AS</label><select id="import-type" className="import-type-select" value={importType} onChange={e=>{const next=e.target.value as ModelType; setImportType(next); if(!customDownloadPath) setDownloadPath(defaultImportDirectory(state.models_root,next));}} disabled={busy}>{IMPORT_TYPES.map(x=><option key={x} value={x}>{x}</option>)}</select></div><div className="import-type-note">Civitai suggests <b>{preview.model.type || 'Unknown'}</b>; Raphael uses the tag you choose for its library category and default folder.</div><div className="section-head">DOWNLOAD LOCATION</div><div className="destination-box"><div className="destination-path" title={downloadPath}>{downloadPath || defaultImportDirectory(state.models_root,importType) || preview.target_directory}</div><button className="primary-btn small" onClick={async()=>{const next=await api.chooseDirectory(downloadPath || defaultImportDirectory(state.models_root,importType) || preview.target_directory); if(next){setDownloadPath(next);setCustomDownloadPath(true);setImportError(null);}}} disabled={busy || api.isWebApp}>{api.isWebApp ? 'DESKTOP ONLY' : 'BROWSE'}</button></div><div className="destination-note">Choose a folder inside your configured ComfyUI models directory. Changing the tag updates the default folder until you manually browse.</div><div className="section-head">ACTIVATION PROMPTS</div><div className="chips">{preview.version.activation_prompts.map(x=><span key={x}>{x}</span>)}</div></div></div><div className="modal-actions"><button className="text-btn" onClick={closeImport}>BACK</button><button className="primary-btn" onClick={install} disabled={busy}>{busy?'STARTING…':'DOWNLOAD & INSTALL'}</button></div>{importError ? <div className="error-box modal-error">{importError}</div> : null}</>}</div></div>}
  </div>;
}
function Background(){return <iframe className="raphael-bg" src="./reference/index.html" title="Raphael background"/>}
export default App;
