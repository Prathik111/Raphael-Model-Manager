import { useEffect, useState } from 'react';
import { api, fileUrl, subscribeToModelChanges } from './tauri';
import type { AppState, CivitaiImportPreview, ModelImage, ModelRecord, ModelType, LibraryCounts, TagRecord } from './types';

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

function fmtBytes(n: number) { if (n < 1024) return `${n} B`; const u=['KB','MB','GB','TB']; let i=-1,v=n; do { v/=1024; i++; } while(v>=1024 && i<u.length-1); return `${v.toFixed(v>=100?0:v>=10?1:2)} ${u[i]}`; }
function initials(s: string) { return s.split(/\s+/).filter(Boolean).slice(0,2).map(x=>x[0]).join('').toUpperCase(); }

function PulseMark() {
  return <div className="pulse-mark" aria-label="Raphael"><span className="pulse-core"/><i/><i/><i/></div>;
}

function Setup({ onReady }: { onReady: (state: AppState)=>void }) {
  const [busy, setBusy] = useState(false);
  const choose = async () => { setBusy(true); try { const p = await api.chooseModelsFolder(); if (p) onReady(await api.setModelsRoot(p)); } finally { setBusy(false); } };
  return <div className="setup-shell">
    <div className="scan-corners"/>
    <PulseMark/>
    <div className="setup-card hud-panel">
      <div className="eyebrow">RAPHAEL MODEL MANAGER</div>
      <h1>Select your ComfyUI models folder</h1>
      <p>The manager watches this folder and everything beneath it. Your model files stay where they are.</p>
      <button className="primary-btn" onClick={choose} disabled={busy}>{busy ? 'OPENING…' : 'BROWSE MODELS FOLDER'}</button>
      <div className="tiny">Example: C:\ComfyUI\models</div>
    </div>
  </div>;
}

function ModelCard({ model, selected, onClick }: { model: ModelRecord; selected: boolean; onClick: ()=>void }) {
  return <button className={`model-card ${selected ? 'selected' : ''}`} onClick={onClick}>
    <div className="thumb model-thumb">{model.thumbnail_path ? <img src={fileUrl(model.thumbnail_path)} alt="" onError={(e)=>{e.currentTarget.style.display="none"; const fallback=e.currentTarget.nextElementSibling as HTMLElement|null; fallback?.classList.remove("hidden-fallback");}}/> : null}<span className={model.thumbnail_path ? "thumb-fallback hidden-fallback" : "thumb-fallback"}>{initials(model.civitai_name || model.filename)}</span></div>
    <div className="card-body">
      <div className="card-title">{model.civitai_name || model.filename.replace(/\.[^.]+$/, '')}</div>
      <div className="card-sub">{model.model_type} · {fmtBytes(model.size_bytes)}</div>
      <div className="card-path">{model.relative_path}</div>
    </div>
    {model.civitai_model_id ? <span className="civitai-dot" title="Linked to Civitai"/> : null}
  </button>;
}

function Gallery({ images }: { images: ModelImage[] }) {
  if (!images.length) return <div className="empty-inline">No cached Civitai images yet.</div>;
  return <div className="gallery-grid">{images.map((img) => <div className="gallery-item" key={img.id}>
    {img.local_path ? <img src={fileUrl(img.local_path)} alt="Civitai example"/> : <div className="thumb placeholder">IMAGE</div>}
    {img.prompt ? <div className="gallery-prompt">{img.prompt}</div> : null}<div className="gallery-meta">
      {img.steps || img.cfg || img.sampler ? <span>{img.sampler || 'sampler'} {img.steps ? `· ${img.steps} steps` : ''}</span> : null}
      {img.prompt ? <button onClick={()=>navigator.clipboard?.writeText(img.prompt!)}>COPY PROMPT</button> : null}
    </div>
  </div>)}</div>;
}


function TagEditor({ model, allTags, onSave, onFilter }: { model: ModelRecord; allTags: TagRecord[]; onSave: (tags: string[])=>Promise<void>; onFilter: (tag: string)=>void }) {
  const [draft, setDraft] = useState<string[]>(model.tags);
  const [input, setInput] = useState('');
  const [saving, setSaving] = useState(false);
  const needle = input.trim().toLowerCase();
  const suggestions = allTags
    .filter(tag => !draft.some(existing => existing.toLowerCase() === tag.name.toLowerCase()))
    .filter(tag => !needle || tag.name.toLowerCase().includes(needle))
    .slice(0, 8);

  const add = (raw: string) => {
    const value = raw.trim();
    if (!value || draft.some(tag => tag.toLowerCase() === value.toLowerCase())) return;
    setDraft(current => [...current, value]);
    setInput('');
  };
  const remove = (tag: string) => setDraft(current => current.filter(existing => existing !== tag));
  const save = async () => {
    setSaving(true);
    try { await onSave(draft); } finally { setSaving(false); }
  };

  return <div className="tag-editor">
    <div className="tag-editor-chips">
      {draft.map(tag => <span className="editable-tag" key={tag}>
        <button className="tag-value" title="Filter by this tag" onClick={()=>onFilter(tag)}>{tag}</button>
        <button className="tag-remove" aria-label={'Remove ' + tag} onClick={()=>remove(tag)}>×</button>
      </span>)}
      {!draft.length ? <span className="empty-inline">No tags assigned.</span> : null}
    </div>
    <div className="tag-input-row">
      <input value={input} onChange={e=>setInput(e.target.value)} onKeyDown={e=>{if(e.key==='Enter'){e.preventDefault();add(input);}}} placeholder="Type a tag…" disabled={saving}/>
      <button className="primary-btn small" onClick={()=>add(input)} disabled={saving || !input.trim()}>ADD</button>
    </div>
    {input.trim() ? <div className="tag-suggestions">
      {suggestions.length ? suggestions.map(tag => <button key={tag.name} onClick={()=>add(tag.name)}><span>{tag.name}</span><b>{tag.count}</b></button>) : <div className="tag-create-hint">Press ENTER to create “{input.trim()}”.</div>}
    </div> : null}
    <div className="tag-editor-footer">
      <span>{draft.length} TAG{draft.length === 1 ? '' : 'S'}</span>
      <button className="text-btn" onClick={save} disabled={saving}>{saving ? 'SAVING…' : 'SAVE TAGS'}</button>
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

function Inspector({ model, images, allTags, onRefresh, onLinkCivitai, onSaveTags, onSaveType, onFilterTag }: { model: ModelRecord; images: ModelImage[]; allTags: TagRecord[]; onRefresh: ()=>void; onLinkCivitai: (url: string)=>Promise<void>; onSaveTags: (tags: string[])=>Promise<void>; onSaveType: (type: string)=>Promise<void>; onFilterTag: (tag: string)=>void }) {
  const [tab, setTab] = useState<'overview'|'examples'|'files'>('overview');
  const [showToken, setShowToken] = useState(false);
  const [token, setToken] = useState('');
  const [civitaiUrl, setCivitaiUrl] = useState(model.civitai_url || '');
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);
  const promptText = model.activation_prompts.join(', ');
  return <aside className="inspector hud-panel">
    <div className="inspector-header"><div><div className="eyebrow">MODEL</div><h2>{model.civitai_name || model.filename}</h2></div><select className="type-select" value={model.model_type} onChange={async e=>{try{await onSaveType(e.target.value);}catch{e.currentTarget.value=model.model_type;}}} aria-label="Model type">{MODEL_TYPES.filter(x=>x.value!=='Auto' || x.value===model.model_type).map(x=><option key={x.value} value={x.value}>{x.label}</option>)}</select></div>
    <div className="inspector-tabs">{(['overview','examples','files'] as const).map(t=><button className={tab===t?'active':''} onClick={()=>setTab(t)} key={t}>{t.toUpperCase()}</button>)}</div>
    {tab==='overview' && <div className="inspector-scroll">
      <section><div className="section-head">DESCRIPTION</div><p className="description">{model.description || 'No description cached from Civitai.'}</p></section>
      <section><div className="section-head">ACTIVATION PROMPTS</div>{promptText ? <><div className="prompt-box">{promptText}</div><button className="text-btn" onClick={()=>navigator.clipboard?.writeText(promptText)}>COPY ALL</button></> : <div className="empty-inline">No activation prompts were published for this version.</div>}</section>
      <section><div className="section-head section-head-row"><span>TAGS</span><span className="section-action">EDITABLE</span></div><TagEditor model={model} allTags={allTags} onSave={onSaveTags} onFilter={onFilterTag}/></section>
      <section><div className="section-head">LOCATION</div><div className="mono-box">{model.path}</div><button className="text-btn" onClick={()=>api.openFolder(model.path)}>OPEN FOLDER</button></section>
      <section><div className="section-head">CIVITAI SOURCE</div>{model.civitai_url ? <div className="civitai-source-panel"><div className="source-line"><span className="source-dot"/><span className="source-label">LINKED SOURCE</span><span className="source-domain">{new URL(model.civitai_url).hostname.replace(/^www\./,'').toUpperCase()}</span></div><div className="mono-box source-url">{model.civitai_url}</div><button className="text-btn" onClick={onRefresh}>REFRESH SOURCE DATA</button></div> : <div className="civitai-link-panel"><div className="source-line"><span className="source-dot"/><span className="source-label">LINK LOCAL MODEL</span><span className="source-domain">CIVITAI</span></div><p className="empty-inline">Paste a Civitai model page to pull its metadata, gallery and thumbnail into Raphael.</p><div className="link-row"><input value={civitaiUrl} onChange={e=>{setCivitaiUrl(e.target.value);setLinkError(null);}} placeholder="civitai.com/models/... or civitai.red/models/..."/><button className="primary-btn small" disabled={linkBusy} onClick={async()=>{if(!civitaiUrl.trim()) return; setLinkBusy(true); setLinkError(null); try { await onLinkCivitai(civitaiUrl.trim()); } catch (e) { setLinkError(String(e)); } finally { setLinkBusy(false); }}}>{linkBusy?'FETCHING…':'FETCH DETAILS'}</button></div>{linkError ? <div className="error-box">{linkError}</div> : null}</div>}</section>
      <section><div className="section-head">API TOKEN</div><button className="text-btn" onClick={()=>setShowToken(v=>!v)}>{showToken?'HIDE':'SET OPTIONAL CIVITAI TOKEN'}</button>{showToken && <div className="token-box"><input value={token} onChange={e=>setToken(e.target.value)} placeholder="Paste token" type="password"/><button className="primary-btn small" onClick={async()=>{await api.setCivitaiToken(token); setToken(''); setShowToken(false);}}>SAVE</button></div>}</section>
    </div>}
    {tab==='examples' && <div className="inspector-scroll"><section><div className="section-head">CACHED CIVITAI GALLERY · {images.length}</div><Gallery images={images}/></section></div>}
    {tab==='files' && <div className="inspector-scroll"><section><div className="section-head">LOCAL FILE</div><div className="kv"><span>SIZE</span><b>{fmtBytes(model.size_bytes)}</b></div><div className="kv"><span>TYPE</span><b>{model.model_type}</b></div><div className="kv"><span>BASE</span><b>{model.base_model || '—'}</b></div><div className="kv"><span>VERSION</span><b>{model.version_name || '—'}</b></div><div className="kv"><span>CREATOR</span><b>{model.creator || '—'}</b></div><div className="kv"><span>SHA256</span><b className="wrap">{model.source_hash || 'Not computed'}</b></div></section></div>}
  </aside>;
}

function App() {
  const [state,setState]=useState<AppState|null>(null); const [models,setModels]=useState<ModelRecord[]>([]); const [selectedId,setSelectedId]=useState<number|null>(null);
  const [type,setType]=useState<ModelType|'All'>('All'); const [query,setQuery]=useState(''); const [activeTags,setActiveTags]=useState<string[]>([]); const [tagPanelOpen,setTagPanelOpen]=useState(false); const [allTags,setAllTags]=useState<TagRecord[]>([]); const [images,setImages]=useState<ModelImage[]>([]); const [importUrl,setImportUrl]=useState(''); const [preview,setPreview]=useState<CivitaiImportPreview|null>(null); const [busy,setBusy]=useState(false); const [sort,setSort]=useState('name'); const [counts,setCounts]=useState<LibraryCounts>({all:0,by_type:{}}); const [importError,setImportError]=useState<string|null>(null); const [downloadPath,setDownloadPath]=useState(''); const [importType,setImportType]=useState<ModelType>('Other'); const [customDownloadPath,setCustomDownloadPath]=useState(false);
  const selected = models.find(m=>m.id===selectedId) || null;
  async function refresh(){ const s=await api.getState(); setState(s); if(!s.models_root){return;} const [list,allCounts,tags]=await Promise.all([api.listModels({type:type==='All'?undefined:type,query:query||undefined,tags:activeTags}),api.getLibraryCounts(),api.getTags()]); list.sort((a,b)=>sort==='size'?b.size_bytes-a.size_bytes:sort==='path'?a.relative_path.localeCompare(b.relative_path):(a.civitai_name||a.filename).localeCompare(b.civitai_name||b.filename)); setModels(list); setCounts(allCounts); setAllTags(tags); setSelectedId(previous=>{ if(previous!==null && list.some(m=>m.id===previous)) return previous; return list[0]?.id ?? null; }); }
  useEffect(()=>{refresh();},[]);
  useEffect(()=>{let stop:undefined|(()=>void); subscribeToModelChanges(()=>refresh()).then(x=>{stop=()=>x();}); return ()=>stop?.();},[type,query,sort,activeTags]);
  useEffect(()=>{if(!selected){setImages([]);return;} api.getImages(selected.id).then(setImages).catch(()=>setImages([])); api.syncModelGallery(selected.id).catch(()=>{}); const timer=window.setInterval(()=>api.getImages(selected.id).then(setImages).catch(()=>{}),2000); return ()=>window.clearInterval(timer);},[selectedId, selected?.civitai_model_id]);
  useEffect(()=>{const t=setTimeout(()=>refresh(),180); return ()=>clearTimeout(t);},[query,type,sort,activeTags]);
  if(!state) return <div className="loading-shell"><PulseMark/></div>;
  if(!state.models_root) return <><Background/><Setup onReady={s=>{setState(s); refresh();}}/></>;
  const doImport = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try { const result=await api.importCivitai(importUrl.trim()); const nextType=civitaiTypeToModelType(result.model.type); setImportType(nextType); setCustomDownloadPath(false); setPreview(result); setDownloadPath(defaultImportDirectory(state.models_root,nextType) || result.target_directory); } catch (e) { setImportError(String(e)); } finally {setBusy(false);} };
  const install = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try { const m=await api.installCivitai(importUrl.trim(),customDownloadPath ? downloadPath : undefined,importType); setPreview(null); setImportUrl(''); setDownloadPath(''); setCustomDownloadPath(false); await refresh(); setSelectedId(m.id); } catch (e) { setImportError(String(e)); } finally { setBusy(false);} };
  return <div className="app-shell"><Background/><div className="noise"/>
    <header className="topbar"><div className="brand"><PulseMark/><span>RAPHAEL MODEL MANAGER</span></div><div className="top-stats"><span>CACHED <b>{fmtBytes(state.storage.cached_bytes)}</b></span><span>TOTAL <b>{fmtBytes(state.storage.total_model_bytes)}</b></span></div><div className="root-path" title={state.models_root}>{state.models_root}</div></header>
    <div className="workspace">
      <aside className="sidebar hud-panel"><div className="side-title">LIBRARY</div><nav>{TYPES.map(t=><button key={t.key} className={type===t.key?'active':''} onClick={()=>setType(t.key)}><span>{t.label}</span><b>{t.key==='All' ? counts.all : (counts.by_type[t.key] ?? 0)}</b></button>)}</nav><div className="sidebar-foot"><button className="text-btn" onClick={async()=>{const p=await api.chooseModelsFolder(); if(p) await api.setModelsRoot(p);}}>CHANGE FOLDER</button></div></aside>
      <main className="library"><div className="library-head"><div><div className="eyebrow">{type.toUpperCase()}</div><h1>{type==='All'?'MODEL LIBRARY':type.toUpperCase()}</h1></div><div className="library-tools"><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search models, tags, tag:…"/><button className={`tag-filter-button ${activeTags.length?'active':''}`} onClick={()=>setTagPanelOpen(v=>!v)}>TAGS{activeTags.length ? ` · ${activeTags.length}` : ''}</button><button className="import-btn" onClick={()=>{setImportError(null);setImportUrl('');setImportType('Other');setCustomDownloadPath(false);setDownloadPath('');setPreview({model:{},version:{id:0,name:'',base_model:null,download_url:'',filename:null,size_bytes:null,activation_prompts:[]},target_directory:'',thumbnail_path:null});}}>IMPORT CIVITAI</button><select value={sort} onChange={e=>setSort(e.target.value)}><option value="name">NAME</option><option value="size">SIZE</option><option value="path">PATH</option></select></div>{tagPanelOpen && <TagFilterPanel tags={allTags} activeTags={activeTags} onToggle={tag=>setActiveTags(current=>current.some(x=>x.toLowerCase()===tag.toLowerCase())?current.filter(x=>x.toLowerCase()!==tag.toLowerCase()):[...current,tag])} onClear={()=>setActiveTags([])}/>}</div>{activeTags.length ? <div className="active-tag-bar">{activeTags.map(tag=><button key={tag} onClick={()=>setActiveTags(current=>current.filter(x=>x.toLowerCase()!==tag.toLowerCase()))}>{tag}<span>×</span></button>)}<span className="active-tag-help">TAG FILTERS</span></div> : null}<div className="grid">{models.map(m=><ModelCard key={m.id} model={m} selected={m.id===selectedId} onClick={()=>setSelectedId(m.id)}/>)}{!models.length&&<div className="empty-state">No models match the current view.</div>}</div></main>
      {selected && <Inspector key={selected.id} model={selected} images={images} allTags={allTags} onRefresh={async()=>{await api.refreshModel(selected.id); await refresh();}} onLinkCivitai={async(url)=>{await api.linkModelCivitai(selected.id,url); await refresh();}} onSaveTags={async(tags)=>{await api.setModelTags(selected.id,tags); await refresh();}} onSaveType={async(nextType)=>{await api.setModelType(selected.id,nextType); await refresh();}} onFilterTag={tag=>{setActiveTags(current=>current.some(x=>x.toLowerCase()===tag.toLowerCase())?current:[...current,tag]);}}/>}
    </div>
    {(preview || importUrl) && <div className="modal-backdrop" onClick={()=>{if(!busy){setPreview(null); setImportUrl(''); setDownloadPath(''); setCustomDownloadPath(false); setImportError(null);}}}><div className="import-modal hud-panel" onClick={e=>e.stopPropagation()}><div className="eyebrow">CIVITAI IMPORT</div><h2>INSTALL A MODEL</h2>{!preview || preview.version.id===0 ? <>{importError ? <div className="error-box modal-error">{importError}</div> : null}<div className="import-row"><input value={importUrl} onChange={e=>setImportUrl(e.target.value)} onKeyDown={e=>e.key==='Enter'&&doImport()} placeholder="civitai.com/models/... or civitai.red/models/..." autoFocus/><button className="primary-btn" onClick={doImport} disabled={busy}>{busy?'ANALYZING…':'ANALYZE'}</button></div></> : <><div className="import-preview-grid"><div className="preview-image">{preview.thumbnail_path ? <img src={fileUrl(preview.thumbnail_path)} alt="" /> : <div className="thumb placeholder"><span>{initials(preview.model.name || preview.version.filename || 'MODEL')}</span></div>}</div><div className="preview-panel"><div className="preview-topline"><span className="type-chip">{importType}</span><span className="source-domain">CIVITAI</span></div><h3>{preview.model.name || preview.version.filename || 'Model'}</h3><div className="preview-meta">{preview.version.base_model || 'Base model unavailable'} · {preview.version.filename || 'Filename automatic'}</div><div className="kv"><span>MODEL FILE</span><b>{preview.version.filename || 'Automatic filename'}</b></div><div className="kv"><span>SIZE</span><b>{preview.version.size_bytes ? fmtBytes(preview.version.size_bytes) : 'Unknown'}</b></div><div className="section-head">LIBRARY TAG</div><div className="import-type-row"><label htmlFor="import-type">CLASSIFY AS</label><select id="import-type" className="import-type-select" value={importType} onChange={e=>{const next=e.target.value as ModelType; setImportType(next); if(!customDownloadPath) setDownloadPath(defaultImportDirectory(state.models_root,next));}} disabled={busy}>{IMPORT_TYPES.map(x=><option key={x} value={x}>{x}</option>)}</select></div><div className="import-type-note">Civitai suggests <b>{preview.model.type || 'Unknown'}</b>; Raphael uses the tag you choose for its library category and default folder.</div><div className="section-head">DOWNLOAD LOCATION</div><div className="destination-box"><div className="destination-path" title={downloadPath}>{downloadPath || defaultImportDirectory(state.models_root,importType) || preview.target_directory}</div><button className="primary-btn small" onClick={async()=>{const next=await api.chooseDirectory(downloadPath || defaultImportDirectory(state.models_root,importType) || preview.target_directory); if(next){setDownloadPath(next);setCustomDownloadPath(true);setImportError(null);}}} disabled={busy}>BROWSE</button></div><div className="destination-note">Choose a folder inside your configured ComfyUI models directory. Changing the tag updates the default folder until you manually browse.</div><div className="section-head">ACTIVATION PROMPTS</div><div className="chips">{preview.version.activation_prompts.map(x=><span key={x}>{x}</span>)}</div></div></div><div className="modal-actions"><button className="text-btn" onClick={()=>{setPreview(null);setImportError(null);}}>BACK</button><button className="primary-btn" onClick={install} disabled={busy}>{busy?'DOWNLOADING…':'DOWNLOAD & INSTALL'}</button></div>{importError ? <div className="error-box modal-error">{importError}</div> : null}</>}</div></div>}
  </div>;
}
function Background(){return <iframe className="raphael-bg" src="./reference/index.html" title="Raphael background"/>}
export default App;
