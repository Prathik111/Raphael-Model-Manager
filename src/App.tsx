import { useEffect, useState } from 'react';
import { api, fileUrl, subscribeToModelChanges } from './tauri';
import type { AppState, CivitaiImportPreview, ModelImage, ModelRecord, ModelType, LibraryCounts } from './types';

const TYPES: Array<{ key: ModelType | 'All'; label: string }> = [
  { key: 'All', label: 'ALL' }, { key: 'Checkpoint', label: 'CHECKPOINTS' }, { key: 'LoRA', label: 'LORAS' },
  { key: 'VAE', label: 'VAE' }, { key: 'ControlNet', label: 'CONTROLNET' }, { key: 'Embedding', label: 'EMBEDDINGS' },
  { key: 'Upscaler', label: 'UPSCALERS' }, { key: 'Other', label: 'OTHER' }
];

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
    <div className="thumb placeholder"><span>{initials(model.civitai_name || model.filename)}</span></div>
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

function Inspector({ model, images, onRefresh, onLinkCivitai }: { model: ModelRecord; images: ModelImage[]; onRefresh: ()=>void; onLinkCivitai: (url: string)=>Promise<void> }) {
  const [tab, setTab] = useState<'overview'|'examples'|'files'>('overview');
  const [showToken, setShowToken] = useState(false);
  const [token, setToken] = useState('');
  const [civitaiUrl, setCivitaiUrl] = useState(model.civitai_url || '');
  const [linkBusy, setLinkBusy] = useState(false);
  const [linkError, setLinkError] = useState<string | null>(null);
  const promptText = model.activation_prompts.join(', ');
  return <aside className="inspector hud-panel">
    <div className="inspector-header"><div><div className="eyebrow">MODEL</div><h2>{model.civitai_name || model.filename}</h2></div><span className="type-chip">{model.model_type}</span></div>
    <div className="inspector-tabs">{(['overview','examples','files'] as const).map(t=><button className={tab===t?'active':''} onClick={()=>setTab(t)} key={t}>{t.toUpperCase()}</button>)}</div>
    {tab==='overview' && <div className="inspector-scroll">
      <section><div className="section-head">DESCRIPTION</div><p className="description">{model.description || 'No description cached from Civitai.'}</p></section>
      <section><div className="section-head">ACTIVATION PROMPTS</div>{promptText ? <><div className="prompt-box">{promptText}</div><button className="text-btn" onClick={()=>navigator.clipboard?.writeText(promptText)}>COPY ALL</button></> : <div className="empty-inline">No activation prompts were published for this version.</div>}</section>
      <section><div className="section-head">TAGS</div><div className="chips">{model.tags.map(x=><span key={x}>{x}</span>)}</div></section>
      <section><div className="section-head">LOCATION</div><div className="mono-box">{model.path}</div><button className="text-btn" onClick={()=>api.openFolder(model.path)}>OPEN FOLDER</button></section>
      <section><div className="section-head">CIVITAI</div>{model.civitai_url ? <><div className="mono-box">{model.civitai_url}</div><button className="text-btn" onClick={onRefresh}>REFRESH CIVITAI DATA</button></> : <><p className="empty-inline">This local model is not linked to Civitai yet.</p><div className="link-row"><input value={civitaiUrl} onChange={e=>{setCivitaiUrl(e.target.value);setLinkError(null);}} placeholder="https://civitai.com/models/..."/><button className="primary-btn small" disabled={linkBusy} onClick={async()=>{if(!civitaiUrl.trim()) return; setLinkBusy(true); setLinkError(null); try { await onLinkCivitai(civitaiUrl.trim()); } catch (e) { setLinkError(String(e)); } finally { setLinkBusy(false); }}}>{linkBusy?'LINKING…':'LINK MODEL'}</button></div>{linkError ? <div className="error-box">{linkError}</div> : null}</>}</section>
      <section><div className="section-head">API TOKEN</div><button className="text-btn" onClick={()=>setShowToken(v=>!v)}>{showToken?'HIDE':'SET OPTIONAL CIVITAI TOKEN'}</button>{showToken && <div className="token-box"><input value={token} onChange={e=>setToken(e.target.value)} placeholder="Paste token" type="password"/><button className="primary-btn small" onClick={async()=>{await api.setCivitaiToken(token); setToken(''); setShowToken(false);}}>SAVE</button></div>}</section>
    </div>}
    {tab==='examples' && <div className="inspector-scroll"><section><div className="section-head">CACHED CIVITAI GALLERY · {images.length}</div><Gallery images={images}/></section></div>}
    {tab==='files' && <div className="inspector-scroll"><section><div className="section-head">LOCAL FILE</div><div className="kv"><span>SIZE</span><b>{fmtBytes(model.size_bytes)}</b></div><div className="kv"><span>TYPE</span><b>{model.model_type}</b></div><div className="kv"><span>BASE</span><b>{model.base_model || '—'}</b></div><div className="kv"><span>VERSION</span><b>{model.version_name || '—'}</b></div><div className="kv"><span>CREATOR</span><b>{model.creator || '—'}</b></div><div className="kv"><span>SHA256</span><b className="wrap">{model.source_hash || 'Not computed'}</b></div></section></div>}
  </aside>;
}

function App() {
  const [state,setState]=useState<AppState|null>(null); const [models,setModels]=useState<ModelRecord[]>([]); const [selectedId,setSelectedId]=useState<number|null>(null);
  const [type,setType]=useState<ModelType|'All'>('All'); const [query,setQuery]=useState(''); const [images,setImages]=useState<ModelImage[]>([]); const [importUrl,setImportUrl]=useState(''); const [preview,setPreview]=useState<CivitaiImportPreview|null>(null); const [busy,setBusy]=useState(false); const [sort,setSort]=useState('name'); const [counts,setCounts]=useState<LibraryCounts>({all:0,by_type:{}}); const [importError,setImportError]=useState<string|null>(null);
  const selected = models.find(m=>m.id===selectedId) || null;
  async function refresh(){ const s=await api.getState(); setState(s); if(!s.models_root){return;} const [list,allCounts]=await Promise.all([api.listModels({type:type==='All'?undefined:type,query:query||undefined}),api.getLibraryCounts()]); list.sort((a,b)=>sort==='size'?b.size_bytes-a.size_bytes:sort==='path'?a.relative_path.localeCompare(b.relative_path):(a.civitai_name||a.filename).localeCompare(b.civitai_name||b.filename)); setModels(list); setCounts(allCounts); if(selectedId && !list.some(m=>m.id===selectedId)) setSelectedId(list[0]?.id??null); else if(!selectedId && list[0]) setSelectedId(list[0].id); }
  useEffect(()=>{refresh();},[]);
  useEffect(()=>{let stop:undefined|(()=>void); subscribeToModelChanges(()=>refresh()).then(x=>{stop=()=>x();}); return ()=>stop?.();},[type,query,sort]);
  useEffect(()=>{if(!selected){setImages([]);return;} api.getImages(selected.id).then(setImages).catch(()=>setImages([])); api.syncModelGallery(selected.id).catch(()=>{}); const timer=window.setInterval(()=>api.getImages(selected.id).then(setImages).catch(()=>{}),2000); return ()=>window.clearInterval(timer);},[selectedId, selected?.civitai_model_id]);
  useEffect(()=>{const t=setTimeout(()=>refresh(),180); return ()=>clearTimeout(t);},[query,type,sort]);
  if(!state) return <div className="loading-shell"><PulseMark/></div>;
  if(!state.models_root) return <><Background/><Setup onReady={s=>{setState(s); refresh();}}/></>;
  const doImport = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try {setPreview(await api.importCivitai(importUrl.trim()));} catch (e) { setImportError(String(e)); } finally {setBusy(false);} };
  const install = async()=>{ if(!importUrl.trim()) return; setBusy(true); setImportError(null); try { const m=await api.installCivitai(importUrl.trim()); setPreview(null); setImportUrl(''); await refresh(); setSelectedId(m.id); } catch (e) { setImportError(String(e)); } finally { setBusy(false);} };
  return <div className="app-shell"><Background/><div className="noise"/>
    <header className="topbar"><div className="brand"><PulseMark/><span>RAPHAEL MODEL MANAGER</span></div><div className="top-stats"><span>CACHED <b>{fmtBytes(state.storage.cached_bytes)}</b></span><span>TOTAL <b>{fmtBytes(state.storage.total_model_bytes)}</b></span></div><div className="root-path" title={state.models_root}>{state.models_root}</div></header>
    <div className="workspace">
      <aside className="sidebar hud-panel"><div className="side-title">LIBRARY</div><nav>{TYPES.map(t=><button key={t.key} className={type===t.key?'active':''} onClick={()=>setType(t.key)}><span>{t.label}</span><b>{t.key==='All' ? counts.all : (counts.by_type[t.key] ?? 0)}</b></button>)}</nav><div className="sidebar-foot"><button className="text-btn" onClick={async()=>{const p=await api.chooseModelsFolder(); if(p) await api.setModelsRoot(p);}}>CHANGE FOLDER</button></div></aside>
      <main className="library"><div className="library-head"><div><div className="eyebrow">{type.toUpperCase()}</div><h1>{type==='All'?'MODEL LIBRARY':type.toUpperCase()}</h1></div><div className="library-tools"><input value={query} onChange={e=>setQuery(e.target.value)} placeholder="Search models..."/><button className="import-btn" onClick={()=>{setImportError(null);setImportUrl('');setPreview({model:{},version:{id:0,name:'',base_model:null,download_url:'',filename:null,size_bytes:null,activation_prompts:[]},target_directory:''});}}>IMPORT CIVITAI</button><select value={sort} onChange={e=>setSort(e.target.value)}><option value="name">NAME</option><option value="size">SIZE</option><option value="path">PATH</option></select></div></div><div className="grid">{models.map(m=><ModelCard key={m.id} model={m} selected={m.id===selectedId} onClick={()=>setSelectedId(m.id)}/>)}{!models.length&&<div className="empty-state">No models match the current view.</div>}</div></main>
      {selected && <Inspector model={selected} images={images} onRefresh={async()=>{await api.refreshModel(selected.id); await refresh();}} onLinkCivitai={async(url)=>{await api.linkModelCivitai(selected.id,url); await refresh();}}/>}
    </div>
    {(preview || importUrl) && <div className="modal-backdrop" onClick={()=>{if(!busy){setPreview(null); setImportUrl('');}}}><div className="import-modal hud-panel" onClick={e=>e.stopPropagation()}><div className="eyebrow">CIVITAI IMPORT</div><h2>INSTALL A MODEL</h2>{!preview || preview.version.id===0 ? <>{importError ? <div className="error-box modal-error">{importError}</div> : null}<div className="import-row"><input value={importUrl} onChange={e=>setImportUrl(e.target.value)} onKeyDown={e=>e.key==='Enter'&&doImport()} placeholder="https://civitai.com/models/..." autoFocus/><button className="primary-btn" onClick={doImport} disabled={busy}>{busy?'ANALYZING…':'ANALYZE'}</button></div></> : <><div className="preview-panel"><h3>{preview.model.name || preview.version.filename || 'Model'}</h3><div className="kv"><span>TYPE</span><b>{preview.model.type || 'Model'}</b></div><div className="kv"><span>BASE</span><b>{preview.version.base_model || '—'}</b></div><div className="kv"><span>FILE</span><b>{preview.version.filename || 'Automatic filename'}</b></div><div className="kv"><span>TARGET</span><b className="wrap">{preview.target_directory}</b></div><div className="section-head">ACTIVATION PROMPTS</div><div className="chips">{preview.version.activation_prompts.map(x=><span key={x}>{x}</span>)}</div></div><div className="modal-actions"><button className="text-btn" onClick={()=>{setPreview(null);setImportError(null);}}>BACK</button><button className="primary-btn" onClick={install} disabled={busy}>{busy?'DOWNLOADING…':'DOWNLOAD & INSTALL'}</button></div></>}</div></div>}
  </div>;
}
function Background(){return <iframe className="raphael-bg" src="./reference/index.html" title="Raphael background"/>}
export default App;
