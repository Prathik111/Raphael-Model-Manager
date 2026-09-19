use chrono::Utc;
use futures_util::StreamExt;
use image::imageops::FilterType;
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use reqwest::{header, Client};
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    fs::{self, File},
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::{Duration, UNIX_EPOCH},
};
use tauri::{AppHandle, Emitter, Manager, State};
use thiserror::Error;
use url::Url;
use walkdir::WalkDir;

const API_BASE: &str = "https://civitai.com/api/v1";
const USER_AGENT: &str = "RaphaelModelManager/0.1.0";

#[derive(Debug, Error)]
enum AppError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("network error: {0}")]
    Network(#[from] reqwest::Error),
    #[error("url error: {0}")]
    Url(#[from] url::ParseError),
    #[error("api error: {0}")]
    Api(String),
    #[error("invalid input: {0}")]
    Invalid(String),
    #[error("keyring error: {0}")]
    Keyring(String),
}

impl serde::Serialize for AppError {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where S: serde::Serializer {
        serializer.serialize_str(&self.to_string())
    }
}

type AppResult<T> = Result<T, AppError>;

#[derive(Clone)]
struct AppStateInner {
    app_data: PathBuf,
    models_root: Arc<RwLock<Option<PathBuf>>>,
    watcher: Arc<Mutex<Option<RecommendedWatcher>>>,
    scan_lock: Arc<Mutex<()>>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelRecord {
    id: i64,
    path: String,
    relative_path: String,
    filename: String,
    model_type: String,
    size_bytes: i64,
    modified_at: i64,
    civitai_model_id: Option<i64>,
    civitai_version_id: Option<i64>,
    civitai_url: Option<String>,
    civitai_name: Option<String>,
    version_name: Option<String>,
    base_model: Option<String>,
    creator: Option<String>,
    description: Option<String>,
    tags: Vec<String>,
    activation_prompts: Vec<String>,
    source_hash: Option<String>,
    updated_at: i64,
}

#[derive(Debug, Serialize, Deserialize)]
struct ModelImage {
    id: i64,
    civitai_image_id: i64,
    local_path: Option<String>,
    thumbnail_path: Option<String>,
    width: Option<i64>,
    height: Option<i64>,
    prompt: Option<String>,
    negative_prompt: Option<String>,
    steps: Option<i64>,
    cfg: Option<f64>,
    sampler: Option<String>,
    seed: Option<i64>,
    meta_json: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CategoryStats { r#type: String, count: i64, bytes: i64 }
#[derive(Debug, Serialize, Deserialize)]
struct StorageStats { total_model_bytes: i64, cached_bytes: i64, categories: Vec<CategoryStats> }
#[derive(Debug, Serialize, Deserialize)]
struct AppStateResponse { models_root: Option<String>, storage: StorageStats }

#[derive(Debug, Serialize, Deserialize)]
struct CivitaiImportPreview {
    model: Value,
    version: Value,
    target_directory: String,
    images_count_hint: Option<i64>,
}

#[derive(Debug, Serialize, Deserialize)]
struct CivitaiEnvelope { metadata: Option<Value>, items: Vec<Value> }

fn now() -> i64 { Utc::now().timestamp() }

fn db_path(app_data: &Path) -> PathBuf { app_data.join("raphael.db") }
fn cache_root(app_data: &Path) -> PathBuf { app_data.join("cache").join("civitai") }
fn open_db(app_data: &Path) -> AppResult<Connection> {
    fs::create_dir_all(app_data)?;
    let c = Connection::open(db_path(app_data))?;
    c.busy_timeout(Duration::from_secs(5))?;
    c.execute_batch(r#"
      PRAGMA journal_mode=WAL;
      PRAGMA foreign_keys=ON;
      CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT NOT NULL);
      CREATE TABLE IF NOT EXISTS models (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        path TEXT NOT NULL UNIQUE,
        relative_path TEXT NOT NULL,
        filename TEXT NOT NULL,
        model_type TEXT NOT NULL,
        size_bytes INTEGER NOT NULL,
        modified_at INTEGER NOT NULL,
        civitai_model_id INTEGER,
        civitai_version_id INTEGER,
        civitai_url TEXT,
        civitai_name TEXT,
        version_name TEXT,
        base_model TEXT,
        creator TEXT,
        description TEXT,
        tags_json TEXT NOT NULL DEFAULT '[]',
        activation_json TEXT NOT NULL DEFAULT '[]',
        source_hash TEXT,
        updated_at INTEGER NOT NULL
      );
      CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
      CREATE INDEX IF NOT EXISTS idx_models_civitai ON models(civitai_model_id, civitai_version_id);
      CREATE TABLE IF NOT EXISTS images (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        model_id INTEGER NOT NULL REFERENCES models(id) ON DELETE CASCADE,
        civitai_image_id INTEGER NOT NULL,
        local_path TEXT,
        thumbnail_path TEXT,
        width INTEGER,
        height INTEGER,
        prompt TEXT,
        negative_prompt TEXT,
        steps INTEGER,
        cfg REAL,
        sampler TEXT,
        seed INTEGER,
        meta_json TEXT,
        UNIQUE(model_id, civitai_image_id)
      );
    "#)?;
    Ok(c)
}

fn setting(c: &Connection, key: &str) -> AppResult<Option<String>> {
    Ok(c.query_row("SELECT value FROM settings WHERE key=?1", [key], |r| r.get(0)).optional()?)
}
fn put_setting(c: &Connection, key: &str, value: &str) -> AppResult<()> {
    c.execute("INSERT INTO settings(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value])?;
    Ok(())
}

fn file_type_from_path(path: &Path, root: &Path) -> String {
    let rel = path.strip_prefix(root).unwrap_or(path);
    let first = rel.components().next().and_then(|c| c.as_os_str().to_str()).unwrap_or("").to_lowercase();
    match first.as_str() {
        "checkpoints" | "checkpoint" | "diffusion_models" | "unet" | "unets" => "Checkpoint",
        "loras" | "lora" | "lycoris" => "LoRA",
        "vae" | "vaes" => "VAE",
        "controlnet" | "controlnets" => "ControlNet",
        "embeddings" | "embedding" | "textual_inversion" => "Embedding",
        "upscale_models" | "upscalers" => "Upscaler",
        "clip" | "text_encoders" | "text_encoder" => "Text Encoder",
        "clip_vision" | "clip_visions" => "CLIP Vision",
        "ipadapter" | "ip_adapter" => "IP-Adapter",
        _ => "Other",
    }.to_string()
}
fn is_model_file(path: &Path) -> bool {
    let ext = path.extension().and_then(|x| x.to_str()).unwrap_or("").to_lowercase();
    matches!(ext.as_str(), "safetensors" | "ckpt" | "pt" | "pth" | "bin" | "gguf" | "onnx")
}
fn mtime(path: &Path) -> i64 {
    fs::metadata(path).and_then(|m| m.modified()).ok().and_then(|t| t.duration_since(UNIX_EPOCH).ok()).map(|d| d.as_secs() as i64).unwrap_or(0)
}
fn sha256_file(path: &Path) -> AppResult<String> {
    let mut reader = BufReader::new(File::open(path)?);
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; 4 * 1024 * 1024];
    loop { let n = reader.read(&mut buf)?; if n == 0 { break; } hasher.update(&buf[..n]); }
    Ok(hex::encode(hasher.finalize()))
}

fn scan_root(app: &AppStateInner, root: &Path) -> AppResult<()> {
    let _guard = app.scan_lock.lock().unwrap();
    let c = open_db(&app.app_data)?;
    let mut seen = Vec::<String>::new();
    for entry in WalkDir::new(root).follow_links(false).into_iter().filter_map(Result::ok) {
        if !entry.file_type().is_file() || !is_model_file(entry.path()) { continue; }
        let path = entry.path().to_path_buf();
        let path_s = path.to_string_lossy().to_string();
        let meta = match fs::metadata(&path) { Ok(m)=>m, Err(_)=>continue };
        let size = meta.len() as i64;
        let modified = mtime(&path);
        seen.push(path_s.clone());
        let old: Option<(i64,i64,i64,Option<String>)> = c.query_row(
            "SELECT id,size_bytes,modified_at,source_hash FROM models WHERE path=?1", [&path_s], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?))
        ).optional()?;
        if old.as_ref().map(|(_,s,m,_)| *s == size && *m == modified).unwrap_or(false) {
            c.execute("UPDATE models SET modified_at=?2, updated_at=?3 WHERE path=?1", params![path_s, modified, now()])?;
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace("\\","/");
        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let mtype = file_type_from_path(&path, root);
        let hash: Option<String> = None;
        if let Some((id,_,_,_)) = old {
            c.execute("UPDATE models SET relative_path=?2,filename=?3,model_type=?4,size_bytes=?5,modified_at=?6,source_hash=?7,updated_at=?8 WHERE id=?1", params![id,rel,filename,mtype,size,modified,hash,now()])?;
        } else {
            c.execute("INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,source_hash,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8)", params![path_s,rel,filename,mtype,size,modified,hash,now()])?;
        }
    }
    let mut stmt = c.prepare("SELECT path FROM models")?;
    let existing: Vec<String> = stmt.query_map([], |r| r.get(0))?.filter_map(Result::ok).collect();
    drop(stmt);
    for path in existing { if !seen.contains(&path) { c.execute("DELETE FROM models WHERE path=?1", [&path])?; } }
    Ok(())
}

fn recursive_scan_and_emit(app: AppStateInner, handle: AppHandle) {
    if let Some(root) = app.models_root.read().unwrap().clone() {
        if scan_root(&app, &root).is_ok() { let _ = handle.emit("models-changed", ()); }
    }
}

fn model_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRecord> {
    let tags: String = r.get(15)?; let activ: String = r.get(16)?;
    Ok(ModelRecord { id:r.get(0)?, path:r.get(1)?, relative_path:r.get(2)?, filename:r.get(3)?, model_type:r.get(4)?, size_bytes:r.get(5)?, modified_at:r.get(6)?, civitai_model_id:r.get(7)?, civitai_version_id:r.get(8)?, civitai_url:r.get(9)?, civitai_name:r.get(10)?, version_name:r.get(11)?, base_model:r.get(12)?, creator:r.get(13)?, description:r.get(14)?, tags:serde_json::from_str(&tags).unwrap_or_default(), activation_prompts:serde_json::from_str(&activ).unwrap_or_default(), source_hash:r.get(17)?, updated_at:r.get(18)? })
}
fn model_by_id(c: &Connection, id: i64) -> AppResult<ModelRecord> {
    Ok(c.query_row("SELECT id,path,relative_path,filename,model_type,size_bytes,modified_at,civitai_model_id,civitai_version_id,civitai_url,civitai_name,version_name,base_model,creator,description,tags_json,activation_json,source_hash,updated_at FROM models WHERE id=?1", [id], model_from_row)?)
}

fn civitai_client(_app: &AppStateInner) -> AppResult<Client> {
    let mut b=Client::builder().user_agent(USER_AGENT);
    let _ = &mut b;
    Ok(b.build()?)
}
fn token() -> Option<String> {
    keyring::Entry::new("Raphael Model Manager", "civitai").ok().and_then(|e| e.get_password().ok())
}
async fn api_get(app: &AppStateInner, url: &str) -> AppResult<Value> {
    let client=civitai_client(app)?;
    let mut req=client.get(url);
    if let Some(t)=token(){ req=req.bearer_auth(t); }
    let res=req.send().await?;
    if !res.status().is_success(){ return Err(AppError::Api(format!("Civitai returned {}",res.status()))); }
    Ok(res.json::<Value>().await?)
}

fn model_id_and_version(url: &str) -> AppResult<(i64,Option<i64>)> {
    let u=Url::parse(url)?;
    if !(u.host_str().unwrap_or("") == "civitai.com" || u.host_str().unwrap_or("").ends_with(".civitai.com")){ return Err(AppError::Invalid("Expected a civitai.com model URL".into())); }
    let parts: Vec<&str>=u.path_segments().map(|s|s.collect()).unwrap_or_default();
    let model_id=parts.iter().position(|x|*x=="models").and_then(|i|parts.get(i+1)).and_then(|x|x.parse().ok()).ok_or_else(||AppError::Invalid("Could not find Civitai model ID".into()))?;
    let version_id=u.query_pairs().find(|(k,_)|k=="modelVersionId").and_then(|(_,v)|v.parse().ok());
    Ok((model_id,version_id))
}

fn selected_file(version: &Value) -> Option<(String,Option<i64>,String)> {
    let files=version.get("files")?.as_array()?;
    let f=files.iter().find(|f|f.get("primary").and_then(Value::as_bool).unwrap_or(false)).or_else(||files.first())?;
    let name=f.get("name").and_then(Value::as_str).map(str::to_string);
    let size=f.get("sizeKB").and_then(Value::as_f64).map(|x|(x*1024.0) as i64);
    let dl=version.get("downloadUrl").and_then(Value::as_str).or_else(||f.get("downloadUrl").and_then(Value::as_str))?.to_string();
    Some((dl,size,name.unwrap_or_default()))
}

async fn fetch_model_and_version(app:&AppStateInner, source:&str) -> AppResult<(Value,Value)> {
    let (mid,vid)=model_id_and_version(source)?;
    let model=api_get(app,&format!("{API_BASE}/models/{mid}")).await?;
    let version=if let Some(v)=vid { api_get(app,&format!("{API_BASE}/model-versions/{v}")).await? } else { model.get("modelVersions").and_then(Value::as_array).and_then(|a|a.first()).cloned().ok_or_else(||AppError::Api("No published model version is available".into()))? };
    Ok((model,version))
}

fn civitai_type_to_folder(t:&str)->&'static str { match t.to_lowercase().as_str(){"checkpoint"=>"checkpoints","lora"|"locon"|"lycoris"=>"loras","vae"=>"vae","controlnet"=>"controlnet","textualinversion"=>"embeddings","upscaler"=>"upscale_models","ipadapter"=>"ipadapter","clip"=>"text_encoders",_=>"other"} }
fn json_strings(v:Option<&Value>)->Vec<String>{v.and_then(Value::as_array).map(|a|a.iter().filter_map(|x|x.as_str().map(str::to_string)).collect()).unwrap_or_default()}
fn strip_html(s:&str)->String{let mut out=String::with_capacity(s.len());let mut in_tag=false;for ch in s.chars(){match ch{ '<'=>in_tag=true,'>'=>in_tag=false,_ if !in_tag=>out.push(ch),_=>{}}}out.replace("&nbsp;"," ").replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")}

#[tauri::command]
fn get_app_state(app: State<AppStateInner>) -> AppResult<AppStateResponse> {
    let c=open_db(&app.app_data)?; let root=setting(&c,"models_root")?; let storage=storage_stats_inner(&app.app_data)?; Ok(AppStateResponse{models_root:root,storage})
}
#[tauri::command]
fn set_models_root(app: State<AppStateInner>, handle: AppHandle, path:String)->AppResult<AppStateResponse>{
    let root=PathBuf::from(&path); if !root.is_dir(){return Err(AppError::Invalid("Selected path is not a folder".into()));}
    let c=open_db(&app.app_data)?; put_setting(&c,"models_root",&path)?; *app.models_root.write().unwrap()=Some(root.clone());
    if let Some(old)=app.watcher.lock().unwrap().take(){ drop(old); }
    let app_clone=app.inner().clone(); let handle_clone=handle.clone();
    let mut watcher=notify::recommended_watcher(move |res:Result<notify::Event,notify::Error>|{ if let Ok(e)=res { match e.kind { EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => { std::thread::sleep(Duration::from_millis(120)); recursive_scan_and_emit(app_clone.clone(),handle_clone.clone()); }, _=>{} } } }).map_err(|e|AppError::Io(io::Error::new(io::ErrorKind::Other,e.to_string())))?;
    watcher.watch(&root,RecursiveMode::Recursive).map_err(|e|AppError::Io(io::Error::new(io::ErrorKind::Other,e.to_string())))?; *app.watcher.lock().unwrap()=Some(watcher);
    scan_root(&app,&root)?; let _=handle.emit("models-changed",()); spawn_hash_enrichment(app.inner().clone(),handle.clone()); Ok(AppStateResponse{models_root:Some(path),storage:storage_stats_inner(&app.app_data)?})
}
#[tauri::command]
fn list_models(app:State<AppStateInner>, r#type:Option<String>, query:Option<String>)->AppResult<Vec<ModelRecord>>{
    let c=open_db(&app.app_data)?; let mut sql="SELECT id,path,relative_path,filename,model_type,size_bytes,modified_at,civitai_model_id,civitai_version_id,civitai_url,civitai_name,version_name,base_model,creator,description,tags_json,activation_json,source_hash,updated_at FROM models WHERE 1=1".to_string(); let mut args:Vec<String>=vec![];
    if let Some(t)=r#type {sql.push_str(" AND model_type=?");args.push(t)}
    if let Some(q)=query {sql.push_str(" AND (filename LIKE ? OR relative_path LIKE ? OR civitai_name LIKE ?)"); let x=format!("%{q}%");args.extend([x.clone(),x.clone(),x]);}
    sql.push_str(" ORDER BY COALESCE(civitai_name,filename) COLLATE NOCASE"); let mut stmt=c.prepare(&sql)?; let rows=stmt.query_map(rusqlite::params_from_iter(args.iter()),model_from_row)?; Ok(rows.filter_map(Result::ok).collect())
}
#[tauri::command]
fn get_model_images(app:State<AppStateInner>, id:i64)->AppResult<Vec<ModelImage>>{
    let c=open_db(&app.app_data)?; let mut stmt=c.prepare("SELECT id,civitai_image_id,local_path,thumbnail_path,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,meta_json FROM images WHERE model_id=?1 ORDER BY id")?; let rows=stmt.query_map([id],|r|Ok(ModelImage{id:r.get(0)?,civitai_image_id:r.get(1)?,local_path:r.get(2)?,thumbnail_path:r.get(3)?,width:r.get(4)?,height:r.get(5)?,prompt:r.get(6)?,negative_prompt:r.get(7)?,steps:r.get(8)?,cfg:r.get(9)?,sampler:r.get(10)?,seed:r.get(11)?,meta_json:r.get(12)?}))?; Ok(rows.filter_map(Result::ok).collect())
}
#[tauri::command]
async fn preview_civitai_import(app:State<'_,AppStateInner>,url:String)->AppResult<CivitaiImportPreview>{
    let (model,version)=fetch_model_and_version(&app,&url).await?; let (dl,size,filename)=selected_file(&version).ok_or_else(||AppError::Api("No downloadable public file found for this version".into()))?;
    let root=app.models_root.read().unwrap().clone().ok_or_else(||AppError::Invalid("Choose your ComfyUI models folder first".into()))?; let typ=model.get("type").and_then(Value::as_str).unwrap_or("Other"); let target=root.join(civitai_type_to_folder(typ));
    let activation=json_strings(version.get("trainedWords")); Ok(CivitaiImportPreview{model:json!({"id":model.get("id"),"name":model.get("name"),"type":typ,"description":model.get("description"),"tags":model.get("tags"),"creator":model.get("creator").and_then(|v|v.get("username"))}),version:json!({"id":version.get("id"),"name":version.get("name"),"base_model":version.get("baseModel"),"download_url":dl,"filename":filename,"size_bytes":size,"activation_prompts":activation}),target_directory:target.to_string_lossy().to_string(),images_count_hint:version.get("images").and_then(Value::as_array).map(|x|x.len() as i64)})
}

async fn download_file(app:&AppStateInner, url:&str, target_dir:&Path)->AppResult<(PathBuf,i64)> {
    fs::create_dir_all(target_dir)?; let client=civitai_client(app).await?; let mut req=client.get(url); if let Some(t)=token(){req=req.bearer_auth(t);} let res=req.send().await?; if !res.status().is_success(){return Err(AppError::Api(format!("Download failed: {}",res.status())))}
    let name=res.headers().get(header::CONTENT_DISPOSITION).and_then(|v|v.to_str().ok()).and_then(|s|s.split("filename=").nth(1)).map(|s|s.trim().trim_matches('"').trim_matches('\'').to_string()).filter(|s|!s.is_empty()).unwrap_or_else(||url.rsplit('/').next().unwrap_or("model.safetensors").split('?').next().unwrap_or("model.safetensors").to_string());
    let path=target_dir.join(name); let partial=path.with_extension(format!("{}.part",path.extension().and_then(|x|x.to_str()).unwrap_or("bin"))); let mut file=File::create(&partial)?; let mut stream=res.bytes_stream(); let mut total=0i64; while let Some(chunk)=stream.next().await {let b=chunk?; total+=b.len() as i64; file.write_all(&b)?;} file.flush()?; fs::rename(&partial,&path)?; Ok((path,total))
}

#[tauri::command]
async fn install_civitai_model(app:State<'_,AppStateInner>, handle:AppHandle, url:String)->AppResult<ModelRecord>{
    let (model,version)=fetch_model_and_version(&app,&url).await?; let version_id=version.get("id").and_then(Value::as_i64); if let Some(vid)=version_id { let c0=open_db(&app.app_data)?; if let Ok(existing_id)=c0.query_row("SELECT id FROM models WHERE civitai_version_id=?1 AND path IS NOT NULL",[vid],|r|r.get::<_,i64>(0)){ return model_by_id(&c0,existing_id); } } let typ=model.get("type").and_then(Value::as_str).unwrap_or("Other").to_string(); let (dl,_,_)=selected_file(&version).ok_or_else(||AppError::Api("No downloadable public file found".into()))?; let root=app.models_root.read().unwrap().clone().ok_or_else(||AppError::Invalid("Choose your ComfyUI models folder first".into()))?; let target=root.join(civitai_type_to_folder(&typ)); let (path,size)=download_file(&app,&dl,&target).await?;
    let hash=sha256_file(&path).ok(); let c=open_db(&app.app_data)?; let rel=path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().replace("\\","/"); let tags=json_strings(model.get("tags")); let activation=json_strings(version.get("trainedWords")); let desc=model.get("description").and_then(Value::as_str).map(strip_html); let creator=model.get("creator").and_then(|v|v.get("username")).and_then(Value::as_str).map(str::to_string); let mid=model.get("id").and_then(Value::as_i64); let vid=version.get("id").and_then(Value::as_i64); let vname=version.get("name").and_then(Value::as_str).map(str::to_string); let base=version.get("baseModel").and_then(Value::as_str).map(str::to_string);
    c.execute("INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,civitai_model_id,civitai_version_id,civitai_url,civitai_name,version_name,base_model,creator,description,tags_json,activation_json,source_hash,updated_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18) ON CONFLICT(path) DO UPDATE SET size_bytes=excluded.size_bytes,modified_at=excluded.modified_at,civitai_model_id=excluded.civitai_model_id,civitai_version_id=excluded.civitai_version_id,civitai_url=excluded.civitai_url,civitai_name=excluded.civitai_name,version_name=excluded.version_name,base_model=excluded.base_model,creator=excluded.creator,description=excluded.description,tags_json=excluded.tags_json,activation_json=excluded.activation_json,source_hash=excluded.source_hash,updated_at=excluded.updated_at",params![path.to_string_lossy(),rel,path.file_name().unwrap_or_default().to_string_lossy(),typ,size,mtime(&path),mid,vid,url,model.get("name").and_then(Value::as_str),vname,base,creator,desc,serde_json::to_string(&tags).unwrap(),serde_json::to_string(&activation).unwrap(),hash,now()])?;
    let rec=model_by_id(&c,c.query_row("SELECT id FROM models WHERE path=?1",[path.to_string_lossy().to_string()],|r|r.get(0))?)?; let _=handle.emit("models-changed",()); let _=sync_gallery_inner(app.inner().clone(),rec.id,handle.clone()).await; Ok(rec)
}

fn parse_meta(meta:&Value,key:&str)->Option<String>{meta.get(key).and_then(Value::as_str).map(str::to_string)}
async fn sync_gallery_inner(app:AppStateInner, model_id:i64, handle:AppHandle)->AppResult<()> {
    { let c=open_db(&app.app_data)?; let cached:i64=c.query_row("SELECT COUNT(*) FROM images WHERE model_id=?1",[model_id],|r|r.get(0))?; let missing:i64=c.query_row("SELECT COUNT(*) FROM images WHERE model_id=?1 AND local_path IS NULL",[model_id],|r|r.get(0))?; if cached>0 && missing==0 { return Ok(()); } }
    let model=model_by_id(&open_db(&app.app_data)?,model_id)?; let civitai_id=match model.civitai_model_id{Some(x)=>x,None=>return Ok(())}; let cache=cache_root(&app.app_data).join(civitai_id.to_string()); fs::create_dir_all(&cache)?; let mut cursor:Option<String>=None; let client=civitai_client(&app)?;
    loop {
        let mut url=format!("{API_BASE}/images?modelId={civitai_id}&limit=200&withMeta=true"); if let Some(c)=&cursor{url.push_str("&cursor=");url.push_str(&urlencoding::encode(c));}
        let mut req=client.get(&url); if let Some(t)=token(){req=req.bearer_auth(t);} let res=req.send().await?; if !res.status().is_success(){return Err(AppError::Api(format!("Image API returned {}",res.status())))} let body:CivitaiEnvelope=res.json().await?; let db=open_db(&app.app_data)?;
        for img in body.items { let iid=match img.get("id").and_then(Value::as_i64){Some(x)=>x,None=>continue}; let remote=img.get("url").and_then(Value::as_str).unwrap_or(""); let ext=remote.split('?').next().and_then(|x|Path::new(x).extension()).and_then(|x|x.to_str()).unwrap_or("jpg"); let local=cache.join(format!("{}.{}",iid,ext)); let thumb=cache.join(format!("{}_thumb.webp",iid));
            let mut local_path=None; if local.exists(){local_path=Some(local.to_string_lossy().to_string())} else if !remote.is_empty(){ if let Ok(resp)=client.get(remote).send().await { if let Ok(resp)=resp.error_for_status() { if let Ok(bytes)=resp.bytes().await { let _=fs::write(&local,&bytes); local_path=Some(local.to_string_lossy().to_string()); } } } }
            if local_path.is_some() && !thumb.exists(){ if let Some(lp)=&local_path { if let Ok(im)=image::open(lp){let t=im.resize(420,420,FilterType::Triangle); let _=t.save_with_format(&thumb,image::ImageFormat::WebP); } } }
            let meta=img.get("meta").cloned().unwrap_or(Value::Null); let prompt=parse_meta(&meta,"prompt"); let neg=parse_meta(&meta,"negativePrompt").or_else(||parse_meta(&meta,"Negative prompt")); let sampler=parse_meta(&meta,"sampler").or_else(||parse_meta(&meta,"Sampler")); let steps=meta.get("steps").and_then(Value::as_i64); let cfg=meta.get("cfgScale").or_else(||meta.get("cfg")).and_then(Value::as_f64); let seed=meta.get("seed").and_then(Value::as_i64); let width=img.get("width").and_then(Value::as_i64); let height=img.get("height").and_then(Value::as_i64);
            db.execute("INSERT INTO images(model_id,civitai_image_id,local_path,thumbnail_path,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,meta_json) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13) ON CONFLICT(model_id,civitai_image_id) DO UPDATE SET local_path=excluded.local_path,thumbnail_path=excluded.thumbnail_path,width=excluded.width,height=excluded.height,prompt=excluded.prompt,negative_prompt=excluded.negative_prompt,steps=excluded.steps,cfg=excluded.cfg,sampler=excluded.sampler,seed=excluded.seed,meta_json=excluded.meta_json",params![model_id,iid,local_path,if thumb.exists(){Some(thumb.to_string_lossy().to_string())}else{None::<String>},width,height,prompt,neg,steps,cfg,sampler,seed,serde_json::to_string(&meta).unwrap()])?;
        }
        cursor=body.metadata.and_then(|m|m.get("nextCursor").and_then(Value::as_str).map(str::to_string)); if cursor.is_none(){break;}
    }
    if let Ok(c)=open_db(&app.app_data){let bytes=dir_size(&cache_root(&app.app_data));let _=put_setting(&c,"cache_bytes",&bytes.to_string());}
    let _=handle.emit("models-changed",()); Ok(())
}

#[tauri::command]
async fn sync_model_gallery(app:State<'_,AppStateInner>,handle:AppHandle,id:i64)->AppResult<()> { let state=app.inner().clone(); tauri::async_runtime::spawn(async move {let _=sync_gallery_inner(state,id,handle).await;}); Ok(()) }
#[tauri::command]
async fn refresh_model_civitai(app:State<'_,AppStateInner>,handle:AppHandle,id:i64)->AppResult<ModelRecord>{
    let current=model_by_id(&open_db(&app.app_data)?,id)?; let url=current.civitai_url.clone().ok_or_else(||AppError::Invalid("This model is not linked to Civitai".into()))?; let (model,version)=fetch_model_and_version(&app,&url).await?; let tags=json_strings(model.get("tags")); let activation=json_strings(version.get("trainedWords")); let desc=model.get("description").and_then(Value::as_str).map(strip_html); let creator=model.get("creator").and_then(|v|v.get("username")).and_then(Value::as_str).map(str::to_string); let c=open_db(&app.app_data)?; c.execute("UPDATE models SET civitai_model_id=?2,civitai_version_id=?3,civitai_name=?4,version_name=?5,base_model=?6,creator=?7,description=?8,tags_json=?9,activation_json=?10,updated_at=?11 WHERE id=?1",params![id,model.get("id").and_then(Value::as_i64),version.get("id").and_then(Value::as_i64),model.get("name").and_then(Value::as_str),version.get("name").and_then(Value::as_str),version.get("baseModel").and_then(Value::as_str),creator,desc,serde_json::to_string(&tags).unwrap(),serde_json::to_string(&activation).unwrap(),now()])?; let rec=model_by_id(&c,id)?; let _=sync_gallery_inner(app.inner().clone(),id,handle.clone()).await; let _=handle.emit("models-changed",()); Ok(rec)
}

fn storage_stats_inner(app_data:&Path)->AppResult<StorageStats>{let c=open_db(app_data)?;let total:i64=c.query_row("SELECT COALESCE(SUM(size_bytes),0) FROM models",[],|r|r.get(0))?;let cached=match setting(&c,"cache_bytes")?{Some(v)=>v.parse::<i64>().unwrap_or(0),None=>{let v=dir_size(&cache_root(app_data));let _=put_setting(&c,"cache_bytes",&v.to_string());v}};let mut stmt=c.prepare("SELECT model_type,COUNT(*),COALESCE(SUM(size_bytes),0) FROM models GROUP BY model_type ORDER BY model_type")?;let categories=stmt.query_map([],|r|Ok(CategoryStats{r#type:r.get(0)?,count:r.get(1)?,bytes:r.get(2)?}))?.filter_map(Result::ok).collect();Ok(StorageStats{total_model_bytes:total,cached_bytes:cached,categories})}
fn dir_size(path:&Path)->i64{if !path.exists(){return 0} WalkDir::new(path).into_iter().filter_map(Result::ok).filter_map(|e|e.metadata().ok()).filter(|m|m.is_file()).map(|m|m.len() as i64).sum()}
#[tauri::command]
fn get_storage_stats(app:State<AppStateInner>)->AppResult<StorageStats>{storage_stats_inner(&app.app_data)}
#[tauri::command]
fn open_in_file_manager(path:String)->AppResult<()>{let p=PathBuf::from(path); let target=if p.is_file(){p.parent().unwrap_or(&p).to_path_buf()}else{p}; #[cfg(target_os="windows")] {std::process::Command::new("explorer").arg(target).spawn()?;} #[cfg(target_os="macos")] {std::process::Command::new("open").arg(target).spawn()?;} #[cfg(target_os="linux")] {std::process::Command::new("xdg-open").arg(target).spawn()?;} Ok(())}
#[tauri::command]
fn set_civitai_token(token:String)->AppResult<()>{let e=keyring::Entry::new("Raphael Model Manager","civitai").map_err(|e|AppError::Keyring(e.to_string()))?; if token.trim().is_empty(){let _=e.delete_credential();}else{e.set_password(token.trim()).map_err(|e|AppError::Keyring(e.to_string()))?;}Ok(())}
#[tauri::command]
fn is_civitai_token_set()->bool{keyring::Entry::new("Raphael Model Manager","civitai").ok().and_then(|e|e.get_password().ok()).map(|x|!x.trim().is_empty()).unwrap_or(false)}

fn spawn_hash_enrichment(app: AppStateInner, handle: AppHandle) {
    tauri::async_runtime::spawn(async move {
        let paths: Vec<(i64, String)> = {
            let connection = match open_db(&app.app_data) {
                Ok(value) => value,
                Err(_) => return,
            };

            let mut statement = match connection.prepare(
                "SELECT id, path FROM models WHERE civitai_model_id IS NULL AND source_hash IS NULL",
            ) {
                Ok(value) => value,
                Err(_) => return,
            };

            match statement.query_map([], |row| Ok((row.get(0)?, row.get(1)?))) {
                Ok(rows) => rows.filter_map(Result::ok).collect(),
                Err(_) => return,
            }
        };

        for (id, path) in paths {
            let client = match civitai_client(&app) {
                Ok(value) => value,
                Err(_) => continue,
            };

            let hash = match sha256_file(Path::new(&path)) {
                Ok(value) => value,
                Err(_) => continue,
            };

            if let Ok(db) = open_db(&app.app_data) {
                let _ = db.execute(
                    "UPDATE models SET source_hash=?2, updated_at=?3 WHERE id=?1",
                    params![id, hash, now()],
                );
            }

            let url = format!("{API_BASE}/model-versions/by-hash/{hash}");
            let mut request = client.get(url);
            if let Some(t) = token() {
                request = request.bearer_auth(t);
            }

            let response = match request.send().await {
                Ok(value) if value.status().is_success() => value,
                _ => continue,
            };

            let version: Value = match response.json().await {
                Ok(value) => value,
                Err(_) => continue,
            };

            let model_id = version.get("modelId").and_then(Value::as_i64);
            let version_id = version.get("id").and_then(Value::as_i64);
            let model_name = version
                .get("model")
                .and_then(|m| m.get("name"))
                .and_then(Value::as_str)
                .or_else(|| version.get("modelName").and_then(Value::as_str));
            let base_model = version.get("baseModel").and_then(Value::as_str);
            let tags = version
                .get("model")
                .and_then(|m| m.get("tags"))
                .cloned()
                .unwrap_or_else(|| json!([]));
            let activation = version
                .get("trainedWords")
                .cloned()
                .unwrap_or_else(|| json!([]));
            let civitai_url = match (model_id, version_id) {
                (Some(mid), Some(vid)) => {
                    Some(format!("https://civitai.com/models/{mid}?modelVersionId={vid}"))
                }
                _ => None,
            };

            if let Ok(db) = open_db(&app.app_data) {
                let _ = db.execute(
                    "UPDATE models
                     SET civitai_model_id=?2,
                         civitai_version_id=?3,
                         civitai_url=?4,
                         civitai_name=?5,
                         version_name=?6,
                         base_model=?7,
                         tags_json=?8,
                         activation_json=?9,
                         updated_at=?10
                     WHERE id=?1",
                    params![
                        id,
                        model_id,
                        version_id,
                        civitai_url,
                        model_name,
                        version.get("name").and_then(Value::as_str),
                        base_model,
                        serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into()),
                        serde_json::to_string(&activation).unwrap_or_else(|_| "[]".into()),
                        now()
                    ],
                );
            }

            let _ = handle.emit("models-changed", ());
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    fn test_state(app_data: PathBuf, models_root: PathBuf) -> AppStateInner {
        AppStateInner {
            app_data,
            models_root: Arc::new(RwLock::new(Some(models_root))),
            watcher: Arc::new(Mutex::new(None)),
            scan_lock: Arc::new(Mutex::new(())),
        }
    }

    #[test]
    fn model_extensions_are_detected() {
        assert!(is_model_file(Path::new("model.safetensors")));
        assert!(is_model_file(Path::new("model.GGUF")));
        assert!(is_model_file(Path::new("model.onnx")));
        assert!(!is_model_file(Path::new("preview.png")));
        assert!(!is_model_file(Path::new("README.txt")));
    }

    #[test]
    fn nested_comfyui_roots_are_classified() {
        let root = PathBuf::from(r"C:\ComfyUI\models");
        assert_eq!(
            file_type_from_path(&root.join(r"loras\characters\megumin\megumin.safetensors"), &root),
            "LoRA"
        );
        assert_eq!(
            file_type_from_path(&root.join(r"checkpoints\anime\model.safetensors"), &root),
            "Checkpoint"
        );
        assert_eq!(
            file_type_from_path(&root.join(r"controlnet\depth\model.safetensors"), &root),
            "ControlNet"
        );
        assert_eq!(
            file_type_from_path(&root.join(r"anything\else\model.gguf"), &root),
            "Other"
        );
    }

    #[test]
    fn civitai_urls_parse_model_and_version() {
        assert_eq!(
            model_id_and_version("https://civitai.com/models/12345?modelVersionId=67890").unwrap(),
            (12345, Some(67890))
        );
        assert_eq!(
            model_id_and_version("https://civitai.com/models/12345").unwrap(),
            (12345, None)
        );
        assert!(model_id_and_version("https://example.com/models/12345").is_err());
        assert!(model_id_and_version("https://civitai.com/images/12345").is_err());
    }

    #[test]
    fn civitai_types_map_to_comfyui_folders() {
        assert_eq!(civitai_type_to_folder("Checkpoint"), "checkpoints");
        assert_eq!(civitai_type_to_folder("LORA"), "loras");
        assert_eq!(civitai_type_to_folder("LoCon"), "loras");
        assert_eq!(civitai_type_to_folder("TextualInversion"), "embeddings");
        assert_eq!(civitai_type_to_folder("Upscaler"), "upscale_models");
        assert_eq!(civitai_type_to_folder("UnknownType"), "other");
    }

    #[test]
    fn html_description_is_stripped_without_panicking() {
        assert_eq!(
            strip_html("<p>Hello &amp; world</p><strong>Raphael</strong>"),
            "Hello & worldRaphael"
        );
    }

    #[test]
    fn recursive_scan_indexes_nested_models_and_removes_deleted_files() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let root = temp.path().join("models");
        fs::create_dir_all(root.join("loras/characters/Megumin")).unwrap();
        fs::create_dir_all(root.join("checkpoints/anime")).unwrap();
        fs::write(root.join("loras/characters/Megumin/megumin.safetensors"), b"fake-lora").unwrap();
        fs::write(root.join("checkpoints/anime/model.safetensors"), b"fake-checkpoint").unwrap();
        fs::write(root.join("loras/characters/Megumin/preview.png"), b"not-a-model").unwrap();

        let state = test_state(app_data.clone(), root.clone());
        scan_root(&state, &root).unwrap();

        let db = open_db(&app_data).unwrap();
        let count: i64 = db.query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0)).unwrap();
        assert_eq!(count, 2);

        let lora_type: String = db.query_row(
            "SELECT model_type FROM models WHERE relative_path LIKE '%megumin.safetensors'",
            [],
            |r| r.get(0),
        ).unwrap();
        assert_eq!(lora_type, "LoRA");

        fs::remove_file(root.join("checkpoints/anime/model.safetensors")).unwrap();
        scan_root(&state, &root).unwrap();

        let count_after_delete: i64 =
            db.query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0)).unwrap();
        assert_eq!(count_after_delete, 1);
    }

    #[test]
    fn database_schema_is_idempotent() {
        let temp = tempfile::tempdir().unwrap();
        let first = open_db(temp.path()).unwrap();
        drop(first);
        let second = open_db(temp.path()).unwrap();
        let tables: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name IN ('settings','models','images')",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tables, 3);
    }
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data=app.path().app_data_dir()?;fs::create_dir_all(&app_data)?;let c=open_db(&app_data)?;let saved=setting(&c,"models_root")?;let state=AppStateInner{app_data:app_data.clone(),models_root:Arc::new(RwLock::new(saved.map(PathBuf::from))),watcher:Arc::new(Mutex::new(None)),scan_lock:Arc::new(Mutex::new(()))};app.manage(state.clone());
            if let Some(root)=state.models_root.read().unwrap().clone(){ if root.is_dir(){let _=scan_root(&state,&root);let handle=app.handle().clone();spawn_hash_enrichment(state.clone(),handle.clone());let state2=state.clone();let handle2=handle.clone();if let Ok(mut watcher)=notify::recommended_watcher(move |res:Result<notify::Event,notify::Error>|{if let Ok(e)=res{match e.kind{EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)=>{std::thread::sleep(Duration::from_millis(120));recursive_scan_and_emit(state2.clone(),handle2.clone());},_=>{}}}}){if watcher.watch(&root,RecursiveMode::Recursive).is_ok(){*state.watcher.lock().unwrap()=Some(watcher)}}}}
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_app_state,set_models_root,list_models,get_model_images,sync_model_gallery,preview_civitai_import,install_civitai_model,refresh_model_civitai,get_storage_stats,open_in_file_manager,set_civitai_token,is_civitai_token_set])
        .run(tauri::generate_context!())
        .expect("error while running Raphael Model Manager");
}
