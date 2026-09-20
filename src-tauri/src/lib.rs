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

mod web;

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
    thumbnail_path: Option<String>,
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
struct LibraryCounts { all: i64, by_type: std::collections::BTreeMap<String, i64> }
#[derive(Debug, Serialize, Deserialize, Clone)]
struct TagRecord { name: String, count: i64 }
#[derive(Debug, Serialize, Deserialize)]
struct AppStateResponse { models_root: Option<String>, storage: StorageStats }

#[derive(Debug, Serialize, Deserialize)]
struct CivitaiImportPreview {
    model: Value,
    version: Value,
    target_directory: String,
    thumbnail_path: Option<String>,
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
        tags_user_modified INTEGER NOT NULL DEFAULT 0,
        model_type_user_modified INTEGER NOT NULL DEFAULT 0,
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
    let has_thumbnail:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='thumbnail_path'",[],|r|r.get(0))?;
    if has_thumbnail==0 { c.execute("ALTER TABLE models ADD COLUMN thumbnail_path TEXT",[])?; }
    let has_tag_lock:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='tags_user_modified'",[],|r|r.get(0))?;
    if has_tag_lock==0 { c.execute("ALTER TABLE models ADD COLUMN tags_user_modified INTEGER NOT NULL DEFAULT 0",[])?; }
    let has_type_lock:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='model_type_user_modified'",[],|r|r.get(0))?;
    if has_type_lock==0 { c.execute("ALTER TABLE models ADD COLUMN model_type_user_modified INTEGER NOT NULL DEFAULT 0",[])?; }
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
        let old: Option<(i64,i64,i64,Option<String>,i64)> = c.query_row(
            "SELECT id,size_bytes,modified_at,source_hash,model_type_user_modified FROM models WHERE path=?1", [&path_s], |r| Ok((r.get(0)?,r.get(1)?,r.get(2)?,r.get(3)?,r.get(4)?))
        ).optional()?;
        if old.as_ref().map(|(_,s,m,_,_)| *s == size && *m == modified).unwrap_or(false) {
            c.execute("UPDATE models SET modified_at=?2, updated_at=?3 WHERE path=?1", params![path_s, modified, now()])?;
            continue;
        }
        let rel = path.strip_prefix(root).unwrap_or(&path).to_string_lossy().replace("\\","/");
        let filename = path.file_name().unwrap_or_default().to_string_lossy().to_string();
        let mtype = file_type_from_path(&path, root);
        let hash: Option<String> = None;
        if let Some((id,_,_,_,_type_locked)) = old {
            c.execute("UPDATE models SET relative_path=?2,filename=?3,model_type=CASE WHEN model_type_user_modified=1 THEN model_type ELSE ?4 END,size_bytes=?5,modified_at=?6,source_hash=?7,updated_at=?8 WHERE id=?1", params![id,rel,filename,mtype,size,modified,hash,now()])?;
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
    Ok(ModelRecord { id:r.get(0)?, path:r.get(1)?, relative_path:r.get(2)?, filename:r.get(3)?, model_type:r.get(4)?, size_bytes:r.get(5)?, modified_at:r.get(6)?, civitai_model_id:r.get(7)?, civitai_version_id:r.get(8)?, civitai_url:r.get(9)?, civitai_name:r.get(10)?, version_name:r.get(11)?, base_model:r.get(12)?, creator:r.get(13)?, description:r.get(14)?, tags:serde_json::from_str(&tags).unwrap_or_default(), activation_prompts:serde_json::from_str(&activ).unwrap_or_default(), source_hash:r.get(17)?, thumbnail_path:r.get(18)?, updated_at:r.get(19)? })
}
const MODEL_SELECT: &str = "SELECT id,path,relative_path,filename,model_type,size_bytes,modified_at,civitai_model_id,civitai_version_id,civitai_url,civitai_name,version_name,base_model,creator,description,tags_json,activation_json,source_hash,thumbnail_path,updated_at FROM models";
fn model_by_id(c: &Connection, id: i64) -> AppResult<ModelRecord> {
    Ok(c.query_row(&format!("{MODEL_SELECT} WHERE id=?1"), [id], model_from_row)?)
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
    let status=res.status();
    if !status.is_success(){
        let body=res.text().await.unwrap_or_default();
        let detail=serde_json::from_str::<Value>(&body)
            .ok()
            .and_then(|v| v.get("message").and_then(Value::as_str).or_else(|| v.get("error").and_then(Value::as_str)).map(str::to_string))
            .filter(|x|!x.is_empty())
            .unwrap_or_else(|| if body.trim().is_empty(){"No response body".into()}else{body.trim().chars().take(220).collect()});
        let hint=match status.as_u16(){401|403=>" Check your Civitai token/permissions.",404=>" Check that the model URL and model version still exist.",_=>""};
        return Err(AppError::Api(format!("Civitai returned {status}: {detail}.{hint}")));
    }
    Ok(res.json::<Value>().await?)
}

fn model_id_and_version(url: &str) -> AppResult<(i64,Option<i64>)> {
    let u=Url::parse(url)?;
    let host=u.host_str().unwrap_or("").to_ascii_lowercase();
    if !(host=="civitai.com" || host=="www.civitai.com" || host=="civitai.red" || host=="www.civitai.red"){ return Err(AppError::Invalid("Expected a civitai.com or civitai.red model URL".into())); }
    let parts: Vec<&str>=u.path_segments().map(|s|s.collect()).unwrap_or_default();
    let model_id=parts.iter().position(|x|*x=="models").and_then(|i|parts.get(i+1)).and_then(|x|x.parse().ok()).ok_or_else(||AppError::Invalid("Could not find Civitai model ID".into()))?;
    let version_id=u.query_pairs().find(|(k,_)|k=="modelVersionId").and_then(|(_,v)|v.parse().ok());
    Ok((model_id,version_id))
}

fn selected_file(version: &Value) -> Option<(String, Option<i64>, String, Option<String>)> {
    let files = version.get("files")?.as_array()?;
    let file = files
        .iter()
        .find(|f| f.get("primary").and_then(Value::as_bool).unwrap_or(false))
        .or_else(|| files.iter().find(|f| {
            f.get("name")
                .and_then(Value::as_str)
                .map(|name| name.ends_with(".safetensors"))
                .unwrap_or(false)
        }))
        .or_else(|| files.first())?;

    let name = file.get("name").and_then(Value::as_str).map(str::to_string);
    let size = file
        .get("sizeKB")
        .and_then(Value::as_f64)
        .map(|x| (x * 1024.0) as i64);
    let download_url = version
        .get("downloadUrl")
        .and_then(Value::as_str)
        .or_else(|| file.get("downloadUrl").and_then(Value::as_str))?
        .to_string();
    let sha256 = file
        .get("hashes")
        .and_then(|h| h.get("SHA256"))
        .and_then(Value::as_str)
        .map(|x| x.to_ascii_lowercase());

    Some((download_url, size, name.unwrap_or_default(), sha256))
}

async fn fetch_model_and_version(app:&AppStateInner, source:&str) -> AppResult<(Value,Value)> {
    let (mid,vid)=model_id_and_version(source)?;
    let model=api_get(app,&format!("{API_BASE}/models/{mid}")).await?;
    let version=if let Some(v)=vid { api_get(app,&format!("{API_BASE}/model-versions/{v}")).await? } else { model.get("modelVersions").and_then(Value::as_array).and_then(|a|a.first()).cloned().ok_or_else(||AppError::Api("No published model version is available".into()))? };
    Ok((model,version))
}

fn civitai_type_to_folder(t:&str)->&'static str { match t.to_lowercase().as_str(){"checkpoint"=>"checkpoints","lora"|"locon"|"lycoris"=>"loras","vae"=>"vae","controlnet"=>"controlnet","textualinversion"|"embedding"=>"embeddings","upscaler"=>"upscale_models","ipadapter"|"ip-adapter"=>"ipadapter","clip"|"text encoder"=>"text_encoders","clipvision"|"clip vision"=>"clip_vision",_=>"other"} }
fn civitai_type_to_model_type(t:&str)->&'static str { match t.to_lowercase().as_str(){"checkpoint"=>"Checkpoint","lora"|"locon"|"lycoris"=>"LoRA","vae"=>"VAE","controlnet"=>"ControlNet","textualinversion"|"embedding"=>"Embedding","upscaler"=>"Upscaler","clip"=>"Text Encoder","clipvision"=>"CLIP Vision","ipadapter"|"ip-adapter"=>"IP-Adapter",_=>"Other"} }
fn normalized_import_type(t:&str)->Option<&'static str> { match t.to_ascii_lowercase().as_str(){"checkpoint"=>Some("Checkpoint"),"lora"=>Some("LoRA"),"vae"=>Some("VAE"),"controlnet"=>Some("ControlNet"),"embedding"=>Some("Embedding"),"upscaler"=>Some("Upscaler"),"text encoder"=>Some("Text Encoder"),"clip vision"=>Some("CLIP Vision"),"ip-adapter"|"ipadapter"=>Some("IP-Adapter"),"other"=>Some("Other"),_=>None} }
fn civitai_host(url:&str)->AppResult<String>{Ok(Url::parse(url)?.host_str().unwrap_or("civitai.com").to_ascii_lowercase().replace("www.",""))}
fn canonical_civitai_url(source:&str,model_id:Option<i64>,version_id:Option<i64>)->AppResult<String>{let host=civitai_host(source)?;Ok(match (model_id,version_id){(Some(mid),Some(vid))=>format!("https://{host}/models/{mid}?modelVersionId={vid}"),(Some(mid),None)=>format!("https://{host}/models/{mid}"),_=>source.to_string()})}
fn json_strings(v:Option<&Value>)->Vec<String>{v.and_then(Value::as_array).map(|a|a.iter().filter_map(|x|x.as_str().map(str::to_string)).collect()).unwrap_or_default()}
fn strip_html(s:&str)->String{let mut out=String::with_capacity(s.len());let mut in_tag=false;for ch in s.chars(){match ch{ '<'=>in_tag=true,'>'=>in_tag=false,_ if !in_tag=>out.push(ch),_=>{}}}out.replace("&nbsp;"," ").replace("&amp;","&").replace("&lt;","<").replace("&gt;",">")}

async fn download_cached_thumbnail(app:&AppStateInner, cache: &Path, url:&str)->AppResult<Option<String>>{
    if cache.exists(){return Ok(Some(cache.to_string_lossy().to_string()));}
    let client=civitai_client(app)?;
    let mut req=client.get(url);
    if let Some(t)=token(){req=req.bearer_auth(t);}
    let res=req.send().await?;
    if !res.status().is_success(){return Ok(None);}
    let bytes=res.bytes().await?;
    if bytes.is_empty(){return Ok(None);}
    if let Some(parent)=cache.parent(){fs::create_dir_all(parent)?;}
    fs::write(cache,&bytes)?;
    Ok(Some(cache.to_string_lossy().to_string()))
}

async fn ensure_model_thumbnail(app:&AppStateInner, model_id:i64, model:&Value, version:&Value, source_url:&str)->AppResult<Option<String>>{
    let root=cache_root(&app.app_data).join(model_id.to_string());
    fs::create_dir_all(&root)?;
    let mut remote:Option<String>=model.get("images").and_then(Value::as_array).and_then(|a|a.iter().find_map(|x|x.get("url").and_then(Value::as_str).map(str::to_string)));
    if remote.is_none(){remote=version.get("images").and_then(Value::as_array).and_then(|a|a.iter().find_map(|x|x.get("url").and_then(Value::as_str).map(str::to_string)));}

    if let Some(url)=remote {
        let ext=Url::parse(&url).ok().and_then(|u|Path::new(u.path()).extension().and_then(|x|x.to_str()).map(str::to_string)).unwrap_or_else(||"jpg".into());
        return download_cached_thumbnail(app,&root.join(format!("thumbnail.{ext}")),&url).await;
    }

    let client=civitai_client(app)?;
    let mut req=client.get(source_url);
    if let Some(t)=token(){req=req.bearer_auth(t);}
    let res=req.send().await?;
    if !res.status().is_success(){return Ok(None);}
    let html=res.text().await?;
    let lower=html.to_ascii_lowercase();
    let marker="property=\"og:image\"";
    if let Some(pos)=lower.find(marker){
        let tail=&html[pos+marker.len()..];
        if let Some(content_pos)=tail.to_ascii_lowercase().find("content=\""){
            let value=&tail[content_pos+9..];
            if let Some(end)=value.find('"'){
                let image_url=&value[..end];
                return download_cached_thumbnail(app,&root.join("thumbnail.jpg"),image_url).await;
            }
        }
    }
    Ok(None)
}

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
    let mut watcher=notify::recommended_watcher(move |res:Result<notify::Event,notify::Error>|{ if let Ok(e)=res { match e.kind { EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => { std::thread::sleep(Duration::from_millis(120)); recursive_scan_and_emit(app_clone.clone(),handle_clone.clone()); }, _=>{} } } }).map_err(|e|AppError::Io(io::Error::other(e.to_string())))?;
    watcher.watch(&root,RecursiveMode::Recursive).map_err(|e|AppError::Io(io::Error::other(e.to_string())))?; *app.watcher.lock().unwrap()=Some(watcher);
    scan_root(&app,&root)?; let _=handle.emit("models-changed",()); spawn_hash_enrichment(app.inner().clone(),handle.clone()); Ok(AppStateResponse{models_root:Some(path),storage:storage_stats_inner(&app.app_data)?})
}
fn normalize_tags(tags: Vec<String>) -> Vec<String> {
    let mut result: Vec<String> = Vec::new();
    for raw in tags {
        let tag = raw.trim();
        if tag.is_empty() { continue; }
        if !result.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
            result.push(tag.to_string());
        }
    }
    result.sort_by_key(|tag| tag.to_ascii_lowercase());
    result
}

fn tokenize_search(query: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    for ch in query.chars() {
        match ch {
            '"' => quoted = !quoted,
            c if c.is_whitespace() && !quoted => {
                if !current.is_empty() { out.push(std::mem::take(&mut current)); }
            }
            c => current.push(c),
        }
    }
    if !current.is_empty() { out.push(current); }
    out
}

fn value_contains(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

fn model_search_match(model: &ModelRecord, query: &str, active_tags: &[String]) -> bool {
    if !active_tags.iter().all(|tag| model.tags.iter().any(|existing| existing.eq_ignore_ascii_case(tag))) {
        return false;
    }
    let searchable = [
        model.filename.as_str(),
        model.relative_path.as_str(),
        model.civitai_name.as_deref().unwrap_or(""),
        model.version_name.as_deref().unwrap_or(""),
        model.base_model.as_deref().unwrap_or(""),
        model.creator.as_deref().unwrap_or(""),
        model.description.as_deref().unwrap_or(""),
    ];
    for raw in tokenize_search(query) {
        if raw.is_empty() { continue; }
        let lower = raw.to_ascii_lowercase();
        let mut exclude = false;
        let mut tag_only = false;
        let value = if let Some(rest) = lower.strip_prefix("-tag:").or_else(|| lower.strip_prefix("-tags:")) {
            exclude = true; tag_only = true; rest
        } else if let Some(rest) = lower.strip_prefix("tag:").or_else(|| lower.strip_prefix("tags:")) {
            tag_only = true; rest
        } else if let Some(rest) = lower.strip_prefix("#") {
            tag_only = true; rest
        } else if let Some(rest) = lower.strip_prefix("-") {
            exclude = true; rest
        } else {
            lower.as_str()
        };
        if value.is_empty() { continue; }
        let matched = if tag_only {
            model.tags.iter().any(|tag| value_contains(tag, value))
        } else {
            searchable.iter().any(|field| value_contains(field, value))
                || model.tags.iter().any(|tag| value_contains(tag, value))
                || model.activation_prompts.iter().any(|prompt| value_contains(prompt, value))
        };
        if exclude {
            if matched { return false; }
        } else if !matched {
            return false;
        }
    }
    true
}

#[tauri::command]
fn list_models(app:State<AppStateInner>, r#type:Option<String>, query:Option<String>, tags:Option<Vec<String>>)->AppResult<Vec<ModelRecord>>{
    let c=open_db(&app.app_data)?;
    let mut sql=format!("{MODEL_SELECT} WHERE 1=1");
    let mut args:Vec<String>=vec![];
    if let Some(t)=r#type { sql.push_str(" AND model_type=?"); args.push(t); }
    sql.push_str(" ORDER BY COALESCE(civitai_name,filename) COLLATE NOCASE");
    let mut stmt=c.prepare(&sql)?;
    let rows=stmt.query_map(rusqlite::params_from_iter(args.iter()),model_from_row)?;
    let query=query.unwrap_or_default();
    let active_tags=tags.unwrap_or_default();
    let mut result:Vec<ModelRecord>=Vec::new();
    for row in rows {
        let model=row?;
        if model_search_match(&model,&query,&active_tags) { result.push(model); }
    }
    Ok(result)
}

#[tauri::command]
fn get_tags(app:State<AppStateInner>)->AppResult<Vec<TagRecord>>{
    let c=open_db(&app.app_data)?;
    let mut stmt=c.prepare("SELECT tags_json FROM models")?;
    let rows=stmt.query_map([],|r|r.get::<_,String>(0))?;
    let mut counts:std::collections::BTreeMap<String,(String,i64)>=std::collections::BTreeMap::new();
    for row in rows {
        let raw=row?;
        let tags:Vec<String>=serde_json::from_str(&raw).unwrap_or_default();
        for tag in normalize_tags(tags) {
            let key=tag.to_ascii_lowercase();
            let entry=counts.entry(key).or_insert_with(||(tag.clone(),0));
            entry.1+=1;
        }
    }
    let mut result:Vec<TagRecord>=counts.into_iter().map(|(_, (name,count))|TagRecord{name,count}).collect();
    result.sort_by(|a,b| b.count.cmp(&a.count).then_with(||a.name.to_ascii_lowercase().cmp(&b.name.to_ascii_lowercase())));
    Ok(result)
}

#[tauri::command]
fn set_model_tags(app:State<AppStateInner>, handle:AppHandle, id:i64, tags:Vec<String>)->AppResult<ModelRecord>{
    let normalized=normalize_tags(tags);
    let c=open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET tags_json=?2,tags_user_modified=1,updated_at=?3 WHERE id=?1",
        params![id,serde_json::to_string(&normalized).unwrap_or_else(|_|"[]".into()),now()],
    )?;
    let rec=model_by_id(&c,id)?;
    let _=handle.emit("models-changed",());
    Ok(rec)
}

fn add_subfolder_tags_inner(app: &AppStateInner) -> AppResult<i64> {
    let c = open_db(&app.app_data)?;
    let rows: Vec<(i64, String, String)> = {
        let mut stmt = c.prepare("SELECT id,relative_path,tags_json FROM models")?;
        stmt.query_map([], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?
        .collect::<rusqlite::Result<Vec<_>>>()?
    };

    let mut updated = 0_i64;
    for (id, relative_path, raw_tags) in rows {
        let parts: Vec<&str> = relative_path
            .split('/')
            .filter(|part| !part.is_empty())
            .collect();
        if parts.len() < 3 {
            continue;
        }

        let existing: Vec<String> = serde_json::from_str(&raw_tags).unwrap_or_default();
        let mut merged = existing.clone();
        for part in &parts[1..parts.len() - 1] {
            let tag = part.trim();
            if !tag.is_empty() && !merged.iter().any(|current| current.eq_ignore_ascii_case(tag)) {
                merged.push(tag.to_string());
            }
        }

        let current_normalized = normalize_tags(existing);
        let next_normalized = normalize_tags(merged);
        if next_normalized == current_normalized {
            continue;
        }

        c.execute(
            "UPDATE models SET tags_json=?2,tags_user_modified=1,updated_at=?3 WHERE id=?1",
            params![
                id,
                serde_json::to_string(&next_normalized).unwrap_or_else(|_| "[]".into()),
                now()
            ],
        )?;
        updated += 1;
    }

    Ok(updated)
}

#[tauri::command]
fn add_subfolder_tags(app: State<AppStateInner>, handle: AppHandle) -> AppResult<i64> {
    let updated = add_subfolder_tags_inner(&app)?;
    let _ = handle.emit("models-changed", ());
    Ok(updated)
}

#[tauri::command]
fn delete_model(app:State<AppStateInner>, handle:AppHandle, id:i64)->AppResult<()> {
    let root = app.models_root.read().unwrap().clone()
        .ok_or_else(|| AppError::Invalid("Choose your ComfyUI models folder first".into()))?;

    let (path, thumbnail_path, image_paths) = {
        let c = open_db(&app.app_data)?;
        let model = model_by_id(&c, id)?;
        let mut stmt = c.prepare("SELECT local_path,thumbnail_path FROM images WHERE model_id=?1")?;
        let rows = stmt.query_map([id], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?))
        })?;
        let image_paths: Vec<(Option<String>,Option<String>)> = rows.filter_map(Result::ok).collect();
        (PathBuf::from(model.path), model.thumbnail_path, image_paths)
    };

    if !path.starts_with(&root) {
        return Err(AppError::Invalid("Refusing to delete a model outside the configured models folder".into()));
    }

    if path.exists() {
        let meta = fs::metadata(&path)?;
        if !meta.is_file() {
            return Err(AppError::Invalid("The model path is not a regular file".into()));
        }
        fs::remove_file(&path)?;
    }

    for (local_path, thumb_path) in image_paths {
        for cached in [local_path, thumb_path] {
            if let Some(cached) = cached {
                let cached_path = PathBuf::from(cached);
                if cached_path.starts_with(&app.app_data) && cached_path.is_file() {
                    let _ = fs::remove_file(cached_path);
                }
            }
        }
    }

    if let Some(thumbnail) = thumbnail_path {
        let thumbnail_path = PathBuf::from(thumbnail);
        if thumbnail_path.starts_with(&app.app_data) && thumbnail_path.is_file() {
            let _ = fs::remove_file(thumbnail_path);
        }
    }

    let c = open_db(&app.app_data)?;
    let deleted = c.execute("DELETE FROM models WHERE id=?1", [id])?;
    if deleted == 0 {
        return Err(AppError::Invalid("Model no longer exists".into()));
    }
    let _ = handle.emit("models-changed", ());
    Ok(())
}

#[tauri::command]
fn set_model_type(app:State<AppStateInner>, handle:AppHandle, id:i64, model_type:String)->AppResult<ModelRecord>{
    let requested=model_type.trim();
    let c=open_db(&app.app_data)?;
    let current_path:String=c.query_row("SELECT path FROM models WHERE id=?1",[id],|r|r.get(0))?;
    let next_type=if requested.eq_ignore_ascii_case("auto") {
        let root=app.models_root.read().unwrap().clone().ok_or_else(||AppError::Invalid("Choose your ComfyUI models folder first".into()))?;
        file_type_from_path(Path::new(&current_path),&root)
    } else {
        match requested {
            "Checkpoint"|"LoRA"|"VAE"|"ControlNet"|"Embedding"|"Upscaler"|"Text Encoder"|"CLIP Vision"|"IP-Adapter"|"Other" => requested.to_string(),
            _ => return Err(AppError::Invalid("Unsupported model type".into())),
        }
    };
    let locked=if requested.eq_ignore_ascii_case("auto"){0}else{1};
    c.execute("UPDATE models SET model_type=?2,model_type_user_modified=?3,updated_at=?4 WHERE id=?1",params![id,next_type,locked,now()])?;
    let rec=model_by_id(&c,id)?;
    let _=handle.emit("models-changed",());
    Ok(rec)
}

#[tauri::command]
fn get_library_counts(app:State<AppStateInner>)->AppResult<LibraryCounts>{
    let c=open_db(&app.app_data)?;
    let all:i64=c.query_row("SELECT COUNT(*) FROM models",[],|r|r.get(0))?;
    let mut stmt=c.prepare("SELECT model_type,COUNT(*) FROM models GROUP BY model_type")?;
    let rows=stmt.query_map([],|r|Ok((r.get::<_,String>(0)?,r.get::<_,i64>(1)?)))?;
    let mut by_type=std::collections::BTreeMap::new();
    for row in rows { let (kind,count)=row?; by_type.insert(kind,count); }
    Ok(LibraryCounts{all,by_type})
}

#[tauri::command]
fn get_model_images(app:State<AppStateInner>, id:i64)->AppResult<Vec<ModelImage>>{
    let c=open_db(&app.app_data)?; let mut stmt=c.prepare("SELECT id,civitai_image_id,local_path,thumbnail_path,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,meta_json FROM images WHERE model_id=?1 ORDER BY id")?; let rows=stmt.query_map([id],|r|Ok(ModelImage{id:r.get(0)?,civitai_image_id:r.get(1)?,local_path:r.get(2)?,thumbnail_path:r.get(3)?,width:r.get(4)?,height:r.get(5)?,prompt:r.get(6)?,negative_prompt:r.get(7)?,steps:r.get(8)?,cfg:r.get(9)?,sampler:r.get(10)?,seed:r.get(11)?,meta_json:r.get(12)?}))?; Ok(rows.filter_map(Result::ok).collect())
}
#[tauri::command]
async fn preview_civitai_import(app:State<'_,AppStateInner>,url:String)->AppResult<CivitaiImportPreview>{
    let (model,version)=fetch_model_and_version(&app,&url).await?; let (dl,size,filename,sha256)=selected_file(&version).ok_or_else(||AppError::Api("No downloadable public file found for this version".into()))?;
    let root=app.models_root.read().unwrap().clone().ok_or_else(||AppError::Invalid("Choose your ComfyUI models folder first".into()))?; let typ=model.get("type").and_then(Value::as_str).unwrap_or("Other"); let target=root.join(civitai_type_to_folder(typ));
    let activation=json_strings(version.get("trainedWords"));
    let mid=model.get("id").and_then(Value::as_i64);
    let thumb=match mid { Some(model_id)=>ensure_model_thumbnail(&app,model_id,&model,&version,&url).await?, None=>None };
    Ok(CivitaiImportPreview{model:json!({"id":model.get("id"),"name":model.get("name"),"type":typ,"description":model.get("description"),"tags":model.get("tags"),"creator":model.get("creator").and_then(|v|v.get("username")),"thumbnail_path":thumb}),version:json!({"id":version.get("id"),"name":version.get("name"),"base_model":version.get("baseModel"),"download_url":dl,"filename":filename,"size_bytes":size,"sha256":sha256,"activation_prompts":activation}),target_directory:target.to_string_lossy().to_string(),thumbnail_path:thumb,images_count_hint:version.get("images").and_then(Value::as_array).map(|x|x.len() as i64)})
}

async fn download_file(
    app: &AppStateInner,
    url: &str,
    target_dir: &Path,
    preferred_name: &str,
    expected_sha256: Option<&str>,
) -> AppResult<(PathBuf, i64, String)> {
    fs::create_dir_all(target_dir)?;
    let client = civitai_client(app)?;
    let mut req = client.get(url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }

    let res = req.send().await?;
    if !res.status().is_success() {
        return Err(AppError::Api(format!("Download failed: {}", res.status())));
    }

    let header_name = res
        .headers()
        .get(header::CONTENT_DISPOSITION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.split("filename=").nth(1))
        .map(|s| s.trim().trim_matches('"').trim_matches('\'').to_string());

    let raw_name = if !preferred_name.is_empty() {
        preferred_name.to_string()
    } else {
        header_name
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| {
                url.rsplit('/')
                    .next()
                    .unwrap_or("model.safetensors")
                    .split('?')
                    .next()
                    .unwrap_or("model.safetensors")
                    .to_string()
            })
    };

    let safe_name = Path::new(&raw_name)
        .file_name()
        .and_then(|x| x.to_str())
        .filter(|x| !x.is_empty())
        .unwrap_or("model.safetensors")
        .to_string();

    let mut path = target_dir.join(&safe_name);
    if path.exists() {
        if let Some(expected) = expected_sha256 {
            if let Ok(existing_hash) = sha256_file(&path) {
                if existing_hash.eq_ignore_ascii_case(expected) {
                    let existing_size = fs::metadata(&path)?.len() as i64;
                    return Ok((path, existing_size, existing_hash));
                }
            }
        }

        let source_name = Path::new(&safe_name);
        let stem = source_name
            .file_stem()
            .and_then(|x| x.to_str())
            .unwrap_or("model");
        let extension = source_name
            .extension()
            .and_then(|x| x.to_str())
            .unwrap_or("");
        let mut index = 1u32;
        loop {
            let candidate_name = if extension.is_empty() {
                format!("{stem} ({index})")
            } else {
                format!("{stem} ({index}).{extension}")
            };
            let candidate = target_dir.join(candidate_name);
            if !candidate.exists() {
                path = candidate;
                break;
            }
            index += 1;
        }
    }

    let partial = path.with_extension(format!(
        "{}.part",
        path.extension()
            .and_then(|x| x.to_str())
            .unwrap_or("bin")
    ));
    let mut file = File::create(&partial)?;
    let mut stream = res.bytes_stream();
    let mut total = 0i64;
    let mut hasher = Sha256::new();

    while let Some(chunk) = stream.next().await {
        let bytes = chunk?;
        total += bytes.len() as i64;
        hasher.update(&bytes);
        file.write_all(&bytes)?;
    }

    file.flush()?;
    let actual_sha256 = hex::encode(hasher.finalize());

    if let Some(expected) = expected_sha256 {
        if actual_sha256 != expected.to_ascii_lowercase() {
            let _ = fs::remove_file(&partial);
            return Err(AppError::Api(format!(
                "Downloaded file failed SHA256 verification: expected {expected}, got {actual_sha256}"
            )));
        }
    }

    fs::rename(&partial, &path)?;
    Ok((path, total, actual_sha256))
}

#[tauri::command]
async fn install_civitai_model(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    url: String,
    target_directory: Option<String>,
    selected_type: Option<String>,
) -> AppResult<ModelRecord> {
    let (model, version) = fetch_model_and_version(&app, &url).await?;
    let version_id = version.get("id").and_then(Value::as_i64);

    if let Some(vid) = version_id {
        let c0 = open_db(&app.app_data)?;
        if let Ok(existing_id) = c0.query_row(
            "SELECT id FROM models WHERE civitai_version_id=?1 AND path IS NOT NULL",
            [vid],
            |r| r.get::<_, i64>(0),
        ) {
            return model_by_id(&c0, existing_id);
        }
    }

    let civitai_typ = model
        .get("type")
        .and_then(Value::as_str)
        .unwrap_or("Other");
    let typ = if let Some(selected) = selected_type {
        normalized_import_type(&selected)
            .ok_or_else(|| AppError::Invalid("Unsupported Raphael library tag".into()))?
            .to_string()
    } else {
        civitai_type_to_model_type(civitai_typ).to_string()
    };
    let (dl, _, filename, sha256) = selected_file(&version)
        .ok_or_else(|| AppError::Api("No downloadable public file found for this version".into()))?;

    let root = app
        .models_root
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| AppError::Invalid("Choose your ComfyUI models folder first".into()))?;

    let target = if let Some(custom) = target_directory {
        let candidate=PathBuf::from(custom);
        fs::create_dir_all(&candidate)?;
        let root_canonical=root.canonicalize()?;
        let target_canonical=candidate.canonicalize()?;
        if !target_canonical.starts_with(&root_canonical){return Err(AppError::Invalid("Download folder must be inside the configured ComfyUI models folder".into()));}
        target_canonical
    } else { root.join(civitai_type_to_folder(&typ)) };
    let (path, size, hash) =
        download_file(&app, &dl, &target, &filename, sha256.as_deref()).await?;

    let rel = path
        .strip_prefix(&root)
        .unwrap_or(&path)
        .to_string_lossy()
        .replace("\\", "/");
    let tags = json_strings(model.get("tags"));
    let activation = json_strings(version.get("trainedWords"));
    let desc = model
        .get("description")
        .and_then(Value::as_str)
        .map(strip_html);
    let creator = model
        .get("creator")
        .and_then(|v| v.get("username"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let vname = version.get("name").and_then(Value::as_str).map(str::to_string);
    let base = version
        .get("baseModel")
        .and_then(Value::as_str)
        .map(str::to_string);

    let mid=model.get("id").and_then(Value::as_i64);
    let vid=version.get("id").and_then(Value::as_i64);
    let civitai_url=canonical_civitai_url(&url,mid,vid)?;
    let thumb=match mid { Some(model_id)=>ensure_model_thumbnail(&app,model_id,&model,&version,&url).await?, None=>None };
    let rec = {
        let c = open_db(&app.app_data)?;
        c.execute(
            "INSERT INTO models(
                path,relative_path,filename,model_type,size_bytes,modified_at,
                civitai_model_id,civitai_version_id,civitai_url,civitai_name,
                version_name,base_model,creator,description,tags_json,
                activation_json,source_hash,thumbnail_path,updated_at
             )
             VALUES(
                ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19
             )
             ON CONFLICT(path) DO UPDATE SET
                size_bytes=excluded.size_bytes,
                modified_at=excluded.modified_at,
                civitai_model_id=excluded.civitai_model_id,
                civitai_version_id=excluded.civitai_version_id,
                civitai_url=excluded.civitai_url,
                civitai_name=excluded.civitai_name,
                version_name=excluded.version_name,
                base_model=excluded.base_model,
                creator=excluded.creator,
                description=excluded.description,
                activation_json=excluded.activation_json,
                source_hash=excluded.source_hash,
                model_type=CASE WHEN models.model_type_user_modified=0 THEN excluded.model_type ELSE models.model_type END,
                tags_json=CASE WHEN models.tags_user_modified=0 THEN excluded.tags_json ELSE models.tags_json END,
                thumbnail_path=excluded.thumbnail_path,
                updated_at=excluded.updated_at",
            params![
                path.to_string_lossy(),
                rel,
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy(),
                typ,
                size,
                mtime(&path),
                mid,
                vid,
                civitai_url,
                model.get("name").and_then(Value::as_str),
                vname,
                base,
                creator,
                desc,
                serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&activation).unwrap_or_else(|_| "[]".into()),
                hash,
                thumb,
                now()
            ],
        )?;

        let id = c.query_row(
            "SELECT id FROM models WHERE path=?1",
            [path.to_string_lossy().to_string()],
            |r| r.get(0),
        )?;
        model_by_id(&c, id)?
    };

    let _ = handle.emit("models-changed", ());
    let _ = sync_gallery_inner(app.inner().clone(), rec.id, handle.clone(), true).await;
    Ok(model_by_id(&open_db(&app.app_data)?, rec.id)?)
}
fn parse_meta(meta:&Value,key:&str)->Option<String>{meta.get(key).and_then(Value::as_str).map(str::to_string)}
async fn sync_gallery_inner(
    app: AppStateInner,
    model_id: i64,
    handle: AppHandle,
    force: bool,
) -> AppResult<()> {
    {
        let c = open_db(&app.app_data)?;
        let cached: i64 =
            c.query_row("SELECT COUNT(*) FROM images WHERE model_id=?1", [model_id], |r| {
                r.get(0)
            })?;
        let missing: i64 = c.query_row(
            "SELECT COUNT(*) FROM images WHERE model_id=?1 AND local_path IS NULL",
            [model_id],
            |r| r.get(0),
        )?;
        if !force && cached > 0 && missing == 0 {
            return Ok(());
        }
    }

    let model = {
        let c = open_db(&app.app_data)?;
        model_by_id(&c, model_id)?
    };

    let civitai_id = match model.civitai_model_id {
        Some(x) => x,
        None => return Ok(()),
    };

    let cache = cache_root(&app.app_data).join(civitai_id.to_string());
    fs::create_dir_all(&cache)?;
    let mut cursor: Option<String> = None;
    let client = civitai_client(&app)?;

    loop {
        let mut url = format!(
            "{API_BASE}/images?modelId={civitai_id}&limit=200&withMeta=true"
        );
        if let Some(c) = &cursor {
            url.push_str("&cursor=");
            url.push_str(&urlencoding::encode(c));
        }

        let mut req = client.get(&url);
        if let Some(t) = token() {
            req = req.bearer_auth(t);
        }

        let res = req.send().await?;
        if !res.status().is_success() {
            return Err(AppError::Api(format!(
                "Image API returned {}",
                res.status()
            )));
        }

        let body: CivitaiEnvelope = res.json().await?;

        for img in body.items {
            let iid = match img.get("id").and_then(Value::as_i64) {
                Some(x) => x,
                None => continue,
            };

            let remote = img.get("url").and_then(Value::as_str).unwrap_or("");
            let ext = remote
                .split('?')
                .next()
                .and_then(|x| Path::new(x).extension())
                .and_then(|x| x.to_str())
                .unwrap_or("jpg");

            let local = cache.join(format!("{iid}.{ext}"));
            let thumb = cache.join(format!("{iid}_thumb.webp"));

            let local_path = if local.exists() {
                Some(local.to_string_lossy().to_string())
            } else if !remote.is_empty() {
                match client.get(remote).send().await {
                    Ok(resp) => match resp.error_for_status() {
                        Ok(resp) => match resp.bytes().await {
                            Ok(bytes) if fs::write(&local, &bytes).is_ok() => {
                                Some(local.to_string_lossy().to_string())
                            }
                            _ => None,
                        },
                        Err(_) => None,
                    },
                    Err(_) => None,
                }
            } else {
                None
            };

            if let Some(lp) = &local_path {
                if !thumb.exists() {
                    if let Ok(im) = image::open(lp) {
                        let t = im.resize(420, 420, FilterType::Triangle);
                        let _ = t.save_with_format(&thumb, image::ImageFormat::WebP);
                    }
                }
            }

            let meta = img.get("meta").cloned().unwrap_or(Value::Null);
            let prompt = parse_meta(&meta, "prompt");
            let neg = parse_meta(&meta, "negativePrompt")
                .or_else(|| parse_meta(&meta, "Negative prompt"));
            let sampler = parse_meta(&meta, "sampler")
                .or_else(|| parse_meta(&meta, "Sampler"));
            let steps = meta.get("steps").and_then(Value::as_i64);
            let cfg = meta
                .get("cfgScale")
                .or_else(|| meta.get("cfg"))
                .and_then(Value::as_f64);
            let seed = meta.get("seed").and_then(Value::as_i64);
            let width = img.get("width").and_then(Value::as_i64);
            let height = img.get("height").and_then(Value::as_i64);

            let c = open_db(&app.app_data)?;
            c.execute(
                "INSERT INTO images(
                    model_id,civitai_image_id,local_path,thumbnail_path,width,height,
                    prompt,negative_prompt,steps,cfg,sampler,seed,meta_json
                 )
                 VALUES(
                    ?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13
                 )
                 ON CONFLICT(model_id,civitai_image_id) DO UPDATE SET
                    local_path=excluded.local_path,
                    thumbnail_path=excluded.thumbnail_path,
                    width=excluded.width,
                    height=excluded.height,
                    prompt=excluded.prompt,
                    negative_prompt=excluded.negative_prompt,
                    steps=excluded.steps,
                    cfg=excluded.cfg,
                    sampler=excluded.sampler,
                    seed=excluded.seed,
                    meta_json=excluded.meta_json",
                params![
                    model_id,
                    iid,
                    local_path,
                    if thumb.exists() {
                        Some(thumb.to_string_lossy().to_string())
                    } else {
                        None::<String>
                    },
                    width,
                    height,
                    prompt,
                    neg,
                    steps,
                    cfg,
                    sampler,
                    seed,
                    serde_json::to_string(&meta).unwrap_or_else(|_| "null".into()),
                ],
            )?;
        }

        cursor = body
            .metadata
            .and_then(|m| m.get("nextCursor").and_then(Value::as_str).map(str::to_string));
        if cursor.is_none() {
            break;
        }
    }

    if let Ok(c) = open_db(&app.app_data) {
        let bytes = dir_size(&cache_root(&app.app_data));
        let _ = put_setting(&c, "cache_bytes", &bytes.to_string());
    }

    let _ = handle.emit("models-changed", ());
    Ok(())
}
#[tauri::command]
async fn sync_model_gallery(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
) -> AppResult<()> {
    let state = app.inner().clone();
    tauri::async_runtime::spawn(async move {
        let _ = sync_gallery_inner(state, id, handle, true).await;
    });
    Ok(())
}

#[tauri::command]
async fn link_model_civitai(
    app:State<'_,AppStateInner>,
    handle:AppHandle,
    id:i64,
    url:String,
)->AppResult<ModelRecord>{
    let trimmed=url.trim();
    let (_mid,_vid)=model_id_and_version(trimmed)?;
    let (model,version)=fetch_model_and_version(&app,trimmed).await?;
    let tags=json_strings(model.get("tags"));
    let activation=json_strings(version.get("trainedWords"));
    let desc=model.get("description").and_then(Value::as_str).map(strip_html);
    let creator=model.get("creator").and_then(|v|v.get("username")).and_then(Value::as_str).map(str::to_string);
    let mid=model.get("id").and_then(Value::as_i64);
    let vid=version.get("id").and_then(Value::as_i64);
    let canonical=canonical_civitai_url(trimmed,mid,vid)?;
    let thumbnail_path=match mid { Some(model_id)=>ensure_model_thumbnail(&app,model_id,&model,&version,trimmed).await?, None=>None };
    let rec={
        let c=open_db(&app.app_data)?;
        c.execute(
            "UPDATE models
             SET civitai_model_id=?2,
                 civitai_version_id=?3,
                 civitai_url=?4,
                 civitai_name=?5,
                 version_name=?6,
                 base_model=?7,
                 creator=?8,
                 description=?9,
                 tags_json=CASE WHEN tags_user_modified=0 THEN ?10 ELSE tags_json END,
                 activation_json=?11,
                 thumbnail_path=?12,
                 updated_at=?13
             WHERE id=?1",
            params![
                id,
                mid,
                vid,
                canonical,
                model.get("name").and_then(Value::as_str),
                version.get("name").and_then(Value::as_str),
                version.get("baseModel").and_then(Value::as_str),
                creator,
                desc,
                serde_json::to_string(&tags).unwrap_or_else(|_|"[]".into()),
                serde_json::to_string(&activation).unwrap_or_else(|_|"[]".into()),
                thumbnail_path,
                now()
            ],
        )?;
        model_by_id(&c,id)?
    };
    sync_gallery_inner(app.inner().clone(),id,handle.clone(),true).await?;
    let _=handle.emit("models-changed",());
    Ok(rec)
}

#[tauri::command]
async fn refresh_model_civitai(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
) -> AppResult<ModelRecord> {
    let current = {
        let c = open_db(&app.app_data)?;
        model_by_id(&c, id)?
    };

    let url = current
        .civitai_url
        .clone()
        .ok_or_else(|| AppError::Invalid("This model is not linked to Civitai".into()))?;

    let (model, version) = fetch_model_and_version(&app, &url).await?;
    let tags = json_strings(model.get("tags"));
    let activation = json_strings(version.get("trainedWords"));
    let desc = model
        .get("description")
        .and_then(Value::as_str)
        .map(strip_html);
    let creator = model
        .get("creator")
        .and_then(|v| v.get("username"))
        .and_then(Value::as_str)
        .map(str::to_string);
    let thumbnail_path=match model.get("id").and_then(Value::as_i64) { Some(mid)=>ensure_model_thumbnail(&app,mid,&model,&version,&url).await?, None=>None };

    let rec = {
        let c = open_db(&app.app_data)?;
        c.execute(
            "UPDATE models
             SET civitai_model_id=?2,
                 civitai_version_id=?3,
                 civitai_name=?4,
                 version_name=?5,
                 base_model=?6,
                 creator=?7,
                 description=?8,
                 tags_json=CASE WHEN tags_user_modified=0 THEN ?9 ELSE tags_json END,
                 activation_json=?10,
                 thumbnail_path=?11,
                 updated_at=?12
             WHERE id=?1",
            params![
                id,
                model.get("id").and_then(Value::as_i64),
                version.get("id").and_then(Value::as_i64),
                model.get("name").and_then(Value::as_str),
                version.get("name").and_then(Value::as_str),
                version.get("baseModel").and_then(Value::as_str),
                creator,
                desc,
                serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into()),
                serde_json::to_string(&activation).unwrap_or_else(|_| "[]".into()),
                thumbnail_path,
                now()
            ],
        )?;
        model_by_id(&c, id)?
    };

    let _ = sync_gallery_inner(app.inner().clone(), id, handle.clone(), true).await;
    let _ = handle.emit("models-changed", ());
    Ok(rec)
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

            let mut rows = match statement.query([]) {
                Ok(value) => value,
                Err(_) => return,
            };

            let mut collected = Vec::new();
            loop {
                match rows.next() {
                    Ok(Some(row)) => {
                        let id: i64 = match row.get(0) {
                            Ok(value) => value,
                            Err(_) => return,
                        };
                        let path: String = match row.get(1) {
                            Ok(value) => value,
                            Err(_) => return,
                        };
                        collected.push((id, path));
                    }
                    Ok(None) => break,
                    Err(_) => return,
                }
            }
            collected
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
                         tags_json=CASE WHEN tags_user_modified=0 THEN ?8 ELSE tags_json END,
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

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(web::WebServerController::default())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data=app.path().app_data_dir()?;fs::create_dir_all(&app_data)?;let c=open_db(&app_data)?;let saved=setting(&c,"models_root")?;let state=AppStateInner{app_data:app_data.clone(),models_root:Arc::new(RwLock::new(saved.map(PathBuf::from))),watcher:Arc::new(Mutex::new(None)),scan_lock:Arc::new(Mutex::new(()))};app.manage(state.clone());
            if let Some(root)=state.models_root.read().unwrap().clone(){ if root.is_dir(){let _=scan_root(&state,&root);let handle=app.handle().clone();spawn_hash_enrichment(state.clone(),handle.clone());let state2=state.clone();let handle2=handle.clone();if let Ok(mut watcher)=notify::recommended_watcher(move |res:Result<notify::Event,notify::Error>|{if let Ok(e)=res{match e.kind{EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)=>{std::thread::sleep(Duration::from_millis(120));recursive_scan_and_emit(state2.clone(),handle2.clone());},_=>{}}}}){if watcher.watch(&root,RecursiveMode::Recursive).is_ok(){*state.watcher.lock().unwrap()=Some(watcher)}}}}
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_app_state,set_models_root,list_models,get_tags,add_subfolder_tags,set_model_tags,set_model_type,delete_model,get_library_counts,get_model_images,sync_model_gallery,preview_civitai_import,install_civitai_model,link_model_civitai,refresh_model_civitai,get_storage_stats,open_in_file_manager,set_civitai_token,is_civitai_token_set,web::get_web_app_status,web::toggle_web_app])
        .run(tauri::generate_context!())
        .expect("error while running Raphael Model Manager");
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
    fn subfolder_tags_are_derived_and_appended() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let root = temp.path().join("models");
        fs::create_dir_all(root.join("checkpoints/Illustrus")).unwrap();
        fs::create_dir_all(root.join("loras/Illustrus/Character")).unwrap();
        fs::write(root.join("checkpoints/Illustrus/model.safetensors"), b"checkpoint").unwrap();
        fs::write(root.join("loras/Illustrus/Character/model.safetensors"), b"lora").unwrap();

        let state = test_state(app_data.clone(), root.clone());
        scan_root(&state, &root).unwrap();

        let updated = add_subfolder_tags_inner(&state).unwrap();
        assert_eq!(updated, 2);

        let db = open_db(&app_data).unwrap();
        let checkpoints_tags: String = db
            .query_row(
                "SELECT tags_json FROM models WHERE relative_path='checkpoints/Illustrus/model.safetensors'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        let lora_tags: String = db
            .query_row(
                "SELECT tags_json FROM models WHERE relative_path='loras/Illustrus/Character/model.safetensors'",
                [],
                |r| r.get(0),
            )
            .unwrap();

        let checkpoints: Vec<String> = serde_json::from_str(&checkpoints_tags).unwrap();
        let lora: Vec<String> = serde_json::from_str(&lora_tags).unwrap();
        assert_eq!(checkpoints, vec!["Illustrus"]);
        assert_eq!(lora, vec!["Character", "Illustrus"]);
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
        assert_eq!(
            model_id_and_version("https://civitai.red/models/12345?modelVersionId=67890").unwrap(),
            (12345, Some(67890))
        );
        assert_eq!(
            model_id_and_version("https://civitai.red/models/12345?modelVersionId=67890").unwrap(),
            (12345, Some(67890))
        );
        assert!(model_id_and_version("https://example.com/models/12345").is_err());
        assert!(model_id_and_version("https://civitai.com/images/12345").is_err());
    }

    #[test]
    fn civitai_types_map_to_comfyui_folders() {
        assert_eq!(civitai_type_to_folder("Checkpoint"), "checkpoints");
        assert_eq!(civitai_type_to_folder("LORA"), "loras");
        assert_eq!(civitai_type_to_folder("LoCon"), "loras");
        assert_eq!(civitai_type_to_model_type("Upscaler"), "Upscaler");
        assert_eq!(civitai_type_to_model_type("TextualInversion"), "Embedding");
        assert_eq!(civitai_type_to_folder("TextualInversion"), "embeddings");
        assert_eq!(civitai_type_to_folder("Upscaler"), "upscale_models");
        assert_eq!(civitai_type_to_folder("UnknownType"), "other");
    }

    #[test]
    fn tag_normalization_is_case_insensitive_and_deduplicated() {
        assert_eq!(
            normalize_tags(vec!["Anime".into(), " anime ".into(), "".into(), "character".into()]),
            vec!["Anime".to_string(), "character".to_string()]
        );
    }

    #[test]
    fn tag_search_supports_positive_negative_and_hash_syntax() {
        let model = ModelRecord {
            id: 1, path: "C:/models/a.safetensors".into(), relative_path: "loras/a.safetensors".into(),
            filename: "a.safetensors".into(), model_type: "LoRA".into(), size_bytes: 1, modified_at: 0,
            civitai_model_id: None, civitai_version_id: None, civitai_url: None, civitai_name: Some("Hero".into()),
            version_name: None, base_model: None, creator: None, description: None,
            tags: vec!["Anime".into(), "Megumin".into()], activation_prompts: vec!["magic".into()],
            source_hash: None, thumbnail_path: None, updated_at: 0,
        };
        assert!(model_search_match(&model, "tag:anime", &[]));
        assert!(model_search_match(&model, "#megumin", &[]));
        assert!(model_search_match(&model, "-tag:realistic", &[]));
        assert!(!model_search_match(&model, "-tag:anime", &[]));
        assert!(model_search_match(&model, "magic", &[]));
    }

    #[test]
    fn delete_model_requires_path_inside_models_root() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let root = temp.path().join("models");
        fs::create_dir_all(&root).unwrap();
        let db = open_db(&app_data).unwrap();
        db.execute(
            "INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,updated_at)
             VALUES(?1,'evil/model.safetensors','model.safetensors','Other',1,0,0)",
            [temp.path().join("outside.safetensors").to_string_lossy().to_string()],
        ).unwrap();
        let stored: String = db.query_row("SELECT path FROM models LIMIT 1", [], |r| r.get(0)).unwrap();
        assert!(!Path::new(&stored).starts_with(&root));
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

        let thumbnail_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='thumbnail_path'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(thumbnail_column, 1);

        let tag_lock_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='tags_user_modified'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(tag_lock_column, 1);

        let type_lock_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='model_type_user_modified'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(type_lock_column, 1);
    }
}
