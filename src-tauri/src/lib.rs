use chrono::Utc;
use futures_util::{stream, StreamExt};
use image::{imageops::FilterType, ImageFormat, ImageReader};
use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};
use reqwest::Client;
use rusqlite::{params, Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{
    collections::{HashMap, HashSet},
    fs::{self, File},
    io::{self, BufReader, Read, Write},
    path::{Path, PathBuf},
    sync::{atomic::{AtomicU64, Ordering}, Arc, Mutex, OnceLock, RwLock},
    time::{Duration, SystemTime, UNIX_EPOCH},
};
use tokio::sync::{Mutex as AsyncMutex, Notify};
use tauri::{AppHandle, Emitter, Manager, State};
use thiserror::Error;
use url::Url;
use registry::RegistryClient;
use walkdir::WalkDir;

mod registry;
mod web;

const API_BASE: &str = "https://civitai.com/api/v1";
const USER_AGENT: &str = "RaphaelModelManager/0.1.0";
static DOWNLOAD_COUNTER: AtomicU64 = AtomicU64::new(1);
static MODEL_CHANGE_REVISION: AtomicU64 = AtomicU64::new(1);
static MODEL_CHANGE_NOTIFY: OnceLock<Notify> = OnceLock::new();

fn model_change_notify() -> &'static Notify {
    MODEL_CHANGE_NOTIFY.get_or_init(Notify::new)
}

pub(crate) fn model_change_revision() -> u64 {
    MODEL_CHANGE_REVISION.load(Ordering::Acquire)
}

fn emit_models_changed(handle: &AppHandle) {
    MODEL_CHANGE_REVISION.fetch_add(1, Ordering::AcqRel);
    model_change_notify().notify_waiters();
    let _ = handle.emit("models-changed", ());
}

#[derive(Debug, Error)]
enum AppError {
    #[error("database error: {0}")]
    Db(#[from] rusqlite::Error),
    #[error("model registry error: {0}")]
    Registry(String),
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

impl From<registry::RegistryError> for AppError {
    fn from(error: registry::RegistryError) -> Self {
        Self::Registry(error.to_string())
    }
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
    downloads: Arc<Mutex<Vec<DownloadProgress>>>,
    active_download_paths: Arc<Mutex<HashSet<PathBuf>>>,
    active_download_versions: Arc<Mutex<HashSet<i64>>>,
    active_downloads: Arc<Mutex<usize>>,
    parallel_downloads: Arc<Mutex<usize>>,
    examples_refresh_state: Arc<Mutex<ExamplesRefreshState>>,
    cache_lock: Arc<AsyncMutex<()>>,
    registry: RegistryClient,
    registry_sync_running: Arc<Mutex<bool>>,
    registry_event_cursor: Arc<Mutex<i64>>,
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
    cover_path: Option<String>,
    cover_source_image_id: Option<i64>,
    cover_position_x: f64,
    cover_position_y: f64,
    downloaded_at: i64,
    updated_at: i64,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
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
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct CacheStats {
    pub(crate) location: String,
    pub(crate) used_bytes: i64,
    pub(crate) max_bytes: i64,
    pub(crate) over_limit: bool,
    pub(crate) files: i64,
    pub(crate) image_files: i64,
    pub(crate) image_bytes: i64,
    pub(crate) featured_files: i64,
    pub(crate) featured_bytes: i64,
    pub(crate) gallery_files: i64,
    pub(crate) gallery_bytes: i64,
    pub(crate) thumbnail_files: i64,
    pub(crate) thumbnail_bytes: i64,
    pub(crate) cover_files: i64,
    pub(crate) cover_bytes: i64,
    pub(crate) other_files: i64,
    pub(crate) other_bytes: i64,
}
#[derive(Debug, Serialize, Deserialize, Clone)]
pub(crate) struct CacheOperationResult {
    pub(crate) deleted_files: i64,
    pub(crate) freed_bytes: i64,
    pub(crate) remaining_bytes: i64,
    pub(crate) over_limit: bool,
}
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

#[derive(Debug, Serialize, Deserialize, Clone)]
struct DownloadProgress {
    visible: bool,
    task_id: Option<String>,
    filename: String,
    phase: String,
    downloaded_bytes: i64,
    total_bytes: Option<i64>,
    percent: Option<f64>,
    error: Option<String>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ExamplesRefreshProgress {
    current: usize,
    total: usize,
    model_id: Option<i64>,
    model_name: Option<String>,
    version_current: usize,
    version_total: usize,
    images_saved: usize,
    status: String,
    done: bool,
    error: Option<String>,
}

#[derive(Clone, Default)]
struct ExamplesRefreshState {
    progress: Option<ExamplesRefreshProgress>,
    running: bool,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
struct ModelImagesResponse {
    images: Vec<ModelImage>,
    has_more: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct CivitaiEnvelope { metadata: Option<Value>, items: Vec<Value> }

fn now() -> i64 { Utc::now().timestamp() }

fn new_download_id() -> String {
    let millis = SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0);
    let sequence = DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    format!("{millis}-{sequence}")
}

fn set_download_progress(
    state: &Arc<Mutex<Vec<DownloadProgress>>>,
    task_id: &str,
    update: impl FnOnce(&mut DownloadProgress),
) {
    if let Ok(mut guard) = state.lock() {
        if let Some(progress) = guard.iter_mut().find(|item| item.task_id.as_deref() == Some(task_id)) {
            update(progress);
        }
    }
}

fn push_download_progress(
    state: &Arc<Mutex<Vec<DownloadProgress>>>,
    progress: DownloadProgress,
) -> AppResult<()> {
    let mut guard = state
        .lock()
        .map_err(|_| AppError::Invalid("Download state is unavailable".into()))?;
    guard.push(progress);
    const MAX_DOWNLOAD_HISTORY: usize = 100;
    if guard.len() > MAX_DOWNLOAD_HISTORY {
        let excess = guard.len() - MAX_DOWNLOAD_HISTORY;
        guard.drain(0..excess);
    }
    Ok(())
}

fn remove_download_progress(
    state: &Arc<Mutex<Vec<DownloadProgress>>>,
    task_id: &str,
) -> AppResult<()> {
    let mut guard = state
        .lock()
        .map_err(|_| AppError::Invalid("Download state is unavailable".into()))?;
    guard.retain(|item| item.task_id.as_deref() != Some(task_id));
    Ok(())
}

fn reserve_download_path(
    active_paths: &Arc<Mutex<HashSet<PathBuf>>>,
    target_dir: &Path,
    filename: &str,
) -> AppResult<PathBuf> {
    let mut guard = active_paths
        .lock()
        .map_err(|_| AppError::Invalid("Download path state is unavailable".into()))?;

    let raw_name = Path::new(filename);
    let stem = raw_name.file_stem().and_then(|x| x.to_str()).unwrap_or("model");
    let extension = raw_name.extension().and_then(|x| x.to_str()).unwrap_or("");

    let mut index = 0u32;
    loop {
        let candidate_name = if index == 0 {
            if extension.is_empty() { stem.to_string() } else { format!("{stem}.{extension}") }
        } else if extension.is_empty() {
            format!("{stem} ({index})")
        } else {
            format!("{stem} ({index}).{extension}")
        };
        let candidate = target_dir.join(candidate_name);
        if !candidate.exists() && !guard.contains(&candidate) {
            guard.insert(candidate.clone());
            return Ok(candidate);
        }
        index += 1;
    }
}

fn release_download_path(
    active_paths: &Arc<Mutex<HashSet<PathBuf>>>,
    path: &Path,
) {
    if let Ok(mut guard) = active_paths.lock() {
        guard.remove(path);
    }
}

struct DownloadSlot {
    active: Arc<Mutex<usize>>,
}

impl Drop for DownloadSlot {
    fn drop(&mut self) {
        if let Ok(mut guard) = self.active.lock() {
            *guard = guard.saturating_sub(1);
        }
    }
}

async fn acquire_download_slot(state: &AppStateInner) -> AppResult<DownloadSlot> {
    loop {
        let limit = *state.parallel_downloads.lock()
            .map_err(|_| AppError::Invalid("Parallel download setting is unavailable".into()))?;
        {
            let mut active = state.active_downloads.lock()
                .map_err(|_| AppError::Invalid("Download worker state is unavailable".into()))?;
            if *active < limit.clamp(1, 8) {
                *active += 1;
                return Ok(DownloadSlot { active: state.active_downloads.clone() });
            }
        }
        tokio::time::sleep(Duration::from_millis(250)).await;
    }
}

fn db_path(app_data: &Path) -> PathBuf { app_data.join("raphael.db") }
fn cache_base_default(app_data: &Path) -> PathBuf { app_data.join("cache") }
pub(crate) fn cache_root(app_data: &Path) -> PathBuf {
    let fallback = cache_base_default(app_data);
    let connection = match Connection::open(db_path(app_data)) {
        Ok(value) => value,
        Err(_) => return fallback,
    };
    let _ = connection.busy_timeout(Duration::from_secs(5));
    setting(&connection, "cache_location")
        .ok()
        .flatten()
        .filter(|value| !value.trim().is_empty())
        .map(PathBuf::from)
        .filter(|path| path.is_absolute())
        .unwrap_or(fallback)
}
static DB_SCHEMA_READY: OnceLock<Mutex<HashSet<PathBuf>>> = OnceLock::new();

fn initialize_db_schema(c: &Connection) -> AppResult<()> {
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
        updated_at INTEGER NOT NULL,
        cover_path TEXT,
        cover_position_x REAL NOT NULL DEFAULT 50,
        cover_position_y REAL NOT NULL DEFAULT 50,
        cover_source_image_id INTEGER,
        downloaded_at INTEGER NOT NULL DEFAULT 0,
        registry_model_id TEXT,
        registry_version_id TEXT,
        registry_file_id TEXT
      );
      CREATE INDEX IF NOT EXISTS idx_models_type ON models(model_type);
      CREATE INDEX IF NOT EXISTS idx_models_civitai ON models(civitai_model_id, civitai_version_id);
      CREATE TABLE IF NOT EXISTS pending_registry_file_removals (
        registry_model_id TEXT NOT NULL,
        registry_file_id TEXT NOT NULL,
        PRIMARY KEY(registry_model_id, registry_file_id)
      );
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
        cached_at INTEGER NOT NULL DEFAULT 0,
        UNIQUE(model_id, civitai_image_id)
      );
    "#)?;

    let has_thumbnail:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='thumbnail_path'",[],|r|r.get(0))?;
    if has_thumbnail==0 { c.execute("ALTER TABLE models ADD COLUMN thumbnail_path TEXT",[])?; }
    let has_tag_lock:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='tags_user_modified'",[],|r|r.get(0))?;
    if has_tag_lock==0 { c.execute("ALTER TABLE models ADD COLUMN tags_user_modified INTEGER NOT NULL DEFAULT 0",[])?; }
    let has_type_lock:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='model_type_user_modified'",[],|r|r.get(0))?;
    if has_type_lock==0 { c.execute("ALTER TABLE models ADD COLUMN model_type_user_modified INTEGER NOT NULL DEFAULT 0",[])?; }
    let has_cover_path:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_path'",[],|r|r.get(0))?;
    if has_cover_path==0 { c.execute("ALTER TABLE models ADD COLUMN cover_path TEXT",[])?; }
    let has_cover_x:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_position_x'",[],|r|r.get(0))?;
    if has_cover_x==0 { c.execute("ALTER TABLE models ADD COLUMN cover_position_x REAL NOT NULL DEFAULT 50",[])?; }
    let has_cover_y:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_position_y'",[],|r|r.get(0))?;
    if has_cover_y==0 { c.execute("ALTER TABLE models ADD COLUMN cover_position_y REAL NOT NULL DEFAULT 50",[])?; }
    let has_cover_source_image_id:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_source_image_id'",[],|r|r.get(0))?;
    if has_cover_source_image_id==0 { c.execute("ALTER TABLE models ADD COLUMN cover_source_image_id INTEGER",[])?; }
    let has_downloaded_at:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='downloaded_at'",[],|r|r.get(0))?;
    if has_downloaded_at==0 {
        c.execute("ALTER TABLE models ADD COLUMN downloaded_at INTEGER NOT NULL DEFAULT 0",[])?;
        c.execute("UPDATE models SET downloaded_at=?1 WHERE downloaded_at=0",[now()])?;
    }
    let has_registry_model_id:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='registry_model_id'",[],|r|r.get(0))?;
    if has_registry_model_id==0 { c.execute("ALTER TABLE models ADD COLUMN registry_model_id TEXT",[])?; }
    let has_registry_version_id:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='registry_version_id'",[],|r|r.get(0))?;
    if has_registry_version_id==0 { c.execute("ALTER TABLE models ADD COLUMN registry_version_id TEXT",[])?; }
    let has_registry_file_id:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='registry_file_id'",[],|r|r.get(0))?;
    if has_registry_file_id==0 { c.execute("ALTER TABLE models ADD COLUMN registry_file_id TEXT",[])?; }
    c.execute("CREATE INDEX IF NOT EXISTS idx_models_registry_model ON models(registry_model_id)",[])?;

    let has_cached_at:i64=c.query_row("SELECT COUNT(*) FROM pragma_table_info('images') WHERE name='cached_at'",[],|r|r.get(0))?;
    if has_cached_at==0 {
        c.execute("ALTER TABLE images ADD COLUMN cached_at INTEGER NOT NULL DEFAULT 0",[])?;
        c.execute("UPDATE images SET cached_at=?1 WHERE cached_at=0",[now()])?;
    }
    c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('parallel_downloads','3')",[])?;
    c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('cache_max_bytes','0')",[])?;
    c.execute("INSERT OR IGNORE INTO settings(key,value) VALUES('example_load_amount','20')",[])?;
    Ok(())
}

fn open_db(app_data: &Path) -> AppResult<Connection> {
    fs::create_dir_all(app_data)?;
    let db_file = db_path(app_data);
    let c = Connection::open(&db_file)?;
    c.busy_timeout(Duration::from_secs(5))?;

    let ready = DB_SCHEMA_READY.get_or_init(|| Mutex::new(HashSet::new()));
    let mut guard = ready
        .lock()
        .map_err(|_| AppError::Invalid("Database initialization state is unavailable".into()))?;
    if !guard.contains(&db_file) {
        initialize_db_schema(&c)?;
        guard.insert(db_file);
    }

    Ok(c)
}
fn setting(c: &Connection, key: &str) -> AppResult<Option<String>> {
    Ok(c.query_row("SELECT value FROM settings WHERE key=?1", [key], |r| r.get(0)).optional()?)
}
fn put_setting(c: &Connection, key: &str, value: &str) -> AppResult<()> {
    c.execute("INSERT INTO settings(key,value) VALUES(?1,?2) ON CONFLICT(key) DO UPDATE SET value=excluded.value", params![key, value])?;
    Ok(())
}

fn clamp_parallel_downloads(value: i64) -> i64 { value.clamp(1, 8) }
fn read_parallel_downloads(c: &Connection) -> AppResult<usize> {
    Ok(setting(c, "parallel_downloads")?.and_then(|value| value.parse::<i64>().ok()).map(clamp_parallel_downloads).unwrap_or(3) as usize)
}


fn read_example_load_amount(c: &Connection) -> AppResult<i64> {
    Ok(setting(c, "example_load_amount")?
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(20)
        .clamp(1, 100))
}

fn read_cache_max_bytes(c: &Connection) -> AppResult<i64> {
    Ok(setting(c, "cache_max_bytes")?
        .and_then(|value| value.parse::<i64>().ok())
        .unwrap_or(0)
        .max(0))
}

fn cache_file_counts(path: &Path) -> (i64, i64) {
    if !path.exists() { return (0, 0); }
    let mut files = 0i64;
    let mut bytes = 0i64;
    for entry in WalkDir::new(path).follow_links(false).into_iter().filter_map(Result::ok) {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                files += 1;
                bytes += meta.len() as i64;
            }
        }
    }
    (files, bytes)
}
fn cache_stats_inner(app_data: &Path) -> AppResult<CacheStats> {
    let c = open_db(app_data)?;
    let root = cache_root(app_data);
    let max_bytes = read_cache_max_bytes(&c)?;
    let mut stats = CacheStats {
        location: root.to_string_lossy().to_string(),
        used_bytes: 0,
        max_bytes,
        over_limit: false,
        files: 0,
        image_files: 0,
        image_bytes: 0,
        featured_files: 0,
        featured_bytes: 0,
        gallery_files: 0,
        gallery_bytes: 0,
        thumbnail_files: 0,
        thumbnail_bytes: 0,
        cover_files: 0,
        cover_bytes: 0,
        other_files: 0,
        other_bytes: 0,
    };
    if !root.exists() { return Ok(stats); }

    for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok) {
        let meta = match entry.metadata() {
            Ok(value) if value.is_file() => value,
            _ => continue,
        };
        let bytes = meta.len() as i64;
        stats.files += 1;
        stats.used_bytes += bytes;
        let relative = entry.path().strip_prefix(&root).unwrap_or(entry.path());
        let components: Vec<String> = relative.components()
            .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_ascii_lowercase()))
            .collect();

        if components.first().map(String::as_str) == Some("covers") {
            stats.cover_files += 1;
            stats.cover_bytes += bytes;
        } else if components.first().map(String::as_str) == Some("civitai") {
            stats.image_files += 1;
            stats.image_bytes += bytes;
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if components.iter().any(|c| c == "featured") ||
               components.iter().any(|c| c == "featured.__staging" || c == "featured.__backup") {
                stats.featured_files += 1;
                stats.featured_bytes += bytes;
            } else if name.starts_with("thumbnail.") || name.contains("_thumb.") {
                stats.thumbnail_files += 1;
                stats.thumbnail_bytes += bytes;
            } else {
                stats.gallery_files += 1;
                stats.gallery_bytes += bytes;
            }
        } else {
            stats.other_files += 1;
            stats.other_bytes += bytes;
        }
    }
    stats.over_limit = stats.max_bytes > 0 && stats.used_bytes > stats.max_bytes;
    Ok(stats)
}

fn remove_path_with_stats(path: &Path) -> AppResult<(i64, i64)> {
    if !path.exists() { return Ok((0, 0)); }
    let (files, bytes) = cache_file_counts(path);
    if path.is_dir() { fs::remove_dir_all(path)?; } else { fs::remove_file(path)?; }
    Ok((files, bytes))
}

fn copy_dir_recursive(source: &Path, target: &Path) -> AppResult<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let src = entry.path();
        let dst = target.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if kind.is_file() {
            fs::copy(&src, &dst)?;
        } else if kind.is_symlink() {
            return Err(AppError::Invalid(format!(
                "Cache relocation encountered an unsupported symbolic link: {}",
                src.to_string_lossy()
            )));
        }
    }
    Ok(())
}

fn rewrite_cache_paths(
    c: &mut Connection,
    old_root: &Path,
    new_root: &Path,
    cache_bytes: i64,
) -> AppResult<()> {
    let old = old_root.to_string_lossy().to_string();
    let new = new_root.to_string_lossy().to_string();
    let separator = std::path::MAIN_SEPARATOR.to_string();
    let alternate_separator = if separator == "/" { "\\".to_string() } else { "/".to_string() };
    let tx = c.transaction()?;

    for column in ["local_path", "thumbnail_path"] {
        let sql = format!(
            "UPDATE images SET {column}=?2 || substr({column}, length(?1)+1)
             WHERE {column}=?1
                OR substr({column},1,length(?1)+1)=?1 || ?3
                OR substr({column},1,length(?1)+1)=?1 || ?4"
        );
        tx.execute(&sql, params![old, new, separator, alternate_separator])?;
    }
    for column in ["thumbnail_path", "cover_path"] {
        let sql = format!(
            "UPDATE models SET {column}=?2 || substr({column}, length(?1)+1)
             WHERE {column}=?1
                OR substr({column},1,length(?1)+1)=?1 || ?3
                OR substr({column},1,length(?1)+1)=?1 || ?4"
        );
        tx.execute(&sql, params![old, new, separator, alternate_separator])?;
    }
    tx.execute(
        "INSERT INTO settings(key,value) VALUES('cache_location',?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [&new],
    )?;
    tx.execute(
        "INSERT INTO settings(key,value) VALUES('cache_bytes',?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [cache_bytes.to_string()],
    )?;
    tx.commit()?;
    Ok(())
}

fn delete_image_records(c: &mut Connection, cache_root: &Path, ids: &[i64]) -> AppResult<(i64, i64)> {
    if ids.is_empty() { return Ok((0, 0)); }
    let mut paths = HashSet::<PathBuf>::new();
    for id in ids {
        if let Ok((local, thumb)) = c.query_row(
            "SELECT local_path,thumbnail_path FROM images WHERE id=?1",
            [id],
            |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?)),
        ) {
            if let Some(value) = local { paths.insert(PathBuf::from(value)); }
            if let Some(value) = thumb { paths.insert(PathBuf::from(value)); }
        }
    }
    {
        let tx = c.transaction()?;
        for id in ids { tx.execute("DELETE FROM images WHERE id=?1", [id])?; }
        tx.commit()?;
    }

    let canonical_root = cache_root.canonicalize().unwrap_or_else(|_| cache_root.to_path_buf());
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    for path in paths {
        if !path.is_file() { continue; }
        let canonical_path = match path.canonicalize() { Ok(value) => value, Err(_) => continue };
        if !canonical_path.starts_with(&canonical_root) { continue; }
        let value = path.to_string_lossy().to_string();
        let image_refs: i64 = c.query_row(
            "SELECT COUNT(*) FROM images WHERE local_path=?1 OR thumbnail_path=?1",
            [&value], |r| r.get(0),
        )?;
        let model_refs: i64 = c.query_row(
            "SELECT COUNT(*) FROM models WHERE thumbnail_path=?1 OR cover_path=?1",
            [&value], |r| r.get(0),
        )?;
        if image_refs == 0 && model_refs == 0 {
            let bytes = fs::metadata(&path).map(|m| m.len() as i64).unwrap_or(0);
            fs::remove_file(&path)?;
            deleted_files += 1;
            freed_bytes += bytes;
        }
    }
    Ok((deleted_files, freed_bytes))
}

fn referenced_cache_paths(c: &Connection) -> AppResult<HashSet<PathBuf>> {
    let mut paths = HashSet::new();
    for sql in [
        "SELECT local_path FROM images WHERE local_path IS NOT NULL",
        "SELECT thumbnail_path FROM images WHERE thumbnail_path IS NOT NULL",
        "SELECT thumbnail_path FROM models WHERE thumbnail_path IS NOT NULL",
        "SELECT cover_path FROM models WHERE cover_path IS NOT NULL",
    ] {
        let mut stmt = c.prepare(sql)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        for value in rows.flatten() {
            let path = PathBuf::from(value);
            if let Ok(canonical) = path.canonicalize() {
                paths.insert(canonical);
            }
            paths.insert(path);
        }
    }
    Ok(paths)
}

fn clean_cache_orphans_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let c = open_db(app_data)?;
    let references = referenced_cache_paths(&c)?;
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    if root.exists() {
        for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            let referenced = references.contains(path)
                || path.canonicalize().map(|canonical| references.contains(&canonical)).unwrap_or(false);
            if !path.is_file() || referenced { continue; }
            let bytes = fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
            if fs::remove_file(path).is_ok() {
                deleted_files += 1;
                freed_bytes += bytes;
            }
        }
    }
    for stale in [root.join("civitai").join("featured.__staging"),root.join("civitai").join("featured.__backup")] {
        if let Ok((files,bytes))=remove_path_with_stats(&stale) {
            deleted_files+=files;
            freed_bytes+=bytes;
        }
    }
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

fn enforce_cache_limit_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let initial = cache_stats_inner(app_data)?;
    if initial.max_bytes == 0 || initial.used_bytes <= initial.max_bytes {
        return Ok(CacheOperationResult { deleted_files: 0, freed_bytes: 0, remaining_bytes: initial.used_bytes, over_limit: false });
    }

    let mut c = open_db(app_data)?;
    let root = cache_root(app_data);
    let mut stmt = c.prepare(
        "SELECT i.id FROM images i
         WHERE NOT EXISTS (SELECT 1 FROM models m WHERE m.cover_source_image_id=i.id)
         ORDER BY CASE WHEN i.meta_json LIKE '%\"featured\":true%' THEN 1 ELSE 0 END, i.cached_at ASC, i.id ASC"
    )?;
    let ids: Vec<i64> = stmt.query_map([], |r| r.get(0))?.filter_map(Result::ok).collect();
    drop(stmt);

    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    let mut current_bytes = initial.used_bytes;
    for id in ids {
        if current_bytes <= initial.max_bytes { break; }
        let (files, bytes) = delete_image_records(&mut c, &root.join("civitai"), &[id])?;
        deleted_files += files;
        freed_bytes += bytes;
        current_bytes = current_bytes.saturating_sub(bytes);
    }

    if current_bytes > initial.max_bytes {
        let result = clean_cache_orphans_inner(app_data)?;
        deleted_files += result.deleted_files;
        freed_bytes += result.freed_bytes;
    }

    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

fn set_cache_max_bytes_inner(app_data: &Path, value: i64) -> AppResult<CacheStats> {
    let value = value.max(0);
    let c = open_db(app_data)?;
    put_setting(&c, "cache_max_bytes", &value.to_string())?;
    drop(c);
    let _ = enforce_cache_limit_inner(app_data)?;
    cache_stats_inner(app_data)
}

fn clear_cache_images_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let image_root = root.join("civitai");
    let (deleted_files, freed_bytes) = remove_path_with_stats(&image_root)?;
    fs::create_dir_all(&image_root)?;
    let c = open_db(app_data)?;
    c.execute_batch("DELETE FROM images;")?;
    c.execute("UPDATE models SET thumbnail_path=NULL,cover_source_image_id=NULL", [])?;
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

fn clear_complete_cache_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let (deleted_files, freed_bytes) = remove_path_with_stats(&root)?;
    fs::create_dir_all(root.join("civitai"))?;
    fs::create_dir_all(root.join("covers"))?;
    let c = open_db(app_data)?;
    c.execute_batch("DELETE FROM images;")?;
    c.execute(
        "UPDATE models SET thumbnail_path=NULL,cover_path=NULL,cover_source_image_id=NULL",
        [],
    )?;
    let _ = put_setting(&c, "cache_bytes", "0");
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

fn prune_cache_images_inner(app_data: &Path, keep_per_model: i64) -> AppResult<CacheOperationResult> {
    let keep = keep_per_model.clamp(0, 10000);
    let mut c = open_db(app_data)?;
    let root = cache_root(app_data);
    let model_ids: Vec<i64> = {
        let mut stmt = c.prepare("SELECT DISTINCT model_id FROM images ORDER BY model_id")?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.filter_map(Result::ok).collect()
    };
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;

    for model_id in model_ids {
        let mut stmt = c.prepare(
            "SELECT i.id FROM images i
             WHERE i.model_id=?1
               AND NOT EXISTS (SELECT 1 FROM models m WHERE m.cover_source_image_id=i.id)
             ORDER BY CASE WHEN i.meta_json LIKE '%\"featured\":true%' THEN 0 ELSE 1 END, i.cached_at DESC, i.id DESC"
        )?;
        let ids: Vec<i64> = stmt.query_map([model_id], |r| r.get(0))?.filter_map(Result::ok).collect();
        drop(stmt);
        if ids.len() <= keep as usize { continue; }
        let remove_ids: Vec<i64> = ids.into_iter().skip(keep as usize).collect();
        let (files, bytes) = delete_image_records(&mut c, &root.join("civitai"), &remove_ids)?;
        deleted_files += files;
        freed_bytes += bytes;
    }

    let orphan_result=clean_cache_orphans_inner(app_data)?;
    deleted_files+=orphan_result.deleted_files;
    freed_bytes+=orphan_result.freed_bytes;
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

fn set_cache_location_inner(app_data: &Path, path: &str) -> AppResult<CacheStats> {
    let target = PathBuf::from(path.trim());
    if target.as_os_str().is_empty() || !target.is_absolute() {
        return Err(AppError::Invalid("Cache location must be an absolute folder path".into()));
    }
    fs::create_dir_all(&target)?;
    let old = cache_root(app_data);
    if !old.exists() { fs::create_dir_all(&old)?; }
    let old_canonical = old.canonicalize().unwrap_or_else(|_| old.clone());
    let target_canonical = target.canonicalize().unwrap_or_else(|_| target.clone());

    if old_canonical == target_canonical {
        let mut c = open_db(app_data)?;
        let bytes = dir_size(&target_canonical);
        rewrite_cache_paths(&mut c, &old, &target, bytes)?;
        return cache_stats_inner(app_data);
    }
    if target_canonical.starts_with(&old_canonical) || old_canonical.starts_with(&target_canonical) {
        return Err(AppError::Invalid("The new cache location cannot contain the current cache location or be inside it".into()));
    }
    if fs::read_dir(&target_canonical)?.next().is_some() {
        return Err(AppError::Invalid("Choose an empty folder for the new Raphael cache location".into()));
    }

    let before = cache_file_counts(&old);
    copy_dir_recursive(&old, &target_canonical)?;
    let after = cache_file_counts(&target_canonical);
    if before != after {
        let _ = fs::remove_dir_all(&target_canonical);
        return Err(AppError::Invalid("Cache relocation verification failed; the original cache was preserved".into()));
    }

    let db_update = (|| {
        let mut c = open_db(app_data)?;
        let bytes = dir_size(&target_canonical);
        rewrite_cache_paths(&mut c, &old, &target, bytes)?;
        Ok::<(), AppError>(())
    })();

    if let Err(error) = db_update {
        let _ = fs::remove_dir_all(&target_canonical);
        return Err(error);
    }

    let _ = fs::remove_dir_all(&old);
    cache_stats_inner(app_data)
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

fn scan_root(app: &AppStateInner, root: &Path) -> AppResult<Vec<(String, Option<String>, Option<String>)>> {
    let _guard = app.scan_lock.lock().unwrap();
    let c = open_db(&app.app_data)?;
    let mut seen = HashSet::<String>::new();
    let mut scan_complete = true;

    for entry in WalkDir::new(root).follow_links(false) {
        let entry = match entry {
            Ok(entry) => entry,
            Err(_) => {
                scan_complete = false;
                continue;
            }
        };
        if !entry.file_type().is_file() || !is_model_file(entry.path()) { continue; }
        let path = entry.path().to_path_buf();
        let path_s = path.to_string_lossy().to_string();
        let meta = match fs::metadata(&path) {
            Ok(m) => m,
            Err(_) => {
                scan_complete = false;
                continue;
            }
        };
        let size = meta.len() as i64;
        let modified = mtime(&path);
        seen.insert(path_s.clone());
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
            c.execute("INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,source_hash,updated_at,downloaded_at) VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9)", params![path_s,rel,filename,mtype,size,modified,hash,now(),now()])?;
        }
    }

    let removed = prune_unseen_models(&c, &seen, scan_complete)?;
    Ok(removed)
}

fn prune_unseen_models(
    c: &Connection,
    seen: &HashSet<String>,
    scan_complete: bool,
) -> AppResult<Vec<(String, Option<String>, Option<String>)>> {
    if !scan_complete {
        return Ok(Vec::new());
    }
    let mut stmt = c.prepare("SELECT path,registry_model_id,registry_file_id FROM models")?;
    let existing: Vec<(String, Option<String>, Option<String>)> = stmt
        .query_map([], |r| Ok((
            r.get::<_, String>(0)?,
            r.get::<_, Option<String>>(1)?,
            r.get::<_, Option<String>>(2)?,
        )))?
        .filter_map(Result::ok)
        .collect();
    drop(stmt);
    let mut removed = Vec::new();
    for (path, registry_model_id, registry_file_id) in existing {
        if !seen.contains(&path) {
            removed.push((path.clone(), registry_model_id, registry_file_id));
            c.execute("DELETE FROM models WHERE path=?1", [&path])?;
        }
    }
    Ok(removed)
}

fn queue_registry_file_removal(
    app: &AppStateInner,
    model_id: &str,
    file_id: &str,
) {
    if let Ok(c) = open_db(&app.app_data) {
        let _ = c.execute(
            "INSERT OR IGNORE INTO pending_registry_file_removals(registry_model_id,registry_file_id) VALUES(?1,?2)",
            params![model_id, file_id],
        );
    }
}

async fn retry_pending_registry_file_removals(app: &AppStateInner) {
    let pending: Vec<(String, String)> = match open_db(&app.app_data).and_then(|c| {
        let mut stmt = c.prepare(
            "SELECT registry_model_id,registry_file_id FROM pending_registry_file_removals"
        )?;
        let rows = stmt.query_map([], |r| Ok((
            r.get::<_, String>(0)?,
            r.get::<_, String>(1)?,
        )))?;
        Ok(rows.filter_map(Result::ok).collect())
    }) {
        Ok(rows) => rows,
        Err(_) => return,
    };

    for (model_id, file_id) in pending {
        if app.registry.remove_file(&model_id, &file_id).await.is_ok() {
            if let Ok(c) = open_db(&app.app_data) {
                let _ = c.execute(
                    "DELETE FROM pending_registry_file_removals WHERE registry_model_id=?1 AND registry_file_id=?2",
                    params![model_id, file_id],
                );
            }
        }
    }
}

async fn reconcile_removed_registry_files(
    app: &AppStateInner,
    removed: Vec<(String, Option<String>, Option<String>)>,
) {
    for (_path, model_id, file_id) in removed {
        if let (Some(model_id), Some(file_id)) = (model_id, file_id) {
            if app.registry.remove_file(&model_id, &file_id).await.is_err() {
                queue_registry_file_removal(app, &model_id, &file_id);
            }
        }
    }
}

fn recursive_scan_and_emit(app: AppStateInner, handle: AppHandle) {
    if let Some(root) = app.models_root.read().unwrap().clone() {
        if let Ok(removed) = scan_root(&app, &root) {
            if !removed.is_empty() {
                let app_state = app.clone();
                tauri::async_runtime::spawn(async move {
                    reconcile_removed_registry_files(&app_state, removed).await;
                });
            }
            spawn_registry_sync(app.clone(), handle.clone());
            emit_models_changed(&handle);
        }
    }
}

fn model_from_row(r: &rusqlite::Row<'_>) -> rusqlite::Result<ModelRecord> {
    let tags: String = r.get(15)?; let activ: String = r.get(16)?;
    Ok(ModelRecord {
        id:r.get(0)?, path:r.get(1)?, relative_path:r.get(2)?, filename:r.get(3)?, model_type:r.get(4)?,
        size_bytes:r.get(5)?, modified_at:r.get(6)?, civitai_model_id:r.get(7)?, civitai_version_id:r.get(8)?,
        civitai_url:r.get(9)?, civitai_name:r.get(10)?, version_name:r.get(11)?, base_model:r.get(12)?,
        creator:r.get(13)?, description:r.get(14)?, tags:serde_json::from_str(&tags).unwrap_or_default(),
        activation_prompts:serde_json::from_str(&activ).unwrap_or_default(), source_hash:r.get(17)?,
        thumbnail_path:r.get(18)?, updated_at:r.get(19)?, cover_path:r.get(20)?,
        cover_source_image_id:r.get(21)?, cover_position_x:r.get(22)?, cover_position_y:r.get(23)?, downloaded_at:r.get(24)?
    })
}
const MODEL_SELECT: &str = "SELECT id,path,relative_path,filename,model_type,size_bytes,modified_at,civitai_model_id,civitai_version_id,civitai_url,civitai_name,version_name,base_model,creator,description,tags_json,activation_json,source_hash,thumbnail_path,updated_at,cover_path,cover_source_image_id,cover_position_x,cover_position_y,downloaded_at FROM models";
fn model_by_id(c: &Connection, id: i64) -> AppResult<ModelRecord> {
    Ok(c.query_row(&format!("{MODEL_SELECT} WHERE id=?1"), [id], model_from_row)?)
}


fn registry_model_type(model_type: &str) -> &'static str {
    match model_type {
        "Checkpoint" => "checkpoint",
        "LoRA" => "lora",
        "VAE" => "vae",
        "ControlNet" => "controlnet",
        "Embedding" => "embedding",
        "Upscaler" => "upscaler",
        "Text Encoder" => "text_encoder",
        "CLIP Vision" => "clip_vision",
        "IP-Adapter" => "ip_adapter",
        _ => "other",
    }
}

fn local_model_type(model_type: &str) -> String {
    match model_type {
        "checkpoint" => "Checkpoint",
        "lora" => "LoRA",
        "vae" => "VAE",
        "controlnet" => "ControlNet",
        "embedding" => "Embedding",
        "upscaler" => "Upscaler",
        "text_encoder" => "Text Encoder",
        "clip_vision" => "CLIP Vision",
        "ip_adapter" => "IP-Adapter",
        _ => "Other",
    }.to_string()
}

async fn hydrate_local_model_from_registry(
    app: &AppStateInner,
    local_id: i64,
    registry_model_id: &str,
    registry_version_id: Option<&str>,
) -> AppResult<ModelRecord> {
    let model = app.registry.get_model(registry_model_id).await?;
    let tags = app.registry.tags(registry_model_id).await.unwrap_or_default();
    let sources = app.registry.sources(registry_model_id).await.unwrap_or_default();
    let versions = app.registry.versions(registry_model_id).await.unwrap_or_default();

    let civitai_source = sources.iter().find(|source| source.provider.eq_ignore_ascii_case("civitai"));
    let selected_version = registry_version_id
        .and_then(|id| versions.iter().find(|version| version.id == id))
        .or_else(|| {
            civitai_source.and_then(|source| {
                source.external_version_id.as_ref().and_then(|external| {
                    versions.iter().find(|version| {
                        version.source_version_id.as_deref() == Some(external.as_str())
                    })
                })
            })
        })
        .or_else(|| versions.first());

    let civitai_model_id = civitai_source
        .and_then(|source| source.external_model_id.as_deref())
        .and_then(|value| value.parse::<i64>().ok());
    let civitai_version_id = civitai_source
        .and_then(|source| source.external_version_id.as_deref())
        .and_then(|value| value.parse::<i64>().ok())
        .or_else(|| selected_version.and_then(|version| version.source_version_id.as_deref()).and_then(|value| value.parse::<i64>().ok()));
    let civitai_url = civitai_source.and_then(|source| source.url.clone());

    let version_name = selected_version.and_then(|version| version.version_name.clone());
    let base_model = selected_version
        .and_then(|version| version.base_model.clone())
        .or_else(|| model.base_model.clone());
    let activation = selected_version
        .map(|version| version.activation_prompts.clone())
        .unwrap_or_default();

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models
         SET registry_model_id=?2,
             registry_version_id=?3,
             model_type=?4,
             civitai_model_id=?5,
             civitai_version_id=?6,
             civitai_url=?7,
             civitai_name=?8,
             version_name=?9,
             base_model=?10,
             creator=?11,
             description=?12,
             tags_json=?13,
             activation_json=?14,
             updated_at=?15
         WHERE id=?1",
        params![
            local_id,
            registry_model_id,
            selected_version.map(|version| version.id.as_str()),
            local_model_type(&model.model_type),
            civitai_model_id,
            civitai_version_id,
            civitai_url,
            model.name,
            version_name,
            base_model,
            model.creator,
            model.description,
            serde_json::to_string(&tags).unwrap_or_else(|_| "[]".into()),
            serde_json::to_string(&activation).unwrap_or_else(|_| "[]".into()),
            now()
        ],
    )?;
    model_by_id(&c, local_id)
}

async fn sync_local_model_to_registry(
    app: &AppStateInner,
    local_id: i64,
) -> AppResult<ModelRecord> {
    let (local, registry_model_id, registry_version_id, registry_file_id): (
        ModelRecord,
        Option<String>,
        Option<String>,
        Option<String>,
    ) = {
        let c = open_db(&app.app_data)?;
        let local = model_by_id(&c, local_id)?;
        let ids = c.query_row(
            "SELECT registry_model_id,registry_version_id,registry_file_id FROM models WHERE id=?1",
            [local_id],
            |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?, r.get::<_, Option<String>>(2)?)),
        )?;
        (local, ids.0, ids.1, ids.2)
    };

    let source_hash = match local.source_hash.clone() {
        Some(hash) if !hash.trim().is_empty() => hash,
        _ => {
            let hash = sha256_file(Path::new(&local.path))?;
            let c = open_db(&app.app_data)?;
            c.execute(
                "UPDATE models SET source_hash=?2,updated_at=?3 WHERE id=?1",
                params![local_id, hash, now()],
            )?;
            hash
        }
    };

    let deterministic_id = registry_model_id.clone().unwrap_or_else(|| {
        if let Some(civitai_id) = local.civitai_model_id {
            format!("civitai_model_{civitai_id}")
        } else {
            format!("sha256_{}", source_hash)
        }
    });

    let mut model = match app.registry.get_model(&deterministic_id).await {
        Ok(model) => model,
        Err(_) => match app.registry.get_model(&deterministic_id).await {
            Ok(existing) => existing,
            Err(_) => {
                let created = app.registry.create_model(
                    &deterministic_id,
                    local.civitai_name.as_deref().unwrap_or(&local.filename),
                    registry_model_type(&local.model_type),
                    local.creator.as_deref(),
                    local.description.as_deref(),
                    local.base_model.as_deref(),
                    json!({
                        "managed_by": "raphael-model-manager",
                        "local_model_id": local_id
                    }),
                ).await?;
                if created.id.is_empty() {
                    return Err(AppError::Registry("Registry created an invalid model id".into()));
                }
                created
            }
        },
    };

    let created_model_id = model.id.clone();

    if let (Some(civitai_model_id), Some(civitai_version_id)) =
        (local.civitai_model_id, local.civitai_version_id)
    {
        let version_id = format!("civitai_version_{civitai_version_id}");
        let versions = app.registry.versions(&created_model_id).await.unwrap_or_default();
        let external_version_text = civitai_version_id.to_string();
        let existing = versions
            .iter()
            .find(|version| {
                version.id == version_id
                    || version.source_version_id.as_deref() == Some(external_version_text.as_str())
            })
            .cloned();

        let version = if let Some(existing) = existing {
            app.registry.update_version(
                &created_model_id,
                &existing.id,
                existing.revision,
                json!({
                    "version_name": local.version_name,
                    "base_model": local.base_model,
                    "source": "civitai",
                    "source_model_id": civitai_model_id.to_string(),
                    "source_version_id": civitai_version_id.to_string(),
                    "source_url": local.civitai_url,
                    "activation_prompts": local.activation_prompts,
                    "metadata": {}
                }),
            ).await.unwrap_or(existing)
        } else {
            app.registry.create_version(
                &created_model_id,
                json!({
                    "id": version_id,
                    "version_name": local.version_name,
                    "base_model": local.base_model,
                    "source": "civitai",
                    "source_model_id": civitai_model_id.to_string(),
                    "source_version_id": civitai_version_id.to_string(),
                    "source_url": local.civitai_url,
                    "activation_prompts": local.activation_prompts,
                    "metadata": {}
                }),
            ).await?
        };

        app.registry.add_source(
            &created_model_id,
            json!({
                "provider": "civitai",
                "external_model_id": civitai_model_id.to_string(),
                "external_version_id": civitai_version_id.to_string(),
                "url": local.civitai_url,
                "metadata": {}
            }),
        ).await?;

        let mut registry_version_id = version.id.clone();
        let files = app.registry.files(&created_model_id).await.unwrap_or_default();

        if let Some(existing_file) = files.iter().find(|file| file.id == registry_file_id.clone().unwrap_or_default()).cloned() {
            if existing_file.path != local.path || existing_file.sha256.as_deref() != Some(source_hash.as_str()) {
                let _ = app.registry.remove_file(&created_model_id, &existing_file.id).await;
                registry_version_id = version.id.clone();
            } else {
                registry_version_id = existing_file.version_id.unwrap_or(version.id.clone());
            }
        }

        let files = app.registry.files(&created_model_id).await.unwrap_or_default();
        let matching = files.iter().find(|file| {
            file.path == local.path || file.sha256.as_deref() == Some(source_hash.as_str())
        }).cloned();

        let file = if let Some(file) = matching {
            file
        } else {
            app.registry.add_file(
                &created_model_id,
                json!({
                    "id": format!("file_{}", source_hash),
                    "version_id": Some(registry_version_id.clone()),
                    "path": local.path,
                    "relative_path": local.relative_path,
                    "filename": local.filename,
                    "size_bytes": local.size_bytes,
                    "modified_at": local.modified_at,
                    "sha256": source_hash,
                    "status": "available"
                }),
            ).await?
        };

        {
            let tags = local.tags.clone();
            for tag in tags {
                let _ = app.registry.add_tag(&created_model_id, &tag).await;
            }
        }

        model = app.registry.get_model(&created_model_id).await?;
        let _ = hydrate_local_model_from_registry(app, local_id, &model.id, Some(&version.id)).await?;
        let c = open_db(&app.app_data)?;
        c.execute(
            "UPDATE models SET registry_model_id=?2,registry_version_id=?3,registry_file_id=?4,source_hash=?5,updated_at=?6 WHERE id=?1",
            params![local_id, model.id, version.id, file.id, source_hash, now()],
        )?;
    } else {
        let files = app.registry.files(&created_model_id).await.unwrap_or_default();

        if let Some(existing_file_id) = registry_file_id.clone() {
            if let Some(existing_file) = files.iter().find(|file| file.id == existing_file_id) {
                if existing_file.path != local.path
                    || existing_file.sha256.as_deref() != Some(source_hash.as_str())
                    || existing_file.size_bytes != local.size_bytes
                    || existing_file.modified_at != local.modified_at
                {
                    let _ = app.registry.remove_file(&created_model_id, &existing_file.id).await;
                }
            }
        }

        let files = app.registry.files(&created_model_id).await.unwrap_or_default();
        let matching = files.iter().find(|file| {
            file.path == local.path
                && file.sha256.as_deref() == Some(source_hash.as_str())
                && file.size_bytes == local.size_bytes
                && file.modified_at == local.modified_at
        }).cloned();

        let file = if let Some(file) = matching {
            file
        } else {
            app.registry.add_file(
                &created_model_id,
                json!({
                    "id": format!("file_{}", source_hash),
                    "path": local.path,
                    "relative_path": local.relative_path,
                    "filename": local.filename,
                    "size_bytes": local.size_bytes,
                    "modified_at": local.modified_at,
                    "sha256": source_hash,
                    "status": "available"
                }),
            ).await?
        };

        let existing_tags = app.registry.tags(&created_model_id).await.unwrap_or_default();
        if existing_tags.is_empty() {
            for tag in local.tags.clone() {
                let _ = app.registry.add_tag(&created_model_id, &tag).await;
            }
        }

        let _ = hydrate_local_model_from_registry(app, local_id, &created_model_id, None).await?;
        let c = open_db(&app.app_data)?;
        c.execute(
            "UPDATE models SET registry_model_id=?2,registry_file_id=?3,source_hash=?4,updated_at=?5 WHERE id=?1",
            params![local_id, created_model_id, file.id, source_hash, now()],
        )?;
    }

    hydrate_local_model_from_registry(app, local_id, &created_model_id, registry_version_id.as_deref()).await
}


async fn apply_civitai_metadata_to_registry(
    app: &AppStateInner,
    local_id: i64,
    model: &Value,
    version: &Value,
    canonical_url: &str,
    tags: &[String],
    activation_prompts: &[String],
    description: Option<&str>,
    creator: Option<&str>,
) -> AppResult<(String, String)> {
    let _ = sync_local_model_to_registry(app, local_id).await?;
    let (registry_model_id, tags_user_modified) = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT registry_model_id,tags_user_modified FROM models WHERE id=?1",
            [local_id],
            |r| Ok((r.get::<_, String>(0)?, r.get::<_, i64>(1)? != 0)),
        )?
    };

    let registry_model = app.registry.get_model(&registry_model_id).await?;
    let model_type = {
        let c = open_db(&app.app_data)?;
        let local_type: String = c.query_row(
            "SELECT model_type FROM models WHERE id=?1",
            [local_id],
            |r| r.get(0),
        )?;
        registry_model_type(&local_type).to_string()
    };

    app.registry.update_model(
        &registry_model_id,
        registry_model.revision,
        json!({
            "name": model.get("name").and_then(Value::as_str),
            "model_type": model_type,
            "creator": creator,
            "description": description,
            "base_model": version.get("baseModel").and_then(Value::as_str)
        }),
    ).await?;

    let external_version_id = version
        .get("id")
        .and_then(Value::as_i64)
        .ok_or_else(|| AppError::Invalid("Civitai response did not include a model version ID".into()))?;
    let external_model_id = model
        .get("id")
        .and_then(Value::as_i64)
        .ok_or_else(|| AppError::Invalid("Civitai response did not include a model ID".into()))?;
    let desired_version_id = format!("civitai_version_{external_version_id}");
    let external_version_text = external_version_id.to_string();

    let versions = app.registry.versions(&registry_model_id).await?;
    let version_payload = json!({
        "version_name": version.get("name").and_then(Value::as_str),
        "base_model": version.get("baseModel").and_then(Value::as_str),
        "source": "civitai",
        "source_model_id": external_model_id.to_string(),
        "source_version_id": external_version_id.to_string(),
        "source_url": canonical_url,
        "activation_prompts": activation_prompts,
        "metadata": {}
    });

    let registry_version_id = if let Some(existing) = versions.iter().find(|item| {
        item.id == desired_version_id
            || item.source_version_id.as_deref() == Some(external_version_text.as_str())
    }) {
        app.registry.update_version(
            &registry_model_id,
            &existing.id,
            existing.revision,
            version_payload,
        ).await?.id
    } else {
        app.registry.create_version(
            &registry_model_id,
            json!({
                "id": desired_version_id,
                "version_name": version.get("name").and_then(Value::as_str),
                "base_model": version.get("baseModel").and_then(Value::as_str),
                "source": "civitai",
                "source_model_id": external_model_id.to_string(),
                "source_version_id": external_version_id.to_string(),
                "source_url": canonical_url,
                "activation_prompts": activation_prompts,
                "metadata": {}
            }),
        ).await?.id
    };

    app.registry.add_source(
        &registry_model_id,
        json!({
            "provider": "civitai",
            "external_model_id": external_model_id.to_string(),
            "external_version_id": external_version_id.to_string(),
            "url": canonical_url,
            "metadata": {}
        }),
    ).await?;

    if !tags_user_modified {
        let existing_tags = app.registry.tags(&registry_model_id).await.unwrap_or_default();
        for tag in existing_tags.iter().filter(|tag| !tags.iter().any(|value| value.eq_ignore_ascii_case(tag))) {
            let _ = app.registry.remove_tag(&registry_model_id, tag).await;
        }
        for tag in tags {
            if !existing_tags.iter().any(|value| value.eq_ignore_ascii_case(tag)) {
                app.registry.add_tag(&registry_model_id, tag).await?;
            }
        }
    }

    let _ = hydrate_local_model_from_registry(
        app,
        local_id,
        &registry_model_id,
        Some(&registry_version_id),
    ).await?;

    Ok((registry_model_id, registry_version_id))
}

fn spawn_registry_sync(app: AppStateInner, handle: AppHandle) {
    let retry_state = app.clone();
    tauri::async_runtime::spawn(async move {
        retry_pending_registry_file_removals(&retry_state).await;
    });
    {
        let mut running = match app.registry_sync_running.lock() {
            Ok(guard) => guard,
            Err(_) => return,
        };
        if *running {
            return;
        }
        *running = true;
    }

    tauri::async_runtime::spawn(async move {
        let ids: Vec<i64> = match open_db(&app.app_data).and_then(|c| {
            let mut stmt = c.prepare("SELECT id FROM models ORDER BY id")?;
            let rows = stmt.query_map([], |r| r.get::<_, i64>(0))?;
            Ok(rows.filter_map(Result::ok).collect())
        }) {
            Ok(ids) => ids,
            Err(_) => {
                if let Ok(mut running) = app.registry_sync_running.lock() {
                    *running = false;
                }
                return;
            }
        };

        let mut changed = false;
        for id in ids {
            if sync_local_model_to_registry(&app, id).await.is_ok() {
                changed = true;
            }
        }

        if changed {
            emit_models_changed(&handle);
        }

        if let Ok(mut running) = app.registry_sync_running.lock() {
            *running = false;
        }
    });
}


fn civitai_client(_app: &AppStateInner) -> AppResult<Client> {
    let mut b=Client::builder().user_agent(USER_AGENT).timeout(Duration::from_secs(30));
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

async fn download_cached_thumbnail(
    app: &AppStateInner,
    directory: &Path,
    url: &str,
) -> AppResult<Option<String>> {
    fs::create_dir_all(directory)?;

    let digest = Sha256::digest(url.as_bytes());
    let key = hex::encode(&digest[..8]);

    // The filename includes the source URL hash, so changing the Civitai
    // thumbnail URL automatically gets a fresh cached file instead of reusing
    // a stale thumbnail from an earlier version/link.
    for ext in ["jpg", "png", "webp", "avif"] {
        let candidate = directory.join(format!("thumbnail-{key}.{ext}"));
        if candidate.is_file() && fs::metadata(&candidate).map(|m| m.len() > 0).unwrap_or(false) {
            return Ok(Some(candidate.to_string_lossy().to_string()));
        }
    }

    let client = civitai_client(app)?;
    let mut req = client.get(url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }

    let res = req.send().await?;
    if !res.status().is_success() {
        return Ok(None);
    }

    let bytes = res.bytes().await?;
    if bytes.is_empty() {
        return Ok(None);
    }

    let format = detect_image_format_from_bytes(&bytes)?;
    let ext = image_format_extension(format)
        .ok_or_else(|| AppError::Invalid("Civitai returned an unsupported thumbnail format".into()))?;
    let final_path = directory.join(format!("thumbnail-{key}.{ext}"));
    let temp_path = directory.join(format!("thumbnail-{key}.{ext}.part"));

    fs::write(&temp_path, &bytes)?;
    if let Err(error) = fs::rename(&temp_path, &final_path) {
        let _ = fs::remove_file(&temp_path);
        if !final_path.is_file() {
            return Err(AppError::Io(error));
        }
    }

    Ok(Some(final_path.to_string_lossy().to_string()))
}

async fn ensure_model_thumbnail(
    app: &AppStateInner,
    model_id: i64,
    model: &Value,
    version: &Value,
    source_url: &str,
) -> AppResult<Option<String>> {
    let root = cache_root(&app.app_data).join(model_id.to_string());
    fs::create_dir_all(&root)?;

    let mut remote: Option<String> = model
        .get("images")
        .and_then(Value::as_array)
        .and_then(|a| {
            a.iter().find_map(|x| {
                x.get("url")
                    .and_then(Value::as_str)
                    .map(str::to_string)
            })
        });

    if remote.is_none() {
        remote = version
            .get("images")
            .and_then(Value::as_array)
            .and_then(|a| {
                a.iter().find_map(|x| {
                    x.get("url")
                        .and_then(Value::as_str)
                        .map(str::to_string)
                })
            });
    }

    if let Some(url) = remote {
        return download_cached_thumbnail(app, &root, &url).await;
    }

    let client = civitai_client(app)?;
    let mut req = client.get(source_url);
    if let Some(t) = token() {
        req = req.bearer_auth(t);
    }
    let res = req.send().await?;
    if !res.status().is_success() {
        return Ok(None);
    }

    let html = res.text().await?;
    let lower = html.to_ascii_lowercase();
    let marker = "property=\"og:image\"";
    if let Some(pos) = lower.find(marker) {
        let tail = &html[pos + marker.len()..];
        if let Some(content_pos) = tail.to_ascii_lowercase().find("content=\"") {
            let value = &tail[content_pos + 9..];
            if let Some(end) = value.find('"') {
                let image_url = &value[..end];
                return download_cached_thumbnail(app, &root, image_url).await;
            }
        }
    }

    Ok(None)
}

fn emit_examples_progress(handle: &AppHandle, progress: ExamplesRefreshProgress) {
    let _ = handle.emit("examples-refresh-progress", progress);
}

fn store_examples_refresh_state(
    state: &Arc<Mutex<ExamplesRefreshState>>,
    handle: &AppHandle,
    progress: ExamplesRefreshProgress,
) {
    if let Ok(mut guard) = state.lock() {
        guard.progress = Some(progress.clone());
        guard.running = !progress.done;
    }
    emit_examples_progress(handle, progress);
}

fn get_examples_refresh_state(
    state: &Arc<Mutex<ExamplesRefreshState>>,
) -> Option<ExamplesRefreshProgress> {
    state.lock().ok().and_then(|guard| guard.progress.clone())
}

type FeaturedImageRecord = (i64, Option<String>, Option<String>, Option<i64>, Option<i64>, Option<String>, Option<String>, Option<i64>, Option<f64>, Option<String>, Option<i64>, String);

const FEATURED_MODEL_CONCURRENCY: usize = 3;
const FEATURED_IMAGE_DOWNLOAD_CONCURRENCY: usize = 8;

#[allow(clippy::large_enum_variant)]
enum FeaturedDownloadResult {
    Saved(FeaturedImageRecord),
    DownloadFailed(String),
    ReadFailed(String),
    EmptyResponse,
    InvalidImage(String),
}

#[allow(clippy::too_many_arguments)]
async fn download_featured_image(
    client: Client,
    remote: String,
    image_id: i64,
    version_id: i64,
    version_name: String,
    image: Value,
    version_dir: PathBuf,
    active: PathBuf,
) -> (i64, FeaturedDownloadResult) {
    let response = match client.get(&remote).timeout(Duration::from_secs(60)).send().await {
        Ok(value) => value,
        Err(error) => return (image_id, FeaturedDownloadResult::DownloadFailed(error.to_string())),
    };

    if !response.status().is_success() {
        return (
            image_id,
            FeaturedDownloadResult::DownloadFailed(format!("Civitai returned {}", response.status())),
        );
    }

    let bytes = match response.bytes().await {
        Ok(value) => value,
        Err(error) => return (image_id, FeaturedDownloadResult::ReadFailed(error.to_string())),
    };

    if bytes.is_empty() {
        return (image_id, FeaturedDownloadResult::EmptyResponse);
    }

    let detected_format = match detect_image_format_from_bytes(&bytes) {
        Ok(value) => value,
        Err(error) => return (image_id, FeaturedDownloadResult::InvalidImage(error.to_string())),
    };

    let guessed_ext = featured_extension(&remote);
    let ext = image_format_extension(detected_format)
        .unwrap_or(guessed_ext.as_str())
        .to_string();
    let local = version_dir.join(format!("{image_id}.{ext}"));

    if let Err(error) = fs::write(&local, &bytes) {
        return (image_id, FeaturedDownloadResult::ReadFailed(error.to_string()));
    }

    let thumbnail_local = ensure_featured_thumbnail(&local);

    let mut meta = image.clone();
    if let Some(map) = meta.as_object_mut() {
        map.insert("featured".into(), json!(true));
        map.insert("civitai_version_id".into(), json!(version_id));
        map.insert("civitai_version_name".into(), json!(version_name));
    }

    let prompt = meta.get("meta").and_then(|m| parse_meta(m, "prompt"));
    let negative_prompt = meta
        .get("meta")
        .and_then(|m| parse_meta(m, "negativePrompt").or_else(|| parse_meta(m, "Negative prompt")));
    let sampler = meta
        .get("meta")
        .and_then(|m| parse_meta(m, "sampler").or_else(|| parse_meta(m, "Sampler")));
    let steps = meta.get("meta").and_then(|m| m.get("steps")).and_then(Value::as_i64);
    let cfg = meta
        .get("meta")
        .and_then(|m| m.get("cfgScale").or_else(|| m.get("cfg")))
        .and_then(Value::as_f64);
    let seed = meta.get("meta").and_then(|m| m.get("seed")).and_then(Value::as_i64);
    let width = image.get("width").and_then(Value::as_i64);
    let height = image.get("height").and_then(Value::as_i64);
    let final_local = active
        .join(version_id.to_string())
        .join(format!("{image_id}.{ext}"))
        .to_string_lossy()
        .to_string();

    (
        image_id,
        FeaturedDownloadResult::Saved((
            image_id,
            Some(final_local),
            thumbnail_local.map(|_| {
                active
                    .join(version_id.to_string())
                    .join(format!("{image_id}_thumb.webp"))
                    .to_string_lossy()
                    .to_string()
            }),
            width,
            height,
            prompt,
            negative_prompt,
            steps,
            cfg,
            sampler,
            seed,
            serde_json::to_string(&meta).unwrap_or_else(|_| "{}".into()),
        )),
    )
}

fn ensure_featured_thumbnail(source: &Path) -> Option<String> {
    let parent = source.parent()?;
    let stem = source.file_stem()?.to_str()?;
    let thumbnail = parent.join(format!("{stem}_thumb.webp"));
    if !thumbnail.is_file() {
        let image = decode_image_file(source).ok()?;
        let resized = image.resize(420, 420, FilterType::Triangle);
        resized.save_with_format(&thumbnail, ImageFormat::WebP).ok()?;
    }
    Some(thumbnail.to_string_lossy().to_string())
}

fn featured_extension(url: &str) -> String {
    Url::parse(url)
        .ok()
        .and_then(|u| Path::new(u.path()).extension().and_then(|x| x.to_str()).map(|x| x.to_ascii_lowercase()))
        .filter(|x| !x.is_empty() && x.len() <= 8)
        .unwrap_or_else(|| "jpeg".into())
}

fn featured_image_key(image: &Value, version_id: i64, index: usize) -> Option<i64> {
    let url = image.get("url").and_then(Value::as_str)?.trim();
    if url.is_empty() {
        return None;
    }

    // modelVersions[].images does not reliably expose an image ID.
    // Use a deterministic negative key derived from the actual image URL
    // and version instead of requiring an optional/absent JSON field.
    let mut hasher = Sha256::new();
    hasher.update(version_id.to_le_bytes());
    hasher.update((index as u64).to_le_bytes());
    hasher.update(url.as_bytes());
    let digest = hasher.finalize();
    let mut bytes = [0u8; 8];
    bytes.copy_from_slice(&digest[..8]);
    let positive = (u64::from_le_bytes(bytes) % (i64::MAX as u64)) + 1;
    Some(-(positive as i64))
}

async fn sync_featured_examples_inner(
    app: AppStateInner,
    model_id: i64,
    handle: AppHandle,
    progress_model: Option<(usize, usize)>,
    maintain_cache: bool,
    emit_model_change: bool,
) -> AppResult<usize> {
    let model_record = { let c = open_db(&app.app_data)?; model_by_id(&c, model_id)? };
    let civitai_id = model_record.civitai_model_id.ok_or_else(|| AppError::Invalid("This model is not linked to Civitai".into()))?;
    let model_name = model_record.civitai_name.clone().unwrap_or_else(|| model_record.filename.clone());

    let existing_featured: HashMap<i64, (PathBuf, String)> = {
        let c = open_db(&app.app_data)?;
        let mut stmt = c.prepare(
            "SELECT civitai_image_id,local_path,COALESCE(meta_json,'')
             FROM images
             WHERE model_id=?1 AND meta_json LIKE '%\"featured\":true%'",
        )?;
        let rows = stmt.query_map([model_id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
                r.get::<_, String>(2)?,
            ))
        })?;
        rows.filter_map(Result::ok)
            .filter_map(|(image_id,path,meta)| {
                let path=PathBuf::from(path?);
                let url=serde_json::from_str::<Value>(&meta)
                    .ok()
                    .and_then(|value| value.get("url").and_then(Value::as_str).map(str::to_string))?;
                Some((image_id,(path,url)))
            })
            .collect()
    };

    let model = api_get(&app, &format!("{API_BASE}/models/{civitai_id}")).await?;
    let mut versions = model.get("modelVersions").and_then(Value::as_array).cloned().unwrap_or_default();
    versions.sort_by(|a,b| {
        let ac=a.get("createdAt").and_then(Value::as_str).unwrap_or("");
        let bc=b.get("createdAt").and_then(Value::as_str).unwrap_or("");
        let ai=a.get("id").and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))).unwrap_or_default();
        let bi=b.get("id").and_then(|v| v.as_i64().or_else(|| v.as_str().and_then(|s| s.parse::<i64>().ok()))).unwrap_or_default();
        bc.cmp(ac).then_with(|| bi.cmp(&ai))
    });

    if versions.is_empty() {
        return Err(AppError::Api("Civitai returned no published model versions".into()));
    }

    // Process every published model version returned by Civitai and every
    // creator-uploaded image in version.images[]. There is intentionally no
    // version limit here: the prototype and the Civitai model endpoint expose
    // featured images on every returned model version.
    let total_versions=versions.len();
    let (model_index, model_total)=progress_model.unwrap_or((0,1));
    let cache=cache_root(&app.app_data).join(civitai_id.to_string());
    fs::create_dir_all(&cache)?;
    let run_id=DOWNLOAD_COUNTER.fetch_add(1,Ordering::Relaxed);
    let staging=cache.join(format!("featured.__staging_{run_id}"));
    let active=cache.join("featured");
    fs::create_dir_all(&staging)?;
    let client=civitai_client(&app)?;

    let mut records:Vec<FeaturedImageRecord>=Vec::new();
    let mut saved_count=0usize;
    let mut image_entries=0usize;
    let mut image_urls=0usize;
    let mut download_failures=0usize;
    let mut read_failures=0usize;
    let mut empty_responses=0usize;

    for (version_index,version) in versions.iter().enumerate(){
        let version_id=version.get("id").and_then(Value::as_i64).ok_or_else(||AppError::Api("Civitai returned a model version without an ID".into()))?;
        let version_name=version.get("name").and_then(Value::as_str).unwrap_or("version").to_string();
        emit_examples_progress(&handle,ExamplesRefreshProgress{current:model_index,total:model_total,model_id:Some(model_id),model_name:Some(model_name.clone()),version_current:version_index,version_total:total_versions,images_saved:saved_count,status:format!("Downloading featured images from {version_name}"),done:false,error:None});

        let images=version.get("images").and_then(Value::as_array).cloned().unwrap_or_default();
        image_entries += images.len();
        let version_dir=staging.join(version_id.to_string());
        fs::create_dir_all(&version_dir)?;

        let mut first_image_error: Option<String> = None;
        let mut jobs: Vec<(String,i64,Value)> = Vec::new();

        for (image_index,image) in images.into_iter().enumerate() {
            let remote=match image.get("url").and_then(Value::as_str).map(str::trim).filter(|url| !url.is_empty()).map(str::to_string) {
                Some(value)=>value,
                None=>continue,
            };
            image_urls += 1;
            let image_id=match featured_image_key(&image,version_id,image_index) {
                Some(value)=>value,
                None=>continue,
            };

            let reused = existing_featured.get(&image_id).and_then(|(source,stored_url)| {
                if stored_url != &remote || !source.is_file() {
                    return None;
                }
                let ext=source.extension().and_then(|value| value.to_str()).unwrap_or("jpg");
                let target=version_dir.join(format!("{image_id}.{ext}"));
                if copy_or_hard_link(source,&target).is_err() {
                    return None;
                }
                let thumbnail = ensure_featured_thumbnail(&target);
                let mut meta=image.clone();
                if let Some(map)=meta.as_object_mut() {
                    map.insert("featured".into(),json!(true));
                    map.insert("civitai_version_id".into(),json!(version_id));
                    map.insert("civitai_version_name".into(),json!(version_name));
                }
                let prompt=meta.get("meta").and_then(|m|parse_meta(m,"prompt"));
                let negative_prompt=meta.get("meta").and_then(|m|parse_meta(m,"negativePrompt").or_else(||parse_meta(m,"Negative prompt")));
                let sampler=meta.get("meta").and_then(|m|parse_meta(m,"sampler").or_else(||parse_meta(m,"Sampler")));
                let steps=meta.get("meta").and_then(|m|m.get("steps")).and_then(Value::as_i64);
                let cfg=meta.get("meta").and_then(|m|m.get("cfgScale").or_else(||m.get("cfg"))).and_then(Value::as_f64);
                let seed=meta.get("meta").and_then(|m|m.get("seed")).and_then(Value::as_i64);
                let width=image.get("width").and_then(Value::as_i64);
                let height=image.get("height").and_then(Value::as_i64);
                let final_local=active.join(version_id.to_string()).join(format!("{image_id}.{ext}")).to_string_lossy().to_string();
                let final_thumbnail = thumbnail.map(|_| active.join(version_id.to_string()).join(format!("{image_id}_thumb.webp")).to_string_lossy().to_string());
                Some((image_id,(image_id,Some(final_local),final_thumbnail,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,serde_json::to_string(&meta).unwrap_or_else(|_|"{}".into()))))
            });

            if let Some((_image_id,record)) = reused {
                records.push(record);
                saved_count+=1;

            } else {
                jobs.push((remote,image_id,image));
            }
        }

        let results = stream::iter(jobs.into_iter().map(|(remote,image_id,image)| {
            let client=client.clone();
            let version_dir=version_dir.clone();
            let active=active.clone();
            let version_name=version_name.clone();
            async move {
                download_featured_image(
                    client,remote,image_id,version_id,version_name,image,version_dir,active
                ).await
            }
        }))
        .buffer_unordered(FEATURED_IMAGE_DOWNLOAD_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

        for (_image_id,result) in results {
            match result {
                FeaturedDownloadResult::Saved(record) => {
                    records.push(record);
                    saved_count+=1;
                }
                FeaturedDownloadResult::DownloadFailed(error) => {
                    download_failures+=1;
                    if first_image_error.is_none() { first_image_error=Some(error); }
                }
                FeaturedDownloadResult::ReadFailed(error) => {
                    read_failures+=1;
                    if first_image_error.is_none() { first_image_error=Some(error); }
                }
                FeaturedDownloadResult::EmptyResponse => {
                    empty_responses+=1;
                }
                FeaturedDownloadResult::InvalidImage(error) => {
                    read_failures+=1;
                    if first_image_error.is_none() { first_image_error=Some(error); }
                }
            }
        }

        emit_examples_progress(&handle,ExamplesRefreshProgress{
            current:model_index,total:model_total,model_id:Some(model_id),model_name:Some(model_name.clone()),
            version_current:version_index+1,version_total:total_versions,images_saved:saved_count,
            status:format!("Finished {version_name}: {saved_count} images ready, {} failures",download_failures+read_failures+empty_responses),
            done:false,error:first_image_error.clone(),
        });
    }

    if records.is_empty(){
        let _=fs::remove_dir_all(&staging);
        emit_examples_progress(&handle,ExamplesRefreshProgress{
            current:model_index,total:model_total,model_id:Some(model_id),model_name:Some(model_name.clone()),
            version_current:total_versions,version_total:total_versions,images_saved:0,
            status:format!("No featured images could be downloaded across {total_versions} published model versions ({} entries, {} usable URLs, {} download failures, {} read failures, {} empty responses)", image_entries, image_urls, download_failures, read_failures, empty_responses),
            done:false,error:None
        });
        return Ok(0);
    }

    let _commit_guard=app.cache_lock.lock().await;
    let mut preserved_featured_cover_key: Option<i64> = None;
    let mut preserved_featured_cover_path: Option<PathBuf> = None;
    let mut clear_stale_featured_cover = false;
    let c = open_db(&app.app_data)?;
    let old_cover: Option<String> = c.query_row(
        "SELECT cover_path FROM models WHERE id=?1",
        [model_id],
        |r| r.get(0),
    )?;
    if let Some(old_cover) = old_cover {
        let old_path = PathBuf::from(old_cover);
        if old_path.starts_with(&active) {
            if old_path.is_file() {
                preserved_featured_cover_path = Some(copy_cached_cover(&app, model_id, &old_path)?);
                preserved_featured_cover_key = c
                    .query_row(
                        "SELECT civitai_image_id FROM images WHERE model_id=?1 AND (local_path=?2 OR thumbnail_path=?2) AND meta_json LIKE '%\"featured\":true%' LIMIT 1",
                        params![model_id, old_path.to_string_lossy().to_string()],
                        |r| r.get::<_, i64>(0),
                    )
                    .optional()?;
            } else {
                clear_stale_featured_cover = true;
            }
        }
    }

    let backup=cache.join(format!("featured.__backup_{run_id}"));
    if backup.exists(){let _=fs::remove_dir_all(&backup);}
    if active.exists(){fs::rename(&active,&backup)?;}
    fs::rename(&staging,&active)?;

    let db_result:AppResult<()>=(||{
        let mut c=open_db(&app.app_data)?;
        let tx=c.transaction()?;
        tx.execute("DELETE FROM images WHERE model_id=?1 AND meta_json LIKE '%\"featured\":true%'", [model_id])?;
        for record in &records{
            tx.execute(
                "INSERT INTO images(model_id,civitai_image_id,local_path,thumbnail_path,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,meta_json,cached_at)
                 VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,CAST(strftime('%s','now') AS INTEGER))
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
                    meta_json=excluded.meta_json,
                    cached_at=excluded.cached_at",
                params![model_id,record.0,record.1,record.2,record.3,record.4,record.5,record.6,record.7,record.8,record.9,record.10,record.11],
            )?;
        }
        if let Some(new_cover) = preserved_featured_cover_path.as_ref() {
            let new_source_id = if let Some(cover_key) = preserved_featured_cover_key {
                tx
                    .query_row(
                        "SELECT id FROM images WHERE model_id=?1 AND civitai_image_id=?2 AND meta_json LIKE '%\"featured\":true%' LIMIT 1",
                        params![model_id, cover_key],
                        |r| r.get::<_, i64>(0),
                    )
                    .optional()?
            } else {
                None
            };
            tx.execute(
                "UPDATE models SET cover_path=?2,cover_source_image_id=?3,updated_at=?4 WHERE id=?1",
                params![model_id, new_cover.to_string_lossy().to_string(), new_source_id, now()],
            )?;
        } else if clear_stale_featured_cover {
            tx.execute(
                "UPDATE models SET cover_path=NULL,cover_source_image_id=NULL,updated_at=?2 WHERE id=?1",
                params![model_id, now()],
            )?;
        }
        tx.commit()?;
        Ok(())
    })();

    match db_result{
        Ok(())=>{
            if backup.exists(){let _=fs::remove_dir_all(&backup);}
            if maintain_cache {
                let bytes=dir_size(&cache_root(&app.app_data));
                if let Ok(c)=open_db(&app.app_data){let _=put_setting(&c,"cache_bytes",&bytes.to_string());}
                let _=enforce_cache_limit_inner(&app.app_data);
            }
            if emit_model_change {
                emit_models_changed(&handle);
            }
            Ok(saved_count)
        }
        Err(error)=>{let _=fs::remove_dir_all(&active);if backup.exists(){let _=fs::rename(&backup,&active);}Err(error)}
    }
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
    let removed=scan_root(&app,&root)?; if !removed.is_empty(){let app_state=app.inner().clone();tauri::async_runtime::spawn(async move{reconcile_removed_registry_files(&app_state,removed).await;});} emit_models_changed(&handle); spawn_registry_sync(app.inner().clone(),handle.clone()); spawn_registry_event_sync(app.inner().clone()); spawn_hash_enrichment(app.inner().clone(),handle.clone()); Ok(AppStateResponse{models_root:Some(path),storage:storage_stats_inner(&app.app_data)?})
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
async fn set_model_tags_inner(
    app: &AppStateInner,
    handle: AppHandle,
    id: i64,
    tags: Vec<String>,
) -> AppResult<ModelRecord> {
    let normalized = normalize_tags(tags);
    let _ = sync_local_model_to_registry(app, id).await?;

    let registry_model_id: String = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT registry_model_id FROM models WHERE id=?1",
            [id],
            |r| r.get::<_, String>(0),
        )?
    };

    let existing = app.registry.tags(&registry_model_id).await.unwrap_or_default();
    for tag in existing.iter().filter(|tag| !normalized.iter().any(|value| value.eq_ignore_ascii_case(tag))) {
        let _ = app.registry.remove_tag(&registry_model_id, tag).await;
    }
    for tag in &normalized {
        if !existing.iter().any(|value| value.eq_ignore_ascii_case(tag)) {
            app.registry.add_tag(&registry_model_id, tag).await?;
        }
    }

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET tags_user_modified=1,updated_at=?2 WHERE id=?1",
        params![id, now()],
    )?;
    drop(c);

    let rec = hydrate_local_model_from_registry(app, id, &registry_model_id, None).await?;
    emit_models_changed(&handle);
    Ok(rec)
}

#[tauri::command]
async fn set_model_tags(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    tags: Vec<String>,
) -> AppResult<ModelRecord> {
    set_model_tags_inner(app.inner(), handle, id, tags).await
}

fn add_subfolder_tags_inner(app: &AppStateInner) -> AppResult<i64> {
    let c = open_db(&app.app_data)?;
    let rows: Vec<(i64, String, String)> = {
        let mut stmt = c.prepare("SELECT id,relative_path,tags_json FROM models")?;
        let rows = stmt
            .query_map([], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, String>(1)?,
                    r.get::<_, String>(2)?,
                ))
            })?
            .collect::<rusqlite::Result<Vec<_>>>()?;
        rows
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
    spawn_registry_sync(app.inner().clone(), handle.clone());
    emit_models_changed(&handle);
    Ok(updated)
}

#[tauri::command]
async fn delete_model_inner(app: &AppStateInner, handle: AppHandle, id: i64) -> AppResult<()> {
    let _guard=app.cache_lock.lock().await;
    let root = app.models_root.read().unwrap().clone()
        .ok_or_else(|| AppError::Invalid("Choose your ComfyUI models folder first".into()))?;

    let (path, thumbnail_path, cover_path, image_paths) = {
        let c = open_db(&app.app_data)?;
        let model = model_by_id(&c, id)?;
        let mut stmt = c.prepare("SELECT local_path,thumbnail_path FROM images WHERE model_id=?1")?;
        let rows = stmt.query_map([id], |r| {
            Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?))
        })?;
        let image_paths: Vec<(Option<String>,Option<String>)> = rows.filter_map(Result::ok).collect();
        (PathBuf::from(model.path), model.thumbnail_path, model.cover_path, image_paths)
    };

    let _ = sync_local_model_to_registry(app, id).await?;
    let (registry_model_id, registry_file_id) = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT registry_model_id,registry_file_id FROM models WHERE id=?1",
            [id],
            |r| Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<String>>(1)?,
            )),
        )?
    };

    let root_canonical = root.canonicalize()?;
    let path_canonical = path.canonicalize().unwrap_or_else(|_| path.clone());
    if !path_canonical.starts_with(&root_canonical) {
        return Err(AppError::Invalid("Refusing to delete a model outside the configured models folder".into()));
    }

    if let (Some(registry_model_id), Some(registry_file_id)) =
        (registry_model_id.as_deref(), registry_file_id.as_deref())
    {
        app.registry
            .remove_file(registry_model_id, registry_file_id)
            .await?;
    }

    if path.exists() {
        let meta = fs::metadata(&path)?;
        if !meta.is_file() {
            return Err(AppError::Invalid("The model path is not a regular file".into()));
        }
        fs::remove_file(&path)?;
    }

    for (local_path, thumb_path) in image_paths {
        for cached in [local_path, thumb_path].into_iter().flatten() {
            let cached_path = PathBuf::from(cached);
            if path_is_in_cache(&app.app_data,&cached_path) && cached_path.is_file() {
                let _ = fs::remove_file(cached_path);
            }
        }
    }

    if let Some(thumbnail) = thumbnail_path {
        let thumbnail_path = PathBuf::from(thumbnail);
        if path_is_in_cache(&app.app_data,&thumbnail_path) && thumbnail_path.is_file() {
            let _ = fs::remove_file(thumbnail_path);
        }
    }

    if let Some(cover) = cover_path {
        let cover_path = PathBuf::from(cover);
        if path_is_in_cache(&app.app_data,&cover_path) && cover_path.is_file() {
            let _ = fs::remove_file(cover_path);
        }
    }

    let c = open_db(&app.app_data)?;
    let deleted = c.execute("DELETE FROM models WHERE id=?1", [id])?;
    if deleted == 0 {
        return Err(AppError::Invalid("Model no longer exists".into()));
    }
    emit_models_changed(&handle);
    Ok(())
}

#[tauri::command]
async fn delete_model(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
) -> AppResult<()> {
    delete_model_inner(app.inner(), handle, id).await
}
#[tauri::command]
async fn set_model_type(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    model_type: String,
) -> AppResult<ModelRecord> {
    let requested = model_type.trim();
    let current = {
        let c = open_db(&app.app_data)?;
        model_by_id(&c, id)?
    };
    let next_type = if requested.eq_ignore_ascii_case("auto") {
        let root = app.models_root.read().unwrap().clone().ok_or_else(|| {
            AppError::Invalid("Choose your ComfyUI models folder first".into())
        })?;
        file_type_from_path(Path::new(&current.path), &root)
    } else {
        match requested {
            "Checkpoint" | "LoRA" | "VAE" | "ControlNet" | "Embedding" | "Upscaler"
            | "Text Encoder" | "CLIP Vision" | "IP-Adapter" | "Other" => requested.to_string(),
            _ => return Err(AppError::Invalid("Unsupported model type".into())),
        }
    };

    let _ = sync_local_model_to_registry(&app, id).await?;
    let registry_model_id = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT registry_model_id FROM models WHERE id=?1",
            [id],
            |r| r.get::<_, String>(0),
        )?
    };
    let registry_model = app.registry.get_model(&registry_model_id).await?;
    app.registry
        .update_model(
            &registry_model_id,
            registry_model.revision,
            json!({"model_type": registry_model_type(&next_type)}),
        )
        .await?;

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET model_type_user_modified=?2,updated_at=?3 WHERE id=?1",
        params![id, if requested.eq_ignore_ascii_case("auto") { 0 } else { 1 }, now()],
    )?;
    drop(c);

    let rec = hydrate_local_model_from_registry(app.inner(), id, &registry_model_id, None).await?;
    emit_models_changed(&handle);
    Ok(rec)
}

fn cover_position(value: f64) -> f64 { value.clamp(0.0, 100.0) }

fn custom_cover_extension(path: &Path) -> Option<&'static str> {
    match path
        .extension()
        .and_then(|x| x.to_str())
        .map(|x| x.to_ascii_lowercase())
        .as_deref()
    {
        Some("png") => Some("png"),
        Some("jpg") | Some("jpeg") => Some("jpg"),
        Some("webp") => Some("webp"),
        Some("avif") => Some("avif"),
        _ => None,
    }
}

fn copy_or_hard_link(source: &Path, target: &Path) -> io::Result<()> {
    match fs::hard_link(source, target) {
        Ok(()) => Ok(()),
        Err(_) => {
            fs::copy(source, target)?;
            Ok(())
        }
    }
}

fn image_format_extension(format: ImageFormat) -> Option<&'static str> {
    match format {
        ImageFormat::Png => Some("png"),
        ImageFormat::Jpeg => Some("jpg"),
        ImageFormat::WebP => Some("webp"),
        ImageFormat::Avif => Some("avif"),
        _ => None,
    }
}

fn detect_image_format_from_bytes(bytes: &[u8]) -> AppResult<ImageFormat> {
    let reader = ImageReader::new(std::io::Cursor::new(bytes))
        .with_guessed_format()
        .map_err(|e| AppError::Invalid(format!("Could not determine image format: {e}")))?;
    reader
        .format()
        .ok_or_else(|| AppError::Invalid("Could not determine image format from image data".into()))
}

fn decode_image_file(path: &Path) -> AppResult<image::DynamicImage> {
    let file = File::open(path)?;
    let reader = ImageReader::new(BufReader::new(file))
        .with_guessed_format()
        .map_err(|e| AppError::Invalid(format!("Could not inspect image: {e}")))?;
    reader
        .decode()
        .map_err(|e| AppError::Invalid(format!("Could not decode image: {e}")))
}

fn copy_custom_cover(app: &AppStateInner, id: i64, source: &Path) -> AppResult<PathBuf> {
    if !source.is_file() {
        return Err(AppError::Invalid("Selected cover image is not a file".into()));
    }

    // Validate the image header/format without fully decoding the image.
    let reader = ImageReader::open(source)
        .map_err(|e| AppError::Invalid(format!("Could not read the custom cover image: {e}")))?
        .with_guessed_format()
        .map_err(|e| AppError::Invalid(format!("Could not inspect the custom cover image: {e}")))?;
    let format = reader
        .format()
        .ok_or_else(|| AppError::Invalid("Could not determine the custom cover image format".into()))?;
    let ext = image_format_extension(format)
        .or_else(|| custom_cover_extension(source))
        .ok_or_else(|| AppError::Invalid("Custom covers must be PNG, JPG, JPEG, WebP, or AVIF images".into()))?;

    let dir = cache_root(&app.app_data).join("covers");
    fs::create_dir_all(&dir)?;
    let sequence = DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let target = dir.join(format!("model_{id}_{sequence}.{ext}"));

    // Keep custom covers independent from the user's source file. A hard link
    // would make later edits/replacements of that external file change the
    // cover unexpectedly.
    fs::copy(source, &target)?;
    Ok(target)
}

fn copy_cached_cover(app: &AppStateInner, id: i64, source: &Path) -> AppResult<PathBuf> {
    let cache_root_path = cache_root(&app.app_data)
        .canonicalize()
        .map_err(|_| AppError::Invalid("Raphael's Civitai cache is unavailable".into()))?;
    let canonical_source = source
        .canonicalize()
        .map_err(|_| AppError::Invalid("That example image is no longer available in Raphael's cache".into()))?;
    if !canonical_source.starts_with(&cache_root_path) || !canonical_source.is_file() {
        return Err(AppError::Invalid("That example image is outside Raphael's Civitai cache".into()));
    }
    let format = ImageReader::open(&canonical_source)
        .map_err(|e| AppError::Invalid(format!("Could not inspect the example image: {e}")))?
        .with_guessed_format()
        .map_err(|e| AppError::Invalid(format!("Could not inspect the example image: {e}")))?
        .format()
        .ok_or_else(|| AppError::Invalid("Could not determine the cached example image format".into()))?;
    let ext = image_format_extension(format)
        .ok_or_else(|| AppError::Invalid("The cached example image has an unsupported format".into()))?;
    let dir = cache_root(&app.app_data).join("covers");
    fs::create_dir_all(&dir)?;
    let sequence = DOWNLOAD_COUNTER.fetch_add(1, Ordering::Relaxed);
    let target = dir.join(format!("model_{id}_{sequence}.{ext}"));
    copy_or_hard_link(&canonical_source, &target)?;
    Ok(target)
}

#[tauri::command]
fn set_model_cover_position(
    app: State<AppStateInner>,
    id: i64,
    x: f64,
    y: f64,
) -> AppResult<ModelRecord> {
    let x = cover_position(x);
    let y = cover_position(y);
    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET cover_position_x=?2,cover_position_y=?3,updated_at=?4 WHERE id=?1",
        params![id, x, y, now()],
    )?;
    model_by_id(&c, id)
}

#[tauri::command]
async fn set_model_custom_cover(
    app: State<'_, AppStateInner>,
    id: i64,
    source_path: String,
) -> AppResult<ModelRecord> {
    let old_cover: Option<String> = {
        let c = open_db(&app.app_data)?;
        c.query_row("SELECT cover_path FROM models WHERE id=?1", [id], |r| r.get(0))?
    };
    let source = PathBuf::from(source_path);
    let target = copy_custom_cover(&app, id, &source)?;
    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET cover_path=?2,cover_source_image_id=NULL,cover_position_x=50,cover_position_y=50,updated_at=?3 WHERE id=?1",
        params![id, target.to_string_lossy().to_string(), now()],
    )?;
    let rec = model_by_id(&c, id)?;
    drop(c);
    if let Some(old) = old_cover {
        let old_path = PathBuf::from(old);
        if old_path != target && path_is_in_cache(&app.app_data, &old_path) && old_path.is_file() {
            let _ = fs::remove_file(old_path);
        }
    }
    Ok(rec)
}

#[tauri::command]
async fn reset_model_cover(
    app: State<'_, AppStateInner>,
    id: i64,
) -> AppResult<ModelRecord> {
    let old_cover = {
        let c = open_db(&app.app_data)?;
        c.query_row("SELECT cover_path FROM models WHERE id=?1", [id], |r| r.get::<_, Option<String>>(0))?
    };

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET cover_path=NULL,cover_source_image_id=NULL,cover_position_x=50,cover_position_y=50,updated_at=?2 WHERE id=?1",
        params![id, now()],
    )?;
    let rec = model_by_id(&c, id)?;
    drop(c);
    if let Some(path) = old_cover {
        let cover = PathBuf::from(path);
        if path_is_in_cache(&app.app_data,&cover) && cover.is_file() {
            let _ = fs::remove_file(cover);
        }
    }
    Ok(rec)
}

#[tauri::command]
async fn set_model_cover_from_image(
    app: State<'_, AppStateInner>,
    id: i64,
    image_id: i64,
) -> AppResult<ModelRecord> {
    let (old_cover, old_cover_source_id, source) = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT m.cover_path,m.cover_source_image_id,COALESCE(i.local_path,i.thumbnail_path)
             FROM models m
             JOIN images i ON i.model_id=m.id
             WHERE m.id=?1 AND i.id=?2",
            params![id, image_id],
            |r| Ok((
                r.get::<_, Option<String>>(0)?,
                r.get::<_, Option<i64>>(1)?,
                r.get::<_, Option<String>>(2)?,
            )),
        )
        .map_err(|_| AppError::Invalid("That example image is not cached for this model".into()))?
    };
    let source = source
        .map(PathBuf::from)
        .ok_or_else(|| AppError::Invalid("That example image has not finished caching yet".into()))?;

    // The selected image is already part of Raphael's protected Civitai cache.
    // Point the model cover directly at it instead of decoding/copying the entire
    // image into a second file. cover_source_image_id protects it from eviction.
    if !path_is_in_cache(&app.app_data, &source) || !source.is_file() {
        return Err(AppError::Invalid("That example image is outside Raphael's Civitai cache".into()));
    }

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET cover_path=?2,cover_source_image_id=?3,cover_position_x=50,cover_position_y=50,updated_at=?4 WHERE id=?1",
        params![id, source.to_string_lossy().to_string(), image_id, now()],
    )?;
    let rec = model_by_id(&c, id)?;
    drop(c);

    // A previous custom/stable cover is no longer referenced. Never delete a
    // gallery-backed cover here: its image remains valid cached content.
    if old_cover_source_id.is_none() {
        if let Some(old) = old_cover {
            let old_path = PathBuf::from(old);
            if old_path != source
                && path_is_in_cache(&app.app_data, &old_path)
                && old_path.is_file()
            {
                let _ = fs::remove_file(old_path);
            }
        }
    }

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
fn get_model_images(app: State<AppStateInner>, id: i64, limit: Option<i64>) -> AppResult<ModelImagesResponse> {
    let c = open_db(&app.app_data)?;
    let limit = limit.unwrap_or(20).clamp(1, 1000);
    let mut stmt = c.prepare(
        "SELECT id,civitai_image_id,local_path,thumbnail_path,width,height,prompt,negative_prompt,steps,cfg,sampler,seed,meta_json
         FROM images
         WHERE model_id=?1
         ORDER BY CASE WHEN meta_json LIKE '%\"featured\":true%' THEN 0 ELSE 1 END, id
         LIMIT ?2",
    )?;
    let rows = stmt.query_map(params![id, limit], |r| Ok(ModelImage {
        id: r.get(0)?,
        civitai_image_id: r.get(1)?,
        local_path: r.get(2)?,
        thumbnail_path: r.get(3)?,
        width: r.get(4)?,
        height: r.get(5)?,
        prompt: r.get(6)?,
        negative_prompt: r.get(7)?,
        steps: r.get(8)?,
        cfg: r.get(9)?,
        sampler: r.get(10)?,
        seed: r.get(11)?,
        meta_json: r.get(12)?,
    }))?;
    let images: Vec<ModelImage> = rows.filter_map(Result::ok).collect();
    Ok(ModelImagesResponse { has_more: images.len() as i64 >= limit, images })
}
#[tauri::command]
async fn preview_civitai_import_inner(app: &AppStateInner, url: String) -> AppResult<CivitaiImportPreview> {
    let (model, version) = fetch_model_and_version(app, &url).await?;
    let (dl, size, filename, sha256) = selected_file(&version)
        .ok_or_else(|| AppError::Api("No downloadable public file found for this version".into()))?;
    let root = app
        .models_root
        .read()
        .unwrap()
        .clone()
        .ok_or_else(|| AppError::Invalid("Choose your ComfyUI models folder first".into()))?;
    let typ = model.get("type").and_then(Value::as_str).unwrap_or("Other");
    let target = root.join(civitai_type_to_folder(typ));
    let activation = json_strings(version.get("trainedWords"));
    let mid = model.get("id").and_then(Value::as_i64);
    let thumb = match mid {
        Some(model_id) => {
            let _guard = app.cache_lock.lock().await;
            let result = ensure_model_thumbnail(app, model_id, &model, &version, &url).await?;
            let _ = enforce_cache_limit_inner(&app.app_data);
            result
        }
        None => None,
    };
    Ok(CivitaiImportPreview {
        model: json!({
            "id": model.get("id"),
            "name": model.get("name"),
            "type": typ,
            "description": model.get("description"),
            "tags": model.get("tags"),
            "creator": model.get("creator").and_then(|v| v.get("username")),
            "thumbnail_path": thumb
        }),
        version: json!({
            "id": version.get("id"),
            "name": version.get("name"),
            "base_model": version.get("baseModel"),
            "download_url": dl,
            "filename": filename,
            "size_bytes": size,
            "sha256": sha256,
            "activation_prompts": activation
        }),
        target_directory: target.to_string_lossy().to_string(),
        thumbnail_path: thumb,
        images_count_hint: version
            .get("images")
            .and_then(Value::as_array)
            .map(|x| x.len() as i64),
    })
}

#[tauri::command]
async fn preview_civitai_import(
    app: State<'_, AppStateInner>,
    url: String,
) -> AppResult<CivitaiImportPreview> {
    preview_civitai_import_inner(app.inner(), url).await
}

#[allow(clippy::too_many_arguments)]
async fn download_file(
    app: &AppStateInner,
    url: &str,
    target_dir: &Path,
    _preferred_name: &str,
    expected_sha256: Option<&str>,
    progress: Arc<Mutex<Vec<DownloadProgress>>>,
    task_id: &str,
    path: PathBuf,
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
    let response_total = res.content_length().map(|x| x as i64);
    set_download_progress(&progress, task_id, |p| {
        p.phase = "DOWNLOADING".into();
        p.total_bytes = response_total;
        p.percent = response_total.filter(|x| *x > 0).map(|_| 0.0);
        p.error = None;
    });

    let partial = path.with_extension(format!(
        "{}.part",
        path.extension().and_then(|x| x.to_str()).unwrap_or("bin")
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
        set_download_progress(&progress, task_id, |p| {
            p.downloaded_bytes = total;
            p.total_bytes = response_total.or(p.total_bytes);
            p.percent = p.total_bytes
                .filter(|x| *x > 0)
                .map(|x| (total as f64 / x as f64 * 100.0).clamp(0.0, 100.0));
        });
    }

    file.flush()?;
    set_download_progress(&progress, task_id, |p| {
        p.downloaded_bytes = total;
        p.percent = Some(100.0);
        p.phase = "VERIFYING".into();
    });
    let actual_sha256 = hex::encode(hasher.finalize());

    if let Some(expected) = expected_sha256 {
        if actual_sha256 != expected.to_ascii_lowercase() {
            let _ = fs::remove_file(&partial);
            return Err(AppError::Api(format!(
                "Downloaded file failed SHA256 verification: expected {expected}, got {actual_sha256}"
            )));
        }
    }

    if let Err(error) = fs::rename(&partial, &path) {
        let _ = fs::remove_file(&partial);
        return Err(AppError::Io(error));
    }
    Ok((path, total, actual_sha256))
}


#[tauri::command]
async fn install_civitai_model(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    url: String,
    target_directory: Option<String>,
    selected_type: Option<String>,
) -> AppResult<DownloadProgress> {
    let (model, version) = fetch_model_and_version(&app, &url).await?;
    let version_id = version.get("id").and_then(Value::as_i64);
    let task_id = new_download_id();

    if let Some(vid) = version_id {
        let c0 = open_db(&app.app_data)?;
        let existing_path: Result<String, rusqlite::Error> = c0.query_row(
            "SELECT path FROM models WHERE civitai_version_id=?1 AND path IS NOT NULL",
            [vid],
            |r| r.get::<_, String>(0),
        );
        if let Ok(existing_path) = existing_path {
            if !Path::new(&existing_path).is_file() {
                // The database can outlive a manually removed model file. In that
                // case allow a fresh download instead of falsely reporting it installed.
            } else {
                let existing_id: i64 = c0.query_row(
                    "SELECT id FROM models WHERE civitai_version_id=?1 AND path=?2",
                    params![vid, existing_path],
                    |r| r.get::<_, i64>(0),
                )?;
                let existing = model_by_id(&c0, existing_id)?;
            let progress = DownloadProgress {
                visible: true,
                task_id: Some(task_id.clone()),
                filename: existing.filename.clone(),
                phase: "ALREADY INSTALLED".into(),
                downloaded_bytes: existing.size_bytes,
                total_bytes: Some(existing.size_bytes),
                percent: Some(100.0),
                error: None,
            };
            push_download_progress(&app.downloads, progress.clone())?;
            let _ = handle.emit("download-progress", progress.clone());
            return Ok(progress);
            }
        }
    }

    let civitai_typ = model.get("type").and_then(Value::as_str).unwrap_or("Other");
    let typ = if let Some(selected) = selected_type {
        normalized_import_type(&selected)
            .ok_or_else(|| AppError::Invalid("Unsupported Raphael library tag".into()))?
            .to_string()
    } else {
        civitai_type_to_model_type(civitai_typ).to_string()
    };

    let (dl, file_size, filename, sha256) = selected_file(&version)
        .ok_or_else(|| AppError::Api("No downloadable public file found for this version".into()))?;

    let root = app.models_root.read().unwrap().clone()
        .ok_or_else(|| AppError::Invalid("Choose your ComfyUI models folder first".into()))?;

    let target = if let Some(custom) = target_directory {
        let candidate = PathBuf::from(custom);
        fs::create_dir_all(&candidate)?;
        let root_canonical = root.canonicalize()?;
        let target_canonical = candidate.canonicalize()?;
        if !target_canonical.starts_with(&root_canonical) {
            return Err(AppError::Invalid("Download folder must be inside the configured ComfyUI models folder".into()));
        }
        target_canonical
    } else {
        root.join(civitai_type_to_folder(&typ))
    };

    let safe_name = Path::new(&filename)
        .file_name()
        .and_then(|x| x.to_str())
        .filter(|x| !x.is_empty())
        .unwrap_or("model.safetensors")
        .to_string();
    let reserved_path = reserve_download_path(&app.active_download_paths, &target, &safe_name)?;

    let version_reserved = if let Some(vid) = version_id {
        let mut versions = app.active_download_versions
            .lock()
            .map_err(|_| AppError::Invalid("Download version state is unavailable".into()))?;
        if versions.contains(&vid) {
            release_download_path(&app.active_download_paths, &reserved_path);
            let progress = DownloadProgress {
                visible: true,
                task_id: Some(task_id.clone()),
                filename: safe_name.clone(),
                phase: "ALREADY QUEUED".into(),
                downloaded_bytes: 0,
                total_bytes: file_size,
                percent: Some(100.0),
                error: None,
            };
            push_download_progress(&app.downloads, progress.clone())?;
            let _ = handle.emit("download-progress", progress.clone());
            return Ok(progress);
        }
        versions.insert(vid);
        Some(vid)
    } else {
        None
    };

    let initial = DownloadProgress {
        visible: true,
        task_id: Some(task_id.clone()),
        filename: reserved_path.file_name().unwrap_or_default().to_string_lossy().to_string(),
        phase: "STARTING".into(),
        downloaded_bytes: 0,
        total_bytes: file_size,
        percent: file_size.filter(|x| *x > 0).map(|_| 0.0),
        error: None,
    };
    if let Err(error) = push_download_progress(&app.downloads, initial.clone()) {
        release_download_path(&app.active_download_paths, &reserved_path);
        if let Some(vid) = version_reserved {
            if let Ok(mut versions) = app.active_download_versions.lock() {
                versions.remove(&vid);
            }
        }
        return Err(error);
    }
    let _ = handle.emit("download-progress", initial.clone());

    let state = app.inner().clone();
    let task_progress = state.downloads.clone();
    let task_active_paths = state.active_download_paths.clone();
    let task_active_versions = state.active_download_versions.clone();
    let task_handle = handle.clone();
    tauri::async_runtime::spawn(async move {
        let result = async {
            set_download_progress(&task_progress, &task_id, |p| { p.phase = "QUEUED".into(); });
            let (path, size, hash) = {
                let _slot = acquire_download_slot(&state).await?;
                set_download_progress(&task_progress, &task_id, |p| { p.phase = "STARTING".into(); });
                download_file(
                &state,
                &dl,
                &target,
                &filename,
                sha256.as_deref(),
                task_progress.clone(),
                &task_id,
                reserved_path.clone(),
                ).await?
            };

            set_download_progress(&task_progress, &task_id, |p| {
                p.phase = "INSTALLING".into();
                p.downloaded_bytes = size;
                p.total_bytes = Some(size);
                p.percent = Some(100.0);
            });

            let rel = path.strip_prefix(&root).unwrap_or(&path).to_string_lossy().replace("\\", "/");
            let tags = json_strings(model.get("tags"));
            let activation = json_strings(version.get("trainedWords"));
            let desc = model.get("description").and_then(Value::as_str).map(strip_html);
            let creator = model.get("creator").and_then(|v| v.get("username")).and_then(Value::as_str).map(str::to_string);
            let vname = version.get("name").and_then(Value::as_str).map(str::to_string);
            let base = version.get("baseModel").and_then(Value::as_str).map(str::to_string);
            let mid = model.get("id").and_then(Value::as_i64);
            let vid = version.get("id").and_then(Value::as_i64);
            let civitai_url = canonical_civitai_url(&url, mid, vid)?;
            let thumb = match mid {
                Some(model_id) => {
                    let _guard=state.cache_lock.lock().await;
                    match ensure_model_thumbnail(&state, model_id, &model, &version, &url).await {
                        Ok(result) => {
                            let _=enforce_cache_limit_inner(&state.app_data);
                            result
                        },
                        Err(_) => None,
                    }
                },
                None => None
            };

            let rec = {
                let c = open_db(&state.app_data)?;
                c.execute(
                    "INSERT INTO models(
                        path,relative_path,filename,model_type,size_bytes,modified_at,
                        civitai_model_id,civitai_version_id,civitai_url,civitai_name,
                        version_name,base_model,creator,description,tags_json,
                        activation_json,source_hash,thumbnail_path,updated_at,downloaded_at
                     )
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,?14,?15,?16,?17,?18,?19,?20)
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
                        path.file_name().unwrap_or_default().to_string_lossy(),
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
                        now(),
                        now()
                    ],
                )?;
                let id = c.query_row("SELECT id FROM models WHERE path=?1", [path.to_string_lossy().to_string()], |r| r.get(0))?;
                model_by_id(&c, id)?
            };

            // The Registry is authoritative. The local SQLite row above is only
            // the physical-file projection needed by the Manager.
            let rec = sync_local_model_to_registry(&state, rec.id).await?;

            emit_models_changed(&task_handle);
            set_download_progress(&task_progress, &task_id, |p| {
                p.phase = "SYNCING GALLERY".into();
            });
            let _ = sync_gallery_inner(state.clone(), rec.id, task_handle.clone(), 20).await;
            Ok::<(), AppError>(())
        }.await;

        let final_progress = {
            let mut guard = match task_progress.lock() {
                Ok(value) => value,
                Err(_) => return,
            };
            let progress = match guard.iter_mut().find(|value| value.task_id.as_deref() == Some(task_id.as_str())) {
                Some(value) => value,
                None => return,
            };
            match result {
                Ok(()) => {
                    progress.phase = "COMPLETED".into();
                    progress.downloaded_bytes = progress.total_bytes.unwrap_or(progress.downloaded_bytes);
                    progress.percent = Some(100.0);
                    progress.error = None;
                }
                Err(error) => {
                    progress.phase = "FAILED".into();
                    progress.error = Some(error.to_string());
                }
            }
            progress.clone()
        };
        release_download_path(&task_active_paths, &reserved_path);
        if let Some(vid) = version_reserved {
            if let Ok(mut versions) = task_active_versions.lock() {
                versions.remove(&vid);
            }
        }
        let _ = task_handle.emit("download-progress", final_progress);
        emit_models_changed(&task_handle);
    });

    Ok(initial)
}

fn parse_meta(meta:&Value,key:&str)->Option<String>{meta.get(key).and_then(Value::as_str).map(str::to_string)}
async fn download_gallery_image(
    client: Client,
    remote: String,
    image_id: i64,
    cache: PathBuf,
) -> (i64, Option<PathBuf>) {
    if remote.is_empty() {
        return (image_id, None);
    }

    let response = match client.get(&remote).send().await {
        Ok(value) => value,
        Err(_) => return (image_id, None),
    };
    let response = match response.error_for_status() {
        Ok(value) => value,
        Err(_) => return (image_id, None),
    };
    let bytes = match response.bytes().await {
        Ok(value) if !value.is_empty() => value,
        _ => return (image_id, None),
    };

    let guessed_ext = remote
        .split('?')
        .next()
        .and_then(|x| Path::new(x).extension())
        .and_then(|x| x.to_str())
        .unwrap_or("jpg");

    let ext = detect_image_format_from_bytes(&bytes)
        .ok()
        .and_then(image_format_extension)
        .unwrap_or(guessed_ext);

    let local = cache.join(format!("{image_id}.{ext}"));
    if fs::write(&local, &bytes).is_err() {
        return (image_id, None);
    }

    let thumb = cache.join(format!("{image_id}_thumb.webp"));
    if !thumb.exists() {
        if let Ok(im) = decode_image_file(&local) {
            let t = im.resize(420, 420, FilterType::Triangle);
            let _ = t.save_with_format(&thumb, ImageFormat::WebP);
        }
    }

    (image_id, Some(local))
}

async fn sync_gallery_inner(
    app: AppStateInner,
    model_id: i64,
    handle: AppHandle,
    target_count: i64,
) -> AppResult<bool> {
    let _guard = app.cache_lock.lock().await;
    let target_count = target_count.clamp(1, 1000);
    let model = {
        let c = open_db(&app.app_data)?;
        model_by_id(&c, model_id)?
    };
    let civitai_id = match model.civitai_model_id {
        Some(x) => x,
        None => return Ok(false),
    };

    let cache = cache_root(&app.app_data).join(civitai_id.to_string());
    fs::create_dir_all(&cache)?;
    let client = civitai_client(&app)?;

    // Read all locally known community examples once. The old implementation
    // performed a separate SQLite query for every Civitai image.
    let existing: HashMap<i64, Option<PathBuf>> = {
        let c = open_db(&app.app_data)?;
        let mut stmt = c.prepare(
            "SELECT civitai_image_id,local_path
             FROM images
             WHERE model_id=?1
             AND meta_json NOT LIKE '%\"featured\":true%'",
        )?;
        let rows = stmt.query_map([model_id], |r| {
            Ok((
                r.get::<_, i64>(0)?,
                r.get::<_, Option<String>>(1)?,
            ))
        })?;
        rows.filter_map(Result::ok)
            .map(|(id,path)| (id, path.map(PathBuf::from)))
            .collect()
    };

    let mut cached_count = existing
        .values()
        .filter(|path| path.as_ref().is_some_and(|p| p.is_file()))
        .count() as i64;

    let mut cursor: Option<String> = None;
    let mut has_more = false;
    let mut first_page = true;

    loop {
        // Always fetch at least one page so an already-cached model can still
        // determine whether Civitai has additional community examples.
        if !first_page && cached_count >= target_count {
            break;
        }
        first_page = false;

        let mut url = format!("{API_BASE}/images?modelId={civitai_id}&limit=200&withMeta=true");
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
            return Err(AppError::Api(format!("Image API returned {}", res.status())));
        }

        let body: CivitaiEnvelope = res.json().await?;
        let page_len = body.items.len();

        // Collect only entries that actually need a local download.
        let mut jobs: Vec<(String, i64, Value)> = Vec::new();
        let mut available_new = 0i64;

        for img in body.items.iter() {
            if cached_count + available_new >= target_count {
                break;
            }

            let iid = match img.get("id").and_then(Value::as_i64) {
                Some(x) => x,
                None => continue,
            };

            if existing
                .get(&iid)
                .and_then(|p| p.as_ref())
                .is_some_and(|p| p.is_file())
            {
                continue;
            }

            let remote = match img
                .get("url")
                .and_then(Value::as_str)
                .map(str::trim)
                .filter(|url| !url.is_empty())
            {
                Some(value) => value.to_string(),
                None => continue,
            };

            jobs.push((remote, iid, img.clone()));
            available_new += 1;
        }

        let results = stream::iter(jobs.into_iter().map(|(remote, iid, img)| {
            let client = client.clone();
            let cache = cache.clone();
            async move {
                let (image_id, local_path) =
                    download_gallery_image(client, remote, iid, cache).await;
                (image_id, local_path, img)
            }
        }))
        .buffer_unordered(FEATURED_IMAGE_DOWNLOAD_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

        let mut successful: Vec<(i64, Option<String>, Option<String>, Value)> = Vec::new();

        for (iid, local_path, img) in results {
            if local_path.is_some() {
                cached_count += 1;
            }
            let thumb_path = local_path.as_ref().map(|path| {
                path.with_file_name(format!(
                    "{}_thumb.webp",
                    path.file_stem().and_then(|s| s.to_str()).unwrap_or("")
                ))
            });
            successful.push((
                iid,
                local_path.map(|p| p.to_string_lossy().to_string()),
                thumb_path
                    .filter(|p| p.is_file())
                    .map(|p| p.to_string_lossy().to_string()),
                img,
            ));
        }

        // One transaction for the whole page instead of one DB connection/query
        // per image.
        if !successful.is_empty() {
            let mut c = open_db(&app.app_data)?;
            let tx = c.transaction()?;

            for (iid, local_path, thumbnail_path, img) in successful {
                let meta = img.get("meta").cloned().unwrap_or(Value::Null);
                let prompt = parse_meta(&meta, "prompt");
                let neg = parse_meta(&meta, "negativePrompt")
                    .or_else(|| parse_meta(&meta, "Negative prompt"));
                let sampler =
                    parse_meta(&meta, "sampler").or_else(|| parse_meta(&meta, "Sampler"));
                let steps = meta.get("steps").and_then(Value::as_i64);
                let cfg = meta
                    .get("cfgScale")
                    .or_else(|| meta.get("cfg"))
                    .and_then(Value::as_f64);
                let seed = meta.get("seed").and_then(Value::as_i64);
                let width = img.get("width").and_then(Value::as_i64);
                let height = img.get("height").and_then(Value::as_i64);

                tx.execute(
                    "INSERT INTO images(
                        model_id,civitai_image_id,local_path,thumbnail_path,width,height,
                        prompt,negative_prompt,steps,cfg,sampler,seed,meta_json,cached_at
                     )
                     VALUES(?1,?2,?3,?4,?5,?6,?7,?8,?9,?10,?11,?12,?13,CAST(strftime('%s','now') AS INTEGER))
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
                        meta_json=CASE
                            WHEN images.meta_json LIKE '%\"featured\":true%' THEN images.meta_json
                            ELSE excluded.meta_json
                        END,
                        cached_at=excluded.cached_at",
                    params![
                        model_id,
                        iid,
                        local_path,
                        thumbnail_path,
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

            tx.commit()?;
        }

        let next_cursor = body
            .metadata
            .as_ref()
            .and_then(|m| m.get("nextCursor").and_then(Value::as_str).map(str::to_string));

        has_more = next_cursor.is_some();

        if cached_count >= target_count {
            break;
        }

        match next_cursor {
            Some(next) => cursor = Some(next),
            None if page_len >= 200 => {
                // Some API responses expose pagination through page size but
                // omit nextCursor. Re-fetching with the same request would loop,
                // so stop conservatively and expose LOAD MORE.
                break;
            }
            None => break,
        }
    }

    // Unlimited caches do not need a full filesystem scan after every
    // gallery request. Perform accounting/pruning only when a cache limit
    // is actually configured.
    let cache_limit = open_db(&app.app_data)
        .ok()
        .and_then(|c| read_cache_max_bytes(&c).ok())
        .unwrap_or(0);
    if cache_limit > 0 {
        if let Ok(c) = open_db(&app.app_data) {
            let bytes = dir_size(&cache_root(&app.app_data));
            let _ = put_setting(&c, "cache_bytes", &bytes.to_string());
        }
        let _ = enforce_cache_limit_inner(&app.app_data);
    }
    emit_models_changed(&handle);
    Ok(has_more)
}
#[tauri::command]
async fn sync_model_gallery(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    target_count: Option<i64>,
) -> AppResult<bool> {
    let target_count = target_count.unwrap_or(20).clamp(1, 200);
    sync_gallery_inner(app.inner().clone(), id, handle, target_count).await
}
#[tauri::command]
async fn load_more_model_examples(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    amount: Option<i64>,
) -> AppResult<bool> {
    let requested = amount.unwrap_or_else(|| {
        open_db(&app.app_data)
            .ok()
            .and_then(|c| read_example_load_amount(&c).ok())
            .unwrap_or(20)
    }).clamp(1, 100);
    let current_count = {
        let c = open_db(&app.app_data)?;
        c.query_row(
            "SELECT COUNT(*) FROM images WHERE model_id=?1 AND meta_json NOT LIKE '%\"featured\":true%'",
            [id],
            |r| r.get::<_, i64>(0),
        )?
    };
    sync_gallery_inner(
        app.inner().clone(),
        id,
        handle,
        current_count.saturating_add(requested).clamp(1, 1000),
    ).await
}

#[tauri::command]
fn get_example_load_amount(app: State<AppStateInner>) -> AppResult<i64> {
    let c = open_db(&app.app_data)?;
    read_example_load_amount(&c)
}

#[tauri::command]
fn set_example_load_amount(app: State<AppStateInner>, amount: i64) -> AppResult<i64> {
    let value = amount.clamp(1, 100);
    let c = open_db(&app.app_data)?;
    put_setting(&c, "example_load_amount", &value.to_string())?;
    Ok(value)
}

#[tauri::command]
async fn link_model_civitai_inner(
    app: &AppStateInner,
    handle:AppHandle,
    id:i64,
    url:String,
)->AppResult<ModelRecord>{
    let _guard=app.cache_lock.lock().await;
    let trimmed=url.trim();
    let (_mid,_vid)=model_id_and_version(trimmed)?;
    let (model,version)=fetch_model_and_version(app,trimmed).await?;
    let tags=json_strings(model.get("tags"));
    let activation=json_strings(version.get("trainedWords"));
    let desc=model.get("description").and_then(Value::as_str).map(strip_html);
    let creator=model.get("creator").and_then(|v|v.get("username")).and_then(Value::as_str).map(str::to_string);
    let mid=model.get("id").and_then(Value::as_i64);
    let vid=version.get("id").and_then(Value::as_i64);
    let canonical=canonical_civitai_url(trimmed,mid,vid)?;
    let thumbnail_path=match mid {
        Some(model_id)=>{
            let result=ensure_model_thumbnail(app,model_id,&model,&version,trimmed).await?;
            let _=enforce_cache_limit_inner(&app.app_data);
            result
        },
        None=>None
    };
    let (_registry_model_id, _registry_version_id) = apply_civitai_metadata_to_registry(
        app,
        id,
        &model,
        &version,
        &canonical,
        &tags,
        &activation,
        desc.as_deref(),
        creator.as_deref(),
    ).await?;

    let c=open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET civitai_model_id=?2,civitai_version_id=?3,civitai_url=?4,thumbnail_path=?5,updated_at=?6 WHERE id=?1",
        params![id,mid,vid,canonical,thumbnail_path,now()],
    )?;
    let rec=model_by_id(&c,id)?;
    drop(c);

    emit_models_changed(&handle);
    drop(_guard);
    Ok(rec)
}

#[tauri::command]
async fn link_model_civitai(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    url: String,
) -> AppResult<ModelRecord> {
    link_model_civitai_inner(app.inner(), handle, id, url).await
}
#[tauri::command]
async fn refresh_model_civitai_inner(
    app: &AppStateInner,
    handle: AppHandle,
    id: i64,
) -> AppResult<ModelRecord> {
    let _guard=app.cache_lock.lock().await;
    let current = {
        let c = open_db(&app.app_data)?;
        model_by_id(&c, id)?
    };

    let url = current
        .civitai_url
        .clone()
        .ok_or_else(|| AppError::Invalid("This model is not linked to Civitai".into()))?;

    let (model, version) = fetch_model_and_version(app, &url).await?;
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
    let thumbnail_path=match model.get("id").and_then(Value::as_i64) {
        Some(mid)=>{
            let result=ensure_model_thumbnail(app,mid,&model,&version,&url).await?;
            let _=enforce_cache_limit_inner(&app.app_data);
            result
        },
        None=>None
    };

    let canonical = canonical_civitai_url(
        &url,
        model.get("id").and_then(Value::as_i64),
        version.get("id").and_then(Value::as_i64),
    )?;

    let (_registry_model_id, _registry_version_id) = apply_civitai_metadata_to_registry(
        app,
        id,
        &model,
        &version,
        &canonical,
        &tags,
        &activation,
        desc.as_deref(),
        creator.as_deref(),
    ).await?;

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET thumbnail_path=?2,updated_at=?3 WHERE id=?1",
        params![id,thumbnail_path,now()],
    )?;
    let rec = model_by_id(&c,id)?;
    drop(c);

    emit_models_changed(&handle);
    drop(_guard);
    Ok(rec)
}
#[tauri::command]
async fn refresh_model_civitai(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
) -> AppResult<ModelRecord> {
    refresh_model_civitai_inner(app.inner(), handle, id).await
}
#[tauri::command]
fn refresh_all_examples(app: State<AppStateInner>, handle: AppHandle) -> AppResult<ExamplesRefreshProgress> {
    let initial = ExamplesRefreshProgress {
        current: 0, total: 0, model_id: None, model_name: None,
        version_current: 0, version_total: 0, images_saved: 0,
        status: "Starting featured example refresh".into(), done: false, error: None,
    };
    {
        let mut guard = app.examples_refresh_state.lock()
            .map_err(|_| AppError::Invalid("Featured example refresh state is unavailable".into()))?;
        if guard.running {
            return Err(AppError::Invalid("Featured example refresh is already running".into()));
        }
        guard.running = true;
        guard.progress = Some(initial.clone());
    }
    emit_examples_progress(&handle, initial.clone());

    let state = app.inner().clone();
    let refresh_state = state.examples_refresh_state.clone();
    tauri::async_runtime::spawn(async move {
        let models: Vec<(i64, String)> = match open_db(&state.app_data).and_then(|c| {
            let mut stmt = c.prepare(
                "SELECT id,COALESCE(civitai_name,filename) FROM models WHERE civitai_model_id IS NOT NULL ORDER BY id",
            )?;
            let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))?;
            Ok(rows.filter_map(Result::ok).collect())
        }) {
            Ok(v) => v,
            Err(e) => {
                store_examples_refresh_state(&refresh_state, &handle, ExamplesRefreshProgress {
                    current: 0, total: 0, model_id: None, model_name: None,
                    version_current: 0, version_total: 0, images_saved: 0,
                    status: "Could not read linked models".into(), done: true, error: Some(e.to_string()),
                });
                return;
            }
        };
        let total = models.len();
        store_examples_refresh_state(&refresh_state, &handle, ExamplesRefreshProgress {
            current: 0, total, model_id: None, model_name: None,
            version_current: 0, version_total: 0,
            images_saved: 0,
            status: if total == 0 { "No Civitai-linked models found".into() } else { "Starting featured example refresh".into() },
            done: total == 0, error: None,
        });
        if total == 0 { return; }

        let mut saved_total = 0usize;
        let mut completed = 0usize;
        let mut first_error: Option<String> = None;

        let results = stream::iter(models.iter().cloned().enumerate().map(|(index,(local_id,name))| {
            let state=state.clone();
            let handle=handle.clone();
            async move {
                store_examples_refresh_state(&state.examples_refresh_state,&handle,ExamplesRefreshProgress{
                    current:index,total,model_id:Some(local_id),model_name:Some(name.clone()),
                    version_current:0,version_total:0,images_saved:0,
                    status:format!("Refreshing {name}"),done:false,error:None,
                });
                let result=sync_featured_examples_inner(state,local_id,handle,Some((index,total)),false,false).await;
                (local_id,name,result)
            }
        }))
        .buffer_unordered(FEATURED_MODEL_CONCURRENCY)
        .collect::<Vec<_>>()
        .await;

        for (local_id,name,result) in results {
            completed+=1;
            match result {
                Ok(saved) => {
                    saved_total+=saved;
                    store_examples_refresh_state(&refresh_state,&handle,ExamplesRefreshProgress{
                        current:completed,total,model_id:Some(local_id),model_name:Some(name.clone()),
                        version_current:0,version_total:0,images_saved:saved_total,
                        status:format!("Finished {name} ({saved} examples)"),done:false,error:None,
                    });
                }
                Err(e) => {
                    let msg=e.to_string();
                    if first_error.is_none(){first_error=Some(msg.clone());}
                    store_examples_refresh_state(&refresh_state,&handle,ExamplesRefreshProgress{
                        current:completed,total,model_id:Some(local_id),model_name:Some(name.clone()),
                        version_current:0,version_total:0,images_saved:saved_total,
                        status:format!("Failed {name}"),done:false,error:Some(msg),
                    });
                }
            }
        }

        {
            let _guard=state.cache_lock.lock().await;
            let bytes=dir_size(&cache_root(&state.app_data));
            if let Ok(c)=open_db(&state.app_data){let _=put_setting(&c,"cache_bytes",&bytes.to_string());}
            let _=enforce_cache_limit_inner(&state.app_data);
        }
        store_examples_refresh_state(&refresh_state, &handle, ExamplesRefreshProgress {
            current: total, total, model_id: None, model_name: None,
            version_current: 0, version_total: 0, images_saved: saved_total,
            status: if first_error.is_some() {
                format!("Finished refreshing {total} model{} with errors", if total == 1 { "" } else { "s" })
            } else {
                format!("Finished refreshing {total} model{}", if total == 1 { "" } else { "s" })
            },
            done: true, error: first_error,
        });
        emit_models_changed(&handle);
    });
    Ok(initial)
}

#[tauri::command]
fn get_examples_refresh_status(app: State<AppStateInner>) -> Option<ExamplesRefreshProgress> {
    get_examples_refresh_state(&app.examples_refresh_state)
}

#[tauri::command]
fn get_download_progress(app: State<AppStateInner>) -> Vec<DownloadProgress> {
    app.downloads.lock().map(|g| g.clone()).unwrap_or_default()
}

#[tauri::command]
fn clear_download_progress(app: State<AppStateInner>, task_id: String) -> AppResult<()> {
    remove_download_progress(&app.downloads, &task_id)
}

fn storage_stats_inner(app_data:&Path)->AppResult<StorageStats>{let c=open_db(app_data)?;let total:i64=c.query_row("SELECT COALESCE(SUM(size_bytes),0) FROM models",[],|r|r.get(0))?;let cached=dir_size(&cache_root(app_data));let _=put_setting(&c,"cache_bytes",&cached.to_string());let mut stmt=c.prepare("SELECT model_type,COUNT(*),COALESCE(SUM(size_bytes),0) FROM models GROUP BY model_type ORDER BY model_type")?;let categories=stmt.query_map([],|r|Ok(CategoryStats{r#type:r.get(0)?,count:r.get(1)?,bytes:r.get(2)?}))?.filter_map(Result::ok).collect();Ok(StorageStats{total_model_bytes:total,cached_bytes:cached,categories})}
fn dir_size(path:&Path)->i64{if !path.exists(){return 0} WalkDir::new(path).into_iter().filter_map(Result::ok).filter_map(|e|e.metadata().ok()).filter(|m|m.is_file()).map(|m|m.len() as i64).sum()}
fn path_is_in_cache(app_data:&Path,path:&Path)->bool{
    let root=cache_root(app_data).canonicalize().unwrap_or_else(|_|cache_root(app_data));
    let candidate=path.canonicalize().unwrap_or_else(|_|path.to_path_buf());
    candidate.starts_with(root)
}
#[tauri::command]
fn get_storage_stats(app:State<AppStateInner>)->AppResult<StorageStats>{storage_stats_inner(&app.app_data)}
#[tauri::command]
fn get_cache_stats(app:State<AppStateInner>)->AppResult<CacheStats>{cache_stats_inner(&app.app_data)}
#[tauri::command]
async fn set_cache_max_bytes(app:State<'_, AppStateInner>, max_bytes:i64)->AppResult<CacheStats>{
    let _guard=app.cache_lock.lock().await;
    set_cache_max_bytes_inner(&app.app_data,max_bytes)
}
#[tauri::command]
async fn set_cache_location(app:State<'_, AppStateInner>, path:String)->AppResult<CacheStats>{
    let _guard=app.cache_lock.lock().await;
    set_cache_location_inner(&app.app_data,&path)
}
#[tauri::command]
async fn clear_cache_images(app:State<'_, AppStateInner>)->AppResult<CacheOperationResult>{
    let _guard=app.cache_lock.lock().await;
    clear_cache_images_inner(&app.app_data)
}
#[tauri::command]
async fn clear_complete_cache(app:State<'_, AppStateInner>)->AppResult<CacheOperationResult>{
    let _guard=app.cache_lock.lock().await;
    clear_complete_cache_inner(&app.app_data)
}
#[tauri::command]
async fn prune_cache_images(app:State<'_, AppStateInner>, keep_per_model:i64)->AppResult<CacheOperationResult>{
    let _guard=app.cache_lock.lock().await;
    prune_cache_images_inner(&app.app_data,keep_per_model)
}
#[tauri::command]
async fn clean_cache_orphans(app:State<'_, AppStateInner>)->AppResult<CacheOperationResult>{
    let _guard=app.cache_lock.lock().await;
    clean_cache_orphans_inner(&app.app_data)
}
#[tauri::command]
fn open_in_file_manager(path:String)->AppResult<()>{let p=PathBuf::from(path); let target=if p.is_file(){p.parent().unwrap_or(&p).to_path_buf()}else{p}; #[cfg(target_os="windows")] {std::process::Command::new("explorer").arg(target).spawn()?;} #[cfg(target_os="macos")] {std::process::Command::new("open").arg(target).spawn()?;} #[cfg(target_os="linux")] {std::process::Command::new("xdg-open").arg(target).spawn()?;} Ok(())}
#[tauri::command]
fn get_parallel_downloads(app: State<AppStateInner>) -> AppResult<i64> {
    let value = app.parallel_downloads.lock().map_err(|_| AppError::Invalid("Parallel download setting is unavailable".into()))?;
    Ok(*value as i64)
}

#[tauri::command]
fn set_parallel_downloads(app: State<AppStateInner>, value: i64) -> AppResult<i64> {
    let next = clamp_parallel_downloads(value);
    let c = open_db(&app.app_data)?;
    put_setting(&c, "parallel_downloads", &next.to_string())?;
    *app.parallel_downloads.lock().map_err(|_| AppError::Invalid("Parallel download setting is unavailable".into()))? = next as usize;
    Ok(next)
}

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

            // Hash enrichment is a discovery step; the Registry still receives
            // the authoritative model/version/file update through the same API.
            let _ = sync_local_model_to_registry(&app, id).await;
            emit_models_changed(&handle);
        }
    });
}


fn spawn_registry_event_sync(app: AppStateInner) {
    tauri::async_runtime::spawn(async move {
        loop {
            let cursor = match app.registry_event_cursor.lock() {
                Ok(value) => *value,
                Err(_) => 0,
            };

            match app.registry.events(cursor).await {
                Ok(events) => {
                    let mut next_cursor = cursor;

                    for event in events {
                        next_cursor = next_cursor.max(event.id);
                        let Some(registry_model_id) = event.model_id.as_deref() else {
                            continue;
                        };

                        let local_ids: Vec<i64> = match open_db(&app.app_data).and_then(|c| {
                            let mut stmt = c.prepare(
                                "SELECT id FROM models WHERE registry_model_id=?1"
                            )?;
                            let rows = stmt.query_map([registry_model_id], |r| r.get::<_, i64>(0))?;
                            Ok(rows.filter_map(Result::ok).collect())
                        }) {
                            Ok(ids) => ids,
                            Err(_) => Vec::new(),
                        };

                        for local_id in local_ids {
                            let _ = hydrate_local_model_from_registry(
                                &app,
                                local_id,
                                registry_model_id,
                                None,
                            ).await;
                        }
                    }

                    if let Ok(mut value) = app.registry_event_cursor.lock() {
                        *value = next_cursor;
                    }
                }
                Err(_) => {}
            }

            tokio::time::sleep(Duration::from_secs(2)).await;
        }
    });
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .manage(web::WebServerController::default())
        .plugin(tauri_plugin_dialog::init())
        .setup(|app| {
            let app_data=app.path().app_data_dir()?;fs::create_dir_all(&app_data)?;let c=open_db(&app_data)?;let saved=setting(&c,"models_root")?;let parallel=read_parallel_downloads(&c)?;let registry=RegistryClient::from_app_data(&app_data).map_err(|e|io::Error::other(e.to_string()))?;
            let state=AppStateInner{app_data:app_data.clone(),models_root:Arc::new(RwLock::new(saved.map(PathBuf::from))),watcher:Arc::new(Mutex::new(None)),scan_lock:Arc::new(Mutex::new(())),
                downloads:Arc::new(Mutex::new(Vec::new())),
                active_download_paths:Arc::new(Mutex::new(HashSet::new())),
                active_download_versions:Arc::new(Mutex::new(HashSet::new())),
                active_downloads:Arc::new(Mutex::new(0)),
                parallel_downloads:Arc::new(Mutex::new(parallel)),
                examples_refresh_state: Arc::new(Mutex::new(ExamplesRefreshState::default())),
                cache_lock: Arc::new(AsyncMutex::new(())),
                registry,
                registry_sync_running: Arc::new(Mutex::new(false)),
                registry_event_cursor: Arc::new(Mutex::new(0)),
            };app.manage(state.clone());
            if let Some(root)=state.models_root.read().unwrap().clone(){ if root.is_dir(){let removed=scan_root(&state,&root).unwrap_or_default(); if !removed.is_empty(){let app_state=state.clone();tauri::async_runtime::spawn(async move{reconcile_removed_registry_files(&app_state,removed).await;});} let handle=app.handle().clone();spawn_registry_sync(state.clone(),handle.clone());spawn_registry_event_sync(state.clone());spawn_hash_enrichment(state.clone(),handle.clone());let state2=state.clone();let handle2=handle.clone();if let Ok(mut watcher)=notify::recommended_watcher(move |res:Result<notify::Event,notify::Error>|{if let Ok(e)=res{match e.kind{EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)=>{std::thread::sleep(Duration::from_millis(120));recursive_scan_and_emit(state2.clone(),handle2.clone());},_=>{}}}}){if watcher.watch(&root,RecursiveMode::Recursive).is_ok(){*state.watcher.lock().unwrap()=Some(watcher)}}}}
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![get_app_state,set_models_root,list_models,get_tags,add_subfolder_tags,set_model_tags,set_model_type,set_model_cover_position,set_model_cover_from_image,set_model_custom_cover,reset_model_cover,delete_model,get_library_counts,get_model_images,sync_model_gallery,refresh_all_examples,get_examples_refresh_status,preview_civitai_import,install_civitai_model,get_download_progress,clear_download_progress,get_parallel_downloads,set_parallel_downloads,link_model_civitai,refresh_model_civitai,get_storage_stats,get_cache_stats,set_cache_max_bytes,set_cache_location,clear_cache_images,clear_complete_cache,prune_cache_images,clean_cache_orphans,get_example_load_amount,set_example_load_amount,load_more_model_examples,open_in_file_manager,set_civitai_token,is_civitai_token_set,web::get_web_app_status,web::toggle_web_app])
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
            downloads: Arc::new(Mutex::new(Vec::new())),
            active_download_paths: Arc::new(Mutex::new(HashSet::new())),
            active_download_versions: Arc::new(Mutex::new(HashSet::new())),
            active_downloads: Arc::new(Mutex::new(0)),
            parallel_downloads: Arc::new(Mutex::new(3)),
            examples_refresh_state: Arc::new(Mutex::new(ExamplesRefreshState::default())),
            cache_lock: Arc::new(AsyncMutex::new(())),
            registry: RegistryClient::from_app_data(&app_data).unwrap(),
            registry_sync_running: Arc::new(Mutex::new(false)),
            registry_event_cursor: Arc::new(Mutex::new(0)),
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
            source_hash: None, thumbnail_path: None, cover_path: None, cover_source_image_id: None, cover_position_x: 50.0, cover_position_y: 50.0,
            downloaded_at: 0, updated_at: 0,
        };
        assert!(model_search_match(&model, "tag:anime", &[]));
        assert!(model_search_match(&model, "#megumin", &[]));
        assert!(model_search_match(&model, "-tag:realistic", &[]));
        assert!(!model_search_match(&model, "-tag:anime", &[]));
        assert!(model_search_match(&model, "magic", &[]));
    }

    #[test]
    fn set_model_tags_are_persistent() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let root = temp.path().join("models");
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("checkpoints/test.safetensors");
        fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        fs::write(&model_path, b"checkpoint").unwrap();

        let state = test_state(app_data.clone(), root.clone());
        scan_root(&state, &root).unwrap();

        let id: i64 = {
            let db = open_db(&app_data).unwrap();
            db.query_row("SELECT id FROM models LIMIT 1", [], |r| r.get(0)).unwrap()
        };

        {
            let db = open_db(&app_data).unwrap();
            let changed = db.execute(
                "UPDATE models SET tags_json=?2,tags_user_modified=1,updated_at=?3 WHERE id=?1",
                params![id, serde_json::to_string(&vec!["Anime", "Megumin"]).unwrap(), now()],
            ).unwrap();
            assert_eq!(changed, 1);
        }

        let db = open_db(&app_data).unwrap();
        let stored: String = db
            .query_row("SELECT tags_json FROM models WHERE id=?1", [id], |r| r.get(0))
            .unwrap();
        let tags: Vec<String> = serde_json::from_str(&stored).unwrap();
        assert_eq!(tags, vec!["Anime", "Megumin"]);
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
    fn incomplete_scan_does_not_prune_existing_models() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let root = temp.path().join("models");
        fs::create_dir_all(&root).unwrap();
        let model_path = root.join("upscale_models/example.safetensors");
        fs::create_dir_all(model_path.parent().unwrap()).unwrap();
        fs::write(&model_path, b"upscaler").unwrap();

        let state = test_state(app_data.clone(), root.clone());
        scan_root(&state, &root).unwrap();

        let db = open_db(&app_data).unwrap();
        let id: i64 = db.query_row("SELECT id FROM models LIMIT 1", [], |r| r.get(0)).unwrap();

        db.execute(
            "DELETE FROM models WHERE id=?1 AND 1=0",
            [id],
        ).unwrap();

        let count_before: i64 = db.query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0)).unwrap();
        assert_eq!(count_before, 1);

        prune_unseen_models(&db, &HashSet::new(), false).unwrap();

        let count_after: i64 = db.query_row("SELECT COUNT(*) FROM models", [], |r| r.get(0)).unwrap();
        assert_eq!(count_after, 1);
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
    fn featured_refresh_state_starts_idle() {
        let state = ExamplesRefreshState::default();
        assert!(!state.running);
        assert!(state.progress.is_none());
    }

    #[test]
    fn featured_version_selection_skips_empty_versions() {
        let mut versions = vec![
            json!({"id": 3, "createdAt": "2026-03-03T00:00:00Z", "images": []}),
            json!({"id": 2, "createdAt": "2026-03-02T00:00:00Z", "images": [{"url": ""}]}),
            json!({"id": 1, "createdAt": "2026-03-01T00:00:00Z", "images": [{"url": "https://example.com/a.jpg"}]}),
        ];
        versions.sort_by(|a,b| {
            let ac=a.get("createdAt").and_then(Value::as_str).unwrap_or("");
            let bc=b.get("createdAt").and_then(Value::as_str).unwrap_or("");
            bc.cmp(ac)
        });
        versions.retain(|version| {
            version.get("images").and_then(Value::as_array).map(|images| {
                images.iter().any(|image| {
                    image.get("url").and_then(Value::as_str).map(|url| !url.trim().is_empty()).unwrap_or(false)
                })
            }).unwrap_or(false)
        });
        assert_eq!(versions.len(), 1);
        assert_eq!(versions[0].get("id").and_then(Value::as_i64), Some(1));
    }

    #[test]
    fn featured_image_key_works_without_civitai_image_id() {
        let image = json!({
            "url": "https://imagecache.civitai.com/example/width=450",
            "width": 832,
            "height": 832
        });
        let key = featured_image_key(&image, 8840, 0).unwrap();
        assert!(key < 0);
        assert_eq!(featured_image_key(&image, 8840, 0), Some(key));
        assert_ne!(featured_image_key(&image, 8840, 1), Some(key));
    }

    #[test]
    fn featured_image_id_accepts_string_ids() {
        let image = json!({"id": "123456789"});
        let id = image.get("id").and_then(|value| {
            value.as_i64().or_else(|| value.as_str().and_then(|value| value.parse::<i64>().ok()))
        });
        assert_eq!(id, Some(123456789));
    }

    #[test]
    fn cached_featured_cover_is_copied_to_stable_cover_storage() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let model_cache = app_data.join("cache").join("civitai").join("123").join("featured").join("456");
        fs::create_dir_all(&model_cache).unwrap();
        let source = model_cache.join("789.jpg");
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 0]));
        image.save(&source).unwrap();

        let state = test_state(app_data.clone(), temp.path().join("models"));
        let copied = copy_cached_cover(&state, 42, &source).unwrap();
        assert!(copied.starts_with(app_data.join("cache").join("covers")));
        assert!(copied.is_file());
        assert_ne!(copied, source);

        let copied_image = image::open(copied).unwrap();
        assert_eq!(copied_image.width(), 2);
        assert_eq!(copied_image.height(), 2);
    }

    #[test]
    fn cached_featured_cover_uses_actual_image_format_not_filename_extension() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let model_cache = app_data.join("cache").join("civitai").join("123").join("featured").join("456");
        fs::create_dir_all(&model_cache).unwrap();
        let source = model_cache.join("789.jpg");
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb([255, 0, 0]));
        image.save_with_format(&source, ImageFormat::Png).unwrap();

        let state = test_state(app_data.clone(), temp.path().join("models"));
        let copied = copy_cached_cover(&state, 43, &source).unwrap();
        assert_eq!(copied.extension().and_then(|x| x.to_str()), Some("png"));
        let copied_image = decode_image_file(&copied).unwrap();
        assert_eq!(copied_image.width(), 2);
        assert_eq!(copied_image.height(), 2);
    }

    #[test]
    fn cache_relocation_updates_files_and_database_paths() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let models_root = temp.path().join("models");
        fs::create_dir_all(&models_root).unwrap();

        let old_file = app_data.join("cache/civitai/123/456.jpg");
        fs::create_dir_all(old_file.parent().unwrap()).unwrap();
        fs::write(&old_file, b"cached-image").unwrap();

        let db = open_db(&app_data).unwrap();
        db.execute(
            "INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,updated_at)
             VALUES('/models/example.safetensors','example.safetensors','example.safetensors','Other',1,0,0)",
            [],
        ).unwrap();
        let model_id = db.last_insert_rowid();
        db.execute(
            "INSERT INTO images(model_id,civitai_image_id,local_path,cached_at)
             VALUES(?1,456,?2,1)",
            params![model_id, old_file.to_string_lossy().to_string()],
        ).unwrap();

        let target = temp.path().join("relocated-cache");
        let stats = set_cache_location_inner(&app_data, &target.to_string_lossy()).unwrap();
        assert_eq!(
            PathBuf::from(stats.location).canonicalize().unwrap(),
            target.canonicalize().unwrap(),
        );
        let new_file = target.join("civitai/123/456.jpg");
        assert!(new_file.is_file());
        assert!(!old_file.exists());

        let db = open_db(&app_data).unwrap();
        let stored: String = db
            .query_row("SELECT local_path FROM images WHERE id=1", [], |r| r.get(0))
            .unwrap();
        assert_eq!(PathBuf::from(stored), new_file);
    }

    #[test]
    fn cache_limit_evicts_oldest_unprotected_image() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let models_root = temp.path().join("models");
        fs::create_dir_all(&models_root).unwrap();

        let old_file = app_data.join("cache/civitai/1/old.jpg");
        let protected_file = app_data.join("cache/civitai/1/protected.jpg");
        fs::create_dir_all(old_file.parent().unwrap()).unwrap();
        fs::write(&old_file, b"1234567890").unwrap();
        fs::write(&protected_file, b"abcdefghij").unwrap();

        let db = open_db(&app_data).unwrap();
        db.execute(
            "INSERT INTO models(path,relative_path,filename,model_type,size_bytes,modified_at,updated_at)
             VALUES('/models/example.safetensors','example.safetensors','example.safetensors','Other',1,0,0)",
            [],
        ).unwrap();
        let model_id = db.last_insert_rowid();
        db.execute(
            "INSERT INTO images(model_id,civitai_image_id,local_path,cached_at)
             VALUES(?1,100,?2,1)",
            params![model_id, old_file.to_string_lossy().to_string()],
        ).unwrap();
        let old_id = db.last_insert_rowid();
        db.execute(
            "INSERT INTO images(model_id,civitai_image_id,local_path,cached_at)
             VALUES(?1,101,?2,2)",
            params![model_id, protected_file.to_string_lossy().to_string()],
        ).unwrap();
        let protected_id = db.last_insert_rowid();
        db.execute(
            "UPDATE models SET cover_source_image_id=?2 WHERE id=?1",
            params![model_id, protected_id],
        ).unwrap();

        put_setting(&db, "cache_max_bytes", "15").unwrap();
        drop(db);

        let result = enforce_cache_limit_inner(&app_data).unwrap();
        assert!(!old_file.exists());
        assert!(protected_file.is_file());
        assert!(result.remaining_bytes <= 15);
        let db = open_db(&app_data).unwrap();
        let old_exists: i64 = db
            .query_row("SELECT COUNT(*) FROM images WHERE id=?1", [old_id], |r| r.get(0))
            .unwrap();
        let protected_exists: i64 = db
            .query_row("SELECT COUNT(*) FROM images WHERE id=?1", [protected_id], |r| r.get(0))
            .unwrap();
        assert_eq!(old_exists, 0);
        assert_eq!(protected_exists, 1);
    }

    #[test]
    fn custom_cover_uses_configured_cache_location() {
        let temp = tempfile::tempdir().unwrap();
        let app_data = temp.path().join("app");
        let models_root = temp.path().join("models");
        fs::create_dir_all(&models_root).unwrap();
        let target = temp.path().join("relocated-cache");
        fs::create_dir_all(&target).unwrap();

        let db = open_db(&app_data).unwrap();
        put_setting(&db, "cache_location", &target.to_string_lossy()).unwrap();
        drop(db);

        let source = temp.path().join("source.png");
        let image = image::RgbImage::from_pixel(2, 2, image::Rgb([0, 255, 0]));
        image.save(&source).unwrap();
        let existing = target.join("covers/model_7.png");
        fs::create_dir_all(existing.parent().unwrap()).unwrap();
        fs::write(&existing, b"old-cover").unwrap();

        let state = test_state(app_data.clone(), models_root);
        let copied = copy_custom_cover(&state, 7, &source).unwrap();
        assert!(copied.starts_with(target.join("covers")));
        assert!(copied.is_file());
        assert_ne!(copied, existing);
        assert_eq!(fs::read(&existing).unwrap(), b"old-cover");
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

        let cover_path_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_path'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cover_path_column, 1);

        let cover_x_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_position_x'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cover_x_column, 1);

        let cover_y_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_position_y'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cover_y_column, 1);

        let cover_source_image_column: i64 = second
            .query_row(
                "SELECT COUNT(*) FROM pragma_table_info('models') WHERE name='cover_source_image_id'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(cover_source_image_column, 1);
    }
}
