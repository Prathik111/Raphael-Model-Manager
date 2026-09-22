use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Path as AxumPath, Query, State as AxumState},
    http::{header, HeaderValue, Request, StatusCode},
    middleware,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::HashMap,
    future::Future,
    net::{IpAddr, SocketAddr, UdpSocket},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, Mutex, RwLock,
    },
    time::{Duration, Instant},
};
use tauri::{path::BaseDirectory, AppHandle, Manager, State};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
    net::TcpListener,
    sync::oneshot,
};
use tower_http::services::ServeDir;

use crate::{
    add_subfolder_tags, clear_download_progress, delete_model, delete_model_inner, get_app_state, get_download_progress, get_library_counts, get_model_images, get_storage_stats,
    get_parallel_downloads, set_parallel_downloads,
    get_cache_stats, set_cache_max_bytes, set_cache_max_bytes_inner, set_cache_location, set_cache_location_inner,
    clear_cache_images, clear_cache_images_inner, clear_complete_cache, clear_complete_cache_inner,
    prune_cache_images, prune_cache_images_inner, clean_cache_orphans, clean_cache_orphans_inner,
    get_tags, install_civitai_model, link_model_civitai, link_model_civitai_inner, list_models, preview_civitai_import, preview_civitai_import_inner,
    refresh_all_examples, get_examples_refresh_status, load_more_model_examples, get_example_load_amount, set_example_load_amount, refresh_model_civitai, refresh_model_civitai_inner, reset_model_cover, set_civitai_token, set_model_cover_position, set_model_cover_from_image,
    set_model_tags, set_model_name, set_model_description, refetch_all_model_tags, refetch_all_model_tags_inner, set_model_type, sync_model_gallery, sync_gallery_inner,
    is_civitai_token_set, open_db, read_example_load_amount, AppError, AppResult, CivitaiImportPreview, ModelRecord,
};

pub const WEB_PORT: u16 = 1421;
const WEB_TASK_TTL: Duration = Duration::from_secs(15 * 60);
const WEB_TASK_MAX_AGE: Duration = Duration::from_secs(2 * 60 * 60);
static WEB_TASK_COUNTER: AtomicU64 = AtomicU64::new(1);

#[derive(Debug, Serialize, Clone)]
struct WebTaskStatus {
    task_id: String,
    state: String,
    result: Option<Value>,
    error: Option<String>,
}

struct WebTaskRecord {
    status: WebTaskStatus,
    updated_at: Instant,
}

#[derive(Clone, Default)]
struct WebTaskStore {
    inner: Arc<Mutex<HashMap<String, WebTaskRecord>>>,
}

impl WebTaskStore {
    fn prune(&self) {
        if let Ok(mut guard) = self.inner.lock() {
            guard.retain(|_, record| {
                if matches!(record.status.state.as_str(), "completed" | "failed") {
                    record.updated_at.elapsed() < WEB_TASK_TTL
                } else {
                    record.updated_at.elapsed() < WEB_TASK_MAX_AGE
                }
            });
        }
    }

    fn start(
        &self,
        operation: std::pin::Pin<Box<dyn Future<Output = AppResult<Value>> + Send>>,
    ) -> String {
        self.prune();
        let task_id = format!("web-{}", WEB_TASK_COUNTER.fetch_add(1, Ordering::Relaxed));
        let status = WebTaskStatus {
            task_id: task_id.clone(),
            state: "queued".into(),
            result: None,
            error: None,
        };
        if let Ok(mut guard) = self.inner.lock() {
            guard.insert(
                task_id.clone(),
                WebTaskRecord {
                    status,
                    updated_at: Instant::now(),
                },
            );
        }

        let store = self.clone();
        let id = task_id.clone();
        tauri::async_runtime::spawn(async move {
            if let Ok(mut guard) = store.inner.lock() {
                if let Some(record) = guard.get_mut(&id) {
                    record.status.state = "running".into();
                    record.updated_at = Instant::now();
                }
            }

            let result = operation.await;

            if let Ok(mut guard) = store.inner.lock() {
                if let Some(record) = guard.get_mut(&id) {
                    record.updated_at = Instant::now();
                    match result {
                        Ok(value) => {
                            record.status.state = "completed".into();
                            record.status.result = Some(value);
                            record.status.error = None;
                        }
                        Err(error) => {
                            record.status.state = "failed".into();
                            record.status.result = None;
                            record.status.error = Some(error.to_string());
                        }
                    }
                }
            }
        });

        task_id
    }

    fn get(&self, task_id: &str) -> Option<WebTaskStatus> {
        self.prune();
        self.inner
            .lock()
            .ok()
            .and_then(|guard| guard.get(task_id).map(|record| record.status.clone()))
    }
}

#[derive(Debug, Serialize, Clone)]
pub struct WebAppStatus {
    pub enabled: bool,
    pub url: Option<String>,
    pub port: u16,
}

#[derive(Clone, Default)]
pub struct WebServerController {
    inner: Arc<WebServerControllerInner>,
}

struct WebServerControllerInner {
    shutdown: Mutex<Option<oneshot::Sender<()>>>,
    enabled: RwLock<bool>,
    url: RwLock<Option<String>>,
    generation: AtomicU64,
    tasks: WebTaskStore,
}

impl Default for WebServerControllerInner {
    fn default() -> Self {
        Self {
            shutdown: Mutex::new(None),
            enabled: RwLock::new(false),
            url: RwLock::new(None),
            generation: AtomicU64::new(0),
            tasks: WebTaskStore::default(),
        }
    }
}

impl WebServerController {
    pub fn status(&self) -> WebAppStatus {
        WebAppStatus {
            enabled: *self.inner.enabled.read().unwrap(),
            url: self.inner.url.read().unwrap().clone(),
            port: WEB_PORT,
        }
    }
}

#[derive(Clone)]
struct WebServerState {
    handle: AppHandle,
    controller: WebServerController,
}

#[derive(Debug, Deserialize)]
struct ListModelsArgs {
    #[serde(rename = "type")]
    r#type: Option<String>,
    query: Option<String>,
    tags: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct IdArgs {
    id: i64,
}

#[derive(Debug, Deserialize)]
struct ModelImagesArgs {
    id: i64,
    limit: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct SyncGalleryArgs {
    id: i64,
    #[serde(rename = "targetCount")]
    target_count: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct LoadMoreArgs {
    id: i64,
    amount: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ExampleLoadAmountArgs {
    amount: Option<i64>,
}

#[derive(Debug, Deserialize)]
struct ParallelDownloadsArgs {
    value: i64,
}

#[derive(Debug, Deserialize)]
struct CacheMaxBytesArgs {
    #[serde(rename = "maxBytes")]
    max_bytes: i64,
}

#[derive(Debug, Deserialize)]
struct CacheLocationArgs {
    path: String,
}

#[derive(Debug, Deserialize)]
struct KeepImagesArgs {
    #[serde(rename = "keepPerModel")]
    keep_per_model: i64,
}

#[derive(Debug, Deserialize)]
struct TagsArgs {
    id: i64,
    tags: Vec<String>,
}

#[derive(Debug, Deserialize)]
struct NameArgs {
    id: i64,
    name: String,
}

#[derive(Debug, Deserialize)]
struct DescriptionArgs {
    id: i64,
    description: String,
}

#[derive(Debug, Deserialize)]
struct TypeArgs {
    id: i64,
    #[serde(rename = "modelType")]
    model_type: String,
}

#[derive(Debug, Deserialize)]
struct CoverPositionArgs {
    id: i64,
    x: f64,
    y: f64,
}

#[derive(Debug, Deserialize)]
struct ImageArgs {
    id: i64,
    #[serde(rename = "imageId")]
    image_id: i64,
}

#[derive(Debug, Deserialize)]
struct UrlArgs {
    url: String,
}

#[derive(Debug, Deserialize)]
struct InstallArgs {
    url: String,
    #[serde(rename = "targetDirectory")]
    target_directory: Option<String>,
    #[serde(rename = "selectedType")]
    selected_type: Option<String>,
}

#[derive(Debug, Deserialize)]
struct LinkArgs {
    id: i64,
    url: String,
}

#[derive(Debug, Deserialize)]
struct TokenArgs {
    token: String,
}

#[derive(Debug, Deserialize)]
struct DownloadProgressArgs {
    #[serde(rename = "taskId")]
    task_id: String,
}

#[derive(Debug, Deserialize)]
struct FileQuery {
    path: String,
}


fn arg<T: DeserializeOwned>(value: Value) -> Result<T, AppError> {
    serde_json::from_value(value).map_err(|e| AppError::Invalid(format!("Invalid web request: {e}")))
}

fn response_ok<T: Serialize>(value: T) -> Response {
    Json(value).into_response()
}

fn response_err(error: AppError) -> Response {
    (
        StatusCode::BAD_REQUEST,
        Json(json!({ "error": error.to_string() })),
    )
        .into_response()
}

fn lan_ip() -> String {
    let fallback = || {
        if let Ok(name) = std::env::var("COMPUTERNAME") {
            if !name.trim().is_empty() {
                return name;
            }
        }
        "localhost".to_string()
    };

    match UdpSocket::bind("0.0.0.0:0")
        .and_then(|socket| {
            socket.connect("8.8.8.8:80")?;
            socket.local_addr()
        })
        .map(|addr| addr.ip())
    {
        Ok(IpAddr::V4(ip)) if !ip.is_loopback() => ip.to_string(),
        Ok(ip) if !ip.is_loopback() => ip.to_string(),
        _ => fallback(),
    }
}

fn web_url() -> String {
    if cfg!(debug_assertions) {
        format!("http://{}:1420/", lan_ip())
    } else {
        format!("http://{}:{}/", lan_ip(), WEB_PORT)
    }
}

async fn validate_cached_file(path: &Path, app_data: &Path) -> AppResult<PathBuf> {
    let cache_root = crate::cache_root(app_data);
    let cache = tokio::fs::canonicalize(&cache_root).await.unwrap_or(cache_root);
    let path = tokio::fs::canonicalize(path).await.map_err(AppError::Io)?;
    let metadata = tokio::fs::metadata(&path).await.map_err(AppError::Io)?;
    if !path.starts_with(&cache) || !metadata.is_file() {
        return Err(AppError::Invalid("Requested file is outside Raphael's configured cache".into()));
    }
    Ok(path)
}

fn parse_single_range(value: &str, size: u64) -> Option<Result<(u64, u64), ()>> {
    let value = value.trim();
    let range = value.strip_prefix("bytes=")?;
    if range.contains(',') {
        return Some(Err(()));
    }

    let (start, end) = range.split_once('-')?;
    if start.is_empty() {
        let suffix = end.parse::<u64>().ok()?;
        if suffix == 0 || size == 0 {
            return Some(Err(()));
        }
        let length = suffix.min(size);
        return Some(Ok((size - length, size - 1)));
    }

    let start = start.parse::<u64>().ok()?;
    if start >= size {
        return Some(Err(()));
    }

    let end = if end.is_empty() {
        size - 1
    } else {
        end.parse::<u64>().ok()?.min(size - 1)
    };

    if end < start {
        return Some(Err(()));
    }

    Some(Ok((start, end)))
}

async fn health_handler() -> Response {
    (
        StatusCode::NO_CONTENT,
        [(header::CACHE_CONTROL, HeaderValue::from_static("no-store, no-cache, must-revalidate"))],
    )
        .into_response()
}

async fn file_handler(
    AxumState(state): AxumState<WebServerState>,
    headers: axum::http::HeaderMap,
    Query(query): Query<FileQuery>,
) -> Response {
    let app_data = match state.handle.path().app_data_dir() {
        Ok(path) => path,
        Err(error) => return response_err(AppError::Io(std::io::Error::other(error.to_string()))),
    };

    let requested = PathBuf::from(query.path);
    let path = match validate_cached_file(&requested, &app_data).await {
        Ok(path) => path,
        Err(error) => return response_err(error),
    };

    let metadata = match tokio::fs::metadata(&path).await {
        Ok(metadata) => metadata,
        Err(error) => return response_err(AppError::Io(error)),
    };
    let size = metadata.len();

    let (status, start, end) = match headers.get(header::RANGE).and_then(|value| value.to_str().ok()) {
        Some(value) => match parse_single_range(value, size) {
            Some(Ok((start, end))) => (StatusCode::PARTIAL_CONTENT, start, end),
            Some(Err(())) => {
                return (
                    StatusCode::RANGE_NOT_SATISFIABLE,
                    [
                        (header::CONTENT_RANGE, format!("bytes */{size}")),
                        (header::CACHE_CONTROL, "public, max-age=300".to_string()),
                    ],
                )
                    .into_response();
            }
            None => (StatusCode::PARTIAL_CONTENT, 0, size.saturating_sub(1)),
        },
        None => (StatusCode::OK, 0, size.saturating_sub(1)),
    };

    let content_type = match path.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        "avif" => "image/avif",
        _ => "application/octet-stream",
    };

    let content_length = if size == 0 { 0 } else { end - start + 1 };
    let mut file = match File::open(&path).await {
        Ok(file) => file,
        Err(error) => return response_err(AppError::Io(error)),
    };

    if start > 0 {
        if let Err(error) = file.seek(SeekFrom::Start(start)).await {
            return response_err(AppError::Io(error));
        }
    }

    let stream_length = content_length;
    let stream = futures_util::stream::unfold(
        (file, 0u64),
        move |(mut file, mut sent)| async move {
            if sent >= stream_length {
                return None;
            }

            let remaining = stream_length - sent;
            let chunk_size = remaining.min(64 * 1024) as usize;
            let mut buffer = vec![0u8; chunk_size];
            match file.read(&mut buffer).await {
                Ok(0) => None,
                Ok(bytes_read) => {
                    buffer.truncate(bytes_read);
                    sent += bytes_read as u64;
                    Some((Ok::<Vec<u8>, std::io::Error>(buffer), (file, sent)))
                }
                Err(error) => Some((Err(error), (file, stream_length))),
            }
        },
    );

    let mut response = Response::new(Body::from_stream(stream));
    *response.status_mut() = status;
    let response_headers = response.headers_mut();
    response_headers.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static(content_type),
    );
    response_headers.insert(
        header::ACCEPT_RANGES,
        HeaderValue::from_static("bytes"),
    );
    response_headers.insert(
        header::CACHE_CONTROL,
        HeaderValue::from_static("public, max-age=300, stale-while-revalidate=60"),
    );
    if let Ok(value) = HeaderValue::from_str(&content_length.to_string()) {
        response_headers.insert(header::CONTENT_LENGTH, value);
    }
    if status == StatusCode::PARTIAL_CONTENT {
        if let Ok(value) = HeaderValue::from_str(&format!("bytes {start}-{end}/{size}")) {
            response_headers.insert(header::CONTENT_RANGE, value);
        }
    }

    response
}



#[derive(Debug, Deserialize)]
struct TaskStartArgs {
    command: String,
    #[serde(default)]
    args: Value,
}

#[derive(Debug, Deserialize)]
struct RevisionQuery {
    since: Option<u64>,
}

async fn model_changes_handler(Query(query): Query<RevisionQuery>) -> Response {
    let since = query.since.unwrap_or(0);
    let current = crate::model_change_revision();
    if current <= since {
        let notification = crate::model_change_notify();
        let _ = tokio::time::timeout(Duration::from_secs(25), notification.notified()).await;
    }

    response_ok(json!({ "revision": crate::model_change_revision() }))
}

async fn task_status_handler(
    AxumState(state): AxumState<WebServerState>,
    AxumPath(task_id): AxumPath<String>,
) -> Response {
    match state.controller.inner.tasks.get(&task_id) {
        Some(status) => response_ok(status),
        None => (
            StatusCode::NOT_FOUND,
            Json(json!({ "error": "Web task not found or expired" })),
        )
            .into_response(),
    }
}

async fn task_start_handler(
    AxumState(state): AxumState<WebServerState>,
    Json(request): Json<TaskStartArgs>,
) -> Response {
    let handle = state.handle.clone();
    let app = handle.state::<crate::AppStateInner>().inner().clone();

    let operation: std::pin::Pin<Box<dyn Future<Output = AppResult<Value>> + Send>> =
        match request.command.as_str() {
            "preview_civitai_import" => {
                let args: UrlArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                Box::pin(async move {
                    let value = preview_civitai_import_inner(&app, args.url).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "sync_model_gallery" => {
                let args: SyncGalleryArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                let task_handle = handle.clone();
                Box::pin(async move {
                    let target_count = args.target_count.unwrap_or(20).clamp(1, 200);
                    let value = sync_gallery_inner(app.clone(), args.id, task_handle, target_count).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "load_more_model_examples" => {
                let args: LoadMoreArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                let task_handle = handle.clone();
                Box::pin(async move {
                    let requested = args.amount.unwrap_or_else(|| {
                        open_db(&app.app_data)
                            .ok()
                            .and_then(|c| read_example_load_amount(&c).ok())
                            .unwrap_or(20)
                    }).clamp(1, 100);
                    let current_count = {
                        let c = open_db(&app.app_data)?;
                        c.query_row(
                            "SELECT COUNT(*) FROM images WHERE model_id=?1 AND meta_json NOT LIKE '%\"featured\":true%'",
                            [args.id],
                            |row| row.get::<_, i64>(0),
                        )?
                    };
                    let value = sync_gallery_inner(
                        app.clone(),
                        args.id,
                        task_handle,
                        current_count.saturating_add(requested).clamp(1, 1000),
                    ).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "add_subfolder_tags" => {
                let task_handle = handle.clone();
                Box::pin(async move {
                    let value = crate::add_subfolder_tags_inner(&app)?;
                    crate::spawn_registry_sync(app.clone(), task_handle.clone());
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "refetch_all_model_tags" => {
                let task_handle = handle.clone();
                Box::pin(async move {
                    let value = refetch_all_model_tags_inner(
                        &app,
                        task_handle,
                    ).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            },
            "delete_model" => {
                let args: IdArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                let task_handle = handle.clone();
                Box::pin(async move {
                    delete_model_inner(&app, task_handle, args.id).await?;
                    Ok(Value::Null)
                })
            }
            "set_cache_max_bytes" => {
                let args: CacheMaxBytesArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                Box::pin(async move {
                    let _guard = app.cache_lock.lock().await;
                    let value = set_cache_max_bytes_inner(&app.app_data, args.max_bytes)?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "set_cache_location" => {
                let args: CacheLocationArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                Box::pin(async move {
                    let _guard = app.cache_lock.lock().await;
                    let value = set_cache_location_inner(&app.app_data, &args.path)?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "clear_cache_images" => Box::pin(async move {
                let _guard = app.cache_lock.lock().await;
                let value = clear_cache_images_inner(&app.app_data)?;
                serde_json::to_value(value)
                    .map_err(|error| AppError::Invalid(error.to_string()))
            }),
            "clear_complete_cache" => Box::pin(async move {
                let _guard = app.cache_lock.lock().await;
                let value = clear_complete_cache_inner(&app.app_data)?;
                serde_json::to_value(value)
                    .map_err(|error| AppError::Invalid(error.to_string()))
            }),
            "prune_cache_images" => {
                let args: KeepImagesArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                Box::pin(async move {
                    let _guard = app.cache_lock.lock().await;
                    let value = prune_cache_images_inner(&app.app_data, args.keep_per_model)?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "clean_cache_orphans" => Box::pin(async move {
                let _guard = app.cache_lock.lock().await;
                let value = clean_cache_orphans_inner(&app.app_data)?;
                serde_json::to_value(value)
                    .map_err(|error| AppError::Invalid(error.to_string()))
            }),
            "link_model_civitai" => {
                let args: LinkArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                let task_handle = handle.clone();
                Box::pin(async move {
                    let value = link_model_civitai_inner(&app, task_handle, args.id, args.url).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            "refresh_model_civitai" => {
                let args: IdArgs = match arg(request.args) {
                    Ok(value) => value,
                    Err(error) => return response_err(error),
                };
                let task_handle = handle.clone();
                Box::pin(async move {
                    let value = refresh_model_civitai_inner(&app, task_handle, args.id).await?;
                    serde_json::to_value(value)
                        .map_err(|error| AppError::Invalid(error.to_string()))
                })
            }
            _ => {
                return response_err(AppError::Invalid(format!(
                    "Command '{}' is not eligible for background web execution",
                    request.command
                )));
            }
        };

    let task_id = state.controller.inner.tasks.start(operation);
    response_ok(json!({ "task_id": task_id, "state": "queued" }))
}

async fn command_handler(
    AxumPath(command): AxumPath<String>,
    AxumState(state): AxumState<WebServerState>,
    Json(args): Json<Value>,
) -> Response {
    let handle = &state.handle;

    let result: AppResult<Value> = match command.as_str() {
        "get_app_state" => get_app_state(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "list_models" => {
            let args: ListModelsArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            list_models(handle.state(), args.r#type, args.query, args.tags).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "get_library_counts" => get_library_counts(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "get_tags" => get_tags(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "add_subfolder_tags" => add_subfolder_tags(handle.state(), handle.clone()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "set_model_tags" => {
            let args: TagsArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_tags(handle.state(), handle.clone(), args.id, args.tags).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_name" => {
            let args: NameArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_name(handle.state(), handle.clone(), args.id, args.name)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_description" => {
            let args: DescriptionArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_description(handle.state(), handle.clone(), args.id, args.description)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "refetch_all_model_tags" => {
            refetch_all_model_tags(handle.state(), handle.clone())
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_type" => {
            let args: TypeArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_type(handle.state(), handle.clone(), args.id, args.model_type).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_cover_position" => {
            let args: CoverPositionArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_cover_position(handle.state(), args.id, args.x, args.y)
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "reset_model_cover" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            reset_model_cover(handle.state(), args.id)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_cover_from_image" => {
            let args: ImageArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_cover_from_image(handle.state(), args.id, args.image_id)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_custom_cover" => {
            Err(AppError::Invalid("Custom cover file selection is only available in the desktop app.".into()))
        }
        "delete_model" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            delete_model(handle.state(), handle.clone(), args.id).await.map(|_| json!(null))
        }
        "get_model_images" => {
            let args: ModelImagesArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            get_model_images(handle.state(), args.id, args.limit).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "sync_model_gallery" => {
            let args: SyncGalleryArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            sync_model_gallery(handle.state(), handle.clone(), args.id, args.target_count)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "load_more_model_examples" => {
            let args: SyncGalleryArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            load_more_model_examples(handle.state(), handle.clone(), args.id, args.target_count)
                .await
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "get_example_load_amount" => get_example_load_amount(handle.state())
            .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "set_example_load_amount" => {
            let args: ExampleLoadAmountArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            let amount=args.amount.unwrap_or(20);
            set_example_load_amount(handle.state(), amount)
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "refresh_all_examples" => {
            refresh_all_examples(handle.state(), handle.clone())
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "get_examples_refresh_status" => {
            serde_json::to_value(get_examples_refresh_status(handle.state()))
                .map_err(|e| AppError::Invalid(e.to_string()))
        },
        "preview_civitai_import" => {
            let args: UrlArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            preview_civitai_import(handle.state(), args.url)
                .await
                .map(|value: CivitaiImportPreview| serde_json::to_value(value).unwrap_or(Value::Null))
        }
        "install_civitai_model" => {
            let args: InstallArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            install_civitai_model(
                handle.state(),
                handle.clone(),
                args.url,
                args.target_directory,
                args.selected_type,
            )
            .await
            .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "get_download_progress" => serde_json::to_value(get_download_progress(handle.state())).map_err(|e| AppError::Invalid(e.to_string())),
        "get_parallel_downloads" => get_parallel_downloads(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "set_parallel_downloads" => {
            let args: ParallelDownloadsArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_parallel_downloads(handle.state(), args.value).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "get_cache_stats" => get_cache_stats(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "set_cache_max_bytes" => {
            let args: CacheMaxBytesArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_cache_max_bytes(handle.state(), args.max_bytes).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "set_cache_location" => {
            let args: CacheLocationArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_cache_location(handle.state(), args.path).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "clear_cache_images" => clear_cache_images(handle.state()).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "clear_complete_cache" => clear_complete_cache(handle.state()).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "prune_cache_images" => {
            let args: KeepImagesArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            prune_cache_images(handle.state(), args.keep_per_model).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        },
        "clean_cache_orphans" => clean_cache_orphans(handle.state()).await.and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "clear_download_progress" => {
            let args: DownloadProgressArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            clear_download_progress(handle.state(), args.task_id).map(|_| Value::Null)
        },
        "link_model_civitai" => {
            let args: LinkArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            link_model_civitai(handle.state(), handle.clone(), args.id, args.url)
                .await
                .map(|value: ModelRecord| serde_json::to_value(value).unwrap_or(Value::Null))
        }
        "refresh_model_civitai" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            refresh_model_civitai(handle.state(), handle.clone(), args.id)
                .await
                .map(|value: ModelRecord| serde_json::to_value(value).unwrap_or(Value::Null))
        }
        "get_storage_stats" => get_storage_stats(handle.state()).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string()))),
        "set_civitai_token" => {
            let args: TokenArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_civitai_token(args.token).map(|_| json!(null))
        }
        "is_civitai_token_set" => Ok(json!(is_civitai_token_set())),
        "check_registry_health" => {
            let app = handle.state::<crate::AppStateInner>();
            Ok(json!(app.registry.is_authenticated().await))
        }
        "open_in_file_manager" => {
            let _ = arg::<serde_json::Map<String, Value>>(args);
            Err(AppError::Invalid("Opening the Windows file manager is only available in the desktop app".into()))
        }
        "chooseModelsFolder" | "chooseDirectory" | "set_models_root" => {
            Err(AppError::Invalid("Folder selection is only available in the desktop app. Configure the models folder from Raphael on the host PC.".into()))
        }
        _ => Err(AppError::Invalid(format!("Unknown web command: {command}"))),
    };

    match result {
        Ok(value) => response_ok(value),
        Err(error) => response_err(error),
    }
}

async fn status_handler(AxumState(state): AxumState<WebServerState>) -> Response {
    response_ok(state.controller.status())
}

fn build_router(handle: AppHandle, controller: WebServerController, web_root: Option<PathBuf>) -> Router {
    let state = WebServerState {
        handle,
        controller,
    };

    let api = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/changes", get(model_changes_handler))
        .route("/api/tasks/{task_id}", get(task_status_handler))
        .route("/api/task/start", post(task_start_handler))
        .route("/api/status", get(status_handler))
        .route("/api/file", get(file_handler))
        .route("/api/command/{command}", post(command_handler))
        .layer(DefaultBodyLimit::max(1024 * 1024))
        .with_state(state.clone());

    let mut router = Router::new().merge(api);
    if let Some(root) = web_root {
        router = router.fallback_service(ServeDir::new(root));
    }
    router
}

pub async fn set_web_app_enabled(
    handle: AppHandle,
    controller: State<'_, WebServerController>,
    enabled: bool,
) -> AppResult<WebAppStatus> {
    let controller = controller.inner().clone();

    if !enabled {
        controller.inner.generation.fetch_add(1, Ordering::AcqRel);
        if let Some(sender) = controller.inner.shutdown.lock().unwrap().take() {
            let _ = sender.send(());
        }
        *controller.inner.enabled.write().unwrap() = false;
        *controller.inner.url.write().unwrap() = None;
        return Ok(controller.status());
    }

    if *controller.inner.enabled.read().unwrap() {
        return Ok(controller.status());
    }

    // Bind once before reporting success so an occupied/unavailable port is
    // surfaced immediately. After the server starts, the supervisor below
    // will re-bind automatically if the listener ever exits unexpectedly.
    let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], WEB_PORT)))
        .await
        .map_err(AppError::Io)?;

    let generation = controller.inner.generation.fetch_add(1, Ordering::AcqRel) + 1;
    let (shutdown_tx, shutdown_rx) = oneshot::channel();
    let url = web_url();

    let web_root = if cfg!(debug_assertions) {
        None
    } else {
        Some(
            handle
                .path()
                .resolve("web", BaseDirectory::Resource)
                .map_err(|error| AppError::Invalid(format!("Could not resolve bundled web app: {error}")))?,
        )
    };

    *controller.inner.shutdown.lock().unwrap() = Some(shutdown_tx);
    *controller.inner.enabled.write().unwrap() = true;
    *controller.inner.url.write().unwrap() = Some(url);

    let task_controller = controller.clone();
    let task_handle = handle.clone();
    tauri::async_runtime::spawn(async move {
        let mut listener = Some(listener);
        let mut shutdown_rx = Some(shutdown_rx);

        loop {
            if !*task_controller.inner.enabled.read().unwrap()
                || task_controller.inner.generation.load(Ordering::Acquire) != generation
            {
                break;
            }

            let router = build_router(task_handle.clone(), task_controller.clone(), web_root.clone());
            let current_shutdown = match shutdown_rx.take() {
                Some(receiver) => receiver,
                None => {
                    eprintln!("Raphael web server supervisor lost its shutdown channel; stopping safely");
                    break;
                }
            };

            let current_listener = match listener.take() {
                Some(listener) => listener,
                None => {
                    eprintln!("Raphael web server supervisor has no listener to serve; stopping safely");
                    break;
                }
            };

            match axum::serve(current_listener, router)
                .with_graceful_shutdown(async move {
                    let _ = current_shutdown.await;
                })
                .await
            {
                Ok(()) => {
                    if *task_controller.inner.enabled.read().unwrap() {
                        eprintln!("Raphael web server listener exited unexpectedly without an error");
                    }
                }
                Err(error) => {
                    eprintln!("Raphael web server listener failed: {error}");
                }
            }

            if !*task_controller.inner.enabled.read().unwrap()
                || task_controller.inner.generation.load(Ordering::Acquire) != generation
            {
                break;
            }

            *task_controller.inner.shutdown.lock().unwrap() = None;

            loop {
                if !*task_controller.inner.enabled.read().unwrap()
                    || task_controller.inner.generation.load(Ordering::Acquire) != generation
                {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(500)).await;

                match TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], WEB_PORT))).await {
                    Ok(next_listener) => {
                        listener = Some(next_listener);
                        *task_controller.inner.url.write().unwrap() = Some(web_url(
                            &task_controller
                                .inner
                                .auth_token
                                .read()
                                .unwrap()
                                .clone()
                                .unwrap_or_default(),
                        ));
                        let (next_shutdown_tx, next_shutdown_rx) = oneshot::channel();
                        *task_controller.inner.shutdown.lock().unwrap() = Some(next_shutdown_tx);
                        shutdown_rx = Some(next_shutdown_rx);
                        break;
                    }
                    Err(error) => {
                        eprintln!("Raphael web server restart failed: {error}");
                    }
                }
            }

            if shutdown_rx.is_none() {
                break;
            }
        }

        if task_controller.inner.generation.load(Ordering::Acquire) == generation {
            *task_controller.inner.shutdown.lock().unwrap() = None;
            *task_controller.inner.enabled.write().unwrap() = false;
            *task_controller.inner.url.write().unwrap() = None;
            *task_controller.inner.auth_token.write().unwrap() = None;
        }
    });
    Ok(controller.status())
}
#[tauri::command]
pub async fn get_web_app_status(
    controller: State<'_, WebServerController>,
) -> AppResult<WebAppStatus> {
    Ok(controller.status())
}

#[tauri::command]
pub async fn toggle_web_app(
    handle: AppHandle,
    controller: State<'_, WebServerController>,
    enabled: bool,
) -> AppResult<WebAppStatus> {
    set_web_app_enabled(handle, controller, enabled).await
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_access_tokens_are_six_digit_numbers() {
        for _ in 0..100 {
            let token = generate_access_token().expect("token generation should succeed");
            assert_eq!(token.len(), 6);
            assert!(token.chars().all(|value| value.is_ascii_digit()));
            let value: u32 = token.parse().expect("token should be numeric");
            assert!((100_000..=999_999).contains(&value));
        }
    }

    #[test]
    fn bearer_and_cookie_tokens_are_parsed_without_accepting_other_schemes() {
        let mut headers = axum::http::HeaderMap::new();
        headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Bearer secret-token"));
        headers.insert(header::COOKIE, HeaderValue::from_static("foo=bar; raphael_auth=secret-token; other=value"));
        assert_eq!(bearer_token(&headers).as_deref(), Some("secret-token"));
        assert_eq!(cookie_token(&headers).as_deref(), Some("secret-token"));

        headers.insert(header::AUTHORIZATION, HeaderValue::from_static("Basic secret-token"));
        assert!(bearer_token(&headers).is_none());
    }

    #[test]
    fn web_urls_keep_the_access_token_in_the_fragment() {
        let url = web_url("abc123");
        assert!(url.contains("#access_token=abc123"));
        assert!(!url.contains("?access_token="));
    }
}
