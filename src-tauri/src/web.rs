use axum::{
    body::Body,
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    net::{IpAddr, SocketAddr, UdpSocket},
    path::{Path, PathBuf},
    sync::{Arc, Mutex, RwLock},
    time::Duration,
};
use tauri::{path::BaseDirectory, AppHandle, Manager, State};
use tokio::{
    fs::File,
    io::{AsyncReadExt, AsyncSeekExt, SeekFrom},
    net::TcpListener,
    sync::oneshot,
};
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::{
    add_subfolder_tags, clear_download_progress, delete_model, get_app_state, get_download_progress, get_library_counts, get_model_images, get_storage_stats,
    get_parallel_downloads, set_parallel_downloads,
    get_cache_stats, set_cache_max_bytes, set_cache_location, clear_cache_images, clear_complete_cache, prune_cache_images, clean_cache_orphans,
    get_tags, install_civitai_model, link_model_civitai, list_models, preview_civitai_import,
    refresh_all_examples, get_examples_refresh_status, load_more_model_examples, get_example_load_amount, set_example_load_amount, refresh_model_civitai, reset_model_cover, set_civitai_token, set_model_cover_position, set_model_cover_from_image,
    set_model_tags, set_model_type, sync_model_gallery,
    is_civitai_token_set, AppError, AppResult, CivitaiImportPreview, ModelRecord,
};

pub const WEB_PORT: u16 = 1421;

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
}

impl Default for WebServerControllerInner {
    fn default() -> Self {
        Self {
            shutdown: Mutex::new(None),
            enabled: RwLock::new(false),
            url: RwLock::new(None),
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
        format!("http://{}:1420", lan_ip())
    } else {
        format!("http://{}:{}", lan_ip(), WEB_PORT)
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
            set_model_tags(handle.state(), handle.clone(), args.id, args.tags).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_type" => {
            let args: TypeArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_type(handle.state(), handle.clone(), args.id, args.model_type).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
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

    let mut router = Router::new()
        .route("/api/health", get(health_handler))
        .route("/api/status", get(status_handler))
        .route("/api/file", get(file_handler))
        .route("/api/command/{command}", post(command_handler))
        .layer(CorsLayer::permissive())
        .with_state(state);

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
        let mut listener = listener;
        let mut shutdown_rx = Some(shutdown_rx);

        loop {
            let router = build_router(task_handle.clone(), task_controller.clone(), web_root.clone());
            let current_shutdown = match shutdown_rx.take() {
                Some(receiver) => receiver,
                None => {
                    eprintln!("Raphael web server supervisor lost its shutdown channel; stopping safely");
                    break;
                }
            };

            let _ = axum::serve(listener, router)
                .with_graceful_shutdown(async move {
                    let _ = current_shutdown.await;
                })
                .await;

            *task_controller.inner.shutdown.lock().unwrap() = None;

            if !*task_controller.inner.enabled.read().unwrap() {
                break;
            }

            // An unexpected listener exit should not leave clients stranded.
            // Retry the bind while the user still has the web app enabled.
            loop {
                if !*task_controller.inner.enabled.read().unwrap() {
                    break;
                }

                tokio::time::sleep(Duration::from_millis(750)).await;

                match TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], WEB_PORT))).await {
                    Ok(next_listener) => {
                        listener = next_listener;
                        *task_controller.inner.url.write().unwrap() = Some(web_url());
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

        *task_controller.inner.shutdown.lock().unwrap() = None;
        *task_controller.inner.enabled.write().unwrap() = false;
        *task_controller.inner.url.write().unwrap() = None;
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
