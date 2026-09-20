use axum::{
    extract::{Path as AxumPath, Query, State as AxumState},
    http::{header, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    net::{IpAddr, SocketAddr, UdpSocket},
    path::PathBuf,
    sync::{Arc, Mutex, RwLock},
};
use tauri::{path::BaseDirectory, AppHandle, Manager, State};
use tokio::{net::TcpListener, sync::oneshot};
use tower_http::{cors::CorsLayer, services::ServeDir};

use crate::{
    add_subfolder_tags, clear_download_progress, delete_model, get_app_state, get_download_progress, get_library_counts, get_model_images, get_storage_stats,
    get_tags, install_civitai_model, link_model_civitai, list_models, preview_civitai_import,
    refresh_model_civitai, reset_model_cover, set_civitai_token, set_model_cover_position, set_model_cover_from_image,
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

fn validate_cached_file(path: &PathBuf, app_data: &PathBuf) -> AppResult<PathBuf> {
    let app_data = app_data
        .canonicalize()
        .map_err(|e| AppError::Io(e))?;
    let path = path.canonicalize().map_err(|e| AppError::Io(e))?;
    if !path.starts_with(&app_data) || !path.is_file() {
        return Err(AppError::Invalid("Requested file is outside Raphael's cache".into()));
    }
    Ok(path)
}

async fn file_handler(
    AxumState(state): AxumState<WebServerState>,
    Query(query): Query<FileQuery>,
) -> Response {
    let app_data = match state.handle.path().app_data_dir() {
        Ok(path) => path,
        Err(error) => return response_err(AppError::Io(std::io::Error::other(error.to_string()))),
    };

    let requested = PathBuf::from(query.path);
    let path = match validate_cached_file(&requested, &app_data) {
        Ok(path) => path,
        Err(error) => return response_err(error),
    };

    let content = match tokio::fs::read(&path).await {
        Ok(content) => content,
        Err(error) => return response_err(AppError::Io(error)),
    };

    let content_type = match path.extension().and_then(|x| x.to_str()).unwrap_or("").to_ascii_lowercase().as_str() {
        "jpg" | "jpeg" => "image/jpeg",
        "png" => "image/png",
        "webp" => "image/webp",
        "gif" => "image/gif",
        _ => "application/octet-stream",
    };

    (
        [(header::CONTENT_TYPE, content_type)],
        content,
    )
        .into_response()
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
            set_model_cover_position(handle.state(), handle.clone(), args.id, args.x, args.y)
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "reset_model_cover" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            reset_model_cover(handle.state(), handle.clone(), args.id)
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_cover_from_image" => {
            let args: ImageArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            set_model_cover_from_image(handle.state(), handle.clone(), args.id, args.image_id)
                .and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "set_model_custom_cover" => {
            Err(AppError::Invalid("Custom cover file selection is only available in the desktop app.".into()))
        }
        "delete_model" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            delete_model(handle.state(), handle.clone(), args.id).map(|_| json!(null))
        }
        "get_model_images" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            get_model_images(handle.state(), args.id).and_then(|value| serde_json::to_value(value).map_err(|e| AppError::Invalid(e.to_string())))
        }
        "sync_model_gallery" => {
            let args: IdArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            sync_model_gallery(handle.state(), handle.clone(), args.id)
                .await
                .map(|_| json!(null))
        }
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
        "clear_download_progress" => {
            let args: DownloadProgressArgs = match arg(args) { Ok(value) => value, Err(error) => return response_err(error) };
            clear_download_progress(handle.state(), args.task_id).and_then(|_| Ok(Value::Null))
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

    let listener = TcpListener::bind(SocketAddr::from(([0, 0, 0, 0], WEB_PORT)))
        .await
        .map_err(|error| AppError::Io(error))?;

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

    let router = build_router(handle, controller.clone(), web_root);
    *controller.inner.shutdown.lock().unwrap() = Some(shutdown_tx);
    *controller.inner.enabled.write().unwrap() = true;
    *controller.inner.url.write().unwrap() = Some(url);

    let task_controller = controller.clone();
    tauri::async_runtime::spawn(async move {
        let _ = axum::serve(listener, router)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;

        *task_controller.inner.enabled.write().unwrap() = false;
        *task_controller.inner.url.write().unwrap() = None;
        *task_controller.inner.shutdown.lock().unwrap() = None;
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
