use reqwest::{Client, Method, StatusCode};
use serde::de::DeserializeOwned;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    env,
    fs,
    path::{Path, PathBuf},
    time::Duration,
};
use thiserror::Error;
use tokio::{
    process::Command,
    sync::Mutex as AsyncMutex,
    time::{sleep, timeout},
};
use url::Url;

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:43217";
const REGISTRY_STARTUP_TIMEOUT: Duration = Duration::from_secs(180);
const REGISTRY_STARTUP_POLL: Duration = Duration::from_millis(150);

#[derive(Debug, Error)]
pub(crate) enum RegistryError {
    #[error("registry HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("registry I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("registry authentication token is not configured")]
    MissingToken,
    #[error("registry returned {status}: {message}")]
    Api { status: StatusCode, message: String },
    #[error("registry response could not be decoded: {0}")]
    Decode(#[from] serde_json::Error),
    #[error("automatic Registry startup is only supported for a local HTTP Registry endpoint")]
    AutoStartUnsupported,
    #[error("could not locate the Raphael Model Registry executable; set RAPHAEL_REGISTRY_EXECUTABLE to its path")]
    ExecutableNotFound,
    #[error("Registry did not become available before the startup timeout")]
    StartupTimeout,
    #[error("Registry is running but does not provide the required Raphael asset API")]
    IncompatibleApi,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub(crate) struct RegistryModel {
    pub id: String,
    pub name: String,
    pub model_type: String,
    pub creator: Option<String>,
    pub description: Option<String>,
    pub base_model: Option<String>,
    pub revision: i64,
    pub created_at: i64,
    pub updated_at: i64,
    #[serde(default)]
    pub extensions: Value,
}

#[derive(Debug, Clone, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct RegistrySearchResult {
    pub items: Vec<RegistryModel>,
    pub total: i64,
    pub limit: i64,
    pub offset: i64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub(crate) struct RegistryVersion {
    pub id: String,
    pub model_id: String,
    pub version_name: Option<String>,
    pub base_model: Option<String>,
    pub revision: i64,
    pub source: Option<String>,
    pub source_model_id: Option<String>,
    pub source_version_id: Option<String>,
    pub source_url: Option<String>,
    #[serde(default)]
    pub activation_prompts: Vec<String>,
    #[serde(default)]
    pub metadata: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub(crate) struct RegistryFile {
    pub id: String,
    pub model_id: String,
    pub version_id: Option<String>,
    pub path: String,
    pub relative_path: Option<String>,
    pub filename: String,
    pub size_bytes: i64,
    pub modified_at: i64,
    pub sha256: Option<String>,
    pub status: String,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
pub(crate) struct RegistrySource {
    pub id: String,
    pub model_id: String,
    pub provider: String,
    pub external_model_id: Option<String>,
    pub external_version_id: Option<String>,
    pub url: Option<String>,
    pub imported_at: i64,
    #[serde(default)]
    pub metadata: Value,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
#[allow(dead_code)]
pub(crate) struct RegistryAsset {
    pub id: String,
    pub model_id: String,
    pub kind: String,
    pub path: String,
    pub source: Option<String>,
    #[serde(default)]
    pub metadata: Value,
    pub created_at: i64,
    pub updated_at: i64,
}

#[derive(Debug, Clone, serde::Deserialize)]
pub(crate) struct RegistryEvent {
    pub id: i64,
    pub model_id: Option<String>,
}

#[derive(Clone)]
pub(crate) struct RegistryClient {
    base_url: String,
    token_file: PathBuf,
    client: Client,
    startup_lock: std::sync::Arc<AsyncMutex<()>>,
}

impl RegistryClient {
    pub(crate) fn from_app_data(app_data: &Path) -> Result<Self, RegistryError> {
        let base_url = env::var("RAPHAEL_REGISTRY_URL")
            .unwrap_or_else(|_| DEFAULT_BASE_URL.to_string())
            .trim_end_matches('/')
            .to_string();
        validate_registry_base_url(&base_url)?;

        let token_file = env::var_os("RAPHAEL_REGISTRY_TOKEN_FILE")
            .map(PathBuf::from)
            .unwrap_or_else(|| default_token_path(app_data));

        Ok(Self {
            base_url,
            token_file,
            client: Client::builder()
                .timeout(Duration::from_secs(30))
                .build()?,
            startup_lock: std::sync::Arc::new(AsyncMutex::new(())),
        })
    }

    pub(crate) async fn ensure_running(&self) -> Result<(), RegistryError> {
        if self.is_ready().await {
            return Ok(());
        }

        let _startup_guard = self.startup_lock.lock().await;

        // Another Manager operation may have started or repaired the Registry
        // while this task was waiting for the startup lock.
        if self.is_ready().await {
            return Ok(());
        }

        // A healthy process that fails the capability check is an older local
        // Registry binary. Replace it before starting the current one.
        if self.is_healthy().await {
            self.stop_stale_local_registry_locked().await?;
        }

        let (bind, port) = local_http_registry_endpoint(&self.base_url)?;
        let data_dir = self
            .token_file
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| PathBuf::from("."));

        let mut child = if let Some(executable) = env::var_os("RAPHAEL_REGISTRY_EXECUTABLE") {
            let executable = PathBuf::from(executable);
            if !executable.is_file() {
                return Err(RegistryError::ExecutableNotFound);
            }
            spawn_registry_executable(&executable, &bind, port, &data_dir, &self.token_file)?
        } else if let Some(executable) = find_registry_executable() {
            spawn_registry_executable(&executable, &bind, port, &data_dir, &self.token_file)?
        } else if let Some(executable) = registry_command_on_path() {
            spawn_registry_executable(&executable, &bind, port, &data_dir, &self.token_file)?
        } else if let Some(manifest) = find_registry_workspace_manifest() {
            spawn_registry_cargo(&manifest, &bind, port, &data_dir, &self.token_file)?
        } else {
            return Err(RegistryError::ExecutableNotFound);
        };

        let ready = timeout(REGISTRY_STARTUP_TIMEOUT, async {
            loop {
                if self.is_ready().await {
                    return Ok::<(), RegistryError>(());
                }

                // The listener is healthy as soon as an old incompatible
                // binary starts, so fail fast instead of waiting 180 seconds.
                if self.is_healthy().await && !self.has_required_asset_api().await {
                    let _ = child.kill().await;
                    return Err(RegistryError::IncompatibleApi);
                }

                sleep(REGISTRY_STARTUP_POLL).await;
            }
        })
        .await;

        match ready {
            Ok(result) => {
                result?;
                let _ = child.id();
                Ok(())
            }
            Err(_) => {
                let _ = child.kill().await;
                Err(RegistryError::StartupTimeout)
            }
        }
    }

    pub(crate) async fn is_healthy(&self) -> bool {
        self.client
            .get(format!("{}/health", self.base_url))
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false)
    }

    async fn has_required_asset_api(&self) -> bool {
        self.client
            .get(format!("{}/api/v1/capabilities", self.base_url))
            .send()
            .await
            .ok()
            .and_then(|response| async {
                if !response.status().is_success() {
                    return None;
                }
                response.json::<RegistryCapabilities>().await.ok()
            }.await)
            .map(|capabilities| {
                capabilities.api_version == "v1" && capabilities.asset_content_upload
            })
            .unwrap_or(false)
    }

    async fn is_ready(&self) -> bool {
        self.is_healthy().await && self.has_required_asset_api().await
    }

    fn token(&self) -> Result<String, RegistryError> {
        if let Ok(value) = env::var("RAPHAEL_REGISTRY_AUTH_TOKEN") {
            if !value.trim().is_empty() {
                return Ok(value.trim().to_string());
            }
        }

        let token = fs::read_to_string(&self.token_file)
            .map_err(|_| RegistryError::MissingToken)?;
        let token = token.trim();
        if token.is_empty() {
            return Err(RegistryError::MissingToken);
        }
        Ok(token.to_string())
    }

    fn request(&self, method: Method, path: &str) -> Result<reqwest::RequestBuilder, RegistryError> {
        Ok(self
            .client
            .request(method, format!("{}{}", self.base_url, path))
            .bearer_auth(self.token()?)
            .header("x-raphael-actor", "model-manager"))
    }

    async fn send_json<T: DeserializeOwned>(
        &self,
        request: reqwest::RequestBuilder,
    ) -> Result<T, RegistryError> {
        let retry = request.try_clone();
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if error.is_connect() => {
                self.ensure_running().await?;
                retry
                    .ok_or_else(|| RegistryError::Http(error))?
                    .send()
                    .await?
            }
            Err(error) => return Err(RegistryError::Http(error)),
        };

        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        if !status.is_success() {
            return Err(RegistryError::Api {
                status,
                message: body,
            });
        }
        Ok(serde_json::from_str(&body)?)
    }

    async fn send_empty(&self, request: reqwest::RequestBuilder) -> Result<(), RegistryError> {
        let retry = request.try_clone();
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if error.is_connect() => {
                self.ensure_running().await?;
                retry
                    .ok_or_else(|| RegistryError::Http(error))?
                    .send()
                    .await?
            }
            Err(error) => return Err(RegistryError::Http(error)),
        };

        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let message = response.text().await.unwrap_or_default();
        Err(RegistryError::Api { status, message })
    }


    #[allow(dead_code)]
    pub(crate) async fn search_models(
        &self,
        query: Option<&str>,
        model_type: Option<&str>,
        tag: Option<&str>,
        limit: i64,
        offset: i64,
    ) -> Result<RegistrySearchResult, RegistryError> {
        let mut request = self.request(Method::GET, "/api/v1/models")?
            .query(&[("limit", limit.clamp(1, 200)), ("offset", offset.max(0))]);
        if let Some(value) = query { request = request.query(&[("q", value)]); }
        if let Some(value) = model_type { request = request.query(&[("model_type", value)]); }
        if let Some(value) = tag { request = request.query(&[("tag", value)]); }
        self.send_json(request).await
    }

    pub(crate) async fn get_model(&self, id: &str) -> Result<RegistryModel, RegistryError> {
        self.send_json(self.request(Method::GET, &format!("/api/v1/models/{id}"))?).await
    }


    #[allow(clippy::too_many_arguments)]
    pub(crate) async fn create_model(
        &self,
        id: &str,
        name: &str,
        model_type: &str,
        creator: Option<&str>,
        description: Option<&str>,
        base_model: Option<&str>,
        extensions: Value,
    ) -> Result<RegistryModel, RegistryError> {
        self.send_json(
            self.request(Method::POST, "/api/v1/models")?
                .json(&json!({
                    "id": id,
                    "name": name,
                    "model_type": model_type,
                    "creator": creator,
                    "description": description,
                    "base_model": base_model,
                    "extensions": extensions
                })),
        )
        .await
    }

    /// Create a model while tolerating the pre-canonical Registry spelling
    /// `control_net` used by older local Registry binaries.
    pub(crate) async fn create_model_with_legacy_type_fallback(
        &self,
        id: &str,
        name: &str,
        model_type: &str,
        creator: Option<&str>,
        description: Option<&str>,
        base_model: Option<&str>,
        extensions: Value,
    ) -> Result<RegistryModel, RegistryError> {
        match self
            .create_model(
                id,
                name,
                model_type,
                creator,
                description,
                base_model,
                extensions.clone(),
            )
            .await
        {
            Err(RegistryError::Api { status: StatusCode::UNPROCESSABLE_ENTITY, message })
                if model_type == "controlnet"
                    && message.contains("unknown variant")
                    && message.contains("control_net")
            => self
                .create_model(
                    id,
                    name,
                    "control_net",
                    creator,
                    description,
                    base_model,
                    extensions,
                )
                .await,
            result => result,
        }
    }

    pub(crate) async fn update_model(
        &self,
        id: &str,
        expected_revision: i64,
        patch: Value,
    ) -> Result<RegistryModel, RegistryError> {
        let mut revision = expected_revision;
        for attempt in 0..=2 {
            let mut body = match patch.clone() {
                Value::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            body.insert("expected_revision".into(), json!(revision));

            match self
                .send_json(
                    self.request(Method::PATCH, &format!("/api/v1/models/{id}"))?
                        .json(&Value::Object(body)),
                )
                .await
            {
                Ok(model) => return Ok(model),
                Err(RegistryError::Api { status, .. })
                    if is_revision_conflict(status) && attempt < 2 =>
                {
                    revision = self.get_model(id).await?.revision;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("registry model update retry loop always returns")
    }

    pub(crate) async fn update_model_with_legacy_type_fallback(
        &self,
        id: &str,
        expected_revision: i64,
        patch: Value,
    ) -> Result<RegistryModel, RegistryError> {
        match self.update_model(id, expected_revision, patch.clone()).await {
            Err(RegistryError::Api { status: StatusCode::UNPROCESSABLE_ENTITY, message })
                if patch
                    .get("model_type")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value == "controlnet")
                    && message.contains("unknown variant")
                    && message.contains("control_net")
            => {
                let mut legacy_patch = match patch {
                    Value::Object(map) => map,
                    _ => serde_json::Map::new(),
                };
                legacy_patch.insert("model_type".into(), json!("control_net"));
                self.update_model(id, expected_revision, Value::Object(legacy_patch)).await
            }
            result => result,
        }
    }

    pub(crate) async fn versions(&self, id: &str) -> Result<Vec<RegistryVersion>, RegistryError> {
        self.send_json(self.request(Method::GET, &format!("/api/v1/models/{id}/versions"))?)
            .await
    }

    pub(crate) async fn create_version(
        &self,
        model_id: &str,
        payload: Value,
    ) -> Result<RegistryVersion, RegistryError> {
        self.send_json(
            self.request(Method::POST, &format!("/api/v1/models/{model_id}/versions"))?
                .json(&payload),
        )
        .await
    }

    pub(crate) async fn update_version(
        &self,
        model_id: &str,
        version_id: &str,
        expected_revision: i64,
        patch: Value,
    ) -> Result<RegistryVersion, RegistryError> {
        let mut revision = expected_revision;
        for attempt in 0..=2 {
            let mut body = match patch.clone() {
                Value::Object(map) => map,
                _ => serde_json::Map::new(),
            };
            body.insert("expected_revision".into(), json!(revision));

            match self
                .send_json(
                    self.request(
                        Method::PATCH,
                        &format!("/api/v1/models/{model_id}/versions/{version_id}"),
                    )?
                    .json(&Value::Object(body)),
                )
                .await
            {
                Ok(version) => return Ok(version),
                Err(RegistryError::Api { status, .. })
                    if is_revision_conflict(status) && attempt < 2 =>
                {
                    revision = self.versions(model_id).await?
                        .into_iter()
                        .find(|version| version.id == version_id)
                        .ok_or_else(|| RegistryError::Api {
                            status: StatusCode::NOT_FOUND,
                            message: format!("Registry version {version_id} disappeared during conflict retry"),
                        })?
                        .revision;
                }
                Err(error) => return Err(error),
            }
        }
        unreachable!("registry version update retry loop always returns")
    }

    pub(crate) async fn files(&self, model_id: &str) -> Result<Vec<RegistryFile>, RegistryError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/api/v1/models/{model_id}/files"),
        )?)
        .await
    }

    pub(crate) async fn add_file(
        &self,
        model_id: &str,
        payload: Value,
    ) -> Result<RegistryFile, RegistryError> {
        self.send_json(
            self.request(Method::POST, &format!("/api/v1/models/{model_id}/files"))?
                .json(&payload),
        )
        .await
    }

    pub(crate) async fn remove_file(
        &self,
        model_id: &str,
        file_id: &str,
    ) -> Result<(), RegistryError> {
        self.send_empty(self.request(
            Method::DELETE,
            &format!("/api/v1/models/{model_id}/files/{file_id}"),
        )?)
        .await
    }

    pub(crate) async fn tags(&self, model_id: &str) -> Result<Vec<String>, RegistryError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/api/v1/models/{model_id}/tags"),
        )?)
        .await
    }

    pub(crate) async fn add_tag(&self, model_id: &str, tag: &str) -> Result<Vec<String>, RegistryError> {
        self.send_json(
            self.request(Method::POST, &format!("/api/v1/models/{model_id}/tags"))?
                .json(&json!({ "tag": tag })),
        )
        .await
    }

    pub(crate) async fn remove_tag(
        &self,
        model_id: &str,
        tag: &str,
    ) -> Result<Vec<String>, RegistryError> {
        self.send_json(
            self.request(
                Method::DELETE,
                &format!("/api/v1/models/{model_id}/tags/{}", urlencoding::encode(tag)),
            )?,
        )
        .await
    }

    pub(crate) async fn sources(&self, model_id: &str) -> Result<Vec<RegistrySource>, RegistryError> {
        self.send_json(self.request(
            Method::GET,
            &format!("/api/v1/models/{model_id}/sources"),
        )?)
        .await
    }

    #[allow(dead_code)]
    pub(crate) async fn assets(&self, model_id: &str) -> Result<Vec<RegistryAsset>, RegistryError> {
        self.send_json(self.request(Method::GET, &format!("/api/v1/models/{model_id}/assets"))?).await
    }

    #[allow(dead_code)]
    pub(crate) async fn delete_asset(
        &self,
        model_id: &str,
        asset_id: &str,
    ) -> Result<(), RegistryError> {
        self.send_empty(self.request(
            Method::DELETE,
            &format!("/api/v1/models/{model_id}/assets/{asset_id}"),
        )?)
        .await
    }

    #[allow(dead_code)]
    pub(crate) async fn asset_content(
        &self,
        model_id: &str,
        asset_id: &str,
    ) -> Result<(String, Vec<u8>), RegistryError> {
        let request = self.request(
            Method::GET,
            &format!("/api/v1/models/{model_id}/assets/{asset_id}/content"),
        )?;
        let retry = request.try_clone();
        let response = match request.send().await {
            Ok(response) => response,
            Err(error) if error.is_connect() => {
                self.ensure_running().await?;
                retry
                    .ok_or_else(|| RegistryError::Http(error))?
                    .send()
                    .await?
            }
            Err(error) => return Err(RegistryError::Http(error)),
        };

        let status = response.status();
        if !status.is_success() {
            let message = response.text().await.unwrap_or_default();
            return Err(RegistryError::Api { status, message });
        }
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|value| value.to_str().ok())
            .unwrap_or("application/octet-stream")
            .to_string();
        let bytes = response.bytes().await?.to_vec();
        Ok((content_type, bytes))
    }

    #[allow(dead_code)]
    pub(crate) async fn upload_asset_content(
        &self,
        model_id: &str,
        kind: &str,
        source: Option<&str>,
        content_type: &str,
        bytes: Vec<u8>,
    ) -> Result<RegistryAsset, RegistryError> {
        let path = format!("/api/v1/models/{model_id}/assets/content");
        let mut request = self
            .request(Method::POST, &path)?
            .header("content-type", content_type)
            .header("x-raphael-asset-kind", kind)
            .body(bytes.clone());
        if let Some(source) = source {
            request = request.header("x-raphael-asset-source", source);
        }

        match self.send_json(request).await {
            Ok(asset) => Ok(asset),
            Err(RegistryError::Api { status, .. }) if status == StatusCode::NOT_FOUND => {
                // A previous Registry binary can still be healthy on the port
                // while not implementing the canonical binary asset endpoint.
                // Serialize replacement so concurrent uploads cannot kill the
                // newly-started Registry underneath each other.
                self.restart_stale_local_registry().await?;

                let mut retry = self
                    .request(Method::POST, &path)?
                    .header("content-type", content_type)
                    .header("x-raphael-asset-kind", kind)
                    .body(bytes);
                if let Some(source) = source {
                    retry = retry.header("x-raphael-asset-source", source);
                }
                self.send_json(retry).await
            }
            Err(error) => Err(error),
        }
    }

    async fn restart_stale_local_registry(&self) -> Result<(), RegistryError> {
        let _startup_guard = self.startup_lock.lock().await;

        // Another concurrent uploader may already have repaired the Registry.
        if self.is_ready().await {
            return Ok(());
        }

        if !self.is_healthy().await {
            drop(_startup_guard);
            return self.ensure_running().await;
        }

        self.stop_stale_local_registry_locked().await?;

        drop(_startup_guard);
        self.ensure_running().await
    }

    async fn stop_stale_local_registry_locked(&self) -> Result<(), RegistryError> {
        let (bind, _port) = local_http_registry_endpoint(&self.base_url)?;
        let pid_path = self
            .token_file
            .parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| RegistryError::ExecutableNotFound)?
            .join("registry.pid");

        let pid = fs::read_to_string(&pid_path)
            .ok()
            .and_then(|value| value.trim().parse::<u32>().ok());

        if let Some(pid) = pid {
            if pid != std::process::id() {
                terminate_local_registry_process(pid).await?;
            }
        } else if self.is_healthy().await {
            return Err(RegistryError::Api {
                status: StatusCode::CONFLICT,
                message: format!(
                    "Registry at {bind} is healthy but does not expose the current asset API, and its managed PID could not be determined"
                ),
            });
        }

        let deadline = tokio::time::Instant::now() + Duration::from_secs(15);
        while self.is_healthy().await {
            if tokio::time::Instant::now() >= deadline {
                return Err(RegistryError::Api {
                    status: StatusCode::CONFLICT,
                    message: "The stale local Registry process could not be stopped".into(),
                });
            }
            sleep(Duration::from_millis(150)).await;
        }

        Ok(())
    }

    #[allow(dead_code)]
    pub(crate) fn asset_content_url(&self, model_id: &str, asset_id: &str) -> String {
        format!("{}/api/v1/models/{model_id}/assets/{asset_id}/content", self.base_url)
    }

    pub(crate) async fn events(&self, after_id: i64) -> Result<Vec<RegistryEvent>, RegistryError> {
        self.send_json(
            self.request(Method::GET, "/api/v1/events/snapshot")?
                .query(&[("after_id", after_id)]),
        )
        .await
    }

    pub(crate) async fn add_source(
        &self,
        model_id: &str,
        payload: Value,
    ) -> Result<RegistrySource, RegistryError> {
        self.send_json(
            self.request(Method::POST, &format!("/api/v1/models/{model_id}/sources"))?
                .json(&payload),
        )
        .await
    }

}

fn local_http_registry_endpoint(base_url: &str) -> Result<(String, u16), RegistryError> {
    let parsed = Url::parse(base_url).map_err(|error| RegistryError::Api {
        status: StatusCode::BAD_REQUEST,
        message: format!("Invalid registry URL: {error}"),
    })?;

    if parsed.scheme() != "http" {
        return Err(RegistryError::AutoStartUnsupported);
    }

    let host = parsed.host_str().unwrap_or_default();
    let host = host.trim_matches(['[', ']']);
    let bind = match host {
        "localhost" | "127.0.0.1" => "127.0.0.1".to_string(),
        "::1" => "::1".to_string(),
        _ => return Err(RegistryError::AutoStartUnsupported),
    };

    Ok((bind, parsed.port().unwrap_or(43217)))
}

fn spawn_registry_executable(
    executable: &Path,
    bind: &str,
    port: u16,
    data_dir: &Path,
    token_file: &Path,
) -> Result<tokio::process::Child, RegistryError> {
    let mut command = Command::new(executable);
    command.arg("server");
    configure_registry_command(&mut command, bind, port, data_dir, token_file);
    Ok(command.spawn()?)
}

fn spawn_registry_cargo(
    manifest: &Path,
    bind: &str,
    port: u16,
    data_dir: &Path,
    token_file: &Path,
) -> Result<tokio::process::Child, RegistryError> {
    let mut command = Command::new("cargo");
    command
        .arg("run")
        .arg("--manifest-path")
        .arg(manifest)
        .arg("-p")
        .arg("registry-server")
        .arg("--")
        .arg("server");
    configure_registry_command(&mut command, bind, port, data_dir, token_file);
    Ok(command.spawn()?)
}

fn configure_registry_command(
    command: &mut Command,
    bind: &str,
    port: u16,
    data_dir: &Path,
    token_file: &Path,
) {
    command
        .env("RAPHAEL_REGISTRY_BIND", bind)
        .env("RAPHAEL_REGISTRY_PORT", port.to_string())
        .env("RAPHAEL_REGISTRY_DATA_DIR", data_dir);

    if let Ok(token) = fs::read_to_string(token_file) {
        let token = token.trim();
        if !token.is_empty() {
            command.env("RAPHAEL_REGISTRY_AUTH_TOKEN", token);
        }
    }

    #[cfg(windows)]
    {
        command.creation_flags(0x08000000);
    }
}

fn registry_command_on_path() -> Option<PathBuf> {
    let name = if cfg!(windows) {
        "raphael-registry.exe"
    } else {
        "raphael-registry"
    };

    std::env::var_os("PATH")?.to_str().and_then(|path| {
        std::env::split_paths(path)
            .map(|entry| entry.join(name))
            .find(|candidate| candidate.is_file())
    })
}

async fn terminate_local_registry_process(pid: u32) -> Result<(), RegistryError> {
    #[cfg(windows)]
    {
        let status = Command::new("taskkill")
            .args(["/PID", &pid.to_string(), "/T", "/F"])
            .status()
            .await?;
        if status.success() {
            return Ok(());
        }
        return Err(RegistryError::Api {
            status: StatusCode::CONFLICT,
            message: format!("taskkill could not terminate Registry process {pid}"),
        });
    }

    #[cfg(unix)]
    {
        let status = Command::new("kill")
            .args(["-TERM", &pid.to_string()])
            .status()
            .await?;
        if status.success() {
            return Ok(());
        }
        return Err(RegistryError::Api {
            status: StatusCode::CONFLICT,
            message: format!("kill could not terminate Registry process {pid}"),
        });
    }

    #[cfg(not(any(windows, unix)))]
    {
        let _ = pid;
        Err(RegistryError::Api {
            status: StatusCode::CONFLICT,
            message: "Automatic stale Registry replacement is unsupported on this platform".into(),
        })
    }
}

fn find_registry_executable() -> Option<PathBuf> {
    let binary_name = if cfg!(windows) {
        "raphael-registry.exe"
    } else {
        "raphael-registry"
    };
    let mut candidates = Vec::new();

    if let Ok(current_exe) = env::current_exe() {
        for ancestor in current_exe.ancestors() {
            if ancestor.file_name().and_then(|name| name.to_str()) == Some("Raphael-Model-Manager") {
                if let Some(projects) = ancestor.parent() {
                    // Development builds should prefer the local Registry
                    // workspace binary so source changes are used immediately.
                    candidates.push(
                        projects
                            .join("Raphael-Model-Registry")
                            .join("target/debug")
                            .join(binary_name),
                    );
                    candidates.push(
                        projects
                            .join("Raphael-Model-Registry")
                            .join("target/release")
                            .join(binary_name),
                    );
                }
            }
        }

        // Packaged Manager builds keep the bundled Registry beside the app.
        candidates.push(current_exe.parent().unwrap_or_else(|| Path::new(".")).join(binary_name));
        candidates.push(
            current_exe
                .parent()
                .unwrap_or_else(|| Path::new("."))
                .join("resources")
                .join(binary_name),
        );
    }

    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../Raphael-Model-Registry/target/debug")
            .join(binary_name),
    );
    candidates.push(
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../Raphael-Model-Registry/target/release")
            .join(binary_name),
    );

    candidates.into_iter().find(|path| path.is_file())
}

fn find_registry_workspace_manifest() -> Option<PathBuf> {
    let candidates = [
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../Raphael-Model-Registry/Cargo.toml"),
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../Raphael-Model-Registry/Cargo.toml"),
    ];

    candidates.into_iter().find(|path| path.is_file())
}

fn is_revision_conflict(status: StatusCode) -> bool {
    matches!(status, StatusCode::CONFLICT | StatusCode::PRECONDITION_FAILED)
}

fn validate_registry_base_url(base_url: &str) -> Result<(), RegistryError> {
    let parsed = url::Url::parse(base_url).map_err(|error| RegistryError::Api {
        status: StatusCode::BAD_REQUEST,
        message: format!("Invalid registry URL: {error}"),
    })?;

    if parsed.scheme() == "https" {
        return Ok(());
    }

    let host = parsed.host_str().unwrap_or_default();
    let loopback = matches!(host, "localhost" | "127.0.0.1" | "::1");
    if parsed.scheme() == "http" && loopback {
        return Ok(());
    }

    Err(RegistryError::Api {
        status: StatusCode::BAD_REQUEST,
        message: "Registry URL must use HTTPS unless it points to the local machine".into(),
    })
}

fn default_token_path(app_data: &Path) -> PathBuf {
    if let Some(local_app_data) = env::var_os("LOCALAPPDATA") {
        return PathBuf::from(local_app_data)
            .join("Raphael")
            .join("ModelRegistry")
            .join("data")
            .join("registry.token");
    }
    app_data.join("registry.token")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_pid_path_is_next_to_token_file() {
        let token = Path::new(r"C:\Users\test\AppData\Local\Raphael\ModelRegistry\data\registry.token");
        let pid = token.parent().unwrap().join("registry.pid");
        assert!(pid.ends_with("ModelRegistry\\data\\registry.pid"));
    }

    #[test]
    fn default_token_path_uses_app_data_when_localappdata_is_missing() {
        std::env::remove_var("LOCALAPPDATA");
        assert!(default_token_path(Path::new("app")).ends_with("registry.token"));
    }

    #[test]
    fn registry_url_requires_https_for_remote_endpoints() {
        assert!(validate_registry_base_url("http://127.0.0.1:43217").is_ok());
        assert!(validate_registry_base_url("http://localhost:43217").is_ok());
        assert!(validate_registry_base_url("https://registry.example.com").is_ok());
        assert!(validate_registry_base_url("http://registry.example.com").is_err());
    }

    #[test]
    fn auto_start_only_accepts_local_http_urls() {
        assert_eq!(
            local_http_registry_endpoint("http://127.0.0.1:43217").unwrap(),
            ("127.0.0.1".to_string(), 43217)
        );
        assert_eq!(
            local_http_registry_endpoint("http://localhost:45123").unwrap(),
            ("127.0.0.1".to_string(), 45123)
        );
        assert_eq!(
            local_http_registry_endpoint("http://[::1]:43217").unwrap(),
            ("::1".to_string(), 43217)
        );
        assert!(matches!(
            local_http_registry_endpoint("https://127.0.0.1:43217"),
            Err(RegistryError::AutoStartUnsupported)
        ));
        assert!(matches!(
            local_http_registry_endpoint("http://registry.example.com:43217"),
            Err(RegistryError::AutoStartUnsupported)
        ));
    }

    #[test]
    fn registry_workspace_manifest_is_detected_from_manager_layout() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../Raphael-Model-Registry/Cargo.toml");
        assert_eq!(
            find_registry_workspace_manifest().is_some(),
            path.is_file()
        );
    }

    #[test]
    fn revision_conflicts_include_precondition_failures() {
        assert!(is_revision_conflict(StatusCode::CONFLICT));
        assert!(is_revision_conflict(StatusCode::PRECONDITION_FAILED));
        assert!(!is_revision_conflict(StatusCode::NOT_FOUND));
    }
}