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
        })
    }

    pub(crate) async fn ensure_running(&self) -> Result<(), RegistryError> {
        if self.is_authenticated().await {
            return Ok(());
        }

        if self.is_healthy().await {
            let _ = self.token()?;
            return Err(RegistryError::Api {
                status: StatusCode::UNAUTHORIZED,
                message: "Registry is reachable but rejected the configured authentication token".into(),
            });
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
                if self.is_healthy().await {
                    return Ok::<(), RegistryError>(());
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

    pub(crate) async fn is_authenticated(&self) -> bool {
        let request = match self.request(Method::GET, "/api/v1/events/snapshot") {
            Ok(request) => request.query(&[("after_id", 0)]),
            Err(_) => return false,
        };

        request
            .send()
            .await
            .map(|response| response.status().is_success())
            .unwrap_or(false)
    }

    fn token(&self) -> Result<String, RegistryError> {
        if let Ok(value) = env::var("RAPHAEL_REGISTRY_AUTH_TOKEN") {
            if !value.trim().is_empty() {
                return Ok(value.trim().to_string());
            }
        }

        let mut token_files = vec![self.token_file.clone()];
        for lock_file in self.registry_lock_candidates() {
            if let Some(token_file) = token_file_from_lock(&lock_file) {
                token_files.push(token_file);
            }
        }

        for token_file in token_files {
            let Ok(value) = fs::read_to_string(&token_file) else {
                continue;
            };
            let token = value.trim();
            if !token.is_empty() {
                return Ok(token.to_string());
            }
        }

        Err(RegistryError::MissingToken)
    }

    fn registry_lock_candidates(&self) -> Vec<PathBuf> {
        let mut candidates = Vec::new();

        if let Some(parent) = self.token_file.parent() {
            candidates.push(parent.join("registry.lock"));
        }

        if let Some(data_dir) = env::var_os("RAPHAEL_REGISTRY_DATA_DIR") {
            candidates.push(PathBuf::from(data_dir).join("registry.lock"));
        }

        candidates
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
        let response = request.send().await?;
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
        let response = request.send().await?;
        let status = response.status();
        if status.is_success() {
            return Ok(());
        }
        let message = response.text().await.unwrap_or_default();
        Err(RegistryError::Api { status, message })
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

#[derive(Debug, serde::Deserialize)]
struct RegistryLockInfo {
    token_file: Option<PathBuf>,
}

fn token_file_from_lock(lock_file: &Path) -> Option<PathBuf> {
    let content = fs::read_to_string(lock_file).ok()?;
    let info: RegistryLockInfo = serde_json::from_str(&content).ok()?;
    let token_file = info.token_file?;
    if token_file.is_absolute() {
        Some(token_file)
    } else {
        lock_file.parent().map(|parent| parent.join(token_file))
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

fn find_registry_executable() -> Option<PathBuf> {
    let binary_name = if cfg!(windows) {
        "raphael-registry.exe"
    } else {
        "raphael-registry"
    };
    let mut candidates = Vec::new();

    if let Ok(current_exe) = env::current_exe() {
        if let Some(parent) = current_exe.parent() {
            candidates.push(parent.join(binary_name));
            candidates.push(parent.join("resources").join(binary_name));
        }

        for ancestor in current_exe.ancestors() {
            if ancestor.file_name().and_then(|name| name.to_str()) == Some("Raphael-Model-Manager") {
                if let Some(projects) = ancestor.parent() {
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