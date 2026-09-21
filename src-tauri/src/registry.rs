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

const DEFAULT_BASE_URL: &str = "http://127.0.0.1:43217";

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
        let mut body = match patch {
            Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        body.insert("expected_revision".into(), json!(expected_revision));
        self.send_json(
            self.request(Method::PATCH, &format!("/api/v1/models/{id}"))?
                .json(&Value::Object(body)),
        )
        .await
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
        let mut body = match patch {
            Value::Object(map) => map,
            _ => serde_json::Map::new(),
        };
        body.insert("expected_revision".into(), json!(expected_revision));
        self.send_json(
            self.request(
                Method::PATCH,
                &format!("/api/v1/models/{model_id}/versions/{version_id}"),
            )?
            .json(&Value::Object(body)),
        )
        .await
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
}
