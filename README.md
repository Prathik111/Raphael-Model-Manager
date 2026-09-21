# Raphael Model Manager

A local-first Windows desktop model manager for ComfyUI with Civitai integration, recursive model discovery, offline example-image caching, and a Raphael-themed interface. Logical model metadata is owned by the separate Raphael Model Registry service; Model Manager communicates with it only through the Registry HTTP API. Raphael Model Manager is a Registry consumer: the separate Raphael Model Registry is the authoritative source for logical model identity, versions, files, tags, sources, and metadata.

## Features

- Remembers the ComfyUI `models` folder after first launch.
- Watches the folder recursively so new, changed, and removed models appear automatically.
- Preserves nested folders under `checkpoints/`, `loras/`, `vae/`, `controlnet/`, `embeddings/`, and other ComfyUI model roots.
- Paste a Civitai model URL to preview, download, and install the correct model file.
- Reads and writes canonical model metadata, versions, tags, sources, and physical-file records through the Raphael Model Registry API.
- Downloads Civitai gallery images and thumbnails into a local cache so model pages remain useful offline.
- Keeps model files in their original ComfyUI location; local SQLite is only a runtime/projection cache and is never the authoritative model registry.
- Synchronizes installed physical files with the Raphael Model Registry over its versioned HTTP API.
- Consumes Registry change events so the local projection follows changes made by other Raphael clients.
- Ships as a Tauri 2 Windows desktop app with NSIS and MSI packaging.

## Development

Requirements:

- Node.js 22+
- Rust stable
- Windows WebView2

Install dependencies:

```powershell
npm install
```

Run the desktop app:

```powershell
npm run tauri:dev
```

Build the web frontend:

```powershell
npm run build
```

Build Windows installers:

```powershell
npm run tauri:build
```

## Release

Pushing a tag matching `v*` runs the GitHub Actions release workflow and publishes the Windows NSIS/MSI installers.

## Data locations

The configured ComfyUI model files are never copied into the app database.

App-managed data is stored under the normal Tauri app-data directory:

- `raphael.db` — local projection/settings/runtime cache. It is disposable and is not the canonical model registry.
- `cache/civitai/` — offline descriptions, gallery images, thumbnails, and local cover data

## Civitai authentication

Public model metadata and public downloads work without a token when Civitai permits them. A Civitai API token can be added from the app and is stored using the Windows credential store.

## Raphael Model Registry

The Manager communicates with the separate Raphael Model Registry through its HTTP API. The Registry owns logical model identity, model metadata, versions, file records, tags, sources, assets, compatibility, search, revisions, and durable events. Raphael Model Manager owns the physical ComfyUI files, filesystem watching, downloads, local image cache, and UI.

### Architecture and verification

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the current runtime/module boundaries. Frontend unit tests run with Vitest via `npm run test`; the Integrity workflow runs those tests together with the Rust checks, Clippy, production build, and Windows Tauri packaging.

### LAN web app security

The optional LAN web app binds to the host's LAN address and protects every `/api/*` endpoint with an installation-specific 256-bit access token. The token is delivered in the URL fragment (`#access_token=...`) so it is not sent as part of the HTTP request URL, and the web client exchanges it for a same-origin, HttpOnly session cookie after startup.

The web API is intentionally not CORS-open. Do not share the generated web-app URL with untrusted users because it grants access to the Manager's API for as long as the web app remains enabled.

By default Raphael connects to:

```text
http://127.0.0.1:43217
```

Override the connection with:

```powershell
$env:RAPHAEL_REGISTRY_URL = "http://127.0.0.1:43217"
$env:RAPHAEL_REGISTRY_TOKEN_FILE = "C:\path\to\registry.token"
# or use RAPHAEL_REGISTRY_AUTH_TOKEN for an explicit bearer token
```

Raphael does not open the Registry SQLite database directly. All Registry mutations go through the `/api/v1` HTTP contract.
