# Raphael Model Manager

A local-first Windows desktop model manager for ComfyUI with Civitai integration, recursive model discovery, offline example-image caching, and a Raphael-themed interface.

## Features

- Remembers the ComfyUI `models` folder after first launch.
- Watches the folder recursively so new, changed, and removed models appear automatically.
- Preserves nested folders under `checkpoints/`, `loras/`, `vae/`, `controlnet/`, `embeddings/`, and other ComfyUI model roots.
- Paste a Civitai model URL to preview, download, and install the correct model file.
- Stores Civitai descriptions, tags, activation prompts, version metadata, and example generation parameters.
- Downloads Civitai gallery images and thumbnails into a local cache so model pages remain useful offline.
- Keeps model files in their original ComfyUI location; the SQLite database and cache live in app data.
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

- `raphael.db` — SQLite index and settings
- `cache/civitai/` — offline descriptions, gallery images, thumbnails, and metadata

## Civitai authentication

Public model metadata and public downloads work without a token when Civitai permits them. A Civitai API token can be added from the app and is stored using the Windows credential store.
