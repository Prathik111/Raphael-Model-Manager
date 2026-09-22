# Raphael Model Manager Architecture

## Runtime boundaries

Raphael is a single Tauri desktop application with a React/TypeScript frontend and a Rust backend. The desktop backend owns the local SQLite index, filesystem scanning, cache, Civitai integration, registry synchronization, and the optional authenticated LAN web API.

The LAN web API is a transport adapter over the same application services; it is not a second business-logic implementation. API requests are authenticated before command dispatch, and the static web assets remain loadable without exposing the command surface.

At desktop startup, `registry.rs` health-checks the configured Registry. For a local HTTP Registry, the Manager can start the local `raphael-registry` process (or the sibling Registry workspace through Cargo), sharing the configured data directory and bearer token. Initial Registry synchronization is started only after the health check succeeds. Remote and HTTPS Registry endpoints remain externally managed.

## Rust module boundaries

- `lib.rs` contains application wiring, shared state, filesystem scanning, registry synchronization orchestration, Civitai operations, command registration, and startup.
- `library.rs` contains model search, tag aggregation, model-tag editing, and subfolder-tag operations.
- `cache.rs` contains cache accounting, relocation, cleanup, eviction, orphan removal, and cache settings.
- `registry.rs` contains the HTTP client and registry data contracts.
- `web.rs` contains the authenticated LAN web server, web tasks, command transport, and static asset serving.

The module split is intentionally incremental: behavior remains centralized around the existing `AppStateInner` and database layer while high-churn feature groups are isolated from the application entry module.

## Frontend boundaries

`App.tsx` remains the top-level screen/state coordinator. Pure model mapping and formatting helpers live in `src/utils/model.ts`, which can be tested without a browser or Tauri runtime.

## Concurrency rules

- Do not hold `cache_lock` across registry or Civitai network requests.
- Cache-mutating filesystem/database operations acquire `cache_lock` for their critical section.
- Filesystem watcher callbacks schedule scans asynchronously instead of blocking the notify callback thread.
- Registry synchronization uses explicit revision checks and retries only for revision conflicts.

## Verification

Frontend unit tests run through Vitest. Rust tests, Clippy, frontend typechecking/build, and Windows Tauri packaging remain part of the Integrity workflow.

## Known follow-up architecture work

The two largest coordinators (`lib.rs` and `App.tsx`) still contain substantial orchestration logic. Further extraction should be done by cohesive capability (Civitai service, registry synchronization, download service, and individual React screens/hooks) rather than by arbitrary line-count splitting.
