use super::*;

pub(crate) fn normalize_tags(tags: Vec<String>) -> Vec<String> {
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

pub(crate) fn tokenize_search(query: &str) -> Vec<String> {
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

pub(crate) fn value_contains(haystack: &str, needle: &str) -> bool {
    haystack.to_ascii_lowercase().contains(&needle.to_ascii_lowercase())
}

pub(crate) fn model_search_match(model: &ModelRecord, query: &str, active_tags: &[String]) -> bool {
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
pub(crate) fn list_models(app:State<AppStateInner>, r#type:Option<String>, query:Option<String>, tags:Option<Vec<String>>)->AppResult<Vec<ModelRecord>>{
    let c=open_db(&app.app_data)?;
    let mut sql=format!("{MODEL_SELECT} WHERE 1=1");
    let mut args:Vec<String>=vec![];
    if let Some(t)=r#type {
        sql.push_str(" AND LOWER(TRIM(model_type))=LOWER(TRIM(?))");
        args.push(t);
    }
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
pub(crate) fn get_tags(app:State<AppStateInner>)->AppResult<Vec<TagRecord>>{
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
pub(crate) async fn set_model_tags_inner(
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

    let existing = app.registry.tags(&registry_model_id).await?;
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
pub(crate) async fn set_model_tags(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    tags: Vec<String>,
) -> AppResult<ModelRecord> {
    set_model_tags_inner(app.inner(), handle, id, tags).await
}

#[tauri::command]
pub(crate) async fn set_model_name(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    name: String,
) -> AppResult<ModelRecord> {
    let name = name.trim().to_string();
    if name.is_empty() {
        return Err(AppError::Invalid("Model name cannot be empty".into()));
    }
    if name.chars().count() > 200 {
        return Err(AppError::Invalid("Model name cannot exceed 200 characters".into()));
    }

    let _ = sync_local_model_to_registry(app.inner(), id).await?;

    let registry_model_id: String = {
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
            serde_json::json!({ "name": name }),
        )
        .await?;

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET civitai_name=?2,civitai_name_user_modified=1,updated_at=?3 WHERE id=?1",
        params![id, name, now()],
    )?;
    drop(c);

    let rec = model_by_id(&open_db(&app.app_data)?, id)?;
    emit_models_changed(&handle);
    Ok(rec)
}

#[tauri::command]
pub(crate) async fn set_model_description(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
    id: i64,
    description: String,
) -> AppResult<ModelRecord> {
    let description = description.trim().to_string();
    let _ = sync_local_model_to_registry(app.inner(), id).await?;

    let registry_model_id: String = {
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
            serde_json::json!({
                "description": if description.is_empty() {
                    serde_json::Value::Null
                } else {
                    serde_json::Value::String(description.clone())
                }
            }),
        )
        .await?;

    let c = open_db(&app.app_data)?;
    c.execute(
        "UPDATE models SET description=?2,description_user_modified=1,updated_at=?3 WHERE id=?1",
        rusqlite::params![
            id,
            if description.is_empty() {
                None::<String>
            } else {
                Some(description)
            },
            now()
        ],
    )?;
    drop(c);

    let rec = model_by_id(&open_db(&app.app_data)?, id)?;
    emit_models_changed(&handle);
    Ok(rec)
}

#[derive(Debug, Serialize, Clone)]
pub struct TagRefreshResult {
    pub models_scanned: i64,
    pub models_updated: i64,
    pub tags_added: i64,
    pub failures: i64,
}

#[tauri::command]
pub(crate) async fn refetch_all_model_tags(
    app: State<'_, AppStateInner>,
    handle: AppHandle,
) -> AppResult<TagRefreshResult> {
    let candidates: Vec<(i64, String, Vec<String>)> = {
        let c = open_db(&app.app_data)?;
        let mut stmt = c.prepare(
            "SELECT id,civitai_url,tags_json
             FROM models
             WHERE civitai_url IS NOT NULL AND TRIM(civitai_url) <> ''",
        )?;
        let rows = stmt.query_map([], |r| {
            let raw: String = r.get(2)?;
            let tags = serde_json::from_str::<Vec<String>>(&raw).unwrap_or_default();
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?, tags))
        })?;
        rows.collect::<Result<Vec<_>, _>>()?
    };

    let mut result = TagRefreshResult {
        models_scanned: candidates.len() as i64,
        models_updated: 0,
        tags_added: 0,
        failures: 0,
    };

    for (id, url, local_tags) in candidates {
        let (model, version) = match fetch_model_and_version(&app, &url).await {
            Ok(value) => value,
            Err(_) => {
                result.failures += 1;
                continue;
            }
        };

        let fetched_tags = civitai_model_tags(&model, &version);

        let _ = sync_local_model_to_registry(&app, id).await;
        let registry_model_id: String = {
            let c = open_db(&app.app_data)?;
            c.query_row(
                "SELECT registry_model_id FROM models WHERE id=?1",
                [id],
                |r| r.get::<_, String>(0),
            )?
        };

        let registry_tags = app.registry.tags(&registry_model_id).await.unwrap_or_default();
        let before = normalize_tags(local_tags);
        let mut merged = before.clone();
        merged.extend(registry_tags.clone());
        merged.extend(fetched_tags);
        let merged = normalize_tags(merged);

        let added_to_local = merged
            .iter()
            .filter(|tag| !before.iter().any(|existing| existing.eq_ignore_ascii_case(tag)))
            .count() as i64;

        for tag in &merged {
            if !registry_tags.iter().any(|existing| existing.eq_ignore_ascii_case(tag)) {
                if app.registry.add_tag(&registry_model_id, tag).await.is_err() {
                    result.failures += 1;
                }
            }
        }

        if merged != before {
            let c = open_db(&app.app_data)?;
            c.execute(
                "UPDATE models SET tags_json=?2,updated_at=?3 WHERE id=?1",
                params![
                    id,
                    serde_json::to_string(&merged).unwrap_or_else(|_| "[]".into()),
                    now()
                ],
            )?;
            result.models_updated += 1;
            result.tags_added += added_to_local;
        }
    }

    emit_models_changed(&handle);
    Ok(result)
}

pub(crate) fn add_subfolder_tags_inner(app: &AppStateInner) -> AppResult<i64> {
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
pub(crate) fn add_subfolder_tags(app: State<AppStateInner>, handle: AppHandle) -> AppResult<i64> {
    let updated = add_subfolder_tags_inner(&app)?;
    spawn_registry_sync(app.inner().clone(), handle.clone());
    emit_models_changed(&handle);
    Ok(updated)
}
