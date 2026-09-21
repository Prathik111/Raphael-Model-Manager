use super::*;

pub(crate) fn cache_file_counts(path: &Path) -> (i64, i64) {
    if !path.exists() { return (0, 0); }
    let mut files = 0i64;
    let mut bytes = 0i64;
    for entry in WalkDir::new(path).follow_links(false).into_iter().filter_map(Result::ok) {
        if let Ok(meta) = entry.metadata() {
            if meta.is_file() {
                files += 1;
                bytes += meta.len() as i64;
            }
        }
    }
    (files, bytes)
}
pub(crate) fn cache_stats_inner(app_data: &Path) -> AppResult<CacheStats> {
    let c = open_db(app_data)?;
    let root = cache_root(app_data);
    let max_bytes = read_cache_max_bytes(&c)?;
    let mut stats = CacheStats {
        location: root.to_string_lossy().to_string(),
        used_bytes: 0,
        max_bytes,
        over_limit: false,
        files: 0,
        image_files: 0,
        image_bytes: 0,
        featured_files: 0,
        featured_bytes: 0,
        gallery_files: 0,
        gallery_bytes: 0,
        thumbnail_files: 0,
        thumbnail_bytes: 0,
        cover_files: 0,
        cover_bytes: 0,
        other_files: 0,
        other_bytes: 0,
    };
    if !root.exists() { return Ok(stats); }

    for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok) {
        let meta = match entry.metadata() {
            Ok(value) if value.is_file() => value,
            _ => continue,
        };
        let bytes = meta.len() as i64;
        stats.files += 1;
        stats.used_bytes += bytes;
        let relative = entry.path().strip_prefix(&root).unwrap_or(entry.path());
        let components: Vec<String> = relative.components()
            .filter_map(|c| c.as_os_str().to_str().map(|s| s.to_ascii_lowercase()))
            .collect();

        if components.first().map(String::as_str) == Some("covers") {
            stats.cover_files += 1;
            stats.cover_bytes += bytes;
        } else if components.first().map(String::as_str) == Some("civitai") {
            stats.image_files += 1;
            stats.image_bytes += bytes;
            let name = entry.file_name().to_string_lossy().to_ascii_lowercase();
            if components.iter().any(|c| c == "featured") ||
               components.iter().any(|c| c == "featured.__staging" || c == "featured.__backup") {
                stats.featured_files += 1;
                stats.featured_bytes += bytes;
            } else if name.starts_with("thumbnail.") || name.contains("_thumb.") {
                stats.thumbnail_files += 1;
                stats.thumbnail_bytes += bytes;
            } else {
                stats.gallery_files += 1;
                stats.gallery_bytes += bytes;
            }
        } else {
            stats.other_files += 1;
            stats.other_bytes += bytes;
        }
    }
    stats.over_limit = stats.max_bytes > 0 && stats.used_bytes > stats.max_bytes;
    Ok(stats)
}

pub(crate) fn remove_path_with_stats(path: &Path) -> AppResult<(i64, i64)> {
    if !path.exists() { return Ok((0, 0)); }
    let (files, bytes) = cache_file_counts(path);
    if path.is_dir() { fs::remove_dir_all(path)?; } else { fs::remove_file(path)?; }
    Ok((files, bytes))
}

pub(crate) fn copy_dir_recursive(source: &Path, target: &Path) -> AppResult<()> {
    fs::create_dir_all(target)?;
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let src = entry.path();
        let dst = target.join(entry.file_name());
        let kind = entry.file_type()?;
        if kind.is_dir() {
            copy_dir_recursive(&src, &dst)?;
        } else if kind.is_file() {
            fs::copy(&src, &dst)?;
        } else if kind.is_symlink() {
            return Err(AppError::Invalid(format!(
                "Cache relocation encountered an unsupported symbolic link: {}",
                src.to_string_lossy()
            )));
        }
    }
    Ok(())
}

pub(crate) fn rewrite_cache_paths(
    c: &mut Connection,
    old_root: &Path,
    new_root: &Path,
    cache_bytes: i64,
) -> AppResult<()> {
    let old = old_root.to_string_lossy().to_string();
    let new = new_root.to_string_lossy().to_string();
    let separator = std::path::MAIN_SEPARATOR.to_string();
    let alternate_separator = if separator == "/" { "\\".to_string() } else { "/".to_string() };
    let tx = c.transaction()?;

    for column in ["local_path", "thumbnail_path"] {
        let sql = format!(
            "UPDATE images SET {column}=?2 || substr({column}, length(?1)+1)
             WHERE {column}=?1
                OR substr({column},1,length(?1)+1)=?1 || ?3
                OR substr({column},1,length(?1)+1)=?1 || ?4"
        );
        tx.execute(&sql, params![old, new, separator, alternate_separator])?;
    }
    for column in ["thumbnail_path", "cover_path"] {
        let sql = format!(
            "UPDATE models SET {column}=?2 || substr({column}, length(?1)+1)
             WHERE {column}=?1
                OR substr({column},1,length(?1)+1)=?1 || ?3
                OR substr({column},1,length(?1)+1)=?1 || ?4"
        );
        tx.execute(&sql, params![old, new, separator, alternate_separator])?;
    }
    tx.execute(
        "INSERT INTO settings(key,value) VALUES('cache_location',?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [&new],
    )?;
    tx.execute(
        "INSERT INTO settings(key,value) VALUES('cache_bytes',?1)
         ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        [cache_bytes.to_string()],
    )?;
    tx.commit()?;
    Ok(())
}

pub(crate) fn delete_image_records(c: &mut Connection, cache_root: &Path, ids: &[i64]) -> AppResult<(i64, i64)> {
    if ids.is_empty() { return Ok((0, 0)); }
    let mut paths = HashSet::<PathBuf>::new();
    for id in ids {
        if let Ok((local, thumb)) = c.query_row(
            "SELECT local_path,thumbnail_path FROM images WHERE id=?1",
            [id],
            |r| Ok((r.get::<_, Option<String>>(0)?, r.get::<_, Option<String>>(1)?)),
        ) {
            if let Some(value) = local { paths.insert(PathBuf::from(value)); }
            if let Some(value) = thumb { paths.insert(PathBuf::from(value)); }
        }
    }
    {
        let tx = c.transaction()?;
        for id in ids { tx.execute("DELETE FROM images WHERE id=?1", [id])?; }
        tx.commit()?;
    }

    let canonical_root = cache_root.canonicalize().unwrap_or_else(|_| cache_root.to_path_buf());
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    for path in paths {
        if !path.is_file() { continue; }
        let canonical_path = match path.canonicalize() { Ok(value) => value, Err(_) => continue };
        if !canonical_path.starts_with(&canonical_root) { continue; }
        let value = path.to_string_lossy().to_string();
        let image_refs: i64 = c.query_row(
            "SELECT COUNT(*) FROM images WHERE local_path=?1 OR thumbnail_path=?1",
            [&value], |r| r.get(0),
        )?;
        let model_refs: i64 = c.query_row(
            "SELECT COUNT(*) FROM models WHERE thumbnail_path=?1 OR cover_path=?1",
            [&value], |r| r.get(0),
        )?;
        if image_refs == 0 && model_refs == 0 {
            let bytes = fs::metadata(&path).map(|m| m.len() as i64).unwrap_or(0);
            fs::remove_file(&path)?;
            deleted_files += 1;
            freed_bytes += bytes;
        }
    }
    Ok((deleted_files, freed_bytes))
}

pub(crate) fn referenced_cache_paths(c: &Connection) -> AppResult<HashSet<PathBuf>> {
    let mut paths = HashSet::new();
    for sql in [
        "SELECT local_path FROM images WHERE local_path IS NOT NULL",
        "SELECT thumbnail_path FROM images WHERE thumbnail_path IS NOT NULL",
        "SELECT thumbnail_path FROM models WHERE thumbnail_path IS NOT NULL",
        "SELECT cover_path FROM models WHERE cover_path IS NOT NULL",
    ] {
        let mut stmt = c.prepare(sql)?;
        let rows = stmt.query_map([], |r| r.get::<_, String>(0))?;
        for value in rows.flatten() {
            let path = PathBuf::from(value);
            if let Ok(canonical) = path.canonicalize() {
                paths.insert(canonical);
            }
            paths.insert(path);
        }
    }
    Ok(paths)
}

pub(crate) fn clean_cache_orphans_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let c = open_db(app_data)?;
    let references = referenced_cache_paths(&c)?;
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    if root.exists() {
        for entry in WalkDir::new(&root).follow_links(false).into_iter().filter_map(Result::ok) {
            let path = entry.path();
            let referenced = references.contains(path)
                || path.canonicalize().map(|canonical| references.contains(&canonical)).unwrap_or(false);
            if !path.is_file() || referenced { continue; }
            let bytes = fs::metadata(path).map(|m| m.len() as i64).unwrap_or(0);
            if fs::remove_file(path).is_ok() {
                deleted_files += 1;
                freed_bytes += bytes;
            }
        }
    }
    for stale in [root.join("civitai").join("featured.__staging"),root.join("civitai").join("featured.__backup")] {
        if let Ok((files,bytes))=remove_path_with_stats(&stale) {
            deleted_files+=files;
            freed_bytes+=bytes;
        }
    }
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

pub(crate) fn enforce_cache_limit_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let initial = cache_stats_inner(app_data)?;
    if initial.max_bytes == 0 || initial.used_bytes <= initial.max_bytes {
        return Ok(CacheOperationResult { deleted_files: 0, freed_bytes: 0, remaining_bytes: initial.used_bytes, over_limit: false });
    }

    let mut c = open_db(app_data)?;
    let root = cache_root(app_data);
    let mut stmt = c.prepare(
        "SELECT i.id FROM images i
         WHERE NOT EXISTS (SELECT 1 FROM models m WHERE m.cover_source_image_id=i.id)
         ORDER BY CASE WHEN i.meta_json LIKE '%\"featured\":true%' THEN 1 ELSE 0 END, i.cached_at ASC, i.id ASC"
    )?;
    let ids: Vec<i64> = stmt.query_map([], |r| r.get(0))?.filter_map(Result::ok).collect();
    drop(stmt);

    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;
    let mut current_bytes = initial.used_bytes;
    for id in ids {
        if current_bytes <= initial.max_bytes { break; }
        let (files, bytes) = delete_image_records(&mut c, &root.join("civitai"), &[id])?;
        deleted_files += files;
        freed_bytes += bytes;
        current_bytes = current_bytes.saturating_sub(bytes);
    }

    if current_bytes > initial.max_bytes {
        let result = clean_cache_orphans_inner(app_data)?;
        deleted_files += result.deleted_files;
        freed_bytes += result.freed_bytes;
    }

    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

pub(crate) fn set_cache_max_bytes_inner(app_data: &Path, value: i64) -> AppResult<CacheStats> {
    let value = value.max(0);
    let c = open_db(app_data)?;
    put_setting(&c, "cache_max_bytes", &value.to_string())?;
    drop(c);
    let _ = enforce_cache_limit_inner(app_data)?;
    cache_stats_inner(app_data)
}

pub(crate) fn clear_cache_images_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let image_root = root.join("civitai");
    let (deleted_files, freed_bytes) = remove_path_with_stats(&image_root)?;
    fs::create_dir_all(&image_root)?;
    let c = open_db(app_data)?;
    c.execute_batch("DELETE FROM images;")?;
    c.execute("UPDATE models SET thumbnail_path=NULL,cover_source_image_id=NULL", [])?;
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

pub(crate) fn clear_complete_cache_inner(app_data: &Path) -> AppResult<CacheOperationResult> {
    let root = cache_root(app_data);
    let (deleted_files, freed_bytes) = remove_path_with_stats(&root)?;
    fs::create_dir_all(root.join("civitai"))?;
    fs::create_dir_all(root.join("covers"))?;
    let c = open_db(app_data)?;
    c.execute_batch("DELETE FROM images;")?;
    c.execute(
        "UPDATE models SET thumbnail_path=NULL,cover_path=NULL,cover_source_image_id=NULL",
        [],
    )?;
    let _ = put_setting(&c, "cache_bytes", "0");
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

pub(crate) fn prune_cache_images_inner(app_data: &Path, keep_per_model: i64) -> AppResult<CacheOperationResult> {
    let keep = keep_per_model.clamp(0, 10000);
    let mut c = open_db(app_data)?;
    let root = cache_root(app_data);
    let model_ids: Vec<i64> = {
        let mut stmt = c.prepare("SELECT DISTINCT model_id FROM images ORDER BY model_id")?;
        let rows = stmt.query_map([], |r| r.get(0))?;
        rows.filter_map(Result::ok).collect()
    };
    let mut deleted_files = 0i64;
    let mut freed_bytes = 0i64;

    for model_id in model_ids {
        let mut stmt = c.prepare(
            "SELECT i.id FROM images i
             WHERE i.model_id=?1
               AND NOT EXISTS (SELECT 1 FROM models m WHERE m.cover_source_image_id=i.id)
             ORDER BY CASE WHEN i.meta_json LIKE '%\"featured\":true%' THEN 0 ELSE 1 END, i.cached_at DESC, i.id DESC"
        )?;
        let ids: Vec<i64> = stmt.query_map([model_id], |r| r.get(0))?.filter_map(Result::ok).collect();
        drop(stmt);
        if ids.len() <= keep as usize { continue; }
        let remove_ids: Vec<i64> = ids.into_iter().skip(keep as usize).collect();
        let (files, bytes) = delete_image_records(&mut c, &root.join("civitai"), &remove_ids)?;
        deleted_files += files;
        freed_bytes += bytes;
    }

    let orphan_result=clean_cache_orphans_inner(app_data)?;
    deleted_files+=orphan_result.deleted_files;
    freed_bytes+=orphan_result.freed_bytes;
    let stats = cache_stats_inner(app_data)?;
    Ok(CacheOperationResult { deleted_files, freed_bytes, remaining_bytes: stats.used_bytes, over_limit: stats.over_limit })
}

pub(crate) fn set_cache_location_inner(app_data: &Path, path: &str) -> AppResult<CacheStats> {
    let target = PathBuf::from(path.trim());
    if target.as_os_str().is_empty() || !target.is_absolute() {
        return Err(AppError::Invalid("Cache location must be an absolute folder path".into()));
    }
    fs::create_dir_all(&target)?;
    let old = cache_root(app_data);
    if !old.exists() { fs::create_dir_all(&old)?; }
    let old_canonical = old.canonicalize().unwrap_or_else(|_| old.clone());
    let target_canonical = target.canonicalize().unwrap_or_else(|_| target.clone());

    if old_canonical == target_canonical {
        let mut c = open_db(app_data)?;
        let bytes = dir_size(&target_canonical);
        rewrite_cache_paths(&mut c, &old, &target, bytes)?;
        return cache_stats_inner(app_data);
    }
    if target_canonical.starts_with(&old_canonical) || old_canonical.starts_with(&target_canonical) {
        return Err(AppError::Invalid("The new cache location cannot contain the current cache location or be inside it".into()));
    }
    if fs::read_dir(&target_canonical)?.next().is_some() {
        return Err(AppError::Invalid("Choose an empty folder for the new Raphael cache location".into()));
    }

    let before = cache_file_counts(&old);
    copy_dir_recursive(&old, &target_canonical)?;
    let after = cache_file_counts(&target_canonical);
    if before != after {
        let _ = fs::remove_dir_all(&target_canonical);
        return Err(AppError::Invalid("Cache relocation verification failed; the original cache was preserved".into()));
    }

    let db_update = (|| {
        let mut c = open_db(app_data)?;
        let bytes = dir_size(&target_canonical);
        rewrite_cache_paths(&mut c, &old, &target, bytes)?;
        Ok::<(), AppError>(())
    })();

    if let Err(error) = db_update {
        let _ = fs::remove_dir_all(&target_canonical);
        return Err(error);
    }

    let _ = fs::remove_dir_all(&old);
    cache_stats_inner(app_data)
}
