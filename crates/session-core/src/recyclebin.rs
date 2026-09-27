use serde::{Deserialize, Serialize};
use std::fs;
use std::path::PathBuf;

use crate::app_dir::{get_recyclebin_items_dir, get_recyclebin_manifest_path};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecycledRelatedPath {
    pub original_path: String,
    pub stored_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecycledItem {
    pub id: String,
    pub item_type: String,
    pub reason: String,
    pub source: String,
    pub project_id: String,
    pub session_title: Option<String>,
    pub project_name: Option<String>,
    pub original_path: String,
    pub stored_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub companion_original_path: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub companion_stored_name: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_paths: Vec<RecycledRelatedPath>,
    pub moved_at: String,
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RecyclebinManifest {
    pub version: u32,
    pub items: Vec<RecycledItem>,
}

impl Default for RecyclebinManifest {
    fn default() -> Self {
        RecyclebinManifest {
            version: 1,
            items: vec![],
        }
    }
}

fn generate_id() -> String {
    use std::time::{SystemTime, UNIX_EPOCH};
    let ts = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("{:x}", ts)
}

pub fn load_manifest() -> RecyclebinManifest {
    let path = match get_recyclebin_manifest_path() {
        Some(p) => p,
        None => return RecyclebinManifest::default(),
    };
    if !path.exists() {
        return RecyclebinManifest::default();
    }
    let data = match fs::read_to_string(&path) {
        Ok(d) => d,
        Err(_) => return RecyclebinManifest::default(),
    };
    serde_json::from_str(&data).unwrap_or_default()
}

pub fn save_manifest(manifest: &RecyclebinManifest) -> Result<(), String> {
    let path = get_recyclebin_manifest_path()
        .ok_or_else(|| "Cannot determine recyclebin path".to_string())?;

    // Ensure directory exists
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)
            .map_err(|e| format!("Failed to create recyclebin dir: {}", e))?;
    }

    let json = serde_json::to_string_pretty(manifest)
        .map_err(|e| format!("Failed to serialize manifest: {}", e))?;
    let tmp_path = path.with_extension("json.tmp");
    fs::write(&tmp_path, &json).map_err(|e| format!("Failed to write manifest tmp: {}", e))?;
    fs::rename(&tmp_path, &path).map_err(|e| format!("Failed to rename manifest: {}", e))?;
    Ok(())
}

/// 移动文件或目录到回收站 items/ 目录，追加 manifest，返回生成的 id。
pub fn move_to_recyclebin(
    original_path: &std::path::Path,
    item_type: &str,
    reason: &str,
    source: &str,
    project_id: &str,
    session_title: Option<String>,
    project_name: Option<String>,
) -> Result<String, String> {
    move_paths_to_recyclebin(
        &[original_path.to_path_buf()],
        item_type,
        reason,
        source,
        project_id,
        session_title,
        project_name,
    )
}

/// Move a primary session path and every provider-owned related path as one
/// recoverable item. All moves are rolled back if any path or manifest update
/// fails, so a session cannot be left half deleted.
pub fn move_paths_to_recyclebin(
    original_paths: &[PathBuf],
    item_type: &str,
    reason: &str,
    source: &str,
    project_id: &str,
    session_title: Option<String>,
    project_name: Option<String>,
) -> Result<String, String> {
    let original_path = original_paths
        .first()
        .ok_or_else(|| "At least one recyclebin path is required".to_string())?;
    let items_dir = get_recyclebin_items_dir()
        .ok_or_else(|| "Cannot determine recyclebin items path".to_string())?;
    fs::create_dir_all(&items_dir)
        .map_err(|e| format!("Failed to create recyclebin items dir: {}", e))?;

    let id = generate_id();

    // 计算 stored_name：目录用 id/，文件用 id.ext
    let stored_name = if original_path.is_dir() {
        id.clone()
    } else {
        match original_path.extension().and_then(|e| e.to_str()) {
            Some(ext) => format!("{}.{}", id, ext),
            None => id.clone(),
        }
    };

    let mut moves = Vec::with_capacity(original_paths.len());
    for (index, path) in original_paths.iter().enumerate() {
        let metadata = fs::symlink_metadata(path).map_err(|error| {
            format!(
                "Failed to inspect recyclebin source {}: {error}",
                path.display()
            )
        })?;
        if metadata.file_type().is_symlink() {
            return Err(format!(
                "Recyclebin source cannot be a symbolic link: {}",
                path.display()
            ));
        }
        if !metadata.is_file() && !metadata.is_dir() {
            return Err(format!(
                "Recyclebin source has unsupported type: {}",
                path.display()
            ));
        }
        let name = if index == 0 {
            stored_name.clone()
        } else {
            format!("{id}.related.{:05}", index - 1)
        };
        let target = items_dir.join(&name);
        if target.exists() {
            return Err(format!("Target already exists: {:?}", target));
        }
        moves.push((path.clone(), target, name));
    }

    let mut moved = 0usize;
    for (source_path, target, _) in &moves {
        if let Err(error) = fs::rename(source_path, target) {
            for (rollback_source, rollback_target, _) in moves[..moved].iter().rev() {
                let _ = fs::rename(rollback_target, rollback_source);
            }
            return Err(format!(
                "Failed to move session files to recyclebin: {error}"
            ));
        }
        moved += 1;
    }

    let item = RecycledItem {
        id: id.clone(),
        item_type: item_type.to_string(),
        reason: reason.to_string(),
        source: source.to_string(),
        project_id: project_id.to_string(),
        session_title,
        project_name,
        original_path: original_path.to_string_lossy().to_string(),
        stored_name,
        companion_original_path: None,
        companion_stored_name: None,
        related_paths: moves
            .iter()
            .skip(1)
            .map(|(original, _, stored)| RecycledRelatedPath {
                original_path: original.to_string_lossy().to_string(),
                stored_name: stored.clone(),
            })
            .collect(),
        moved_at: chrono::Utc::now().to_rfc3339(),
    };

    let mut manifest = load_manifest();
    manifest.items.push(item);
    if let Err(error) = save_manifest(&manifest) {
        let mut rollback_errors = Vec::new();
        for (rollback_source, rollback_target, _) in moves.iter().rev() {
            if let Err(rollback_error) = fs::rename(rollback_target, rollback_source) {
                rollback_errors.push(rollback_error.to_string());
            }
        }
        if rollback_errors.is_empty() {
            return Err(format!("Failed to record recyclebin item: {error}"));
        }
        return Err(format!(
            "Failed to record recyclebin item: {error}; rollback failed: {}",
            rollback_errors.join("; ")
        ));
    }

    Ok(id)
}

fn session_artifact_dir(session_path: &std::path::Path) -> Result<Option<PathBuf>, String> {
    let artifact_path = session_path.with_extension("");
    match fs::symlink_metadata(&artifact_path) {
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(format!(
            "Failed to inspect session artifact directory: {error}"
        )),
        Ok(metadata) if metadata.file_type().is_symlink() => {
            Err("OMP session artifact directory must not be a symbolic link".to_string())
        }
        Ok(metadata) if metadata.is_dir() => Ok(Some(artifact_path)),
        Ok(_) => Err("OMP session artifact path is not a directory".to_string()),
    }
}

/// Move an OMP session JSONL and its sibling artifact directory as one
/// recoverable recycle-bin item. The artifact directory is optional because
/// plain text-only sessions do not create one.
pub fn move_omp_session_to_recyclebin(
    session_path: &std::path::Path,
    project_id: &str,
    session_title: Option<String>,
    project_name: Option<String>,
) -> Result<String, String> {
    if !session_path.is_file() {
        return Err("OMP session file not found".to_string());
    }
    let artifact_path = session_artifact_dir(session_path)?;
    let items_dir = get_recyclebin_items_dir()
        .ok_or_else(|| "Cannot determine recyclebin items path".to_string())?;
    fs::create_dir_all(&items_dir)
        .map_err(|error| format!("Failed to create recyclebin items dir: {error}"))?;

    let id = generate_id();
    let stored_name = format!("{id}.jsonl");
    let stored_path = items_dir.join(&stored_name);
    let companion_stored_name = artifact_path.as_ref().map(|_| format!("{id}.artifacts"));
    let companion_stored_path = companion_stored_name
        .as_ref()
        .map(|name| items_dir.join(name));
    if stored_path.exists()
        || companion_stored_path
            .as_ref()
            .is_some_and(|path| path.exists())
    {
        return Err("Recyclebin target already exists".to_string());
    }

    fs::rename(session_path, &stored_path)
        .map_err(|error| format!("Failed to move OMP session to recyclebin: {error}"))?;
    if let (Some(artifact_path), Some(companion_stored_path)) =
        (artifact_path.as_ref(), companion_stored_path.as_ref())
    {
        if let Err(error) = fs::rename(artifact_path, companion_stored_path) {
            let rollback = fs::rename(&stored_path, session_path);
            return match rollback {
                Ok(()) => Err(format!("Failed to move OMP session artifacts to recyclebin: {error}")),
                Err(rollback_error) => Err(format!(
                    "Failed to move OMP session artifacts to recyclebin: {error}; failed to restore session file: {rollback_error}"
                )),
            };
        }
    }

    let item = RecycledItem {
        id: id.clone(),
        item_type: "session".to_string(),
        reason: "ManualDelete".to_string(),
        source: "omp".to_string(),
        project_id: project_id.to_string(),
        session_title,
        project_name,
        original_path: session_path.to_string_lossy().to_string(),
        stored_name,
        companion_original_path: artifact_path
            .as_ref()
            .map(|path| path.to_string_lossy().to_string()),
        companion_stored_name,
        related_paths: Vec::new(),
        moved_at: chrono::Utc::now().to_rfc3339(),
    };
    let mut manifest = load_manifest();
    manifest.items.push(item);
    if let Err(error) = save_manifest(&manifest) {
        let mut rollback_errors = Vec::new();
        if let (Some(artifact_path), Some(companion_stored_path)) =
            (artifact_path.as_ref(), companion_stored_path.as_ref())
        {
            if let Err(rollback_error) = fs::rename(companion_stored_path, artifact_path) {
                rollback_errors.push(rollback_error.to_string());
            }
        }
        if let Err(rollback_error) = fs::rename(&stored_path, session_path) {
            rollback_errors.push(rollback_error.to_string());
        }
        if rollback_errors.is_empty() {
            return Err(format!("Failed to record OMP recyclebin item: {error}"));
        }
        return Err(format!(
            "Failed to record OMP recyclebin item: {error}; rollback failed: {}",
            rollback_errors.join("; ")
        ));
    }

    Ok(id)
}

/// 列出所有回收站条目，按 movedAt 倒序排列。
pub fn list_items() -> Vec<RecycledItem> {
    let mut items = load_manifest().items;
    items.sort_by(|a, b| b.moved_at.cmp(&a.moved_at));
    items
}

/// 将条目还原到 original_path，自动创建父目录。
/// 还原成功后失效对应数据源的 sessions 缓存，避免 UI 不刷新。
pub fn restore_item(id: &str) -> Result<(), String> {
    let mut manifest = load_manifest();
    let pos = manifest
        .items
        .iter()
        .position(|item| item.id == id)
        .ok_or_else(|| format!("Item not found: {id}"))?;
    let item = manifest.items[pos].clone();
    let legacy_companion = match (&item.companion_original_path, &item.companion_stored_name) {
        (None, None) => None,
        (Some(original), Some(stored)) => Some((PathBuf::from(original), stored.clone())),
        _ => return Err("Recyclebin companion metadata is incomplete".to_string()),
    };

    let items_dir = get_recyclebin_items_dir()
        .ok_or_else(|| "Cannot determine recyclebin items path".to_string())?;
    let mut restore_pairs = vec![(
        items_dir.join(&item.stored_name),
        PathBuf::from(&item.original_path),
    )];
    if let Some((original, stored)) = legacy_companion {
        restore_pairs.push((items_dir.join(stored), original));
    }
    restore_pairs.extend(item.related_paths.iter().map(|related| {
        (
            items_dir.join(&related.stored_name),
            PathBuf::from(&related.original_path),
        )
    }));

    for (stored, original) in &restore_pairs {
        if !stored.exists() {
            return Err(format!("Stored session path not found: {stored:?}"));
        }
        if original.exists() {
            return Err(format!(
                "Session restore destination already exists: {original:?}"
            ));
        }
        if let Some(parent) = original.parent() {
            fs::create_dir_all(parent)
                .map_err(|error| format!("Failed to create session parent dir: {error}"))?;
        }
    }

    let mut restored = 0usize;
    for (stored, original) in &restore_pairs {
        if let Err(error) = restore_path(stored, original) {
            let mut rollback_errors = Vec::new();
            for (rollback_stored, rollback_original) in restore_pairs[..restored].iter().rev() {
                if let Err(rollback_error) = restore_path(rollback_original, rollback_stored) {
                    rollback_errors.push(rollback_error);
                }
            }
            if rollback_errors.is_empty() {
                return Err(format!("Failed to restore session files: {error}"));
            }
            return Err(format!(
                "Failed to restore session files: {error}; rollback failed: {}",
                rollback_errors.join("; ")
            ));
        }
        restored += 1;
    }

    manifest.items.remove(pos);
    save_manifest(&manifest)?;

    match item.source.as_str() {
        "claude" => crate::provider::claude::invalidate_cache(),
        "codex" => crate::provider::codex::invalidate_sessions_cache(),
        "grok" => crate::provider::grok::invalidate_sessions_cache(),
        "dsh" => crate::provider::dsh::invalidate_paths(&[]),
        "kiro" => crate::provider::kiro::invalidate_paths(&[]),
        "omp" => crate::provider::omp::invalidate_sessions_cache(),
        _ => {}
    }
    Ok(())
}

fn cross_device_error(err: &std::io::Error) -> bool {
    // EXDEV on Unix; Windows uses a different error string
    matches!(err.raw_os_error(), Some(18))
        || err.to_string().to_lowercase().contains("different device")
        || err.to_string().to_lowercase().contains("different volume")
}

fn restore_path(source: &std::path::Path, destination: &std::path::Path) -> Result<(), String> {
    if let Err(rename_error) = fs::rename(source, destination) {
        if !cross_device_error(&rename_error) {
            return Err(format!("Failed to restore item: {rename_error}"));
        }
        copy_path(source, destination)
            .map_err(|error| format!("Failed to restore item across volumes: {error}"))?;
        remove_path(source)
            .map_err(|error| format!("Restored, but failed to clean recyclebin entry: {error}"))?;
    }
    Ok(())
}

fn copy_path(src: &std::path::Path, dst: &std::path::Path) -> std::io::Result<()> {
    if src.is_dir() {
        fs::create_dir_all(dst)?;
        for entry in fs::read_dir(src)? {
            let entry = entry?;
            let to = dst.join(entry.file_name());
            copy_path(&entry.path(), &to)?;
        }
        Ok(())
    } else {
        if let Some(parent) = dst.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(src, dst).map(|_| ())
    }
}

fn remove_path(p: &std::path::Path) -> std::io::Result<()> {
    if p.is_dir() {
        fs::remove_dir_all(p)
    } else {
        fs::remove_file(p)
    }
}

/// 永久删除条目（从 items/ 删文件 + manifest 移除）。
pub fn permanently_delete_item(id: &str) -> Result<(), String> {
    let mut manifest = load_manifest();
    let pos = manifest
        .items
        .iter()
        .position(|item| item.id == id)
        .ok_or_else(|| format!("Item not found: {id}"))?;
    let item = manifest.items[pos].clone();
    let items_dir = get_recyclebin_items_dir()
        .ok_or_else(|| "Cannot determine recyclebin items path".to_string())?;
    if let Some(companion_stored_name) = &item.companion_stored_name {
        let companion_stored_path = items_dir.join(companion_stored_name);
        if companion_stored_path.exists() {
            remove_path(&companion_stored_path)
                .map_err(|error| format!("Failed to delete stored companion: {error}"))?;
        }
    }
    for related in &item.related_paths {
        let stored_path = items_dir.join(&related.stored_name);
        if stored_path.exists() {
            remove_path(&stored_path)
                .map_err(|error| format!("Failed to delete related session path: {error}"))?;
        }
    }
    let stored_path = items_dir.join(&item.stored_name);
    if stored_path.exists() {
        remove_path(&stored_path)
            .map_err(|error| format!("Failed to delete stored item: {error}"))?;
    }

    manifest.items.remove(pos);
    save_manifest(&manifest)?;
    Ok(())
}

/// 清空回收站所有条目，返回删除数量。
pub fn empty_recyclebin() -> Result<usize, String> {
    let manifest = load_manifest();
    let count = manifest.items.len();
    if count == 0 {
        return Ok(0);
    }

    let items_dir = get_recyclebin_items_dir()
        .ok_or_else(|| "Cannot determine recyclebin items path".to_string())?;

    for item in &manifest.items {
        let stored_path = items_dir.join(&item.stored_name);
        if stored_path.exists() {
            if stored_path.is_dir() {
                let _ = fs::remove_dir_all(&stored_path);
            } else {
                let _ = fs::remove_file(&stored_path);
            }
        }
        if let Some(companion_stored_name) = &item.companion_stored_name {
            let companion_stored_path = items_dir.join(companion_stored_name);
            if companion_stored_path.exists() {
                let _ = remove_path(&companion_stored_path);
            }
        }
        for related in &item.related_paths {
            let stored_path = items_dir.join(&related.stored_name);
            if stored_path.exists() {
                let _ = remove_path(&stored_path);
            }
        }
    }

    save_manifest(&RecyclebinManifest::default())?;
    Ok(count)
}
