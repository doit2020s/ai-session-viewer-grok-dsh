use std::collections::HashSet;
use std::fs;
use std::path::{Path, PathBuf};

use crate::provider::codex;

fn add_existing(
    paths: &mut Vec<PathBuf>,
    seen: &mut HashSet<PathBuf>,
    path: PathBuf,
) -> Result<(), String> {
    let metadata = match fs::symlink_metadata(&path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("无法检查会话关联文件 {}：{error}", path.display())),
    };
    if metadata.file_type().is_symlink() {
        return Err(format!("会话关联路径不能是符号链接：{}", path.display()));
    }
    if !metadata.is_file() && !metadata.is_dir() {
        return Err(format!("会话关联路径类型无效：{}", path.display()));
    }
    if seen.insert(path.clone()) {
        paths.push(path);
    }
    Ok(())
}

fn collect_claude_related(primary: &Path, session_id: &str) -> Result<Vec<PathBuf>, String> {
    let parent = primary.parent().ok_or("Claude 会话路径无效")?;
    let mut paths = vec![primary.to_path_buf()];
    let mut seen = HashSet::from([primary.to_path_buf()]);
    add_existing(&mut paths, &mut seen, parent.join(session_id))?;

    let prefix = format!("{session_id}.");
    for entry in
        fs::read_dir(parent).map_err(|error| format!("无法遍历 Claude 会话目录：{error}"))?
    {
        let entry = entry.map_err(|error| format!("无法读取 Claude 会话目录项：{error}"))?;
        let name = entry.file_name();
        let Some(name) = name.to_str() else { continue };
        if name.starts_with(&prefix) && entry.path() != primary {
            add_existing(&mut paths, &mut seen, entry.path())?;
        }
    }
    Ok(paths)
}

fn find_named_descendants(
    root: &Path,
    name: &str,
    output: &mut Vec<PathBuf>,
) -> Result<(), String> {
    let entries = match fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(error) => return Err(format!("无法遍历会话产物目录 {}：{error}", root.display())),
    };
    for entry in entries {
        let entry = entry.map_err(|error| format!("无法读取会话产物目录项：{error}"))?;
        let metadata = entry
            .file_type()
            .map_err(|error| format!("无法检查会话产物类型：{error}"))?;
        if metadata.is_symlink() {
            continue;
        }
        if entry.file_name().to_str() == Some(name) {
            output.push(entry.path());
        } else if metadata.is_dir() {
            find_named_descendants(&entry.path(), name, output)?;
        }
    }
    Ok(())
}

fn collect_codex_related(primary: &Path, session_id: &str) -> Result<Vec<PathBuf>, String> {
    let codex_home = codex::get_sessions_dir()
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .ok_or("无法确定 Codex 数据目录")?;
    let mut paths = vec![primary.to_path_buf()];
    let mut seen = HashSet::from([primary.to_path_buf()]);

    add_existing(
        &mut paths,
        &mut seen,
        codex_home
            .join("thread-writer-locks")
            .join(format!("{session_id}.lock")),
    )?;

    let mut visualization_paths = Vec::new();
    find_named_descendants(
        &codex_home.join("visualizations"),
        session_id,
        &mut visualization_paths,
    )?;
    for path in visualization_paths {
        add_existing(&mut paths, &mut seen, path)?;
    }
    Ok(paths)
}

/// Return every filesystem path owned by one validated session. The result is
/// deliberately limited to provider-managed roots and exact session IDs; paths
/// mentioned inside chat content are never followed into the user's workspace.
pub fn collect_session_paths(
    source: &str,
    primary: &Path,
    session_id: &str,
) -> Result<Vec<PathBuf>, String> {
    match source {
        "grok" | "dsh" | "kiro" => Ok(vec![primary
            .parent()
            .ok_or("目录型会话路径无效")?
            .to_path_buf()]),
        "omp" => {
            let mut paths = vec![primary.to_path_buf()];
            let mut seen = HashSet::from([primary.to_path_buf()]);
            add_existing(&mut paths, &mut seen, primary.with_extension(""))?;
            Ok(paths)
        }
        "claude" => collect_claude_related(primary, session_id),
        "codex" => collect_codex_related(primary, session_id),
        _ => Err(format!("Unknown source: {source}")),
    }
}

pub fn permanently_delete_session_files(
    source: &str,
    primary: &Path,
    session_id: &str,
) -> Result<usize, String> {
    let paths = collect_session_paths(source, primary, session_id)?;
    for path in paths.iter().rev() {
        if path.is_dir() {
            fs::remove_dir_all(path)
        } else {
            fs::remove_file(path)
        }
        .map_err(|error| format!("删除会话文件 {} 失败：{error}", path.display()))?;
    }
    Ok(paths.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temporary_dir() -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!("ai-session-files-{unique}"))
    }

    #[test]
    fn claude_collection_includes_all_exact_session_siblings_only() {
        let root = temporary_dir();
        fs::create_dir_all(root.join("session-a/subagents")).unwrap();
        fs::write(root.join("session-a.jsonl"), "chat").unwrap();
        fs::write(root.join("session-a.desktop-released.json"), "state").unwrap();
        fs::write(root.join("session-ab.jsonl"), "other").unwrap();
        fs::write(root.join("unrelated.json"), "other").unwrap();

        let paths = collect_claude_related(&root.join("session-a.jsonl"), "session-a").unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths.contains(&root.join("session-a.jsonl")));
        assert!(paths.contains(&root.join("session-a")));
        assert!(paths.contains(&root.join("session-a.desktop-released.json")));
        assert!(!paths.contains(&root.join("session-ab.jsonl")));

        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn named_descendant_scan_returns_exact_directories_without_descending_into_them() {
        let root = temporary_dir();
        fs::create_dir_all(root.join("2026/09/session-id/nested/session-id")).unwrap();
        fs::create_dir_all(root.join("2026/09/other")).unwrap();
        let mut found = Vec::new();
        find_named_descendants(&root, "session-id", &mut found).unwrap();
        assert_eq!(found, vec![root.join("2026/09/session-id")]);
        fs::remove_dir_all(root).unwrap();
    }
}
