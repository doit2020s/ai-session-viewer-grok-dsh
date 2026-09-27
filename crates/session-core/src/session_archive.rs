use std::fs::{self, File};
use std::io::{BufRead, BufReader, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use zip::write::SimpleFileOptions;

use crate::paths::validate_session_file;
use crate::provider::{codex, dsh, grok, kiro, omp};

const MANIFEST_NAME: &str = "ai-session-backup.json";
const ARCHIVE_VERSION: u32 = 1;
const MAX_ARCHIVE_BYTES: u64 = 4 * 1024 * 1024 * 1024;
const MAX_ENTRIES: usize = 100_000;

#[derive(Debug, Clone, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BackupSessionInput {
    pub session_id: String,
    pub file_path: String,
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArchiveSession {
    session_id: String,
    title: Option<String>,
    archive_dir: String,
    primary_name: String,
    directory_backed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ArchiveManifest {
    format: String,
    version: u32,
    source: String,
    project_id: String,
    project_path: Option<String>,
    created_at: String,
    sessions: Vec<ArchiveSession>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct BackupResult {
    pub path: String,
    pub session_count: usize,
    pub file_count: usize,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct RestoreResult {
    pub restored: usize,
    pub skipped: usize,
    pub restored_session_ids: Vec<String>,
}

fn directory_backed(source: &str) -> bool {
    matches!(source, "grok" | "dsh" | "kiro")
}

fn add_file(
    zip: &mut zip::ZipWriter<File>,
    disk_path: &Path,
    archive_path: &str,
) -> Result<(), String> {
    if !fs::symlink_metadata(disk_path)
        .map_err(|e| e.to_string())?
        .is_file()
    {
        return Err(format!("备份对象不是普通文件：{}", disk_path.display()));
    }
    zip.start_file(
        archive_path.replace('\\', "/"),
        SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
    )
    .map_err(|e| format!("创建压缩条目失败：{e}"))?;
    let mut file = File::open(disk_path).map_err(|e| format!("读取会话文件失败：{e}"))?;
    std::io::copy(&mut file, zip).map_err(|e| format!("写入压缩包失败：{e}"))?;
    Ok(())
}

fn add_directory(
    zip: &mut zip::ZipWriter<File>,
    disk_dir: &Path,
    archive_dir: &str,
    count: &mut usize,
) -> Result<(), String> {
    for entry in fs::read_dir(disk_dir).map_err(|e| format!("读取会话目录失败：{e}"))? {
        let entry = entry.map_err(|e| e.to_string())?;
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        let name = entry.file_name().to_string_lossy().into_owned();
        let archive_path = format!("{archive_dir}/{name}");
        if ty.is_dir() {
            add_directory(zip, &entry.path(), &archive_path, count)?;
        } else if ty.is_file() {
            add_file(zip, &entry.path(), &archive_path)?;
            *count += 1;
        } else {
            return Err("会话目录包含链接，无法安全备份".to_string());
        }
    }
    Ok(())
}

pub fn create_backup(
    source: &str,
    project_id: &str,
    project_path: Option<&str>,
    sessions: &[BackupSessionInput],
    output_path: &Path,
) -> Result<BackupResult, String> {
    if sessions.is_empty() {
        return Err("至少选择一个会话".to_string());
    }
    if output_path
        .extension()
        .and_then(|v| v.to_str())
        .map(|v| v.eq_ignore_ascii_case("zip"))
        != Some(true)
    {
        return Err("备份文件必须使用 .zip 后缀".to_string());
    }
    let parent = output_path.parent().ok_or("备份路径无效")?;
    if !parent.is_dir() {
        return Err("备份目标目录不存在".to_string());
    }
    let temporary = output_path.with_extension(format!("{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let file = File::create(&temporary).map_err(|e| format!("创建备份失败：{e}"))?;
        let mut zip = zip::ZipWriter::new(file);
        let mut manifest_sessions = Vec::with_capacity(sessions.len());
        let mut file_count = 0usize;
        for (index, input) in sessions.iter().enumerate() {
            let primary = validate_session_file(source, &input.file_path)?;
            let archive_dir = format!("sessions/{index:05}");
            let primary_name = primary
                .file_name()
                .and_then(|v| v.to_str())
                .ok_or("会话文件名无效")?
                .to_string();
            if directory_backed(source) {
                add_directory(
                    &mut zip,
                    primary.parent().ok_or("会话目录无效")?,
                    &archive_dir,
                    &mut file_count,
                )?;
            } else {
                add_file(&mut zip, &primary, &format!("{archive_dir}/{primary_name}"))?;
                file_count += 1;
                if source == "omp" {
                    let artifacts = primary.with_extension("");
                    if artifacts.is_dir() {
                        add_directory(
                            &mut zip,
                            &artifacts,
                            &format!("{archive_dir}/artifacts"),
                            &mut file_count,
                        )?;
                    }
                }
            }
            manifest_sessions.push(ArchiveSession {
                session_id: input.session_id.clone(),
                title: input.title.clone(),
                archive_dir,
                primary_name,
                directory_backed: directory_backed(source),
            });
        }
        let manifest = ArchiveManifest {
            format: "ai-session-viewer-backup".to_string(),
            version: ARCHIVE_VERSION,
            source: source.to_string(),
            project_id: project_id.to_string(),
            project_path: project_path.map(str::to_string),
            created_at: chrono::Utc::now().to_rfc3339(),
            sessions: manifest_sessions,
        };
        zip.start_file(
            MANIFEST_NAME,
            SimpleFileOptions::default().compression_method(zip::CompressionMethod::Deflated),
        )
        .map_err(|e| e.to_string())?;
        zip.write_all(&serde_json::to_vec_pretty(&manifest).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        zip.finish().map_err(|e| format!("完成压缩包失败：{e}"))?;
        Ok::<_, String>(BackupResult {
            path: output_path.to_string_lossy().into_owned(),
            session_count: sessions.len(),
            file_count,
        })
    })();
    match result {
        Ok(value) => {
            if output_path.exists() {
                fs::remove_file(output_path).map_err(|e| format!("替换旧备份失败：{e}"))?;
            }
            fs::rename(&temporary, output_path).map_err(|e| format!("保存备份失败：{e}"))?;
            Ok(value)
        }
        Err(error) => {
            let _ = fs::remove_file(&temporary);
            Err(error)
        }
    }
}

fn target_directory(source: &str, anchor: &Path) -> Result<PathBuf, String> {
    match source {
        "claude" | "codex" | "omp" => anchor.parent().map(Path::to_path_buf),
        "grok" | "dsh" | "kiro" => anchor
            .parent()
            .and_then(Path::parent)
            .map(Path::to_path_buf),
        _ => None,
    }
    .ok_or_else(|| "无法确定目标工作区会话目录".to_string())
}

fn rewrite_jsonl_cwd(path: &Path, source: &str, cwd: &str) -> Result<(), String> {
    let text = fs::read_to_string(path).map_err(|e| e.to_string())?;
    let mut output = String::new();
    for line in text.lines() {
        let mut row: Value =
            serde_json::from_str(line).map_err(|e| format!("会话 JSONL 损坏：{e}"))?;
        if source == "claude" && row.get("cwd").is_some() {
            row["cwd"] = Value::String(cwd.to_string());
        }
        if source == "codex" && row.get("type").and_then(Value::as_str) == Some("session_meta") {
            row["payload"]["cwd"] = Value::String(cwd.to_string());
        }
        if source == "omp" && row.get("type").and_then(Value::as_str) == Some("session") {
            row["cwd"] = Value::String(cwd.to_string());
        }
        output.push_str(&row.to_string());
        output.push('\n');
    }
    fs::write(path, output).map_err(|e| format!("更新目标工作区路径失败：{e}"))
}

fn rewrite_workspace(
    source: &str,
    staged: &Path,
    primary_name: &str,
    cwd: Option<&str>,
) -> Result<(), String> {
    let Some(cwd) = cwd.filter(|v| !v.trim().is_empty()) else {
        return Ok(());
    };
    match source {
        "claude" | "codex" | "omp" => rewrite_jsonl_cwd(&staged.join(primary_name), source, cwd),
        "grok" => {
            let path = staged.join("summary.json");
            let mut value: Value =
                serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            value["info"]["cwd"] = Value::String(cwd.to_string());
            fs::write(
                path,
                serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())
        }
        "kiro" => {
            let path = staged.join("session.json");
            let mut value: Value =
                serde_json::from_slice(&fs::read(&path).map_err(|e| e.to_string())?)
                    .map_err(|e| e.to_string())?;
            value["workspacePaths"] = serde_json::json!([cwd]);
            fs::write(
                path,
                serde_json::to_vec_pretty(&value).map_err(|e| e.to_string())?,
            )
            .map_err(|e| e.to_string())
        }
        "dsh" => rewrite_dsh_cwd(&staged.join(primary_name), cwd),
        _ => Ok(()),
    }
}

fn rewrite_dsh_cwd(path: &Path, cwd: &str) -> Result<(), String> {
    let decoded = zstd::stream::decode_all(File::open(path).map_err(|e| e.to_string())?)
        .map_err(|e| e.to_string())?;
    let mut output = Vec::new();
    for line in BufReader::new(decoded.as_slice()).lines() {
        let mut row: Value =
            serde_json::from_str(&line.map_err(|e| e.to_string())?).map_err(|e| e.to_string())?;
        if row.get("type").and_then(Value::as_str) == Some("session") {
            row["cwd"] = Value::String(cwd.to_string());
        }
        writeln!(&mut output, "{}", row).map_err(|e| e.to_string())?;
    }
    let encoded = zstd::stream::encode_all(output.as_slice(), 3).map_err(|e| e.to_string())?;
    fs::write(path, encoded).map_err(|e| e.to_string())
}

fn extract_prefix(
    zip: &mut zip::ZipArchive<File>,
    prefix: &str,
    destination: &Path,
) -> Result<(), String> {
    let prefix = format!("{}/", prefix.trim_end_matches('/'));
    let mut matched = false;
    for index in 0..zip.len() {
        let mut file = zip.by_index(index).map_err(|e| e.to_string())?;
        let Some(name) = file.enclosed_name() else {
            return Err("压缩包包含不安全路径".to_string());
        };
        let normalized = name.to_string_lossy().replace('\\', "/");
        let Some(relative) = normalized.strip_prefix(&prefix) else {
            continue;
        };
        if relative.is_empty() {
            continue;
        }
        matched = true;
        let output = destination.join(relative);
        if file.is_dir() {
            fs::create_dir_all(&output).map_err(|e| e.to_string())?;
            continue;
        }
        if let Some(parent) = output.parent() {
            fs::create_dir_all(parent).map_err(|e| e.to_string())?;
        }
        let mut out = File::create(&output).map_err(|e| e.to_string())?;
        std::io::copy(&mut file, &mut out).map_err(|e| e.to_string())?;
    }
    matched
        .then_some(())
        .ok_or_else(|| "备份中缺少会话文件".to_string())
}

pub fn restore_backup(
    archive_path: &Path,
    target_source: &str,
    target_project_path: Option<&str>,
    target_anchor_path: &str,
) -> Result<RestoreResult, String> {
    let metadata = fs::metadata(archive_path).map_err(|e| format!("读取备份失败：{e}"))?;
    if metadata.len() > MAX_ARCHIVE_BYTES {
        return Err("备份文件超过 4GB 限制".to_string());
    }
    let anchor = validate_session_file(target_source, target_anchor_path)?;
    let target_dir = target_directory(target_source, &anchor)?;
    let file = File::open(archive_path).map_err(|e| e.to_string())?;
    let mut zip = zip::ZipArchive::new(file).map_err(|e| format!("无效的 ZIP 备份：{e}"))?;
    if zip.len() > MAX_ENTRIES {
        return Err("备份条目过多".to_string());
    }
    let manifest: ArchiveManifest = {
        let mut item = zip
            .by_name(MANIFEST_NAME)
            .map_err(|_| "缺少会话备份清单".to_string())?;
        let mut bytes = Vec::new();
        item.read_to_end(&mut bytes).map_err(|e| e.to_string())?;
        serde_json::from_slice(&bytes).map_err(|e| format!("备份清单损坏：{e}"))?
    };
    if manifest.format != "ai-session-viewer-backup" || manifest.version != ARCHIVE_VERSION {
        return Err("不支持的会话备份版本".to_string());
    }
    if manifest.source != target_source {
        return Err(format!(
            "该备份属于 {}，不能导入当前 {} 工作区",
            manifest.source, target_source
        ));
    }
    let mut restored = Vec::new();
    let mut skipped = 0usize;
    for session in &manifest.sessions {
        let destination = if session.directory_backed {
            target_dir.join(&session.session_id)
        } else {
            target_dir.join(&session.primary_name)
        };
        let artifact_destination = (target_source == "omp").then(|| destination.with_extension(""));
        if destination.exists() || artifact_destination.as_ref().is_some_and(|p| p.exists()) {
            skipped += 1;
            continue;
        }
        let staging = target_dir.join(format!(".asv-restore-{}", uuid::Uuid::new_v4()));
        fs::create_dir(&staging).map_err(|e| e.to_string())?;
        let restored_one = (|| {
            extract_prefix(&mut zip, &session.archive_dir, &staging)?;
            if !staging.join(&session.primary_name).is_file() {
                return Err(format!("备份会话 {} 缺少主文件", session.session_id));
            }
            rewrite_workspace(
                target_source,
                &staging,
                &session.primary_name,
                target_project_path,
            )?;
            if session.directory_backed {
                fs::rename(&staging, &destination).map_err(|e| e.to_string())?;
            } else {
                let primary = staging.join(&session.primary_name);
                fs::rename(&primary, &destination).map_err(|e| e.to_string())?;
                if let Some(artifact_destination) = &artifact_destination {
                    let artifacts = staging.join("artifacts");
                    if artifacts.is_dir() {
                        fs::rename(artifacts, artifact_destination).map_err(|e| e.to_string())?;
                    }
                }
                let _ = fs::remove_dir(&staging);
            }
            Ok::<(), String>(())
        })();
        if let Err(error) = restored_one {
            let _ = fs::remove_dir_all(&staging);
            return Err(error);
        }
        restored.push(session.session_id.clone());
    }
    match target_source {
        "claude" => crate::provider::claude::invalidate_cache(),
        "codex" => codex::invalidate_sessions_cache(),
        "grok" => grok::invalidate_sessions_cache(),
        "dsh" => dsh::invalidate_paths(&[]),
        "kiro" => kiro::invalidate_paths(&[]),
        "omp" => omp::invalidate_sessions_cache(),
        _ => {}
    }
    Ok(RestoreResult {
        restored: restored.len(),
        skipped,
        restored_session_ids: restored,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zip_prefix_round_trip_and_kiro_workspace_rewrite() {
        let root = std::env::temp_dir().join(format!("asv-archive-{}", uuid::Uuid::new_v4()));
        let source = root.join("source");
        let output = root.join("output");
        fs::create_dir_all(&source).unwrap();
        fs::create_dir_all(&output).unwrap();
        fs::write(
            source.join("session.json"),
            r#"{"id":"sess_test","workspacePaths":["C:\\old"]}"#,
        )
        .unwrap();
        fs::write(source.join("messages.jsonl"), "{\"id\":\"m1\"}\n").unwrap();

        let archive = root.join("test.zip");
        let file = File::create(&archive).unwrap();
        let mut writer = zip::ZipWriter::new(file);
        let mut count = 0;
        add_directory(&mut writer, &source, "sessions/00000", &mut count).unwrap();
        writer.finish().unwrap();
        assert_eq!(count, 2);

        let mut reader = zip::ZipArchive::new(File::open(&archive).unwrap()).unwrap();
        extract_prefix(&mut reader, "sessions/00000", &output).unwrap();
        rewrite_workspace("kiro", &output, "messages.jsonl", Some(r"D:\new-workspace")).unwrap();
        let metadata: Value =
            serde_json::from_slice(&fs::read(output.join("session.json")).unwrap()).unwrap();
        assert_eq!(metadata["workspacePaths"][0], r"D:\new-workspace");
        assert_eq!(
            fs::read_to_string(output.join("messages.jsonl")).unwrap(),
            "{\"id\":\"m1\"}\n"
        );
        let _ = fs::remove_dir_all(root);
    }
}
