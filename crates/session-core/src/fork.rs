//! Fork persisted conversations without ever modifying the source transcript.
//! Both HTTP and Tauri use this service; launching a terminal is a separate action.

use std::collections::{HashMap, HashSet};
use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::Path;

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::paths::{validate_session_file, SessionSourceKind};
use crate::provider::{claude, codex, dsh, grok, kiro, omp};

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ForkRequest {
    pub source: String,
    pub original_file_path: String,
    pub user_msg_uuid: String,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ForkResult {
    pub new_session_id: String,
    pub new_file_path: String,
    pub project_path: String,
    pub project_id: String,
}

/// Physical line + content digest: independent of pagination, and rejects a
/// stale selection when another process rewrites the same line in place.
pub fn line_message_id(line: usize, row: &Value) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in row.to_string().bytes() {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("line-{line}-{hash:016x}")
}

struct Record {
    line: usize,
    value: Value,
}

fn read_records(path: &Path) -> Result<Vec<Record>, String> {
    let file = fs::File::open(path).map_err(|e| format!("读取会话失败：{e}"))?;
    let mut lines = BufReader::new(file).lines().enumerate().peekable();
    let mut rows = Vec::new();
    while let Some((line, text)) = lines.next() {
        let text = text.map_err(|e| format!("读取会话失败：{e}"))?;
        if text.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(&text) {
            Ok(value) if value.is_object() => rows.push(Record { line, value }),
            Err(_) if lines.peek().is_none() => break, // interrupted final write
            _ => return Err(format!("会话第 {} 行损坏，无法可靠分叉", line + 1)),
        }
    }
    Ok(rows)
}

fn read_records_for_kind(kind: SessionSourceKind, path: &Path) -> Result<Vec<Record>, String> {
    if kind == SessionSourceKind::Dsh {
        return Ok(dsh::decode_rows(path)?
            .into_iter()
            .enumerate()
            .map(|(line, value)| Record { line, value })
            .collect());
    }
    read_records(path)
}

fn string<'a>(row: &'a Value, key: &str) -> Option<&'a str> {
    row.get(key).and_then(Value::as_str)
}

fn is_user(kind: SessionSourceKind, row: &Value) -> bool {
    match kind {
        SessionSourceKind::Claude => {
            string(row, "type") == Some("user")
                && row.get("isMeta").and_then(Value::as_bool) != Some(true)
                && row.pointer("/message/content").is_some_and(|content| {
                    content.is_string()
                        || content.as_array().is_some_and(|blocks| {
                            blocks
                                .iter()
                                .any(|b| string(b, "type") != Some("tool_result"))
                        })
                })
        }
        SessionSourceKind::Codex => {
            string(row, "type") == Some("response_item")
                && row.pointer("/payload/type").and_then(Value::as_str) == Some("message")
                && row.pointer("/payload/role").and_then(Value::as_str) == Some("user")
        }
        SessionSourceKind::Grok => {
            string(row, "type") == Some("user")
                && row.get("synthetic_reason").is_none_or(Value::is_null)
        }
        SessionSourceKind::Dsh => string(row, "type") == Some("user/message"),
        SessionSourceKind::Kiro => {
            row.pointer("/payload/type").and_then(Value::as_str) == Some("user")
        }
        SessionSourceKind::Omp => {
            string(row, "type") == Some("message")
                && row.pointer("/message/role").and_then(Value::as_str) == Some("user")
        }
    }
}

fn target_index(kind: SessionSourceKind, rows: &[Record], target: &str) -> Result<usize, String> {
    rows.iter()
        .position(|record| {
            let id = match kind {
                SessionSourceKind::Claude => string(&record.value, "uuid").map(str::to_owned),
                SessionSourceKind::Omp | SessionSourceKind::Kiro => {
                    string(&record.value, "id").map(str::to_owned)
                }
                SessionSourceKind::Dsh => record
                    .value
                    .pointer("/data/id")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
                    .or_else(|| {
                        record
                            .value
                            .get("seq")
                            .map(|seq| format!("dsh-{seq}"))
                    }),
                _ => Some(line_message_id(record.line, &record.value)),
            };
            id.as_deref() == Some(target) && is_user(kind, &record.value)
        })
        .ok_or_else(|| "找不到分叉提问，会话可能已更新，请刷新后重试".to_string())
}

/// Include the whole selected round, including tool results. Tree-backed
/// formats retain only the selected ancestry, never sibling conversations.
fn history_through_round(
    kind: SessionSourceKind,
    rows: &[Record],
    target: usize,
) -> Result<Vec<Value>, String> {
    let end = rows
        .iter()
        .enumerate()
        .skip(target + 1)
        .find(|(_, row)| is_user(kind, &row.value))
        .map_or(rows.len(), |(index, _)| index);
    let (id_key, parent_key) = match kind {
        SessionSourceKind::Claude => ("uuid", "parentUuid"),
        SessionSourceKind::Omp => ("id", "parentId"),
        _ => return Ok(rows[..end].iter().map(|r| r.value.clone()).collect()),
    };
    let mut leaf = target;
    let mut descendants = HashSet::new();
    if let Some(id) = string(&rows[target].value, id_key) {
        descendants.insert(id);
    }
    for (index, row) in rows.iter().enumerate().take(end).skip(target + 1) {
        if string(&row.value, parent_key).is_some_and(|id| descendants.contains(id)) {
            if let Some(id) = string(&row.value, id_key) {
                descendants.insert(id);
                leaf = index;
            }
        }
    }
    let by_id: HashMap<_, _> = rows[..end]
        .iter()
        .enumerate()
        .filter_map(|(i, row)| string(&row.value, id_key).map(|id| (id, i)))
        .collect();
    let mut keep = HashSet::new();
    let mut cursor = Some(leaf);
    while let Some(index) = cursor {
        if !keep.insert(index) {
            return Err("会话父子关系存在循环，无法分叉".to_string());
        }
        cursor = string(&rows[index].value, parent_key)
            .filter(|id| !id.is_empty())
            .map(|id| {
                by_id
                    .get(id)
                    .copied()
                    .ok_or_else(|| "会话缺少祖先记录，无法完整分叉".to_string())
            })
            .transpose()?;
    }
    // Older linear Claude logs may omit parentUuid entirely.
    if kind == SessionSourceKind::Claude
        && !rows[..end]
            .iter()
            .any(|r| r.value.get(parent_key).is_some())
    {
        return Ok(rows[..end].iter().map(|r| r.value.clone()).collect());
    }
    Ok(rows[..=leaf]
        .iter()
        .enumerate()
        .filter(|(i, row)| {
            keep.contains(i)
                || string(&row.value, id_key).is_none()
                || (kind == SessionSourceKind::Omp && string(&row.value, "type") == Some("session"))
        })
        .map(|(_, row)| row.value.clone())
        .collect())
}

fn write_new(path: &Path, contents: &[u8]) -> Result<(), String> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| format!("创建分叉文件失败：{e}"))?;
    let result = file.write_all(contents).and_then(|_| file.sync_all());
    drop(file);
    if let Err(error) = result {
        let _ = fs::remove_file(path); // this call successfully claimed the file
        return Err(format!("写入分叉文件失败：{error}"));
    }
    Ok(())
}

fn jsonl(rows: &[Value]) -> Vec<u8> {
    let mut output = String::new();
    for row in rows {
        output.push_str(&row.to_string());
        output.push('\n');
    }
    output.into_bytes()
}

// Only copy regular files and directories. Never follow links out of a session.
fn copy_directory(source: &Path, destination: &Path) -> Result<(), String> {
    if !source.exists() {
        return Ok(());
    }
    if !fs::symlink_metadata(source)
        .map_err(|e| e.to_string())?
        .is_dir()
    {
        return Err("会话附件目录不是普通目录".to_string());
    }
    fs::create_dir(destination).map_err(|e| e.to_string())?;
    let result = (|| {
        for entry in fs::read_dir(source).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let ty = entry.file_type().map_err(|e| e.to_string())?;
            let dest = destination.join(entry.file_name());
            if ty.is_dir() {
                copy_directory(&entry.path(), &dest)?;
            } else if ty.is_file() {
                fs::copy(entry.path(), dest).map_err(|e| e.to_string())?;
            } else {
                return Err("会话附件包含链接，无法安全复制".to_string());
            }
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(destination);
    }
    result
}

fn register_claude_fork(
    path: &Path,
    id: &str,
    project: &str,
    history: &[Value],
) -> Result<(), String> {
    // Serialize Viewer-created index updates. Preserve unknown index fields.
    static INDEX_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    let _guard = INDEX_LOCK.lock().map_err(|e| e.to_string())?;
    let parent = path.parent().ok_or("Claude 会话目录不存在")?;
    let index_path = parent.join("sessions-index.json");
    let mut index: Value = if index_path.exists() {
        serde_json::from_slice(&fs::read(&index_path).map_err(|e| e.to_string())?)
            .map_err(|e| format!("Claude 会话索引损坏：{e}"))?
    } else {
        json!({ "version": 1, "entries": [], "originalPath": project })
    };
    let entries = index
        .get_mut("entries")
        .and_then(Value::as_array_mut)
        .ok_or("Claude 会话索引缺少 entries")?;
    let first_prompt = history
        .iter()
        .find(|r| is_user(SessionSourceKind::Claude, r))
        .and_then(|r| r.pointer("/message/content"))
        .map(input_text)
        .unwrap_or_default();
    let now = chrono::Utc::now().to_rfc3339();
    entries.push(json!({
        "sessionId": id, "fullPath": path.to_string_lossy(),
        "firstPrompt": first_prompt.chars().take(200).collect::<String>(),
        "messageCount": history.iter().filter(|r| matches!(string(r, "type"), Some("user" | "assistant"))).count(),
        "created": now, "modified": now, "projectPath": project, "isSidechain": false,
    }));
    let temporary = parent.join(format!("sessions-index-{id}.tmp"));
    write_new(&temporary, index.to_string().as_bytes())?;
    if let Err(error) = fs::rename(&temporary, &index_path) {
        let _ = fs::remove_file(&temporary);
        return Err(format!("更新 Claude 会话索引失败：{error}"));
    }
    Ok(())
}

fn copy_session_extras(
    source: &Path,
    destination: &Path,
    skipped_names: &[&str],
) -> Result<(), String> {
    fs::create_dir(destination).map_err(|e| format!("创建分叉暂存目录失败：{e}"))?;
    let result = (|| {
        for entry in fs::read_dir(source).map_err(|e| e.to_string())? {
            let entry = entry.map_err(|e| e.to_string())?;
            let name = entry.file_name();
            if name
                .to_str()
                .is_some_and(|name| skipped_names.contains(&name))
            {
                continue;
            }
            let ty = entry.file_type().map_err(|e| e.to_string())?;
            let target = destination.join(&name);
            if ty.is_dir() {
                copy_directory(&entry.path(), &target)?;
            } else if ty.is_file() {
                fs::copy(entry.path(), target).map_err(|e| e.to_string())?;
            } else {
                return Err("会话目录包含链接，无法安全分叉".to_string());
            }
        }
        Ok::<_, String>(())
    })();
    if result.is_err() {
        let _ = fs::remove_dir_all(destination);
    }
    result
}

fn publish_directory_fork(staging: &Path, destination: &Path) -> Result<(), String> {
    if destination.exists() {
        let _ = fs::remove_dir_all(staging);
        return Err("分叉目标已存在，请重试".to_string());
    }
    if let Err(error) = fs::rename(staging, destination) {
        let _ = fs::remove_dir_all(staging);
        return Err(format!("发布分叉会话失败：{error}"));
    }
    Ok(())
}

fn fork_dsh(path: &Path, rows: &[Record], target: usize) -> Result<ForkResult, String> {
    let mut history = history_through_round(SessionSourceKind::Dsh, rows, target)?;
    let header = history
        .iter_mut()
        .find(|row| string(row, "type") == Some("session"))
        .ok_or("DeepSeek Harness 会话头缺失")?;
    let project = string(header, "cwd")
        .ok_or("DeepSeek Harness 会话缺少 cwd")?
        .to_string();
    let old_id = string(header, "id")
        .ok_or("DeepSeek Harness 会话缺少 id")?
        .to_string();
    let id = format!("session-{}", uuid::Uuid::new_v4());
    header["id"] = json!(id);
    header["createdAt"] = json!(chrono::Utc::now().timestamp_millis());
    header["parentSession"] = json!(old_id);

    let source_dir = path.parent().ok_or("DeepSeek Harness 会话目录不存在")?;
    let project_dir = source_dir
        .parent()
        .ok_or("DeepSeek Harness 工作区目录不存在")?;
    let project_id = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("DeepSeek Harness 工作区标识无效")?
        .to_string();
    let destination = project_dir.join(&id);
    let staging = project_dir.join(format!(".{id}.fork-tmp"));
    let source_file_name = path
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| matches!(*name, "session.v3.jsonl.zstd" | "session.v4.jsonl.zstd"))
        .ok_or("DeepSeek Harness 会话文件名无效")?;
    copy_session_extras(
        source_dir,
        &staging,
        &["session.v3.jsonl.zstd", "session.v4.jsonl.zstd"],
    )?;
    let compressed = dsh::encode_rows(&history)
        .map_err(|e| format!("压缩 DeepSeek Harness 分叉失败：{e}"));
    let result = compressed.and_then(|bytes| {
        write_new(&staging.join(source_file_name), &bytes)?;
        publish_directory_fork(&staging, &destination)
    });
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(ForkResult {
        new_session_id: id,
        new_file_path: destination
            .join(source_file_name)
            .to_string_lossy()
            .into_owned(),
        project_path: project,
        project_id,
    })
}

fn fork_kiro(path: &Path, rows: &[Record], target: usize) -> Result<ForkResult, String> {
    if path.file_name().and_then(|name| name.to_str()) != Some("messages.jsonl") {
        return Err("Kiro 会话缺少 messages.jsonl，无法从历史消息分叉".to_string());
    }
    let history = history_through_round(SessionSourceKind::Kiro, rows, target)?;
    let source_dir = path.parent().ok_or("Kiro 会话目录不存在")?;
    let project_dir = source_dir.parent().ok_or("Kiro 工作区目录不存在")?;
    let project_id = project_dir
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or("Kiro 工作区标识无效")?
        .to_string();
    let mut metadata: Value = serde_json::from_slice(
        &fs::read(source_dir.join("session.json"))
            .map_err(|e| format!("读取 Kiro 会话配置失败：{e}"))?,
    )
    .map_err(|e| format!("Kiro 会话配置损坏：{e}"))?;
    let project = metadata
        .get("workspacePaths")
        .and_then(Value::as_array)
        .and_then(|paths| paths.iter().find_map(Value::as_str))
        .unwrap_or_default()
        .to_string();
    let id = format!("sess_{}", uuid::Uuid::new_v4());
    let now = chrono::Utc::now().to_rfc3339();
    metadata["id"] = json!(id);
    metadata["createdAt"] = json!(now);
    metadata["lastModifiedAt"] = json!(now);
    metadata["status"] = json!("idle");
    if let Some(title) = metadata.get("title").and_then(Value::as_str) {
        metadata["title"] = json!(format!("{title}（分叉）"));
    }
    metadata["description"] = json!("从历史对话记录创建的分叉");

    let destination = project_dir.join(&id);
    let staging = project_dir.join(format!(".{id}.fork-tmp"));
    copy_session_extras(
        source_dir,
        &staging,
        &["messages.jsonl", "session.json", "publish.cursor"],
    )?;
    let result = (|| {
        write_new(&staging.join("messages.jsonl"), &jsonl(&history))?;
        write_new(
            &staging.join("session.json"),
            serde_json::to_vec_pretty(&metadata)
                .map_err(|e| e.to_string())?
                .as_slice(),
        )?;
        publish_directory_fork(&staging, &destination)
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&staging);
        return Err(error);
    }
    Ok(ForkResult {
        new_session_id: id,
        new_file_path: destination
            .join("messages.jsonl")
            .to_string_lossy()
            .into_owned(),
        project_path: project,
        project_id,
    })
}

fn fork_files(
    kind: SessionSourceKind,
    path: &Path,
    rows: &[Record],
    target: usize,
) -> Result<ForkResult, String> {
    let mut history = history_through_round(kind, rows, target)?;
    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    let parent = path.parent().ok_or("会话目录不存在")?;
    let (project, new_path) = match kind {
        SessionSourceKind::Claude => {
            let project = rows
                .iter()
                .find_map(|r| string(&r.value, "cwd"))
                .map(str::to_owned)
                .or_else(|| {
                    let index: Value =
                        serde_json::from_slice(&fs::read(parent.join("sessions-index.json")).ok()?)
                            .ok()?;
                    string(&index, "originalPath").map(str::to_owned)
                })
                .or_else(|| {
                    let name = parent.file_name()?.to_str()?;
                    let decoded = crate::parser::path_encoder::decode_project_path_validated(name);
                    decoded.path_exists.then_some(decoded.display_path)
                })
                .ok_or("会话中缺少项目路径")?;
            for row in &mut history {
                if row.get("sessionId").is_some() {
                    row["sessionId"] = json!(id);
                }
            }
            (project, parent.join(format!("{id}.jsonl")))
        }
        SessionSourceKind::Omp => {
            let header = history
                .iter_mut()
                .find(|r| string(r, "type") == Some("session"))
                .ok_or("OMP 会话头缺失")?;
            let project = string(header, "cwd").ok_or("OMP 会话缺少 cwd")?.to_string();
            let old_id = string(header, "id").ok_or("OMP 会话缺少 id")?.to_string();
            header["id"] = json!(id);
            header["timestamp"] = json!(now);
            header["parentSession"] = json!(old_id);
            if header.get("providerPromptCacheKey").is_none() {
                header["providerPromptCacheKey"] = json!(old_id);
            }
            (
                project,
                parent.join(format!("{}_{id}.jsonl", now.replace([':', '.'], "-"))),
            )
        }
        SessionSourceKind::Grok => {
            let mut summary: Value = serde_json::from_slice(
                &fs::read(parent.join("summary.json")).map_err(|e| e.to_string())?,
            )
            .map_err(|e| format!("Grok 会话摘要损坏：{e}"))?;
            let original_id = summary
                .pointer("/info/id")
                .and_then(Value::as_str)
                .filter(|id| !id.trim().is_empty())
                .ok_or("Grok 会话摘要缺少有效的 info.id")?
                .to_string();
            let original_title = grok::native_display_name_for_path(path)
                .or_else(|| {
                    summary
                        .get("session_summary")
                        .and_then(Value::as_str)
                        .map(str::trim)
                        .filter(|title| !title.is_empty())
                        .map(str::to_owned)
                })
                .or_else(|| {
                    rows.iter().find_map(|row| {
                        let row = &row.value;
                        if string(row, "type") != Some("user")
                            || row.get("synthetic_reason").is_some_and(|value| !value.is_null())
                        {
                            return None;
                        }
                        let content = row.get("content")?;
                        let text = if let Some(text) = content.as_str() {
                            text.to_string()
                        } else {
                            content
                                .as_array()?
                                .iter()
                                .filter_map(|block| block.get("text").and_then(Value::as_str))
                                .collect::<Vec<_>>()
                                .join("\n")
                        };
                        let text = text.trim();
                        (!text.is_empty()).then(|| text.chars().take(200).collect())
                    })
                })
                .unwrap_or(original_id);
            // The native name registry rejects control characters and names
            // above 512 characters. Generated summaries can exceed that limit.
            let fork_title: String = format!("(fork){original_title}")
                .chars()
                .map(|ch| if ch.is_control() { ' ' } else { ch })
                .take(512)
                .collect();
            let project = summary
                .pointer("/info/cwd")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string();
            summary["info"]["id"] = json!(id);
            summary["created_at"] = json!(now);
            summary["updated_at"] = json!(now);
            summary["last_active_at"] = json!(now);
            summary["num_messages"] = json!(history.len());
            summary["num_chat_messages"] = json!(history
                .iter()
                .filter(|r| matches!(string(r, "type"), Some("user" | "assistant")))
                .count());
            // These describe the *old endpoint*, not the selected history.
            summary["session_summary"] = json!("");
            summary["request_id"] = json!(uuid::Uuid::new_v4().to_string());
            summary["next_trace_turn"] = json!(1);
            let destination = parent.parent().ok_or("Grok 项目目录不存在")?.join(&id);
            fs::create_dir(&destination).map_err(|e| e.to_string())?;
            let result = (|| {
                for name in ["prompt_context.json", "system_prompt.txt"] {
                    let file = parent.join(name);
                    if file.exists() {
                        if !fs::symlink_metadata(&file)
                            .map_err(|e| e.to_string())?
                            .is_file()
                        {
                            return Err("Grok 上下文文件不是普通文件".to_string());
                        }
                        fs::copy(file, destination.join(name)).map_err(|e| e.to_string())?;
                    }
                }
                write_new(&destination.join("chat_history.jsonl"), &jsonl(&history))?;
                // Publish summary last: the provider cannot discover partial forks.
                write_new(
                    &destination.join("summary.json"),
                    summary.to_string().as_bytes(),
                )?;
                grok::set_custom_name_for_path(
                    &destination.join("chat_history.jsonl"),
                    Some(&fork_title),
                )
            })();
            if let Err(error) = result {
                let _ = fs::remove_dir_all(&destination); // newly allocated UUID directory only
                return Err(error);
            }
            return Ok(ForkResult {
                new_session_id: id,
                new_file_path: destination
                    .join("chat_history.jsonl")
                    .to_string_lossy()
                    .into_owned(),
                project_id: if project.is_empty() {
                    "<grok-unrooted>".to_string()
                } else {
                    project.clone()
                },
                project_path: project,
            });
        }
        SessionSourceKind::Codex => return Err("Codex 分叉必须通过原生协议创建".to_string()),
        SessionSourceKind::Dsh => return fork_dsh(path, rows, target),
        SessionSourceKind::Kiro => return fork_kiro(path, rows, target),
    };
    let staging = new_path.with_extension("fork-tmp");
    let artifacts = new_path.with_extension("");
    let mut staging_created = false;
    let mut artifacts_created = false;
    let result = (|| {
        write_new(&staging, &jsonl(&history))?;
        staging_created = true;
        if kind == SessionSourceKind::Omp {
            copy_directory(&path.with_extension(""), &artifacts)?;
            artifacts_created = true;
        }
        // UUID destination and create_new staging prevent overwriting sessions.
        fs::rename(&staging, &new_path).map_err(|e| e.to_string())
    })();
    if let Err(error) = result {
        if staging_created {
            let _ = fs::remove_file(&staging);
        }
        if artifacts_created {
            let _ = fs::remove_dir_all(&artifacts);
        }
        return Err(error);
    }
    if kind == SessionSourceKind::Claude {
        if let Err(error) = register_claude_fork(&new_path, &id, &project, &history) {
            let _ = fs::remove_file(&new_path);
            return Err(error);
        }
    }
    Ok(ForkResult {
        new_session_id: id,
        new_file_path: new_path.to_string_lossy().into_owned(),
        project_id: if kind == SessionSourceKind::Claude {
            parent
                .file_name()
                .and_then(|name| name.to_str())
                .ok_or("Claude 项目目录名称无效")?
                .to_string()
        } else {
            project.clone()
        },
        project_path: project,
    })
}

pub async fn fork_session(request: ForkRequest) -> Result<ForkResult, String> {
    let kind = SessionSourceKind::parse(&request.source)?;
    let (path, rows, target) = tokio::task::spawn_blocking(move || {
        let path = validate_session_file(&request.source, &request.original_file_path)?;
        let rows = read_records_for_kind(kind, &path)?;
        let target = target_index(kind, &rows, &request.user_msg_uuid)?;
        Ok::<_, String>((path, rows, target))
    })
    .await
    .map_err(|e| e.to_string())??;
    let result = if kind == SessionSourceKind::Codex {
        fork_codex(&path, &rows, target).await?
    } else {
        tokio::task::spawn_blocking(move || fork_files(kind, &path, &rows, target))
            .await
            .map_err(|e| e.to_string())??
    };
    tokio::task::spawn_blocking(move || {
        let changed = [std::path::PathBuf::from(&result.new_file_path)];
        match kind {
            SessionSourceKind::Claude => claude::invalidate_project(&result.project_id),
            SessionSourceKind::Codex => codex::invalidate_paths(&changed),
            SessionSourceKind::Grok => grok::invalidate_paths(&changed),
            SessionSourceKind::Dsh => dsh::invalidate_paths(&changed),
            SessionSourceKind::Kiro => kiro::invalidate_paths(&changed),
            SessionSourceKind::Omp => omp::invalidate_paths(&changed),
        }
        result
    })
    .await
    .map_err(|e| e.to_string())
}

async fn fork_codex(path: &Path, rows: &[Record], target: usize) -> Result<ForkResult, String> {
    let meta = codex::extract_session_meta(path).ok_or("Codex 会话元数据缺失")?;
    let creds = crate::cli_config::resolve_credentials("codex", None, None)?;
    let server = crate::codex_app_server::CodexAppServer::global();
    let thread = server.read_thread(&creds, &meta.id).await?;
    let turns = thread
        .pointer("/thread/turns")
        .and_then(Value::as_array)
        .ok_or("Codex 未返回会话轮次")?;
    let turn_index = codex_turn_index(rows, target, turns)?;
    let turn = &turns[turn_index];
    let turn_id = string(turn, "id").ok_or("Codex 轮次缺少 id")?;
    if string(turn, "status") == Some("inProgress") {
        return Err("该轮回复尚未结束，请等待完成后分叉".to_string());
    }
    let fork = server.fork_thread(&creds, &meta.id, turn_id).await?;
    let id = fork
        .pointer("/thread/id")
        .and_then(Value::as_str)
        .ok_or("Codex 未返回新会话 ID")?
        .to_string();
    if id == meta.id {
        return Err("Codex 未创建独立会话".to_string());
    }
    let checked = server
        .read_thread(&creds, &id)
        .await
        .map_err(|e| format!("已创建 Codex 分叉 {id}，但无法验证历史：{e}"))?;
    let actual = checked
        .pointer("/thread/turns")
        .and_then(Value::as_array)
        .ok_or("Codex 分叉未返回轮次")?;
    let expected: Vec<_> = turns[..=turn_index]
        .iter()
        .map(|t| string(t, "id"))
        .collect();
    let actual_ids: Vec<_> = actual.iter().map(|t| string(t, "id")).collect();
    if !actual_ids.starts_with(&expected) {
        return Err(format!("已创建 Codex 分叉 {id}，但历史边界验证失败"));
    }
    // Older app-server versions ignore lastTurnId. Roll back ONLY the new fork
    // and verify the persisted result; never silently include later turns.
    let checked = if actual.len() > expected.len() {
        server
            .rollback_thread(&creds, &id, actual.len() - expected.len())
            .await
            .map_err(|e| format!("已创建 Codex 分叉 {id}，但截断后续轮次失败：{e}"))?;
        let trimmed = server.read_thread(&creds, &id).await?;
        let ids: Vec<_> = trimmed
            .pointer("/thread/turns")
            .and_then(Value::as_array)
            .ok_or("Codex 分叉缺少轮次")?
            .iter()
            .map(|t| string(t, "id"))
            .collect();
        if ids != expected {
            return Err(format!("Codex 分叉 {id} 的历史边界不符"));
        }
        trimmed
    } else {
        checked
    };
    let file = checked
        .pointer("/thread/path")
        .and_then(Value::as_str)
        .or_else(|| fork.pointer("/thread/path").and_then(Value::as_str))
        .ok_or_else(|| format!("已创建 Codex 分叉 {id}，但未返回存储路径"))?;
    let new_path = validate_session_file("codex", file)?;
    let new_meta = codex::extract_session_meta(&new_path).ok_or("Codex 分叉元数据缺失")?;
    if new_path == path || new_meta.id != id {
        return Err("Codex 返回的分叉文件与新会话 ID 不一致".to_string());
    }
    Ok(ForkResult {
        new_session_id: id,
        new_file_path: new_path.to_string_lossy().into_owned(),
        project_id: codex::project_id_for_session(&new_path)
            .ok_or("无法确定 Codex 分叉所属项目")?,
        project_path: new_meta.cwd,
    })
}

fn codex_turn_index(rows: &[Record], target: usize, turns: &[Value]) -> Result<usize, String> {
    let row = &rows[target].value;
    let content = row
        .pointer("/payload/content")
        .ok_or("Codex 提问内容缺失")?;
    let text = input_text(content);
    let matches_text = |turn: &Value| {
        turn.get("items")
            .and_then(Value::as_array)
            .is_some_and(|items| {
                items.iter().any(|item| {
                    string(item, "type") == Some("userMessage")
                        && item.get("content").is_some_and(|c| input_text(c) == text)
                })
            })
    };
    // A persisted turn id disambiguates repeated prompts, steering inputs and
    // rollbacks. Text matching is only a fallback for old files without ids.
    let native_id = rows[..=target].iter().rev().find_map(|record| {
        let row = &record.value;
        let is_context = string(row, "type") == Some("turn_context");
        let is_start = string(row, "type") == Some("event_msg")
            && matches!(
                row.pointer("/payload/type").and_then(Value::as_str),
                Some("task_started" | "turn_started")
            );
        (is_context || is_start)
            .then(|| row.pointer("/payload/turn_id").and_then(Value::as_str))
            .flatten()
    });
    if let Some(id) = native_id {
        return turns
            .iter()
            .position(|turn| string(turn, "id") == Some(id) && matches_text(turn))
            .ok_or_else(|| "所选 Codex 轮次已更新或已被回退，请刷新后重试".to_string());
    }
    if text.is_empty() {
        return Err("该消息缺少可定位的 Codex 轮次信息".to_string());
    }
    let occurrence = rows[..=target]
        .iter()
        .filter(|r| {
            is_user(SessionSourceKind::Codex, &r.value)
                && r.value
                    .pointer("/payload/content")
                    .is_some_and(|c| input_text(c) == text)
        })
        .count();
    let matching: Vec<_> = turns
        .iter()
        .enumerate()
        .filter(|(_, turn)| matches_text(turn))
        .collect();
    let (turn_index, _) = matching
        .get(occurrence.saturating_sub(1))
        .copied()
        .ok_or("无法将所选消息对应到 Codex 原生轮次，请刷新会话或升级 Codex CLI")?;
    Ok(turn_index)
}

fn input_text(content: &Value) -> String {
    if let Some(text) = content.as_str() {
        return text.trim().to_string();
    }
    content
        .as_array()
        .map(|items| {
            items
                .iter()
                .filter_map(|item| string(item, "text"))
                .collect::<Vec<_>>()
                .join("\n")
                .trim()
                .to_string()
        })
        .unwrap_or_default()
}

#[cfg(test)]
mod tests;
