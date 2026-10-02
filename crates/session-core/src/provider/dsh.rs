use std::fs;
use std::io::{BufRead, BufReader, Cursor, Write};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde_json::Value;

use crate::models::message::{
    DisplayContentBlock, DisplayMessage, PaginatedMessages, RangeMessages,
};
use crate::models::project::ProjectEntry;
use crate::models::session::{SessionIndexEntry, SessionStatus};

const SESSION_FILES: [&str; 2] = ["session.v4.jsonl.zstd", "session.v3.jsonl.zstd"];

pub fn get_sessions_dir() -> Option<PathBuf> {
    std::env::var_os("DSH_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".dsh")))
        .map(|home| home.join("sessions"))
}

pub(crate) fn decode_rows(path: &Path) -> Result<Vec<Value>, String> {
    let file = fs::File::open(path).map_err(|error| format!("打开 DSH 会话失败：{error}"))?;
    let decoded = zstd::stream::decode_all(file)
        .map_err(|error| format!("解压 DSH 会话失败：{error}"))?;
    BufReader::new(Cursor::new(decoded))
        .lines()
        .enumerate()
        .filter_map(|(index, line)| match line {
            Ok(line) if line.trim().is_empty() => None,
            Ok(line) => Some(
                serde_json::from_str::<Value>(&line)
                    .map_err(|error| format!("DSH 会话第 {} 行损坏：{error}", index + 1)),
            ),
            Err(error) => Some(Err(format!("读取 DSH 会话失败：{error}"))),
        })
        .collect()
}

fn millis_to_rfc3339(value: Option<i64>) -> Option<String> {
    DateTime::<Utc>::from_timestamp_millis(value?).map(|value| value.to_rfc3339())
}

fn json_text(value: &Value) -> String {
    value
        .as_str()
        .map(ToString::to_string)
        .unwrap_or_else(|| serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string()))
}

fn display_blocks(content: &Value) -> Vec<DisplayContentBlock> {
    let Some(blocks) = content.as_array() else {
        return content
            .as_str()
            .filter(|text| !text.trim().is_empty())
            .map(|text| vec![DisplayContentBlock::Text { text: text.to_string() }])
            .unwrap_or_default();
    };
    blocks
        .iter()
        .filter_map(|block| match block.get("type").and_then(Value::as_str) {
            Some("text") => block
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(|text| DisplayContentBlock::Text { text: text.to_string() }),
            Some("reasoning") => block
                .get("text")
                .and_then(Value::as_str)
                .filter(|text| !text.trim().is_empty())
                .map(|text| DisplayContentBlock::Reasoning { text: text.to_string() }),
            Some("tool-call") => Some(DisplayContentBlock::FunctionCall {
                name: block.get("name").and_then(Value::as_str).unwrap_or("tool").to_string(),
                arguments: block.get("arguments").map(json_text).unwrap_or_default(),
                call_id: block.get("id").and_then(Value::as_str).unwrap_or_default().to_string(),
            }),
            Some("tool-result") => Some(DisplayContentBlock::FunctionCallOutput {
                call_id: block
                    .get("toolCallId")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_string(),
                output: block.get("content").map(json_text).unwrap_or_default(),
            }),
            _ => None,
        })
        .collect()
}

fn display_message(row: &Value) -> Option<DisplayMessage> {
    let event_type = row.get("type")?.as_str()?;
    let data = row.get("data")?;
    let (message, role) = match event_type {
        "user/message" => (data, "user"),
        "assistant/message" => (data.get("message")?, "assistant"),
        "tool/result" => (data.get("message")?, "tool"),
        _ => return None,
    };
    let content = display_blocks(message.get("content")?);
    if content.is_empty() {
        return None;
    }
    let source = message.get("source");
    Some(DisplayMessage {
        uuid: message
            .get("id")
            .and_then(Value::as_str)
            .map(ToString::to_string)
            .or_else(|| row.get("seq").map(|seq| format!("dsh-{seq}"))),
        parent_uuid: None,
        role: role.to_string(),
        timestamp: millis_to_rfc3339(row.get("time").and_then(Value::as_i64)),
        model: source
            .and_then(|value| value.get("model"))
            .and_then(Value::as_str)
            .map(ToString::to_string),
        content,
    })
}

pub fn parse_all_messages(path: &Path) -> Result<Vec<DisplayMessage>, String> {
    Ok(decode_rows(path)?
        .iter()
        .filter_map(display_message)
        .collect())
}

fn session_files(project_dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(project_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter_map(|entry| {
            let session_dir = entry.path();
            SESSION_FILES
                .iter()
                .map(|name| session_dir.join(name))
                .find(|path| path.is_file())
        })
        .collect()
}

fn header_for(path: &Path) -> Option<Value> {
    let file = fs::File::open(path).ok()?;
    let decoder = zstd::stream::read::Decoder::new(file).ok()?;
    let mut reader = BufReader::new(decoder);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    serde_json::from_str(&line).ok()
}

fn session_entry(path: &Path) -> Option<SessionIndexEntry> {
    let rows = decode_rows(path).ok()?;
    let header = rows.first()?;
    if header.get("type").and_then(Value::as_str) != Some("session") {
        return None;
    }
    let session_id = header.get("id")?.as_str()?.to_string();
    let cwd = header.get("cwd").and_then(Value::as_str).map(ToString::to_string);
    let messages: Vec<_> = rows.iter().filter_map(display_message).collect();
    let first_prompt = messages.iter().find_map(|message| {
        (message.role == "user").then(|| {
            message.content.iter().find_map(|block| match block {
                DisplayContentBlock::Text { text } => Some(text.chars().take(200).collect()),
                _ => None,
            })
        })?
    });
    let title = rows.iter().rev().find_map(|row| {
        (row.get("type").and_then(Value::as_str) == Some("session/title"))
            .then(|| row.pointer("/data/title").and_then(Value::as_str))?
            .filter(|title| !title.trim().is_empty())
            .map(ToString::to_string)
    });
    let modified = rows
        .iter()
        .rev()
        .find_map(|row| row.get("time").and_then(Value::as_i64))
        .and_then(|time| millis_to_rfc3339(Some(time)));
    let model_provider = rows.iter().rev().find_map(|row| {
        row.pointer("/data/message/source/provider")
            .and_then(Value::as_str)
            .map(ToString::to_string)
    });

    Some(SessionIndexEntry {
        source: "dsh".to_string(),
        session_id,
        file_path: path.to_string_lossy().into_owned(),
        first_prompt,
        thread_name: title,
        message_count: messages.len() as u32,
        created: millis_to_rfc3339(header.get("createdAt").and_then(Value::as_i64)),
        modified,
        git_branch: None,
        project_path: cwd.clone(),
        is_sidechain: None,
        cwd,
        model_provider,
        cli_version: None,
        alias: None,
        tags: None,
        status: if messages.is_empty() { SessionStatus::Empty } else { SessionStatus::Valid },
    })
}

fn project_dir(project_id: &str) -> Result<PathBuf, String> {
    if project_id.is_empty() || project_id.contains('/') || project_id.contains('\\') {
        return Err("无效的 DSH 工作区标识".to_string());
    }
    let root = get_sessions_dir().ok_or_else(|| "找不到 DSH 会话目录".to_string())?;
    let dir = root.join(project_id);
    if !dir.is_dir() {
        return Err("DSH 工作区不存在".to_string());
    }
    Ok(dir)
}

pub fn get_projects() -> Result<Vec<ProjectEntry>, String> {
    let root = match get_sessions_dir() {
        Some(root) if root.is_dir() => root,
        _ => return Ok(Vec::new()),
    };
    let mut projects: Vec<_> = fs::read_dir(root)
        .map_err(|error| format!("读取 DSH 工作区失败：{error}"))?
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let dir = entry.path();
            let encoded_name = entry.file_name().to_string_lossy().into_owned();
            let files = session_files(&dir);
            let header = files.first().and_then(|path| header_for(path));
            let display_path = header
                .as_ref()
                .and_then(|value| value.get("cwd"))
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .unwrap_or_else(|| encoded_name.clone());
            let last_modified = files
                .iter()
                .filter_map(|path| fs::metadata(path).ok()?.modified().ok())
                .max()
                .map(DateTime::<Utc>::from)
                .map(|value| value.to_rfc3339());
            Some(ProjectEntry {
                source: "dsh".to_string(),
                id: encoded_name,
                short_name: Path::new(&display_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or(&display_path)
                    .to_string(),
                display_path: display_path.clone(),
                session_count: files.len(),
                last_modified,
                model_provider: Some("DeepSeek Harness".to_string()),
                alias: None,
                path_exists: Path::new(&display_path).exists(),
                is_virtual: false,
            })
        })
        .collect();
    projects.sort_by(|left, right| right.last_modified.cmp(&left.last_modified));
    Ok(projects)
}

pub fn refresh_projects_cache() -> Result<Vec<ProjectEntry>, String> { get_projects() }
pub fn rebuild_projects_cache() -> Result<Vec<ProjectEntry>, String> { get_projects() }
pub fn invalidate_paths(_paths: &[PathBuf]) {}

pub fn get_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    let files = session_files(&project_dir(project_id)?);
    let mut sessions: Vec<_> = files.par_iter().filter_map(|path| session_entry(path)).collect();
    sessions.sort_by(|left, right| right.modified.cmp(&left.modified));
    Ok(sessions)
}

pub fn refresh_sessions_cache(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    get_sessions(project_id)
}

pub fn get_invalid_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    Ok(get_sessions(project_id)?
        .into_iter()
        .filter(|session| session.status != SessionStatus::Valid)
        .collect())
}

pub fn delete_project(project_id: &str) -> Result<super::claude::DeleteResult, String> {
    let dir = project_dir(project_id)?;
    let sessions = get_sessions(project_id)?;
    let project_name = sessions
        .first()
        .and_then(|session| session.cwd.as_deref().or(session.project_path.as_deref()))
        .and_then(|path| Path::new(path).file_name())
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(project_id)
        .to_string();
    crate::recyclebin::move_to_recyclebin(
        &dir,
        "project",
        "ManualDelete",
        "dsh",
        project_id,
        None,
        Some(project_name),
    )?;
    for session in &sessions {
        let _ = crate::metadata::remove_session_meta("dsh", project_id, &session.session_id);
    }
    invalidate_paths(&[]);
    Ok(super::claude::DeleteResult {
        sessions_deleted: sessions.len(),
        config_cleaned: false,
        bookmarks_removed: 0,
    })
}

struct PreparedWrite {
    path: PathBuf,
    bytes: Vec<u8>,
}

fn message_value(row: &Value) -> Option<(&str, &Value)> {
    let data = row.get("data")?;
    match row.get("type").and_then(Value::as_str)? {
        "user/message" => Some(("user", data)),
        "assistant/message" => Some(("assistant", data.get("message")?)),
        _ => None,
    }
}

fn row_message_id(index: usize, row: &Value) -> Option<String> {
    let (_, message) = message_value(row)?;
    message
        .get("id")
        .and_then(Value::as_str)
        .map(ToString::to_string)
        .or_else(|| row.get("seq").map(|seq| format!("dsh-{seq}")))
        .or_else(|| Some(format!("dsh-line-{index}")))
}

fn content_text(content: &Value) -> Option<String> {
    if let Some(text) = content.as_str() {
        return (!text.trim().is_empty()).then(|| text.to_string());
    }
    let text = content
        .as_array()?
        .iter()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("text"))
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n\n");
    (!text.is_empty()).then_some(text)
}

fn visible_text(row: &Value) -> Option<(&str, String)> {
    let (role, message) = message_value(row)?;
    Some((role, content_text(message.get("content")?)?))
}

fn replace_content_text(content: &mut Value, text: &str) -> Result<(), String> {
    match content {
        Value::String(value) => {
            *value = text.to_string();
            Ok(())
        }
        Value::Array(blocks) => {
            let first_text = blocks
                .iter()
                .position(|block| block.get("type").and_then(Value::as_str) == Some("text"))
                .ok_or_else(|| "该 DeepSeek 消息没有可编辑的文本块".to_string())?;
            blocks[first_text]["text"] = Value::String(text.to_string());
            // Keep the original block order and count: native DSH events can
            // interleave text with reasoning and tool calls.
            for block in blocks.iter_mut().skip(first_text + 1) {
                if block.get("type").and_then(Value::as_str) == Some("text") {
                    block["text"] = Value::String(String::new());
                }
            }
            Ok(())
        }
        _ => Err("该 DeepSeek 消息的 content 格式不支持编辑".to_string()),
    }
}

fn sync_inbox_copy(rows: &mut [Value], message_id: &str, text: Option<&str>) -> Result<(), String> {
    for row in rows {
        if row.get("type").and_then(Value::as_str) != Some("agent/inbox/spliced") {
            continue;
        }
        let Some(inserted) = row.pointer_mut("/data/inserted").and_then(Value::as_array_mut)
        else {
            continue;
        };
        if let Some(text) = text {
            for message in inserted.iter_mut() {
                if message.get("id").and_then(Value::as_str) != Some(message_id) {
                    continue;
                }
                let content = message
                    .get_mut("content")
                    .ok_or_else(|| "DeepSeek 收件箱副本缺少 content 字段".to_string())?;
                replace_content_text(content, text)?;
            }
        } else {
            inserted.retain(|message| message.get("id").and_then(Value::as_str) != Some(message_id));
        }
    }
    Ok(())
}

fn edit_rows(rows: &mut [Value], message_id: &str, text: &str) -> Result<(String, bool, Option<i64>), String> {
    let target = rows
        .iter()
        .enumerate()
        .find_map(|(index, row)| (row_message_id(index, row).as_deref() == Some(message_id)).then_some(index))
        .ok_or_else(|| "找不到要编辑的 DeepSeek 消息，可能会话已被其他进程修改".to_string())?;
    let role = message_value(&rows[target])
        .map(|(role, _)| role.to_string())
        .ok_or_else(|| "只能编辑 DeepSeek 用户消息或助手消息".to_string())?;
    let target_seq = rows[target].get("seq").and_then(Value::as_i64);
    let is_last = rows
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, row)| {
            visible_text(row)
                .filter(|(candidate_role, _)| *candidate_role == role.as_str())
                .map(|_| index)
        })
        == Some(target);

    let data = rows[target]
        .get_mut("data")
        .ok_or_else(|| "DeepSeek 消息缺少 data 字段".to_string())?;
    let message = if role == "assistant" {
        data.get_mut("message")
            .ok_or_else(|| "DeepSeek 助手消息缺少 message 字段".to_string())?
    } else {
        data
    };
    let content = message
        .get_mut("content")
        .ok_or_else(|| "该 DeepSeek 消息没有 content 字段".to_string())?;
    replace_content_text(content, text)?;
    if role == "user" {
        sync_inbox_copy(rows, message_id, Some(text))?;
    }
    Ok((role, is_last, target_seq))
}

fn delete_rows(rows: &mut Vec<Value>, message_id: &str) -> Result<(String, bool, Option<i64>), String> {
    let target = rows
        .iter()
        .enumerate()
        .find_map(|(index, row)| (row_message_id(index, row).as_deref() == Some(message_id)).then_some(index))
        .ok_or_else(|| "找不到要删除的 DeepSeek 消息，可能会话已被其他进程修改".to_string())?;
    let role = message_value(&rows[target])
        .map(|(role, _)| role.to_string())
        .ok_or_else(|| "只能删除 DeepSeek 用户消息或助手消息".to_string())?;
    let target_seq = rows[target].get("seq").and_then(Value::as_i64);
    let call_ids: std::collections::HashSet<String> = message_value(&rows[target])
        .and_then(|(_, message)| message.get("content"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|block| block.get("type").and_then(Value::as_str) == Some("tool-call"))
        .map(|block| {
            block
                .get("id")
                .and_then(Value::as_str)
                .map(ToString::to_string)
                .ok_or_else(|| "DeepSeek 工具调用缺少 ID，无法安全删除".to_string())
        })
        .collect::<Result<_, _>>()?;
    let is_last = rows
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, row)| {
            visible_text(row)
                .filter(|(candidate_role, _)| *candidate_role == role.as_str())
                .map(|_| index)
        })
        == Some(target);
    rows.remove(target);
    if role == "user" {
        sync_inbox_copy(rows, message_id, None)?;
    }
    if !call_ids.is_empty() {
        rows.retain(|row| {
            let call_id = match row.get("type").and_then(Value::as_str) {
                Some("tool/call") => row.pointer("/data/callId").and_then(Value::as_str),
                Some("tool/result") => row.pointer("/data/message/toolCallId").and_then(Value::as_str),
                _ => None,
            };
            !call_id.is_some_and(|id| call_ids.contains(id))
        });
    }
    Ok((role, is_last, target_seq))
}

fn encode_rows(rows: &[Value]) -> Result<Vec<u8>, String> {
    let mut jsonl = String::new();
    for row in rows {
        jsonl.push_str(
            &serde_json::to_string(row)
                .map_err(|error| format!("序列化 DeepSeek 会话失败：{error}"))?,
        );
        jsonl.push('\n');
    }
    zstd::stream::encode_all(Cursor::new(jsonl.into_bytes()), 3)
        .map_err(|error| format!("压缩 DeepSeek 会话失败：{error}"))
}

fn companion_paths(path: &Path) -> Vec<PathBuf> {
    let mut paths = vec![path.to_path_buf()];
    if let Some(parent) = path.parent() {
        for name in SESSION_FILES {
            let candidate = parent.join(name);
            if candidate.is_file() && !paths.contains(&candidate) {
                paths.push(candidate);
            }
        }
    }
    paths
}

fn projection_cache_path(path: &Path) -> Option<PathBuf> {
    let session_dir = path.parent()?;
    let session_id = session_dir.file_name()?.to_str()?;
    let sessions_dir = session_dir.parent()?.parent()?;
    (sessions_dir.file_name()?.to_str()? == "sessions").then(|| {
        sessions_dir
            .parent()
            .unwrap_or(sessions_dir)
            .join("storages")
            .join("session_projcache")
            .join("sessions")
            .join(format!("{session_id}.json"))
    })
}

fn compact_projection_text(text: &str) -> String {
    let trimmed = text.trim();
    let mut compact: String = trimmed.chars().take(200).collect();
    if trimmed.chars().count() > 200 {
        compact.push('…');
    }
    compact
}

fn projection_write(
    path: &Path,
    role: &str,
    target_seq: Option<i64>,
    rows: &[Value],
) -> Result<Option<PreparedWrite>, String> {
    let Some(cache_path) = projection_cache_path(path) else {
        return Ok(None);
    };
    if !cache_path.is_file() {
        return Ok(None);
    }
    let mut cache: Value = serde_json::from_slice(
        &fs::read(&cache_path)
            .map_err(|error| format!("读取 DeepSeek 会话配置失败：{error}"))?,
    )
    .map_err(|error| format!("解析 DeepSeek 会话配置失败：{error}"))?;
    let Some(turns) = cache
        .pointer_mut("/record/rows/turnOutline/val/turns")
        .and_then(Value::as_array_mut)
    else {
        return Ok(None);
    };
    let Some(last_turn) = turns
        .last_mut()
        .and_then(Value::as_object_mut)
    else {
        return Ok(None);
    };
    let Some(turn_start) = last_turn
        .get("seq")
        .and_then(Value::as_i64)
    else {
        return Ok(None);
    };
    // Auto-continuation turns may have no new user message. Never copy a
    // prompt or answer from an earlier turn into their projection.
    if !target_seq.is_some_and(|seq| seq >= turn_start) {
        return Ok(None);
    }
    let text = rows
        .iter()
        .rev()
        .filter(|row| row.get("seq").and_then(Value::as_i64).is_some_and(|seq| seq >= turn_start))
        .find_map(|row| visible_text(row).filter(|(candidate, _)| *candidate == role))
        .map(|(_, text)| text)
        .unwrap_or_default();
    let field = if role == "user" { "prompt" } else { "response" };
    last_turn.insert(field.to_string(), Value::String(compact_projection_text(&text)));
    let bytes = serde_json::to_vec_pretty(&cache)
        .map_err(|error| format!("序列化 DeepSeek 会话配置失败：{error}"))?;
    Ok(Some(PreparedWrite { path: cache_path, bytes }))
}

fn sibling_path(path: &Path, suffix: &str) -> Result<PathBuf, String> {
    let name = path
        .file_name()
        .and_then(|name| name.to_str())
        .ok_or_else(|| "DeepSeek 会话文件名无效".to_string())?;
    Ok(path.with_file_name(format!("{name}.{suffix}")))
}

fn commit_writes(writes: Vec<PreparedWrite>) -> Result<(), String> {
    let transaction = uuid::Uuid::new_v4();
    let mut staged = Vec::with_capacity(writes.len());
    for write in &writes {
        if !write.path.is_file() {
            for (temp, _) in &staged {
                let _ = fs::remove_file(temp);
            }
            return Err(format!("DeepSeek 原文件已消失，已停止保存：{}", write.path.display()));
        }
        let temp = sibling_path(&write.path, &format!("asv-write-{transaction}.tmp"))?;
        let backup = sibling_path(&write.path, &format!("asv-write-{transaction}.bak"))?;
        let stage_result = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temp)
            .and_then(|mut file| file.write_all(&write.bytes));
        if let Err(error) = stage_result {
            let _ = fs::remove_file(&temp);
            for (temp, _) in &staged {
                let _ = fs::remove_file(temp);
            }
            return Err(format!("写入 DeepSeek 临时文件失败：{error}"));
        }
        staged.push((temp, backup));
    }

    for (index, write) in writes.iter().enumerate() {
        let (temp, backup) = &staged[index];
        if let Err(error) = fs::rename(&write.path, backup) {
            let restore_errors = rollback_writes(&writes, &staged, index);
            return Err(format!("备份 DeepSeek 原文件失败：{error}{restore_errors}"));
        }
        if let Err(error) = fs::rename(temp, &write.path) {
            let restore_errors = rollback_writes(&writes, &staged, index + 1);
            return Err(format!("保存 DeepSeek 修改失败：{error}{restore_errors}"));
        }
    }
    for (_, backup) in staged {
        let _ = fs::remove_file(backup);
    }
    Ok(())
}

fn rollback_writes(
    writes: &[PreparedWrite],
    staged: &[(PathBuf, PathBuf)],
    backup_count: usize,
) -> String {
    let mut errors = Vec::new();
    for index in (0..backup_count).rev() {
        let original = &writes[index].path;
        let backup = &staged[index].1;
        if original.exists() {
            if let Err(error) = fs::remove_file(original) {
                errors.push(format!("无法移除失败后的文件 {}：{error}", original.display()));
                continue;
            }
        }
        if let Err(error) = fs::rename(backup, original) {
            errors.push(format!("无法从备份 {} 恢复：{error}", backup.display()));
        }
    }
    for (temp, _) in staged {
        let _ = fs::remove_file(temp);
    }
    if errors.is_empty() {
        String::new()
    } else {
        format!("；自动恢复未完成：{}", errors.join("；"))
    }
}

pub fn edit_message(path: &Path, message_id: &str, text: &str) -> Result<(), String> {
    if text.trim().is_empty() {
        return Err("DeepSeek 消息内容不能为空".to_string());
    }
    let paths = companion_paths(path);
    let mut writes = Vec::with_capacity(paths.len() + 1);
    let mut primary_rows = None;
    let mut primary_role = None;
    let mut primary_is_last = false;
    let mut primary_target_seq = None;
    for candidate in paths {
        let mut rows = decode_rows(&candidate)?;
        let contains_message = rows
            .iter()
            .enumerate()
            .any(|(index, row)| row_message_id(index, row).as_deref() == Some(message_id));
        if candidate.as_path() != path && !contains_message {
            continue;
        }
        let (role, is_last, target_seq) = edit_rows(&mut rows, message_id, text)?;
        if candidate.as_path() == path {
            primary_role = Some(role);
            primary_is_last = is_last;
            primary_target_seq = target_seq;
            primary_rows = Some(rows.clone());
        }
        writes.push(PreparedWrite { path: candidate, bytes: encode_rows(&rows)? });
    }
    if primary_is_last {
        let role = primary_role.as_deref().unwrap_or("assistant");
        if let Some(cache_write) = projection_write(
            path,
            role,
            primary_target_seq,
            primary_rows.as_deref().unwrap_or_default(),
        )? {
            writes.push(cache_write);
        }
    }
    commit_writes(writes)
}

pub fn delete_message(path: &Path, message_id: &str) -> Result<(), String> {
    let paths = companion_paths(path);
    let mut writes = Vec::with_capacity(paths.len() + 1);
    let mut primary_rows = None;
    let mut primary_role = None;
    let mut primary_was_last = false;
    let mut primary_target_seq = None;
    for candidate in paths {
        let mut rows = decode_rows(&candidate)?;
        let contains_message = rows
            .iter()
            .enumerate()
            .any(|(index, row)| row_message_id(index, row).as_deref() == Some(message_id));
        if candidate.as_path() != path && !contains_message {
            continue;
        }
        let (role, was_last, target_seq) = delete_rows(&mut rows, message_id)?;
        if candidate.as_path() == path {
            primary_role = Some(role);
            primary_was_last = was_last;
            primary_target_seq = target_seq;
            primary_rows = Some(rows.clone());
        }
        writes.push(PreparedWrite { path: candidate, bytes: encode_rows(&rows)? });
    }
    if primary_was_last {
        let role = primary_role.as_deref().unwrap_or("assistant");
        if let Some(cache_write) = projection_write(
            path,
            role,
            primary_target_seq,
            primary_rows.as_deref().unwrap_or_default(),
        )? {
            writes.push(cache_write);
        }
    }
    commit_writes(writes)
}

pub fn parse_session_messages(
    path: &Path,
    page: usize,
    page_size: usize,
    from_end: bool,
) -> Result<PaginatedMessages, String> {
    let all = parse_all_messages(path)?;
    let total = all.len();
    let (start, end) = if from_end {
        (
            total.saturating_sub((page + 1).saturating_mul(page_size)),
            total.saturating_sub(page.saturating_mul(page_size)),
        )
    } else {
        let start = page.saturating_mul(page_size).min(total);
        (start, start.saturating_add(page_size).min(total))
    };
    Ok(PaginatedMessages {
        messages: all[start..end].to_vec(),
        total,
        page,
        page_size,
        has_more: if from_end { start > 0 } else { end < total },
    })
}

pub fn parse_messages_range(path: &Path, start: usize, end: usize) -> Result<RangeMessages, String> {
    let all = parse_all_messages(path)?;
    let total = all.len();
    let start = start.min(total);
    let end = end.min(total).max(start);
    Ok(RangeMessages { messages: all[start..end].to_vec(), total, start, end })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    #[test]
    fn parses_v3_compressed_messages_and_title() {
        let root = std::env::temp_dir().join(format!("dsh-provider-{}", uuid::Uuid::new_v4()));
        let session_dir = root.join("workspace").join("session-1");
        fs::create_dir_all(&session_dir).unwrap();
        let path = session_dir.join(SESSION_FILES[1]);
        let file = fs::File::create(&path).unwrap();
        let mut encoder = zstd::stream::write::Encoder::new(file, 1).unwrap();
        writeln!(encoder, r#"{{"type":"session","version":3,"id":"session-1","createdAt":1000,"cwd":"C:\\\\work"}}"#).unwrap();
        writeln!(encoder, r#"{{"type":"user/message","time":2000,"data":{{"id":"u1","content":[{{"type":"text","text":"hello"}}]}}}}"#).unwrap();
        writeln!(encoder, r#"{{"type":"assistant/message","time":3000,"data":{{"message":{{"id":"a1","content":[{{"type":"reasoning","text":"think"}},{{"type":"text","text":"world"}}],"source":{{"provider":"deepseek","model":"deepseek-chat"}}}}}}}}"#).unwrap();
        writeln!(encoder, r#"{{"type":"session/title","time":4000,"data":{{"title":"Named session"}}}}"#).unwrap();
        encoder.finish().unwrap();

        let entry = session_entry(&path).unwrap();
        assert_eq!(entry.session_id, "session-1");
        assert_eq!(entry.thread_name.as_deref(), Some("Named session"));
        assert_eq!(entry.first_prompt.as_deref(), Some("hello"));
        assert_eq!(entry.message_count, 2);
        assert_eq!(entry.model_provider.as_deref(), Some("deepseek"));
        let messages = parse_all_messages(&path).unwrap();
        assert_eq!(messages.len(), 2);
        assert_eq!(messages[1].content.len(), 2);

        let _ = fs::remove_dir_all(root);
    }

    fn write_compressed(path: &Path, rows: &[Value]) {
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, encode_rows(rows).unwrap()).unwrap();
    }

    #[test]
    fn edits_and_deletes_both_versions_and_syncs_last_turn_projection() {
        let root = std::env::temp_dir().join(format!("dsh-edit-provider-{}", uuid::Uuid::new_v4()));
        let session_dir = root.join("sessions").join("workspace").join("session-1");
        let v4 = session_dir.join(SESSION_FILES[0]);
        let v3 = session_dir.join(SESSION_FILES[1]);
        let rows = vec![
            serde_json::json!({"type":"session","version":4,"id":"session-1","seq":0,"createdAt":1000,"cwd":"C:\\work"}),
            serde_json::json!({"type":"agent/inbox/spliced","seq":1,"data":{"inserted":[{"id":"u1","content":[{"type":"text","text":"question"}]}]}}),
            serde_json::json!({"type":"user/message","seq":1,"time":2000,"data":{"id":"u1","content":[{"type":"text","text":"question"}]}}),
            serde_json::json!({"type":"assistant/message","seq":2,"time":3000,"data":{"turn":1,"message":{"id":"a1","content":[{"type":"text","text":"first answer"}],"source":{"provider":"deepseek","model":"deepseek-chat"}}}}),
            serde_json::json!({"type":"assistant/message","seq":3,"time":4000,"data":{"turn":1,"message":{"id":"a2","content":[{"type":"reasoning","text":"private"},{"type":"text","text":"old answer"},{"type":"tool-call","id":"call-1","name":"test"}],"source":{"provider":"deepseek","model":"deepseek-chat"}}}}),
            serde_json::json!({"type":"tool/call","seq":4,"data":{"callId":"call-1","name":"test"}}),
            serde_json::json!({"type":"tool/result","seq":5,"data":{"message":{"toolCallId":"call-1","content":[]}}}),
        ];
        write_compressed(&v4, &rows);
        write_compressed(&v3, &rows);

        let cache = root
            .join("storages")
            .join("session_projcache")
            .join("sessions")
            .join("session-1.json");
        fs::create_dir_all(cache.parent().unwrap()).unwrap();
        fs::write(
            &cache,
            serde_json::to_vec_pretty(&serde_json::json!({
                "version": 1,
                "record": {"rows": {"turnOutline": {"val": {"turns": [
                    {"turn": 1, "seq": 0, "prompt": "question", "response": "old answer"}
                ]}}}}
            }))
            .unwrap(),
        )
        .unwrap();

        edit_message(&v4, "a2", "updated answer").unwrap();
        for path in [&v4, &v3] {
            let messages = parse_all_messages(path).unwrap();
            assert_eq!(messages.len(), 3);
            assert!(messages[2].content.iter().any(|block| {
                matches!(block, DisplayContentBlock::Text { text } if text == "updated answer")
            }));
            assert!(messages[2].content.iter().any(|block| {
                matches!(block, DisplayContentBlock::Reasoning { text } if text == "private")
            }));
        }
        let projection: Value = serde_json::from_slice(&fs::read(&cache).unwrap()).unwrap();
        assert_eq!(
            projection.pointer("/record/rows/turnOutline/val/turns/0/response").and_then(Value::as_str),
            Some("updated answer")
        );

        delete_message(&v4, "a2").unwrap();
        for path in [&v4, &v3] {
            let messages = parse_all_messages(path).unwrap();
            assert_eq!(messages.len(), 2);
            assert_eq!(messages[1].uuid.as_deref(), Some("a1"));
            assert!(!decode_rows(path).unwrap().iter().any(|row| {
                matches!(row.get("type").and_then(Value::as_str), Some("tool/call" | "tool/result"))
            }));
        }
        let projection: Value = serde_json::from_slice(&fs::read(&cache).unwrap()).unwrap();
        assert_eq!(
            projection.pointer("/record/rows/turnOutline/val/turns/0/response").and_then(Value::as_str),
            Some("first answer")
        );
        assert_eq!(session_files(&root.join("sessions").join("workspace")), vec![v4.clone()]);

        edit_message(&v4, "u1", "changed question").unwrap();
        for path in [&v4, &v3] {
            let rows = decode_rows(path).unwrap();
            assert_eq!(rows[1].pointer("/data/inserted/0/content/0/text").and_then(Value::as_str), Some("changed question"));
            assert_eq!(rows[2].pointer("/data/content/0/text").and_then(Value::as_str), Some("changed question"));
        }
        delete_message(&v4, "u1").unwrap();
        for path in [&v4, &v3] {
            let rows = decode_rows(path).unwrap();
            assert!(rows[1].pointer("/data/inserted").and_then(Value::as_array).unwrap().is_empty());
            assert_eq!(parse_all_messages(path).unwrap().len(), 1);
        }

        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn editing_earlier_turn_does_not_fill_auto_continuation_prompt() {
        let root = std::env::temp_dir().join(format!("dsh-auto-turn-{}", uuid::Uuid::new_v4()));
        let path = root
            .join("sessions")
            .join("workspace")
            .join("session-1")
            .join(SESSION_FILES[0]);
        write_compressed(&path, &[
            serde_json::json!({"type":"session","id":"session-1","cwd":"C:\\work"}),
            serde_json::json!({"type":"turn/start","seq":1,"data":{"turn":1}}),
            serde_json::json!({"type":"user/message","seq":2,"data":{"id":"u1","content":[{"type":"text","text":"old"}]}}),
            serde_json::json!({"type":"turn/start","seq":5,"data":{"turn":2}}),
            serde_json::json!({"type":"assistant/message","seq":6,"data":{"turn":2,"message":{"id":"a1","content":[{"type":"text","text":"answer"}]}}}),
        ]);
        let cache = root
            .join("storages")
            .join("session_projcache")
            .join("sessions")
            .join("session-1.json");
        fs::create_dir_all(cache.parent().unwrap()).unwrap();
        fs::write(&cache, serde_json::json!({
            "record":{"rows":{"turnOutline":{"val":{"turns":[
                {"turn":1,"seq":1,"prompt":"old","response":""},
                {"turn":2,"seq":5,"prompt":"","response":"answer"}
            ]}}}}
        }).to_string()).unwrap();
        let old_backup = sibling_path(&path, "asv-write.bak").unwrap();
        fs::write(&old_backup, b"previous recovery copy").unwrap();

        edit_message(&path, "u1", "new").unwrap();
        let projection: Value = serde_json::from_slice(&fs::read(&cache).unwrap()).unwrap();
        assert_eq!(projection.pointer("/record/rows/turnOutline/val/turns/1/prompt").and_then(Value::as_str), Some(""));
        assert_eq!(fs::read(&old_backup).unwrap(), b"previous recovery copy");
        assert_eq!(parse_all_messages(&path).unwrap()[0].content.len(), 1);

        let _ = fs::remove_dir_all(root);
    }
}
