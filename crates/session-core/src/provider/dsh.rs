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
    let bytes = fs::read(path).map_err(|error| format!("打开 DSH 会话失败：{error}"))?;
    decode_rows_from_bytes(&bytes)
}

fn decode_rows_from_bytes(bytes: &[u8]) -> Result<Vec<Value>, String> {
    let decoded = zstd::stream::decode_all(Cursor::new(bytes))
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
    original: Vec<u8>,
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
            for message in inserted.iter_mut() {
                if message.get("id").and_then(Value::as_str) == Some(message_id) {
                    message["content"] = Value::Array(Vec::new());
                }
            }
        }
    }
    Ok(())
}

fn sync_assistant_text_stream(data: &mut Value, text: &str) -> Result<(), String> {
    let unsupported = || "该 DeepSeek 助手消息的流格式无法安全同步，已保留原文件".to_string();
    if data.get("interrupted").and_then(Value::as_bool) == Some(true) {
        return Err(unsupported());
    }
    let content = data
        .pointer("/message/content")
        .and_then(Value::as_array)
        .ok_or_else(unsupported)?;
    let stream = data
        .get("stream")
        .and_then(Value::as_array)
        .ok_or_else(unsupported)?;

    // BlockAssembler orders blocks by the first occurrence of their index, not
    // by block-end order. Keep that mapping before changing any stream record.
    let mut block_order = Vec::new();
    let mut block_ends = Vec::new();
    let mut text_runs = Vec::new();
    for (position, record) in stream.iter().enumerate() {
        let kind = record.get("type").and_then(Value::as_str).ok_or_else(unsupported)?;
        let (index, chunk) = match kind {
            "chunk" => {
                let chunk = record.get("chunk").ok_or_else(unsupported)?;
                let chunk_type = chunk.get("type").and_then(Value::as_str).ok_or_else(unsupported)?;
                match chunk_type {
                    "usage" | "finish" => continue,
                    "text-delta" => return Err(unsupported()),
                    "block-start" | "block-end" | "reasoning-delta" | "tool-call-delta" => {
                        (chunk.get("index").and_then(Value::as_u64).ok_or_else(unsupported)?, Some(chunk))
                    }
                    _ => return Err(unsupported()),
                }
            }
            "text-chunks" | "reasoning-chunks" | "tool-call-chunks" => {
                (record.get("index").and_then(Value::as_u64).ok_or_else(unsupported)?, None)
            }
            _ => return Err(unsupported()),
        };
        if index > 9_007_199_254_740_991 {
            return Err(unsupported());
        }
        if !block_order.contains(&index) {
            block_order.push(index);
        }
        if kind == "text-chunks" {
            text_runs.push((index, position));
        }
        if let Some(chunk) = chunk {
            if chunk.get("type").and_then(Value::as_str) == Some("block-end") {
                let block_type = chunk.pointer("/block/type").and_then(Value::as_str).ok_or_else(unsupported)?;
                if block_ends.iter().any(|(seen, _, _)| *seen == index) {
                    return Err(unsupported());
                }
                block_ends.push((index, position, block_type));
            }
        }
    }
    if block_order.len() != content.len() || block_ends.len() != content.len() {
        return Err(unsupported());
    }

    let mut edits = Vec::new();
    for (position, block) in content.iter().enumerate() {
        let index = block_order[position];
        let block_type = block.get("type").and_then(Value::as_str).ok_or_else(unsupported)?;
        let (_, end_position, end_type) = block_ends
            .iter()
            .find(|(seen, _, _)| *seen == index)
            .ok_or_else(unsupported)?;
        if block_type != *end_type {
            return Err(unsupported());
        }
        if block_type != "text" {
            continue;
        }
        let old_text = block.get("text").and_then(Value::as_str).ok_or_else(unsupported)?;
        if stream[*end_position].pointer("/chunk/block/text").and_then(Value::as_str) != Some(old_text) {
            return Err(unsupported());
        }
        let runs: Vec<_> = text_runs.iter().filter(|(seen, _)| *seen == index).collect();
        if runs.len() != 1 || runs[0].1 >= *end_position {
            return Err(unsupported());
        }
        let run_position = runs[0].1;
        let run = &stream[run_position];
        if run.get("time0").and_then(Value::as_i64).is_none()
            || !run.get("dt").is_some_and(Value::is_array)
            || !run.get("texts").is_some_and(|value| {
                value.as_array().is_some_and(|items| !items.is_empty() && items.iter().all(Value::is_string))
            })
        {
            return Err(unsupported());
        }
        let replacement = if edits.is_empty() { text } else { "" };
        edits.push((run_position, *end_position, replacement.to_string()));
    }
    if edits.is_empty() {
        return Err("该 DeepSeek 消息没有可编辑的文本块".to_string());
    }
    let stream = data
        .get_mut("stream")
        .and_then(Value::as_array_mut)
        .ok_or_else(unsupported)?;
    for (run_position, end_position, replacement) in edits {
        stream[run_position]["texts"] = Value::Array(vec![Value::String(replacement.clone())]);
        stream[run_position]["dt"] = Value::Array(Vec::new());
        stream[end_position]["chunk"]["block"]["text"] = Value::String(replacement);
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
    if role == "assistant" {
        sync_assistant_text_stream(data, text)?;
    }
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
    let has_tool_calls_in_content = message_value(&rows[target])
        .and_then(|(_, message)| message.get("content"))
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|block| block.get("type").and_then(Value::as_str) == Some("tool-call"));
    let has_tool_calls_in_stream = rows[target]
        .pointer("/data/stream")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|entry| {
            entry.pointer("/chunk/type").and_then(Value::as_str) == Some("tool-call-delta")
                || entry.pointer("/chunk/block/type").and_then(Value::as_str) == Some("tool-call")
        });
    if has_tool_calls_in_content || has_tool_calls_in_stream {
        return Err("这条 DeepSeek 回复含工具调用；删除它会破坏原生会话关系，已保留原文件".to_string());
    }
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
    // Native DSH logs are append-only and every event seq is used as a stable
    // reference. Erasing a physical row leaves a gap and can invalidate later
    // surface, turn, and fork references. Empty the message payload in place;
    // the viewer hides empty messages and DSH no longer receives their text.
    let data = rows[target]
        .get_mut("data")
        .ok_or_else(|| "DeepSeek 消息缺少 data 字段".to_string())?;
    if role == "assistant" {
        if let Some(object) = data.as_object_mut() {
            if object.contains_key("stream") {
                object.insert("stream".to_string(), Value::Array(Vec::new()));
            }
            object.remove("usage");
        }
    }
    let message = if role == "assistant" {
        data.get_mut("message")
            .ok_or_else(|| "DeepSeek 助手消息缺少 message 字段".to_string())?
    } else {
        data
    };
    message["content"] = Value::Array(Vec::new());
    if role == "assistant" {
        if let Some(source) = message.get_mut("source").and_then(Value::as_object_mut) {
            source.remove("replayState");
        }
    }
    if role == "user" {
        sync_inbox_copy(rows, message_id, None)?;
    }
    Ok((role, is_last, target_seq))
}

fn encode_frame(bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut encoder = zstd::stream::write::Encoder::new(Vec::new(), 3)
        .map_err(|error| format!("压缩 DeepSeek 会话失败：{error}"))?;
    encoder
        .include_checksum(true)
        .map_err(|error| format!("设置 DeepSeek 帧校验失败：{error}"))?;
    encoder
        .write_all(bytes)
        .map_err(|error| format!("写入 DeepSeek 压缩帧失败：{error}"))?;
    encoder
        .finish()
        .map_err(|error| format!("完成 DeepSeek 压缩帧失败：{error}"))
}

/// DSH reads the first zstd frame independently and requires it to contain
/// exactly the header line. Later frames contain event batches.
pub(crate) fn encode_rows(rows: &[Value]) -> Result<Vec<u8>, String> {
    let (header, events) = rows
        .split_first()
        .ok_or_else(|| "DeepSeek 会话缺少头记录".to_string())?;
    if header.get("type").and_then(Value::as_str) != Some("session") {
        return Err("DeepSeek 会话首行不是头记录".to_string());
    }
    if header.get("version").and_then(Value::as_u64) == Some(4) {
        for (index, event) in events.iter().enumerate() {
            if event.get("seq").and_then(Value::as_u64) != Some(index as u64) {
                return Err(format!(
                    "DeepSeek v4 事件序号在第 {} 条断裂，已拒绝写入以保护原会话",
                    index + 1
                ));
            }
        }
    }
    let mut header_line = serde_json::to_vec(header)
        .map_err(|error| format!("序列化 DeepSeek 会话头失败：{error}"))?;
    header_line.push(b'\n');
    let mut compressed = encode_frame(&header_line)?;
    if !events.is_empty() {
        let mut body = Vec::new();
        for event in events {
            let mut event = event.clone();
            // The older pentest plugin emitted this extension without the
            // compatibility marker now used by its current version.
            if event.get("type").and_then(Value::as_str) == Some("plugin:pentest-submission")
                && event.get("ignorable").is_none()
            {
                event["ignorable"] = Value::Bool(true);
            }
            serde_json::to_writer(&mut body, &event)
                .map_err(|error| format!("序列化 DeepSeek 会话事件失败：{error}"))?;
            body.push(b'\n');
        }
        compressed.extend(encode_frame(&body)?);
    }
    Ok(compressed)
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
    let original = fs::read(&cache_path)
        .map_err(|error| format!("读取 DeepSeek 会话配置失败：{error}"))?;
    let mut cache: Value = serde_json::from_slice(&original)
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
    Ok(Some(PreparedWrite { path: cache_path, bytes, original }))
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
        if !fs::read(&write.path)
            .is_ok_and(|current| current.as_slice() == write.original.as_slice())
        {
            for (temp, _) in &staged {
                let _ = fs::remove_file(temp);
            }
            return Err(format!("DeepSeek 会话在编辑期间已被其他进程修改，已停止保存：{}", write.path.display()));
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
        if !fs::read(&write.path)
            .is_ok_and(|current| current.as_slice() == write.original.as_slice())
        {
            let restore_errors = rollback_writes(&writes, &staged, index);
            return Err(format!("DeepSeek 会话在写入期间已被其他进程修改，已停止保存：{}{restore_errors}", write.path.display()));
        }
        if let Err(error) = fs::rename(&write.path, backup) {
            let restore_errors = rollback_writes(&writes, &staged, index);
            return Err(format!("备份 DeepSeek 原文件失败：{error}{restore_errors}"));
        }
        if let Err(error) = fs::rename(temp, &write.path) {
            let restore_errors = rollback_writes(&writes, &staged, index + 1);
            return Err(format!("保存 DeepSeek 修改失败：{error}{restore_errors}"));
        }
    }
    // An open DSH handle may continue appending to the inode after it has
    // been moved to the backup path. Keep every backup if either side changed
    // while the replacement was being published.
    for (index, write) in writes.iter().enumerate() {
        let backup = &staged[index].1;
        let backup_unchanged = fs::read(backup)
            .is_ok_and(|current| current.as_slice() == write.original.as_slice());
        let replacement_unchanged = fs::read(&write.path)
            .is_ok_and(|current| current.as_slice() == write.bytes.as_slice());
        if !backup_unchanged || !replacement_unchanged {
            return Err(format!(
                "DeepSeek 会话在保存期间被其他进程修改，已保留备份供恢复：{}",
                backup.display()
            ));
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
    let mut preserved = Vec::new();
    for index in (0..backup_count).rev() {
        let original = &writes[index].path;
        let backup = &staged[index].1;
        let backup_bytes = match fs::read(backup) {
            Ok(bytes) => bytes,
            Err(error) => {
                errors.push(format!("无法读取备份 {}：{error}", backup.display()));
                continue;
            }
        };
        let backup_changed = backup_bytes != writes[index].original;
        match fs::read(original) {
            Ok(current) if current != writes[index].bytes => {
                errors.push(format!("文件 {} 已被其他进程修改，未覆盖；备份已保留", original.display()));
                continue;
            }
            Ok(_) => {
                // Move our replacement aside instead of deleting it. An open
                // handle can append between the byte check and this rename;
                // those bytes then remain available in the preserved sibling.
                let kept = match sibling_path(
                    original,
                    &format!("asv-rollback-preserved-{}", uuid::Uuid::new_v4()),
                ) {
                    Ok(path) => path,
                    Err(error) => {
                        errors.push(error);
                        continue;
                    }
                };
                if let Err(error) = fs::rename(original, &kept) {
                    errors.push(format!("无法保留失败后的文件 {}：{error}", original.display()));
                    continue;
                }
                preserved.push(kept);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => {
                errors.push(format!("无法读取失败后的文件 {}：{error}", original.display()));
                continue;
            }
        }
        // hard_link claims an absent destination without replacing a file
        // another process may have created after we moved our copy aside.
        if let Err(error) = fs::hard_link(backup, original) {
            errors.push(format!("无法从备份 {} 恢复：{error}", backup.display()));
            continue;
        }
        if backup_changed {
            // DSH may have appended through an open handle after original was
            // renamed to backup. Both names now refer to those latest bytes;
            // retain the backup for review and keep the session path usable.
            errors.push(format!("备份 {} 已被其他进程修改，已恢复原路径并保留备份", backup.display()));
            continue;
        }
        if let Err(error) = fs::remove_file(backup) {
            errors.push(format!("已恢复文件，但无法清理备份 {}：{error}", backup.display()));
        }
    }
    for (temp, _) in staged {
        let _ = fs::remove_file(temp);
    }
    let kept_note = if preserved.is_empty() {
        String::new()
    } else {
        format!(
            "；失败写入文件已保留：{}",
            preserved.iter().map(|path| path.display().to_string()).collect::<Vec<_>>().join("、")
        )
    };
    if errors.is_empty() {
        kept_note
    } else {
        format!("；自动恢复未完成：{}{kept_note}", errors.join("；"))
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
        let original = fs::read(&candidate)
            .map_err(|error| format!("读取 DeepSeek 原会话失败：{error}"))?;
        let mut rows = decode_rows_from_bytes(&original)?;
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
        writes.push(PreparedWrite { path: candidate, bytes: encode_rows(&rows)?, original });
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
        let original = fs::read(&candidate)
            .map_err(|error| format!("读取 DeepSeek 原会话失败：{error}"))?;
        let mut rows = decode_rows_from_bytes(&original)?;
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
        writes.push(PreparedWrite { path: candidate, bytes: encode_rows(&rows)?, original });
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
    fn encoded_log_has_independent_header_frame_and_legacy_plugin_marker() {
        let rows = [
            serde_json::json!({"type":"session","version":4,"id":"session-1","createdAt":1000,"cwd":"C:\\work"}),
            serde_json::json!({"type":"turn/start","seq":0,"time":1001,"data":{"turn":1}}),
            serde_json::json!({"type":"plugin:pentest-submission","seq":1,"time":1002,"data":{"submissionId":"test"}}),
        ];
        let bytes = encode_rows(&rows).unwrap();
        let first_size = zstd::zstd_safe::find_frame_compressed_size(&bytes).unwrap();
        assert!(first_size < bytes.len(), "event rows require a second frame");
        let header = zstd::stream::decode_all(&bytes[..first_size]).unwrap();
        let header_text = std::str::from_utf8(&header).unwrap();
        assert!(header_text.ends_with('\n'));
        assert_eq!(header_text.lines().count(), 1);
        assert_eq!(serde_json::from_str::<Value>(header_text).unwrap(), rows[0]);
        let body = zstd::stream::decode_all(&bytes[first_size..]).unwrap();
        let events = body
            .split(|byte| *byte == b'\n')
            .filter(|line| !line.is_empty())
            .map(|line| serde_json::from_slice::<Value>(line).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0], rows[1]);
        assert_eq!(events[1].get("ignorable"), Some(&Value::Bool(true)));
    }

    #[test]
    fn refuses_to_remove_a_tool_call_message_without_touching_its_log() {
        let root = std::env::temp_dir().join(format!("dsh-tool-delete-{}", uuid::Uuid::new_v4()));
        let path = root.join("sessions").join("workspace").join("session-1").join(SESSION_FILES[0]);
        write_compressed(&path, &[
            serde_json::json!({"type":"session","version":4,"id":"session-1","cwd":"C:\\work"}),
            serde_json::json!({"type":"assistant/message","seq":0,"data":{"message":{"id":"a1","content":[{"type":"tool-call","id":"call-1","name":"test","arguments":"{}"}]}}}),
            serde_json::json!({"type":"tool/call","seq":1,"data":{"callId":"call-1","name":"test"}}),
        ]);
        let before = fs::read(&path).unwrap();
        assert!(delete_message(&path, "a1").unwrap_err().contains("工具调用"));
        assert_eq!(fs::read(&path).unwrap(), before);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_to_edit_an_assistant_without_a_matching_stream() {
        let root = std::env::temp_dir().join(format!("dsh-unsafe-edit-{}", uuid::Uuid::new_v4()));
        let path = root.join("sessions").join("workspace").join("session-1").join(SESSION_FILES[0]);
        write_compressed(&path, &[
            serde_json::json!({"type":"session","version":4,"id":"session-1","cwd":"C:\\work"}),
            serde_json::json!({"type":"assistant/message","seq":0,"data":{"stream":[{"type":"text-chunks","time0":1,"index":0,"dt":[],"texts":["old"]}],"message":{"id":"a1","content":[{"type":"text","text":"old"}]}}}),
        ]);
        let before = fs::read(&path).unwrap();
        assert!(edit_message(&path, "a1", "new").unwrap_err().contains("流格式无法安全同步"));
        assert_eq!(fs::read(&path).unwrap(), before);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn refuses_to_overwrite_an_external_append() {
        let root = std::env::temp_dir().join(format!("dsh-concurrent-write-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("session.v4.jsonl.zstd");
        fs::write(&path, b"original").unwrap();
        let prepared = PreparedWrite {
            path: path.clone(),
            bytes: b"viewer edit".to_vec(),
            original: b"original".to_vec(),
        };
        fs::write(&path, b"original plus DSH append").unwrap();
        assert!(commit_writes(vec![prepared]).unwrap_err().contains("其他进程修改"));
        assert_eq!(fs::read(&path).unwrap(), b"original plus DSH append");
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rollback_preserves_viewer_copy_and_never_removes_concurrent_content() {
        let root = std::env::temp_dir().join(format!("dsh-rollback-preserve-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("session.v4.jsonl.zstd");
        let backup = sibling_path(&path, "test.bak").unwrap();
        let temp = sibling_path(&path, "test.tmp").unwrap();
        let write = PreparedWrite {
            path: path.clone(),
            bytes: b"viewer edit".to_vec(),
            original: b"original".to_vec(),
        };
        let staged = vec![(temp, backup.clone())];

        fs::write(&path, &write.bytes).unwrap();
        fs::write(&backup, &write.original).unwrap();
        let note = rollback_writes(std::slice::from_ref(&write), &staged, 1);
        assert!(note.contains("失败写入文件已保留"));
        assert_eq!(fs::read(&path).unwrap(), write.original);
        assert!(!backup.exists());
        let kept = fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|file| file.to_string_lossy().contains("asv-rollback-preserved-"))
            .collect::<Vec<_>>();
        assert_eq!(kept.len(), 1);
        assert_eq!(fs::read(&kept[0]).unwrap(), write.bytes);

        fs::write(&path, b"viewer edit plus DSH append").unwrap();
        fs::write(&backup, &write.original).unwrap();
        let error = rollback_writes(std::slice::from_ref(&write), &staged, 1);
        assert!(error.contains("未覆盖"));
        assert_eq!(fs::read(&path).unwrap(), b"viewer edit plus DSH append");
        assert_eq!(fs::read(&backup).unwrap(), write.original);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn rollback_restores_a_backup_appended_after_rename() {
        let root = std::env::temp_dir().join(format!("dsh-rollback-appended-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("session.v4.jsonl.zstd");
        let backup = sibling_path(&path, "test.bak").unwrap();
        let temp = sibling_path(&path, "test.tmp").unwrap();
        let write = PreparedWrite {
            path: path.clone(),
            bytes: b"viewer edit".to_vec(),
            original: b"original".to_vec(),
        };
        let staged = vec![(temp, backup.clone())];
        let appended = b"original plus DSH append";

        // original was renamed to backup, DSH appended through its open
        // handle, and publishing the staged Viewer file failed.
        fs::write(&backup, appended).unwrap();
        let note = rollback_writes(std::slice::from_ref(&write), &staged, 1);
        assert!(note.contains("已恢复原路径并保留备份"));
        assert_eq!(fs::read(&path).unwrap(), appended);
        assert_eq!(fs::read(&backup).unwrap(), appended);

        // The same recovery must preserve a Viewer replacement as well.
        fs::remove_file(&path).unwrap();
        fs::write(&path, &write.bytes).unwrap();
        let note = rollback_writes(std::slice::from_ref(&write), &staged, 1);
        assert!(note.contains("失败写入文件已保留"));
        assert_eq!(fs::read(&path).unwrap(), appended);
        assert_eq!(fs::read(&backup).unwrap(), appended);
        let kept = fs::read_dir(&root)
            .unwrap()
            .filter_map(Result::ok)
            .map(|entry| entry.path())
            .filter(|file| file.to_string_lossy().contains("asv-rollback-preserved-"))
            .collect::<Vec<_>>();
        assert_eq!(kept.len(), 1);
        assert_eq!(fs::read(&kept[0]).unwrap(), write.bytes);
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn edits_and_deletes_both_versions_and_syncs_last_turn_projection() {
        let root = std::env::temp_dir().join(format!("dsh-edit-provider-{}", uuid::Uuid::new_v4()));
        let session_dir = root.join("sessions").join("workspace").join("session-1");
        let v4 = session_dir.join(SESSION_FILES[0]);
        let v3 = session_dir.join(SESSION_FILES[1]);
        let rows = vec![
            serde_json::json!({"type":"session","version":4,"id":"session-1","seq":0,"createdAt":1000,"cwd":"C:\\work"}),
            serde_json::json!({"type":"agent/inbox/spliced","seq":0,"data":{"inserted":[{"id":"u1","content":[{"type":"text","text":"question"}]}]}}),
            serde_json::json!({"type":"user/message","seq":1,"time":2000,"data":{"id":"u1","content":[{"type":"text","text":"question"}]}}),
            serde_json::json!({"type":"assistant/message","seq":2,"time":3000,"data":{"turn":1,"message":{"id":"a1","content":[{"type":"text","text":"first answer"}],"source":{"provider":"deepseek","model":"deepseek-chat"}}}}),
            serde_json::json!({"type":"assistant/message","seq":3,"time":4000,"data":{"turn":1,"stream":[
                {"type":"chunk","time":4000,"chunk":{"type":"block-start","index":0,"blockType":"reasoning"}},
                {"type":"chunk","time":4001,"chunk":{"type":"block-end","index":0,"block":{"type":"reasoning","text":"private"}}},
                {"type":"chunk","time":4002,"chunk":{"type":"block-start","index":1,"blockType":"text"}},
                {"type":"text-chunks","time0":4003,"index":1,"dt":[],"texts":["old answer"]},
                {"type":"chunk","time":4004,"chunk":{"type":"block-end","index":1,"block":{"type":"text","text":"old answer"}}}
            ],"message":{"id":"a2","content":[{"type":"reasoning","text":"private"},{"type":"text","text":"old answer"}],"source":{"provider":"deepseek","model":"deepseek-chat"}}}}),
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
            let rows = decode_rows(path).unwrap();
            assert_eq!(rows[4].pointer("/data/stream/3/texts/0").and_then(Value::as_str), Some("updated answer"));
            assert_eq!(rows[4].pointer("/data/stream/4/chunk/block/text").and_then(Value::as_str), Some("updated answer"));
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
            let rows = decode_rows(path).unwrap();
            assert_eq!(rows[4].pointer("/data/message/content"), Some(&serde_json::json!([])));
            assert_eq!(rows[4].pointer("/data/stream"), Some(&serde_json::json!([])));
            assert!(rows.iter().skip(1).enumerate().all(|(index, row)| {
                row.get("seq").and_then(Value::as_u64) == Some(index as u64)
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
            assert_eq!(rows[1].pointer("/data/inserted/0/content"), Some(&serde_json::json!([])));
            assert_eq!(rows[2].pointer("/data/content"), Some(&serde_json::json!([])));
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
            serde_json::json!({"type":"session","version":4,"id":"session-1","cwd":"C:\\work"}),
            serde_json::json!({"type":"turn/start","seq":0,"data":{"turn":1}}),
            serde_json::json!({"type":"user/message","seq":1,"data":{"id":"u1","content":[{"type":"text","text":"old"}]}}),
            serde_json::json!({"type":"turn/start","seq":2,"data":{"turn":2}}),
            serde_json::json!({"type":"assistant/message","seq":3,"data":{"turn":2,"message":{"id":"a1","content":[{"type":"text","text":"answer"}]}}}),
        ]);
        let cache = root
            .join("storages")
            .join("session_projcache")
            .join("sessions")
            .join("session-1.json");
        fs::create_dir_all(cache.parent().unwrap()).unwrap();
        fs::write(&cache, serde_json::json!({
            "record":{"rows":{"turnOutline":{"val":{"turns":[
                {"turn":1,"seq":0,"prompt":"old","response":""},
                {"turn":2,"seq":2,"prompt":"","response":"answer"}
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
