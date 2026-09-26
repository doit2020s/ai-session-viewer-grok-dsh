use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use parking_lot::Mutex;
use serde_json::Value;

use crate::models::message::{
    DisplayContentBlock, DisplayMessage, PaginatedMessages, RangeMessages,
};
use crate::models::project::ProjectEntry;
use crate::models::session::{SessionIndexEntry, SessionStatus};
use crate::state::file_modified_key;

const CHAT_HISTORY_FILE: &str = "chat_history.jsonl";
const UNROOTED_PROJECT: &str = "<grok-unrooted>";
// v2 includes Grok's client-state customName values in the displayed title.
const DISK_CACHE_VERSION: u32 = 2;

#[derive(serde::Serialize, serde::Deserialize, Clone, Default)]
#[serde(rename_all = "camelCase")]
struct GrokDiskCache {
    version: u32,
    sessions_by_dir: HashMap<String, CachedGrokSession>,
}

#[derive(serde::Serialize, serde::Deserialize, Clone)]
#[serde(rename_all = "camelCase")]
struct CachedGrokSession {
    summary_modified_key: u64,
    history_modified_key: u64,
    entry: SessionIndexEntry,
}

fn sessions_cache() -> &'static Mutex<Option<GrokDiskCache>> {
    static CACHE: OnceLock<Mutex<Option<GrokDiskCache>>> = OnceLock::new();
    CACHE.get_or_init(|| Mutex::new(None))
}

fn disk_cache_path() -> Option<PathBuf> {
    let dir = dirs::config_dir()?.join("ai-session-viewer");
    let _ = fs::create_dir_all(&dir);
    Some(dir.join("grok-list-cache.json"))
}

fn read_disk_cache() -> GrokDiskCache {
    let Some(path) = disk_cache_path() else {
        return GrokDiskCache::default();
    };
    let Ok(content) = fs::read_to_string(path) else {
        return GrokDiskCache::default();
    };
    let Ok(cache) = serde_json::from_str::<GrokDiskCache>(&content) else {
        return GrokDiskCache::default();
    };
    if cache.version == DISK_CACHE_VERSION {
        cache
    } else {
        GrokDiskCache::default()
    }
}

fn save_disk_cache(cache: &GrokDiskCache) {
    let Some(path) = disk_cache_path() else {
        return;
    };
    let Ok(json) = serde_json::to_string(cache) else {
        return;
    };
    let tmp_path = path.with_extension("json.tmp");
    if fs::write(&tmp_path, json).is_err() {
        return;
    }
    if fs::rename(&tmp_path, &path).is_err() && fs::copy(&tmp_path, &path).is_ok() {
        let _ = fs::remove_file(tmp_path);
    }
}

pub struct SessionMeta {
    pub id: String,
    pub cwd: Option<String>,
}

pub fn get_sessions_dir() -> Option<PathBuf> {
    std::env::var_os("GROK_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".grok")))
        .map(|home| home.join("sessions"))
}

/// Grok's UI renames (including fork names) live outside the session folder.
/// They are stored in ~/.grok/client-state/session-meta.json and are not
/// copied into summary.json. Read the value by session id so forked sessions
/// do not inherit the parent's generated title in viewers.
fn custom_name_for(session_id: &str) -> Option<String> {
    let sessions = get_sessions_dir()?;
    let path = sessions.parent()?.join("client-state").join("session-meta.json");
    let value: Value = serde_json::from_str(&fs::read_to_string(path).ok()?).ok()?;
    let meta = value.get(session_id)?;
    if meta.get("provider").and_then(Value::as_str) != Some("grok") {
        return None;
    }
    meta.get("customName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|name| !name.is_empty())
        .map(ToString::to_string)
}

fn session_dirs() -> Vec<PathBuf> {
    let Some(root) = get_sessions_dir() else {
        return Vec::new();
    };
    let Ok(projects) = fs::read_dir(root) else {
        return Vec::new();
    };

    projects
        .flatten()
        .filter_map(|project| fs::read_dir(project.path()).ok())
        .flat_map(|sessions| sessions.flatten().map(|session| session.path()))
        .filter(|path| {
            path.join("summary.json").is_file() && path.join(CHAT_HISTORY_FILE).is_file()
        })
        .collect()
}

pub fn extract_session_meta(chat_history_path: &Path) -> Option<SessionMeta> {
    let summary_path = chat_history_path.parent()?.join("summary.json");
    let summary: Value = serde_json::from_str(&fs::read_to_string(summary_path).ok()?).ok()?;
    let info = summary.get("info")?;
    Some(SessionMeta {
        id: info.get("id")?.as_str()?.to_string(),
        cwd: info
            .get("cwd")
            .and_then(Value::as_str)
            .map(ToString::to_string),
    })
}

fn text_content(value: &Value) -> Option<String> {
    if let Some(text) = value.as_str() {
        return (!text.trim().is_empty()).then(|| text.to_string());
    }

    let text = value
        .as_array()?
        .iter()
        .filter_map(|block| block.get("text").and_then(Value::as_str))
        .filter(|text| !text.trim().is_empty())
        .collect::<Vec<_>>()
        .join("\n");
    (!text.is_empty()).then_some(text)
}

fn display_message_from_row(row: &Value) -> Option<DisplayMessage> {
    let row_type = row.get("type")?.as_str()?;
    let (role, model, content) = match row_type {
        "user"
            if row
                .get("synthetic_reason")
                .and_then(Value::as_str)
                .is_none() =>
        {
            (
                "user",
                None,
                DisplayContentBlock::Text {
                    text: text_content(row.get("content")?)?,
                },
            )
        }
        "assistant" => (
            "assistant",
            row.get("model_id")
                .and_then(Value::as_str)
                .map(ToString::to_string),
            DisplayContentBlock::Text {
                text: text_content(row.get("content")?)?,
            },
        ),
        "reasoning" => (
            "assistant",
            None,
            DisplayContentBlock::Reasoning {
                text: text_content(row.get("summary")?)?,
            },
        ),
        _ => return None,
    };

    Some(DisplayMessage {
        uuid: None,
        parent_uuid: None,
        role: role.to_string(),
        timestamp: None,
        model,
        content: vec![content],
    })
}

pub fn parse_all_messages(path: &Path) -> Result<Vec<DisplayMessage>, String> {
    let file =
        fs::File::open(path).map_err(|error| format!("Failed to open Grok session: {error}"))?;

    Ok(BufReader::new(file)
        .lines()
        .enumerate()
        .map_while(|(index, line)| line.ok().map(|line| (index, line)))
        .filter_map(|(index, line)| {
            serde_json::from_str::<Value>(&line)
                .ok()
                .map(|row| (index, row))
        })
        .filter_map(|(index, row)| {
            let mut message = display_message_from_row(&row)?;
            message.uuid = Some(crate::fork::line_message_id(index, &row));
            Some(message)
        })
        .collect())
}

fn replace_exact_text(value: &mut Value, old: &str, new: &str) -> usize {
    match value {
        Value::String(current) if current == old => {
            *current = new.to_string();
            1
        }
        Value::Array(items) => items
            .iter_mut()
            .map(|item| replace_exact_text(item, old, new))
            .sum(),
        Value::Object(map) => map
            .values_mut()
            .map(|item| replace_exact_text(item, old, new))
            .sum(),
        _ => 0,
    }
}

fn sync_updates_log(path: &Path, old_text: &str, new_text: &str) -> Result<usize, String> {
    let updates_path = path
        .parent()
        .ok_or_else(|| "Grok 会话目录不存在".to_string())?
        .join("updates.jsonl");
    if !updates_path.is_file() {
        return Ok(0);
    }

    // User entries in chat_history include the display wrapper, while ACP's
    // authoritative update stores only the inner query text.
    let update_old = old_text
        .strip_prefix("<user_query>\n")
        .and_then(|value| value.strip_suffix("\n</user_query>"))
        .unwrap_or(old_text);
    let update_new = if update_old != old_text {
        new_text
            .strip_prefix("<user_query>\n")
            .and_then(|value| value.strip_suffix("\n</user_query>"))
            .unwrap_or(new_text)
    } else {
        new_text
    };

    let content = fs::read_to_string(&updates_path)
        .map_err(|error| format!("读取 Grok 恢复日志失败：{error}"))?;
    let mut replacements = 0usize;
    let mut rendered = String::with_capacity(content.len());
    for line in content.lines() {
        let mut row = serde_json::from_str::<Value>(line)
            .map_err(|error| format!("Grok 恢复日志包含无效 JSON：{error}"))?;
        replacements += replace_exact_text(&mut row, update_old, update_new);
        rendered.push_str(
            &serde_json::to_string(&row)
                .map_err(|error| format!("序列化 Grok 恢复日志失败：{error}"))?,
        );
        rendered.push('\n');
    }
    if !content.ends_with('\n') {
        rendered.pop();
    }
    if replacements == 0 {
        return Ok(0);
    }
    let tmp_path = updates_path.with_extension("jsonl.codex-edit.tmp");
    fs::write(&tmp_path, rendered).map_err(|error| format!("写入 Grok 恢复日志临时文件失败：{error}"))?;
    fs::rename(&tmp_path, &updates_path)
        .map_err(|error| format!("保存 Grok 恢复日志失败：{error}"))?;
    Ok(replacements)
}

fn normalized_update_text(text: &str) -> &str {
    text.strip_prefix("<user_query>\n")
        .and_then(|value| value.strip_suffix("\n</user_query>"))
        .unwrap_or(text)
}

fn value_contains_exact_text(value: &Value, expected: &str) -> bool {
    match value {
        Value::String(current) => current == expected,
        Value::Array(items) => items
            .iter()
            .any(|item| value_contains_exact_text(item, expected)),
        Value::Object(map) => map
            .values()
            .any(|item| value_contains_exact_text(item, expected)),
        _ => false,
    }
}

fn remove_update_event(
    path: &Path,
    row_type: &str,
    old_text: &str,
    occurrence: usize,
) -> Result<usize, String> {
    let updates_path = path
        .parent()
        .ok_or_else(|| "Grok 会话目录不存在".to_string())?
        .join("updates.jsonl");
    if !updates_path.is_file() {
        return Ok(0);
    }

    let expected_update_type = match row_type {
        "user" => "user_message_chunk",
        "assistant" => "agent_message_chunk",
        "reasoning" => "agent_thought_chunk",
        _ => return Err("该 Grok 记录类型不支持删除".to_string()),
    };
    let expected_text = normalized_update_text(old_text);
    let content = fs::read_to_string(&updates_path)
        .map_err(|error| format!("读取 Grok 恢复日志失败：{error}"))?;
    let mut matched = 0usize;
    let mut removed = 0usize;
    let mut rendered = String::with_capacity(content.len());

    for line in content.lines() {
        let row = serde_json::from_str::<Value>(line)
            .map_err(|error| format!("Grok 恢复日志包含无效 JSON：{error}"))?;
        let update = row.pointer("/params/update");
        let is_match = update
            .and_then(|value| value.get("sessionUpdate"))
            .and_then(Value::as_str)
            == Some(expected_update_type)
            && update.is_some_and(|value| value_contains_exact_text(value, expected_text));

        if is_match {
            if matched == occurrence {
                removed += 1;
                matched += 1;
                continue;
            }
            matched += 1;
        }
        rendered.push_str(line);
        rendered.push('\n');
    }
    if !content.ends_with('\n') {
        rendered.pop();
    }
    if removed == 0 {
        return Ok(0);
    }

    let tmp_path = updates_path.with_extension("jsonl.codex-delete.tmp");
    fs::write(&tmp_path, rendered)
        .map_err(|error| format!("写入 Grok 恢复日志临时文件失败：{error}"))?;
    fs::rename(&tmp_path, &updates_path)
        .map_err(|error| format!("保存 Grok 恢复日志失败：{error}"))?;
    Ok(removed)
}

fn refresh_tail_summary(path: &Path, lines: &[String]) -> Result<(), String> {
    let Some(summary_path) = path.parent().map(|parent| parent.join("summary.json")) else {
        return Ok(());
    };
    if !summary_path.is_file() {
        return Ok(());
    }
    let summary_text = fs::read_to_string(&summary_path)
        .map_err(|error| format!("读取 summary.json 失败：{error}"))?;
    let mut summary = serde_json::from_str::<Value>(&summary_text)
        .map_err(|error| format!("解析 summary.json 失败：{error}"))?;
    let tail = lines.iter().rev().find_map(|line| {
        let row = serde_json::from_str::<Value>(line).ok()?;
        match row.get("type").and_then(Value::as_str) {
            Some("assistant") => text_content(row.get("content")?),
            Some("user") if row.get("synthetic_reason").is_none() => {
                text_content(row.get("content")?)
            }
            _ => None,
        }
    });
    let compact: String = tail
        .unwrap_or_default()
        .trim()
        .chars()
        .take(500)
        .collect();
    summary["last_turn_summary"] = Value::String(compact.clone());
    summary["last_recap"] = Value::String(compact);
    summary["updated_at"] = Value::String(chrono::Utc::now().to_rfc3339());
    let rendered = serde_json::to_string_pretty(&summary)
        .map_err(|error| format!("序列化 summary.json 失败：{error}"))?;
    let tmp_path = summary_path.with_extension("json.codex-delete.tmp");
    fs::write(&tmp_path, rendered)
        .map_err(|error| format!("写入 summary.json 临时文件失败：{error}"))?;
    fs::rename(&tmp_path, &summary_path)
        .map_err(|error| format!("保存 summary.json 失败：{error}"))?;
    Ok(())
}

pub fn delete_message(path: &Path, message_id: &str) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|error| format!("读取 Grok 会话失败：{error}"))?;
    let mut lines: Vec<String> = content.lines().map(ToString::to_string).collect();
    let mut target = None;

    for (index, line) in lines.iter().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let row = serde_json::from_str::<Value>(line)
            .map_err(|error| format!("会话第 {} 行不是有效 JSON：{error}", index + 1))?;
        let row_type = row.get("type").and_then(Value::as_str).unwrap_or("");
        if row_type != "user" && row_type != "assistant" && row_type != "reasoning" {
            continue;
        }
        let value = if row_type == "reasoning" {
            row.get("summary")
        } else {
            row.get("content")
        };
        let Some(row_text) = value.and_then(text_content) else {
            continue;
        };
        if crate::fork::line_message_id(index, &row) == message_id {
            target = Some((index, row_type.to_string(), row_text));
            break;
        }
    }

    let Some((target_index, row_type, old_text)) = target else {
        return Err("找不到要删除的消息，可能会话已被其他进程修改".to_string());
    };
    let occurrence = lines[..target_index]
        .iter()
        .filter_map(|line| serde_json::from_str::<Value>(line).ok())
        .filter(|row| row.get("type").and_then(Value::as_str) == Some(row_type.as_str()))
        .filter_map(|row| {
            let value = if row_type == "reasoning" {
                row.get("summary")
            } else {
                row.get("content")
            }?;
            text_content(value)
        })
        .filter(|text| normalized_update_text(text) == normalized_update_text(&old_text))
        .count();
    let updates_path = path.parent().map(|parent| parent.join("updates.jsonl"));
    let has_updates_log = updates_path.as_ref().is_some_and(|candidate| candidate.is_file());
    let removed_updates = remove_update_event(path, &row_type, &old_text, occurrence)?;
    if has_updates_log && removed_updates == 0 {
        return Err("在 updates.jsonl 中找不到对应上下文，未删除以避免终端与界面不一致".to_string());
    }

    lines.remove(target_index);
    let mut output = lines.join("\n");
    if content.ends_with('\n') {
        output.push('\n');
    }
    let tmp_path = path.with_extension("jsonl.codex-delete.tmp");
    fs::write(&tmp_path, output).map_err(|error| format!("写入临时会话失败：{error}"))?;
    fs::rename(&tmp_path, path).map_err(|error| format!("保存 Grok 会话失败：{error}"))?;
    refresh_tail_summary(path, &lines)
}

/// Edit the visible text of one Grok JSONL record in place. Grok identifies
/// records by the stable line/digest id exposed to the frontend. When the
/// edited record is the last visible turn, keep summary.json in sync so the
/// next CLI resume sees the same tail context.
pub fn edit_message(path: &Path, message_id: &str, text: &str) -> Result<(), String> {
    let content = fs::read_to_string(path).map_err(|error| format!("读取 Grok 会话失败：{error}"))?;
    let mut lines: Vec<String> = content.lines().map(ToString::to_string).collect();
    let mut target_line = None;
    let mut target_type = None;
    let mut old_text = None;

    for (index, line) in lines.iter_mut().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let Ok(mut row) = serde_json::from_str::<Value>(line) else {
            return Err(format!("会话第 {} 行不是有效 JSON，已停止保存", index + 1));
        };
        if crate::fork::line_message_id(index, &row) != message_id {
            continue;
        }
        let row_type = row
            .get("type")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        if row_type != "user" && row_type != "assistant" {
            return Err("只能编辑用户消息或助手消息".to_string());
        }
        let Some(content_value) = row.get_mut("content") else {
            return Err("该消息没有可编辑的 content 字段".to_string());
        };
        let previous_text = text_content(content_value).ok_or_else(|| "该消息没有可编辑的文本内容".to_string())?;
        match content_value {
            Value::String(value) => *value = text.to_string(),
            Value::Array(blocks) => {
                let Some(block) = blocks.iter_mut().find(|block| {
                    block.get("type").and_then(Value::as_str) == Some("text")
                }) else {
                    return Err("该消息没有可编辑的文本块".to_string());
                };
                block["text"] = Value::String(text.to_string());
            }
            _ => return Err("该消息的 content 格式不支持编辑".to_string()),
        }
        *line = serde_json::to_string(&row).map_err(|error| format!("序列化消息失败：{error}"))?;
        target_line = Some(index);
        target_type = Some(row_type);
        old_text = Some(previous_text);
        break;
    }

    let Some(target_line) = target_line else {
        return Err("找不到要编辑的消息，可能会话已被其他进程修改".to_string());
    };

    let last_conversation_line = lines
        .iter()
        .enumerate()
        .rev()
        .find_map(|(index, line)| {
            let row = serde_json::from_str::<Value>(line).ok()?;
            (row.get("type").and_then(Value::as_str) == Some("user")).then_some(index)
        });

    let mut output = lines.join("\n");
    if content.ends_with('\n') {
        output.push('\n');
    }
    let tmp_path = path.with_extension("jsonl.codex-edit.tmp");
    fs::write(&tmp_path, output).map_err(|error| format!("写入临时会话失败：{error}"))?;
    fs::rename(&tmp_path, path).map_err(|error| format!("保存 Grok 会话失败：{error}"))?;

    if let Some(previous_text) = old_text.as_deref() {
        let updates_path = path.parent().map(|parent| parent.join("updates.jsonl"));
        let has_updates_log = updates_path.as_ref().is_some_and(|candidate| candidate.is_file());
        let replacements = sync_updates_log(path, previous_text, text)?;
        if has_updates_log && replacements == 0 {
            return Err("展示文件已保存，但在 updates.jsonl 中找不到对应消息，已停止以避免终端继续使用旧上下文".to_string());
        }
    }

    if last_conversation_line.is_some_and(|last| target_line >= last) {
        let summary_path = path.parent().map(|parent| parent.join("summary.json"));
        if let Some(summary_path) = summary_path {
            if let Ok(summary_text) = fs::read_to_string(&summary_path) {
                if let Ok(mut summary) = serde_json::from_str::<Value>(&summary_text) {
                    let summary_text = text.trim();
                    let compact: String = summary_text.chars().take(500).collect();
                    summary["last_turn_summary"] = Value::String(compact.clone());
                    summary["last_recap"] = Value::String(compact);
                    summary["updated_at"] = Value::String(chrono::Utc::now().to_rfc3339());
                    let summary_tmp = summary_path.with_extension("json.codex-edit.tmp");
                    let rendered = serde_json::to_string_pretty(&summary)
                        .map_err(|error| format!("序列化 summary.json 失败：{error}"))?;
                    fs::write(&summary_tmp, rendered)
                        .map_err(|error| format!("写入 summary.json 临时文件失败：{error}"))?;
                    fs::rename(&summary_tmp, &summary_path)
                        .map_err(|error| format!("保存 summary.json 失败：{error}"))?;
                }
            }
        }
    }

    let _ = target_type;
    Ok(())
}

fn text_message_count(messages: &[DisplayMessage]) -> u32 {
    messages
        .iter()
        .filter(|message| {
            message
                .content
                .iter()
                .any(|block| matches!(block, DisplayContentBlock::Text { .. }))
        })
        .count() as u32
}

pub fn count_messages(path: &Path) -> u32 {
    parse_all_messages(path)
        .map(|messages| text_message_count(&messages))
        .unwrap_or(0)
}

fn session_entry(dir: &Path) -> Option<SessionIndexEntry> {
    let summary: Value =
        serde_json::from_str(&fs::read_to_string(dir.join("summary.json")).ok()?).ok()?;
    let info = summary.get("info")?;
    let session_id = info.get("id")?.as_str()?.to_string();
    let cwd = info
        .get("cwd")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    let file_path = dir.join(CHAT_HISTORY_FILE);

    let messages = parse_all_messages(&file_path).unwrap_or_default();
    let message_count = text_message_count(&messages);
    let first_prompt = messages.iter().find_map(|message| {
        if message.role != "user" {
            return None;
        }
        message.content.iter().find_map(|block| match block {
            DisplayContentBlock::Text { text } => Some(text.chars().take(200).collect()),
            _ => None,
        })
    });

    let generated_title = summary
        .get("session_summary")
        .and_then(Value::as_str)
        .filter(|title| !title.trim().is_empty())
        .map(ToString::to_string);
    let display_title = custom_name_for(&session_id).or(generated_title);

    Some(SessionIndexEntry {
        source: "grok".to_string(),
        session_id,
        file_path: file_path.to_string_lossy().into_owned(),
        first_prompt,
        thread_name: display_title,
        message_count,
        created: summary
            .get("created_at")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        modified: summary
            .get("updated_at")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        git_branch: None,
        project_path: cwd.clone(),
        is_sidechain: None,
        cwd,
        model_provider: summary
            .get("current_model_id")
            .and_then(Value::as_str)
            .map(ToString::to_string),
        cli_version: None,
        alias: None,
        tags: None,
        status: if message_count == 0 {
            SessionStatus::Empty
        } else {
            SessionStatus::Valid
        },
    })
}

fn session_modified_keys(dir: &Path) -> Option<(u64, u64)> {
    Some((
        file_modified_key(&dir.join("summary.json")).ok()?,
        file_modified_key(&dir.join(CHAT_HISTORY_FILE)).ok()?,
    ))
}

fn cached_session_for_dir(dir: &Path) -> Option<CachedGrokSession> {
    let entry = session_entry(dir)?;
    let (summary_modified_key, history_modified_key) = session_modified_keys(dir)?;
    Some(CachedGrokSession {
        summary_modified_key,
        history_modified_key,
        entry,
    })
}

fn reconcile_sessions_cache_with<F>(
    mut cache: GrokDiskCache,
    dirs: Vec<PathBuf>,
    mut scan: F,
) -> (GrokDiskCache, bool)
where
    F: FnMut(&Path) -> Option<CachedGrokSession>,
{
    let mut cached_by_dir = std::mem::take(&mut cache.sessions_by_dir);
    let mut sessions_by_dir = HashMap::with_capacity(dirs.len());
    let mut changed = cache.version != DISK_CACHE_VERSION;
    cache.version = DISK_CACHE_VERSION;

    for dir in dirs {
        let key = dir.to_string_lossy().into_owned();
        let modified_keys = session_modified_keys(&dir);
        match (cached_by_dir.remove(&key), modified_keys) {
            (Some(cached), Some((summary_key, history_key)))
                if cached.summary_modified_key == summary_key
                    && cached.history_modified_key == history_key =>
            {
                sessions_by_dir.insert(key, cached);
            }
            _ => {
                changed = true;
                if let Some(session) = scan(&dir) {
                    sessions_by_dir.insert(key, session);
                }
            }
        }
    }

    if !cached_by_dir.is_empty() {
        changed = true;
    }
    cache.sessions_by_dir = sessions_by_dir;
    (cache, changed)
}

fn reconcile_sessions_cache(cache: GrokDiskCache, dirs: Vec<PathBuf>) -> (GrokDiskCache, bool) {
    reconcile_sessions_cache_with(cache, dirs, cached_session_for_dir)
}

fn sessions_from_cache(cache: &GrokDiskCache) -> Vec<SessionIndexEntry> {
    cache
        .sessions_by_dir
        .values()
        .map(|cached| cached.entry.clone())
        .collect()
}

fn load_all_sessions() -> Vec<SessionIndexEntry> {
    let mut state = sessions_cache().lock();
    let base = state.take().unwrap_or_else(read_disk_cache);
    let (cache, changed) = reconcile_sessions_cache(base, session_dirs());
    if changed {
        save_disk_cache(&cache);
    }
    let sessions = sessions_from_cache(&cache);
    *state = Some(cache);
    sessions
}

fn rebuild_all_sessions() -> Vec<SessionIndexEntry> {
    let (cache, _) = reconcile_sessions_cache(GrokDiskCache::default(), session_dirs());
    save_disk_cache(&cache);
    let sessions = sessions_from_cache(&cache);
    *sessions_cache().lock() = Some(cache);
    sessions
}

fn projects_from_sessions(sessions: Vec<SessionIndexEntry>) -> Vec<ProjectEntry> {
    let mut grouped: BTreeMap<String, Vec<SessionIndexEntry>> = BTreeMap::new();
    for entry in sessions {
        grouped
            .entry(
                entry
                    .cwd
                    .clone()
                    .unwrap_or_else(|| UNROOTED_PROJECT.to_string()),
            )
            .or_default()
            .push(entry);
    }

    grouped
        .into_iter()
        .map(|(id, sessions)| {
            let short_name = Path::new(&id)
                .file_name()
                .and_then(|name| name.to_str())
                .filter(|name| !name.is_empty())
                .unwrap_or(&id)
                .to_string();
            ProjectEntry {
                source: "grok".to_string(),
                id: id.clone(),
                display_path: id.clone(),
                short_name,
                session_count: sessions
                    .iter()
                    .filter(|session| session.status == SessionStatus::Valid)
                    .count(),
                last_modified: sessions
                    .iter()
                    .filter_map(|session| session.modified.clone())
                    .max(),
                model_provider: None,
                alias: None,
                path_exists: Path::new(&id).exists(),
                is_virtual: id == UNROOTED_PROJECT,
            }
        })
        .collect()
}

pub fn get_projects() -> Result<Vec<ProjectEntry>, String> {
    Ok(projects_from_sessions(load_all_sessions()))
}

pub fn refresh_projects_cache() -> Result<Vec<ProjectEntry>, String> {
    get_projects()
}

pub fn rebuild_projects_cache() -> Result<Vec<ProjectEntry>, String> {
    Ok(projects_from_sessions(rebuild_all_sessions()))
}

pub fn invalidate_sessions_cache() {
    *sessions_cache().lock() = None;
}

fn changed_session_dirs(paths: &[PathBuf]) -> HashSet<PathBuf> {
    paths
        .iter()
        .filter_map(|path| {
            let file_name = path.file_name().and_then(|name| name.to_str());
            if file_name == Some("summary.json") || file_name == Some(CHAT_HISTORY_FILE) {
                path.parent().map(Path::to_path_buf)
            } else if path.join("summary.json").exists() || path.join(CHAT_HISTORY_FILE).exists() {
                Some(path.clone())
            } else {
                None
            }
        })
        .collect()
}

/// Update only changed Grok session directories when the shared snapshot is
/// already warm. A cold snapshot is reconciled lazily against the disk cache.
pub fn invalidate_paths(paths: &[PathBuf]) {
    let changed_dirs = changed_session_dirs(paths);
    if changed_dirs.is_empty() {
        return;
    }

    let snapshot = {
        let mut state = sessions_cache().lock();
        let Some(cache) = state.as_mut() else {
            return;
        };
        for dir in changed_dirs {
            let key = dir.to_string_lossy().into_owned();
            cache.sessions_by_dir.remove(&key);
            if let Some(session) = cached_session_for_dir(&dir) {
                cache.sessions_by_dir.insert(key, session);
            }
        }
        cache.clone()
    };
    save_disk_cache(&snapshot);
}

pub fn get_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    let mut sessions: Vec<_> = load_all_sessions()
        .into_iter()
        .filter(|entry| entry.cwd.as_deref().unwrap_or(UNROOTED_PROJECT) == project_id)
        .collect();
    sessions.sort_by(|left, right| right.modified.cmp(&left.modified));
    Ok(sessions)
}

pub fn refresh_sessions_cache(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    // Project refresh and file watchers rebuild/invalidate the shared snapshot.
    // Reuse it here so one frontend refresh cycle never parses every history twice.
    get_sessions(project_id)
}

pub fn get_invalid_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    Ok(get_sessions(project_id)?
        .into_iter()
        .filter(|session| session.status != SessionStatus::Valid)
        .collect())
}

pub fn delete_project(project_id: &str) -> Result<super::claude::DeleteResult, String> {
    if project_id.is_empty() {
        return Err("Invalid project id".to_string());
    }

    let sessions = get_sessions(project_id)?;
    let project_name = Path::new(project_id)
        .file_name()
        .and_then(|name| name.to_str())
        .filter(|name| !name.is_empty())
        .unwrap_or(project_id)
        .to_string();
    let mut sessions_deleted = 0;

    for session in &sessions {
        let path = Path::new(&session.file_path);
        if !path.exists() {
            continue;
        }
        let Some(session_dir) = path.parent() else {
            continue;
        };
        if crate::recyclebin::move_to_recyclebin(
            session_dir,
            "project",
            "ManualDelete",
            "grok",
            project_id,
            None,
            Some(project_name.clone()),
        )
        .is_ok()
        {
            sessions_deleted += 1;
            let _ = crate::metadata::remove_session_meta("grok", project_id, &session.session_id);
        }
    }

    invalidate_sessions_cache();

    Ok(super::claude::DeleteResult {
        sessions_deleted,
        config_cleaned: false,
        bookmarks_removed: 0,
    })
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

pub fn parse_messages_range(
    path: &Path,
    start: usize,
    end: usize,
) -> Result<RangeMessages, String> {
    let all = parse_all_messages(path)?;
    let total = all.len();
    let start = start.min(total);
    let end = end.min(total).max(start);
    Ok(RangeMessages {
        messages: all[start..end].to_vec(),
        total,
        start,
        end,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn write_session(dir: &Path, id: &str, cwd: &str, text: Option<&str>) {
        fs::create_dir_all(dir).unwrap();
        let summary = serde_json::json!({
            "info": { "id": id, "cwd": cwd },
            "created_at": "2026-08-27T00:00:00Z",
            "updated_at": "2026-08-27T00:00:00Z"
        });
        fs::write(dir.join("summary.json"), summary.to_string()).unwrap();
        let history = text
            .map(|text| serde_json::json!({ "type": "user", "content": text }).to_string())
            .unwrap_or_default();
        fs::write(dir.join(CHAT_HISTORY_FILE), history).unwrap();
    }

    #[test]
    fn parses_visible_history_and_paginates_from_end() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-{}-{unique}.jsonl",
            std::process::id()
        ));
        let rows = [
            serde_json::json!({"type":"system","content":"hidden"}),
            serde_json::json!({"type":"user","content":[{"type":"text","text":"hello"}]}),
            serde_json::json!({"type":"user","content":[{"type":"text","text":"hidden"}],"synthetic_reason":"system_reminder"}),
            serde_json::json!({"type":"reasoning","summary":[{"type":"summary_text","text":"thinking"}]}),
            serde_json::json!({"type":"assistant","content":"world","model_id":"grok-test"}),
        ];
        fs::write(
            &path,
            rows.iter()
                .map(Value::to_string)
                .collect::<Vec<_>>()
                .join("\n"),
        )
        .unwrap();

        assert_eq!(count_messages(&path), 2);
        let page = parse_session_messages(&path, 0, 2, true).unwrap();
        assert_eq!(page.total, 3);
        assert_eq!(page.messages.len(), 2);
        assert!(page.has_more);

        fs::remove_file(path).unwrap();
    }

    #[test]
    fn edits_last_message_and_updates_summary() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-edit-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let history = dir.join(CHAT_HISTORY_FILE);
        fs::write(
            &history,
            "{\"type\":\"user\",\"content\":\"old prompt\"}\n{\"type\":\"assistant\",\"content\":\"old answer\"}\n",
        )
        .unwrap();
        fs::write(
            dir.join("summary.json"),
            serde_json::json!({"last_turn_summary":"old answer","last_recap":"old answer"}).to_string(),
        )
        .unwrap();
        let rows: Vec<Value> = fs::read_to_string(&history)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let id = crate::fork::line_message_id(1, &rows[1]);
        edit_message(&history, &id, "new answer").unwrap();
        let saved = fs::read_to_string(&history).unwrap();
        assert!(saved.contains("new answer"));
        let summary: Value = serde_json::from_str(&fs::read_to_string(dir.join("summary.json")).unwrap()).unwrap();
        assert_eq!(summary["last_turn_summary"], "new answer");
        assert_eq!(summary["last_recap"], "new answer");
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn deletes_reasoning_from_history_and_authoritative_updates() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-delete-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).unwrap();
        let history = dir.join(CHAT_HISTORY_FILE);
        fs::write(
            &history,
            "{\"type\":\"user\",\"content\":\"question\"}\n{\"type\":\"reasoning\",\"summary\":[{\"type\":\"summary_text\",\"text\":\"private thought\"}]}\n{\"type\":\"assistant\",\"content\":\"answer\"}\n",
        )
        .unwrap();
        fs::write(
            dir.join("updates.jsonl"),
            "{\"params\":{\"update\":{\"sessionUpdate\":\"user_message_chunk\",\"content\":{\"type\":\"text\",\"text\":\"question\"}}}}\n{\"params\":{\"update\":{\"sessionUpdate\":\"agent_thought_chunk\",\"content\":{\"type\":\"text\",\"text\":\"private thought\"}}}}\n{\"params\":{\"update\":{\"sessionUpdate\":\"agent_message_chunk\",\"content\":{\"type\":\"text\",\"text\":\"answer\"}}}}\n",
        )
        .unwrap();
        fs::write(
            dir.join("summary.json"),
            serde_json::json!({"last_turn_summary":"answer","last_recap":"answer"}).to_string(),
        )
        .unwrap();

        let rows: Vec<Value> = fs::read_to_string(&history)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        let id = crate::fork::line_message_id(1, &rows[1]);
        delete_message(&history, &id).unwrap();

        assert!(!fs::read_to_string(&history).unwrap().contains("private thought"));
        let updates = fs::read_to_string(dir.join("updates.jsonl")).unwrap();
        assert!(!updates.contains("private thought"));
        assert!(updates.contains("answer"));
        fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn project_count_excludes_empty_sessions() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-project-count-{}-{unique}",
            std::process::id()
        ));
        let empty = root.join("empty");
        let valid = root.join("valid");
        let cwd = r"C:\projects\grok-count-test";
        write_session(&empty, "empty", cwd, None);
        write_session(&valid, "valid", cwd, Some("有效会话"));

        let projects = projects_from_sessions(vec![
            session_entry(&empty).unwrap(),
            session_entry(&valid).unwrap(),
        ]);

        assert_eq!(projects.len(), 1);
        assert_eq!(projects[0].session_count, 1);
        fs::remove_dir_all(root).unwrap();
    }

    #[test]
    fn reconcile_reuses_unchanged_sessions_and_rescans_only_changes() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-cache-{}-{unique}",
            std::process::id()
        ));
        let first = root.join("first");
        let second = root.join("second");
        write_session(&first, "first", r"C:\projects\first", Some("one"));
        write_session(&second, "second", r"C:\projects\second", Some("two"));
        let dirs = vec![first.clone(), second.clone()];

        let (cache, changed) = reconcile_sessions_cache(GrokDiskCache::default(), dirs.clone());
        assert!(changed);
        assert_eq!(cache.sessions_by_dir.len(), 2);

        let mut scans = 0;
        let (cache, changed) = reconcile_sessions_cache_with(cache, dirs.clone(), |dir| {
            scans += 1;
            cached_session_for_dir(dir)
        });
        assert!(!changed);
        assert_eq!(scans, 0);

        fs::write(
            first.join(CHAT_HISTORY_FILE),
            serde_json::json!({ "type": "user", "content": "changed" }).to_string(),
        )
        .unwrap();
        filetime::set_file_mtime(
            first.join(CHAT_HISTORY_FILE),
            filetime::FileTime::from_unix_time(2_000_000_000, 0),
        )
        .unwrap();

        scans = 0;
        let (cache, changed) = reconcile_sessions_cache_with(cache, dirs, |dir| {
            scans += 1;
            cached_session_for_dir(dir)
        });
        assert!(changed);
        assert_eq!(scans, 1);

        fs::remove_dir_all(&second).unwrap();
        scans = 0;
        let (cache, changed) = reconcile_sessions_cache_with(cache, vec![first], |dir| {
            scans += 1;
            cached_session_for_dir(dir)
        });
        assert!(changed);
        assert_eq!(scans, 0);
        assert_eq!(cache.sessions_by_dir.len(), 1);
        fs::remove_dir_all(root).unwrap();
    }
}
