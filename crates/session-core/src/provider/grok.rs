use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs;
use std::io::{BufRead, BufReader, Write};
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
// v3 caches only summary titles. Grok's client-state names are overlaid at
// read time so a rename or a cleared name never leaves a stale disk title.
const DISK_CACHE_VERSION: u32 = 3;

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

fn native_meta_path(grok_home: &Path) -> PathBuf {
    grok_home.join("client-state").join("session-meta.json")
}

#[derive(Clone, Default)]
struct NativeSessionNames {
    custom: Option<String>,
    auto: Option<String>,
}

fn native_session_names(grok_home: &Path) -> HashMap<String, NativeSessionNames> {
    let Ok(bytes) = fs::read(native_meta_path(grok_home)) else {
        return HashMap::new();
    };
    let Ok(Value::Object(sessions)) = serde_json::from_slice::<Value>(&bytes) else {
        return HashMap::new();
    };
    sessions
        .into_iter()
        .filter_map(|(id, meta)| {
            if meta.get("provider").and_then(Value::as_str) != Some("grok") {
                return None;
            }
            let name = |key| {
                meta.get(key)
                    .and_then(Value::as_str)
                    .map(str::trim)
                    .filter(|name| !name.is_empty())
                    .map(ToString::to_string)
            };
            let names = NativeSessionNames {
                custom: name("customName"),
                auto: name("autoName"),
            };
            (names.custom.is_some() || names.auto.is_some()).then_some((id, names))
        })
        .collect()
}

fn grok_home_for_history(chat_history_path: &Path) -> Result<(PathBuf, String, Option<String>), String> {
    let path = chat_history_path
        .canonicalize()
        .map_err(|error| format!("Grok 会话文件不存在：{error}"))?;
    if !path.is_file() || path.file_name().and_then(|name| name.to_str()) != Some(CHAT_HISTORY_FILE) {
        return Err("Grok 会话文件必须是 chat_history.jsonl".to_string());
    }
    let session_dir = path.parent().ok_or("Grok 会话目录不存在")?;
    let project_dir = session_dir.parent().ok_or("Grok 项目目录不存在")?;
    let sessions_dir = project_dir.parent().ok_or("Grok sessions 目录不存在")?;
    // Prefer the configured home even if its sessions directory is a junction
    // to another path whose parent also happens to be called `sessions`.
    let configured_home = get_sessions_dir().and_then(|configured| {
        (configured.canonicalize().ok().as_deref() == Some(sessions_dir))
            .then(|| configured.parent().map(Path::to_path_buf))
            .flatten()
    });
    let home = if let Some(home) = configured_home {
        home
    } else if sessions_dir.file_name().and_then(|name| name.to_str()) == Some("sessions") {
        // Isolated fixture or another Grok home passed by the internal fork
        // function; public routes always resolve a configured session first.
        sessions_dir.parent().ok_or("Grok 主目录不存在")?.to_path_buf()
    } else {
        return Err("Grok 会话路径不是 sessions/<project>/<session-id>/chat_history.jsonl".to_string());
    };
    let summary: Value = serde_json::from_slice(
        &fs::read(session_dir.join("summary.json"))
            .map_err(|error| format!("读取 Grok 会话摘要失败：{error}"))?,
    )
    .map_err(|error| format!("解析 Grok 会话摘要失败：{error}"))?;
    let id = summary
        .pointer("/info/id")
        .and_then(Value::as_str)
        .ok_or("Grok 会话摘要缺少 info.id")?;
    crate::metadata::validate_session_id(id)?;
    if session_dir.file_name().and_then(|name| name.to_str()) != Some(id) {
        return Err("Grok 会话摘要 ID 与目录名称不一致".to_string());
    }
    let cwd = summary
        .pointer("/info/cwd")
        .and_then(Value::as_str)
        .map(ToString::to_string);
    Ok((home, id.to_string(), cwd))
}

/// Grok's own UI stores names in `client-state/session-meta.json`, not in
/// `summary.json`. Resolve the name against the session's Grok home so forks
/// and tests can use isolated homes without changing `GROK_HOME` globally.
pub fn custom_name_for_path(chat_history_path: &Path) -> Option<String> {
    let (home, id, _) = grok_home_for_history(chat_history_path).ok()?;
    native_session_names(&home).remove(&id)?.custom
}

/// Preferred native title for a Grok session: explicit rename, then the
/// CLI's automatically generated name. This is also the fork title source.
pub fn native_display_name_for_path(chat_history_path: &Path) -> Option<String> {
    let (home, id, _) = grok_home_for_history(chat_history_path).ok()?;
    let names = native_session_names(&home).remove(&id)?;
    names.custom.or(names.auto)
}

fn rename_lock() -> &'static Mutex<()> {
    static LOCK: OnceLock<Mutex<()>> = OnceLock::new();
    LOCK.get_or_init(|| Mutex::new(()))
}

/// Store a session name in Grok's native client-state registry. The caller
/// supplies an existing history path; its summary ID must match its directory.
/// Existing metadata (usage, activeAt, etc.) is preserved verbatim as JSON.
pub fn set_custom_name_for_path(chat_history_path: &Path, name: Option<&str>) -> Result<(), String> {
    let (home, id, cwd) = grok_home_for_history(chat_history_path)?;
    let name = name.map(str::trim).filter(|name| !name.is_empty());
    if let Some(name) = name {
        if name.chars().count() > 512 || name.chars().any(char::is_control) {
            return Err("Grok 会话名不能超过 512 字或包含控制字符".to_string());
        }
    }

    let _guard = rename_lock().lock();
    let path = native_meta_path(&home);
    let before = match fs::read(&path) {
        Ok(bytes) => Some(bytes),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
        Err(error) => return Err(format!("读取 Grok 会话元数据失败：{error}")),
    };
    let mut registry = match before.as_deref() {
        Some(bytes) => serde_json::from_slice::<Value>(bytes)
            .map_err(|error| format!("Grok 会话元数据损坏，未覆盖原文件：{error}"))?,
        None => Value::Object(serde_json::Map::new()),
    };
    let sessions = registry
        .as_object_mut()
        .ok_or("Grok 会话元数据不是 JSON 对象，未覆盖原文件")?;
    if !sessions.contains_key(&id) && name.is_none() {
        return Ok(());
    }
    let entry = sessions.entry(id).or_insert_with(|| {
        let mut meta = serde_json::Map::new();
        meta.insert("provider".to_string(), Value::String("grok".to_string()));
        if let Some(cwd) = cwd {
            meta.insert("providerCwd".to_string(), Value::String(cwd));
        }
        Value::Object(meta)
    });
    let meta = entry
        .as_object_mut()
        .ok_or("Grok 会话元数据条目不是 JSON 对象，未覆盖原文件")?;
    if meta.get("provider").and_then(Value::as_str) != Some("grok") {
        return Err("同 ID 元数据属于其他提供商，未覆盖原文件".to_string());
    }
    let previous = meta
        .get("customName")
        .and_then(Value::as_str)
        .map(str::trim)
        .filter(|value| !value.is_empty());
    if previous == name {
        return Ok(());
    }
    if let Some(name) = name {
        meta.insert("customName".to_string(), Value::String(name.to_string()));
    } else {
        meta.remove("customName");
    }
    let rendered = serde_json::to_vec_pretty(&registry)
        .map_err(|error| format!("序列化 Grok 会话元数据失败：{error}"))?;
    if before.as_deref() == Some(rendered.as_slice()) {
        return Ok(());
    }
    let parent = path.parent().ok_or("Grok client-state 目录不存在")?;
    fs::create_dir_all(parent)
        .map_err(|error| format!("创建 Grok client-state 目录失败：{error}"))?;
    let temporary = parent.join(format!("session-meta.{}.tmp", uuid::Uuid::new_v4()));
    let result = (|| {
        let mut file = fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&temporary)
            .map_err(|error| format!("创建 Grok 会话元数据临时文件失败：{error}"))?;
        file.write_all(&rendered)
            .and_then(|_| file.sync_all())
            .map_err(|error| format!("写入 Grok 会话元数据临时文件失败：{error}"))?;
        drop(file);
        // Grok CLI may update this registry independently. Never knowingly
        // replace bytes that differ from the snapshot we edited.
        let current = match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(format!("重新读取 Grok 会话元数据失败：{error}")),
        };
        if current != before {
            return Err("Grok CLI 同时修改了会话元数据，请刷新后重试".to_string());
        }
        if let Some(bytes) = before.as_deref() {
            let backup = parent.join(format!("session-meta.{}.asv-backup.json", uuid::Uuid::new_v4()));
            fs::write(&backup, bytes)
                .map_err(|error| format!("备份 Grok 会话元数据失败：{error}"))?;
        }
        let latest = match fs::read(&path) {
            Ok(bytes) => Some(bytes),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => None,
            Err(error) => return Err(format!("保存前读取 Grok 会话元数据失败：{error}")),
        };
        if latest != before {
            return Err("Grok CLI 同时修改了会话元数据，请刷新后重试".to_string());
        }
        fs::rename(&temporary, &path)
            .map_err(|error| format!("保存 Grok 会话元数据失败（原文件保留）：{error}"))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    } else {
        invalidate_sessions_cache();
    }
    result
}

/// Resolve a Grok session by its summary ID across every project. The UI's
/// project ID is not needed (it can be `<grok-unrooted>` for legacy sessions).
pub fn set_custom_name(session_id: &str, name: Option<&str>) -> Result<(), String> {
    crate::metadata::validate_session_id(session_id)?;
    let matches = load_all_sessions()
        .into_iter()
        .filter(|session| session.session_id == session_id)
        .collect::<Vec<_>>();
    if matches.len() != 1 {
        return Err(if matches.is_empty() {
            "找不到 Grok 会话，请刷新列表后重试".to_string()
        } else {
            "存在重复 Grok 会话 ID，无法确定要重命名的会话".to_string()
        });
    }
    set_custom_name_for_path(Path::new(&matches[0].file_path), name)
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
    Some(SessionIndexEntry {
        source: "grok".to_string(),
        session_id,
        file_path: file_path.to_string_lossy().into_owned(),
        first_prompt,
        thread_name: generated_title,
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
    let names = get_sessions_dir()
        .and_then(|sessions| sessions.parent().map(native_session_names))
        .unwrap_or_default();
    sessions_from_cache_with_names(cache, &names)
}

fn sessions_from_cache_with_names(
    cache: &GrokDiskCache,
    names: &HashMap<String, NativeSessionNames>,
) -> Vec<SessionIndexEntry> {
    cache
        .sessions_by_dir
        .values()
        .map(|cached| {
            let mut entry = cached.entry.clone();
            if let Some(native) = names.get(&entry.session_id) {
                if let Some(name) = native.custom.as_ref() {
                    entry.thread_name = Some(name.clone());
                    // Only a user-assigned native name is editable as an alias.
                    entry.alias = Some(name.clone());
                } else if let Some(name) = native.auto.as_ref() {
                    entry.thread_name = Some(name.clone());
                }
            }
            entry
        })
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
    let sessions_deleted = sessions.len();
    let mut project_dirs = Vec::new();
    for session in &sessions {
        let path = crate::paths::validate_session_file("grok", &session.file_path)?;
        let project_dir = path
            .parent()
            .and_then(Path::parent)
            .ok_or_else(|| "Grok 工作区目录无效".to_string())?
            .to_path_buf();
        if !project_dirs.contains(&project_dir) {
            project_dirs.push(project_dir);
        }
    }
    crate::recyclebin::move_paths_to_recyclebin(
        &project_dirs,
        "project",
        "ManualDelete",
        "grok",
        project_id,
        None,
        Some(project_name),
    )?;
    for session in &sessions {
        let _ = crate::metadata::remove_session_meta("grok", project_id, &session.session_id);
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

    #[test]
    fn native_rename_preserves_other_metadata_and_can_clear_name() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-native-name-{}-{unique}",
            std::process::id()
        ));
        let session_dir = home.join("sessions").join("project").join("session-1");
        write_session(&session_dir, "session-1", "/project", Some("prompt"));
        let history = session_dir.join(CHAT_HISTORY_FILE);
        let native = native_meta_path(&home);
        fs::create_dir_all(native.parent().unwrap()).unwrap();
        fs::write(
            &native,
            serde_json::json!({
                "other-provider": { "provider": "claude", "customName": "keep" },
                "session-1": {
                    "provider": "grok", "customName": "before", "autoName": "automatic", "usage": {"tokens": 7},
                    "providerCwd": "/project"
                }
            })
            .to_string(),
        )
        .unwrap();

        assert_eq!(custom_name_for_path(&history).as_deref(), Some("before"));
        assert_eq!(native_display_name_for_path(&history).as_deref(), Some("before"));
        set_custom_name_for_path(&history, Some(" 新名称 ")).unwrap();
        let after: Value = serde_json::from_slice(&fs::read(&native).unwrap()).unwrap();
        assert_eq!(after["session-1"]["customName"], "新名称");
        assert_eq!(after["session-1"]["usage"]["tokens"], 7);
        assert_eq!(after["other-provider"]["customName"], "keep");
        assert_eq!(custom_name_for_path(&history).as_deref(), Some("新名称"));
        let already_named = fs::read(&native).unwrap();
        set_custom_name_for_path(&history, Some("新名称")).unwrap();
        assert_eq!(fs::read(&native).unwrap(), already_named);

        set_custom_name_for_path(&history, None).unwrap();
        let cleared: Value = serde_json::from_slice(&fs::read(&native).unwrap()).unwrap();
        assert!(cleared["session-1"].get("customName").is_none());
        assert_eq!(cleared["session-1"]["usage"]["tokens"], 7);
        assert_eq!(custom_name_for_path(&history), None);
        assert_eq!(native_display_name_for_path(&history).as_deref(), Some("automatic"));
        let already_cleared = fs::read(&native).unwrap();
        set_custom_name_for_path(&history, None).unwrap();
        assert_eq!(fs::read(&native).unwrap(), already_cleared);

        let second_dir = home.join("sessions").join("project").join("session-2");
        write_session(&second_dir, "session-2", "/project", Some("other prompt"));
        let second_history = second_dir.join(CHAT_HISTORY_FILE);
        set_custom_name_for_path(&second_history, Some("新会话")).unwrap();
        let with_new_entry: Value = serde_json::from_slice(&fs::read(&native).unwrap()).unwrap();
        assert_eq!(with_new_entry["session-2"]["provider"], "grok");
        assert_eq!(with_new_entry["session-2"]["providerCwd"], "/project");
        assert_eq!(with_new_entry["session-2"]["customName"], "新会话");
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn native_rename_requires_matching_summary_and_never_replaces_corrupt_registry() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let home = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-rename-guard-{}-{unique}",
            std::process::id()
        ));
        let session_dir = home.join("sessions").join("project").join("session-1");
        write_session(&session_dir, "different", "/project", Some("prompt"));
        let history = session_dir.join(CHAT_HISTORY_FILE);
        assert!(set_custom_name_for_path(&history, Some("title")).is_err());
        fs::write(
            session_dir.join("summary.json"),
            serde_json::json!({"info":{"id":"session-1","cwd":"/project"}}).to_string(),
        )
        .unwrap();
        let native = native_meta_path(&home);
        fs::create_dir_all(native.parent().unwrap()).unwrap();
        fs::write(&native, b"not valid JSON").unwrap();
        assert!(set_custom_name_for_path(&history, Some("title")).is_err());
        assert_eq!(fs::read(&native).unwrap(), b"not valid JSON");
        let foreign = serde_json::json!({"session-1":{"provider":"claude","customName":"keep"}}).to_string();
        fs::write(&native, &foreign).unwrap();
        assert!(set_custom_name_for_path(&history, Some("title")).is_err());
        assert_eq!(fs::read_to_string(&native).unwrap(), foreign);
        fs::remove_dir_all(home).unwrap();
    }

    #[test]
    fn native_name_overlay_updates_cached_title_and_clear_restores_summary_title() {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "ai-session-viewer-grok-cache-title-{}-{unique}",
            std::process::id()
        ));
        write_session(&dir, "session-1", "/project", Some("prompt"));
        let mut summary: Value = serde_json::from_slice(&fs::read(dir.join("summary.json")).unwrap()).unwrap();
        summary["session_summary"] = Value::String("generated".to_string());
        fs::write(dir.join("summary.json"), summary.to_string()).unwrap();
        let cached = cached_session_for_dir(&dir).unwrap();
        assert_eq!(cached.entry.thread_name.as_deref(), Some("generated"));
        let mut cache = GrokDiskCache::default();
        cache.sessions_by_dir.insert(dir.to_string_lossy().into_owned(), cached);
        let names = HashMap::from([("session-1".to_string(), NativeSessionNames {
            custom: Some("custom".to_string()),
            auto: Some("automatic".to_string()),
        })]);
        assert_eq!(sessions_from_cache_with_names(&cache, &names)[0].thread_name.as_deref(), Some("custom"));
        assert_eq!(sessions_from_cache_with_names(&cache, &names)[0].alias.as_deref(), Some("custom"));
        let auto_only = HashMap::from([("session-1".to_string(), NativeSessionNames {
            custom: None,
            auto: Some("automatic".to_string()),
        })]);
        assert_eq!(sessions_from_cache_with_names(&cache, &auto_only)[0].thread_name.as_deref(), Some("automatic"));
        assert_eq!(sessions_from_cache_with_names(&cache, &auto_only)[0].alias, None);
        assert_eq!(sessions_from_cache_with_names(&cache, &HashMap::new())[0].thread_name.as_deref(), Some("generated"));
        assert_eq!(sessions_from_cache_with_names(&cache, &HashMap::new())[0].alias, None);
        fs::remove_dir_all(dir).unwrap();
    }
}
