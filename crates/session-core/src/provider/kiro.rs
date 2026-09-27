use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde_json::Value;

use crate::models::message::{
    DisplayContentBlock, DisplayMessage, PaginatedMessages, RangeMessages,
};
use crate::models::project::ProjectEntry;
use crate::models::session::{SessionIndexEntry, SessionStatus};

const SESSION_FILE: &str = "session.json";
const MESSAGES_FILE: &str = "messages.jsonl";

pub fn get_sessions_dir() -> Option<PathBuf> {
    std::env::var_os("KIRO_HOME")
        .map(PathBuf::from)
        .or_else(|| dirs::home_dir().map(|home| home.join(".kiro")))
        .map(|home| home.join("sessions"))
}

fn read_json(path: &Path) -> Option<Value> {
    serde_json::from_slice(&fs::read(path).ok()?).ok()
}

fn string(value: &Value, key: &str) -> Option<String> {
    value
        .get(key)
        .and_then(Value::as_str)
        .filter(|s| !s.trim().is_empty())
        .map(str::to_string)
}

fn timestamp(value: Option<&Value>) -> Option<String> {
    let value = value?;
    if let Some(text) = value.as_str() {
        return Some(text.to_string());
    }
    let number = value.as_i64()?;
    let seconds = if number > 10_000_000_000 {
        number / 1000
    } else {
        number
    };
    DateTime::<Utc>::from_timestamp(seconds, 0).map(|v| v.to_rfc3339())
}

fn json_text(value: &Value) -> String {
    value.as_str().map(str::to_string).unwrap_or_else(|| {
        serde_json::to_string_pretty(value).unwrap_or_else(|_| value.to_string())
    })
}

fn display_message(row: &Value, model: Option<&str>) -> Option<DisplayMessage> {
    let payload = row.get("payload")?;
    let kind = payload.get("type")?.as_str()?;
    let content = match kind {
        "user" | "assistant" => {
            let text = payload.get("content").map(json_text).unwrap_or_default();
            if text.trim().is_empty() {
                return None;
            }
            vec![DisplayContentBlock::Text { text }]
        }
        "tool_call" => vec![DisplayContentBlock::FunctionCall {
            name: payload
                .get("toolName")
                .and_then(Value::as_str)
                .unwrap_or("tool")
                .to_string(),
            arguments: payload.get("args").map(json_text).unwrap_or_default(),
            call_id: payload
                .get("toolCallId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
        }],
        "tool_result" => vec![DisplayContentBlock::FunctionCallOutput {
            call_id: payload
                .get("toolCallId")
                .and_then(Value::as_str)
                .unwrap_or_default()
                .to_string(),
            output: payload.get("content").map(json_text).unwrap_or_default(),
        }],
        _ => return None,
    };
    let role = match kind {
        "user" => "user",
        "assistant" => "assistant",
        "tool_call" => "assistant",
        "tool_result" => "tool",
        _ => unreachable!(),
    };
    Some(DisplayMessage {
        uuid: string(row, "id"),
        parent_uuid: None,
        role: role.to_string(),
        timestamp: timestamp(row.get("timestamp")),
        model: (role == "assistant")
            .then(|| model.map(str::to_string))
            .flatten(),
        content,
    })
}

pub fn parse_all_messages(path: &Path) -> Result<Vec<DisplayMessage>, String> {
    let session = read_json(&path.with_file_name(SESSION_FILE));
    let model = session
        .as_ref()
        .and_then(|v| v.get("modelId"))
        .and_then(Value::as_str);
    let file = fs::File::open(path).map_err(|e| format!("打开 Kiro 会话失败：{e}"))?;
    Ok(BufReader::new(file)
        .lines()
        .filter_map(Result::ok)
        .filter_map(|line| serde_json::from_str::<Value>(&line).ok())
        .filter_map(|row| display_message(&row, model))
        .collect())
}

fn session_files(project_dir: &Path) -> Vec<PathBuf> {
    fs::read_dir(project_dir)
        .into_iter()
        .flatten()
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let messages = entry.path().join(MESSAGES_FILE);
            if messages.is_file() {
                Some(messages)
            } else {
                let metadata = entry.path().join(SESSION_FILE);
                metadata.is_file().then_some(metadata)
            }
        })
        .collect()
}

fn workspace_path(metadata: &Value) -> Option<String> {
    metadata
        .get("workspacePaths")?
        .as_array()?
        .iter()
        .find_map(Value::as_str)
        .map(str::to_string)
}

fn session_entry(path: &Path) -> Option<SessionIndexEntry> {
    let metadata = read_json(&path.with_file_name(SESSION_FILE))?;
    let session_id = string(&metadata, "id")
        .or_else(|| path.parent()?.file_name()?.to_str().map(str::to_string))?;
    let messages = parse_all_messages(path).ok()?;
    let first_prompt = messages.iter().find(|m| m.role == "user").and_then(|m| {
        m.content.iter().find_map(|b| match b {
            DisplayContentBlock::Text { text } => Some(text.chars().take(200).collect()),
            _ => None,
        })
    });
    let cwd = workspace_path(&metadata);
    Some(SessionIndexEntry {
        source: "kiro".to_string(),
        session_id,
        file_path: path.to_string_lossy().into_owned(),
        first_prompt,
        thread_name: string(&metadata, "title"),
        message_count: messages.len() as u32,
        created: timestamp(metadata.get("createdAt")),
        modified: timestamp(metadata.get("lastModifiedAt")),
        git_branch: None,
        project_path: cwd.clone(),
        is_sidechain: None,
        cwd,
        model_provider: string(&metadata, "modelId"),
        cli_version: None,
        alias: None,
        tags: None,
        status: if messages.is_empty() {
            SessionStatus::Empty
        } else {
            SessionStatus::Valid
        },
    })
}

fn project_dir(project_id: &str) -> Result<PathBuf, String> {
    if project_id.is_empty()
        || project_id.contains('/')
        || project_id.contains('\\')
        || project_id == "."
        || project_id == ".."
    {
        return Err("无效的 Kiro 工作区标识".to_string());
    }
    let dir = get_sessions_dir()
        .ok_or_else(|| "找不到 Kiro 会话目录".to_string())?
        .join(project_id);
    if !dir.is_dir() {
        return Err("Kiro 工作区不存在".to_string());
    }
    Ok(dir)
}

pub fn get_projects() -> Result<Vec<ProjectEntry>, String> {
    let root = match get_sessions_dir() {
        Some(root) if root.is_dir() => root,
        _ => return Ok(Vec::new()),
    };
    let mut projects: Vec<_> = fs::read_dir(root)
        .map_err(|e| format!("读取 Kiro 工作区失败：{e}"))?
        .flatten()
        .filter(|entry| entry.path().is_dir())
        .filter_map(|entry| {
            let id = entry.file_name().to_string_lossy().into_owned();
            let files = session_files(&entry.path());
            if files.is_empty() {
                return None;
            }
            let display_path = files
                .iter()
                .find_map(|p| {
                    read_json(&p.with_file_name(SESSION_FILE)).and_then(|m| workspace_path(&m))
                })
                .unwrap_or_else(|| {
                    if id == "_global" {
                        "Kiro 全局".to_string()
                    } else {
                        id.clone()
                    }
                });
            let last_modified = files
                .iter()
                .filter_map(|p| fs::metadata(p).ok()?.modified().ok())
                .max()
                .map(DateTime::<Utc>::from)
                .map(|v| v.to_rfc3339());
            Some(ProjectEntry {
                source: "kiro".to_string(),
                id,
                short_name: Path::new(&display_path)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&display_path)
                    .to_string(),
                display_path: display_path.clone(),
                session_count: files.len(),
                last_modified,
                model_provider: Some("Kiro".to_string()),
                alias: None,
                path_exists: Path::new(&display_path).exists(),
                is_virtual: false,
            })
        })
        .collect();
    projects.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    Ok(projects)
}

pub fn refresh_projects_cache() -> Result<Vec<ProjectEntry>, String> {
    get_projects()
}
pub fn rebuild_projects_cache() -> Result<Vec<ProjectEntry>, String> {
    get_projects()
}
pub fn invalidate_paths(_paths: &[PathBuf]) {}
pub fn get_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    let mut sessions: Vec<_> = session_files(&project_dir(project_id)?)
        .par_iter()
        .filter_map(|p| session_entry(p))
        .collect();
    sessions.sort_by(|a, b| b.modified.cmp(&a.modified));
    Ok(sessions)
}
pub fn refresh_sessions_cache(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    get_sessions(project_id)
}
pub fn get_invalid_sessions(project_id: &str) -> Result<Vec<SessionIndexEntry>, String> {
    Ok(get_sessions(project_id)?
        .into_iter()
        .filter(|s| s.status != SessionStatus::Valid)
        .collect())
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
    #[test]
    fn parses_kiro_metadata_messages_and_tools() {
        let root = std::env::temp_dir().join(format!("kiro-provider-{}", uuid::Uuid::new_v4()));
        let dir = root.join("workspace").join("sess_test");
        fs::create_dir_all(&dir).unwrap();
        fs::write(dir.join(SESSION_FILE), r#"{"id":"sess_test","title":"Named Kiro session","workspacePaths":["C:\\work"],"createdAt":"2026-01-01T00:00:00Z","lastModifiedAt":"2026-01-01T00:01:00Z","modelId":"claude-sonnet"}"#).unwrap();
        fs::write(dir.join(MESSAGES_FILE), concat!(
            r#"{"id":"u1","timestamp":"2026-01-01T00:00:01Z","payload":{"type":"user","content":"hello"}}"#, "\n",
            r#"{"id":"a1","timestamp":"2026-01-01T00:00:02Z","payload":{"type":"assistant","content":"world"}}"#, "\n",
            r#"{"id":"t1","timestamp":"2026-01-01T00:00:03Z","payload":{"type":"tool_call","toolName":"read","toolCallId":"call1","args":{"path":"x"}}}"#, "\n",
            r#"{"id":"r1","timestamp":"2026-01-01T00:00:04Z","payload":{"type":"tool_result","toolCallId":"call1","content":"ok","success":true}}"#, "\n")).unwrap();
        let entry = session_entry(&dir.join(MESSAGES_FILE)).unwrap();
        assert_eq!(entry.thread_name.as_deref(), Some("Named Kiro session"));
        assert_eq!(entry.first_prompt.as_deref(), Some("hello"));
        assert_eq!(entry.message_count, 4);
        let messages = parse_all_messages(&dir.join(MESSAGES_FILE)).unwrap();
        assert_eq!(messages.len(), 4);
        assert!(matches!(
            messages[2].content[0],
            DisplayContentBlock::FunctionCall { .. }
        ));
        let _ = fs::remove_dir_all(root);
    }
}
