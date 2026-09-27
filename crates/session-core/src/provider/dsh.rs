use std::fs;
use std::io::{BufRead, BufReader, Cursor};
use std::path::{Path, PathBuf};

use chrono::{DateTime, Utc};
use rayon::prelude::*;
use serde_json::Value;

use crate::models::message::{
    DisplayContentBlock, DisplayMessage, PaginatedMessages, RangeMessages,
};
use crate::models::project::ProjectEntry;
use crate::models::session::{SessionIndexEntry, SessionStatus};

const SESSION_FILE: &str = "session.v3.jsonl.zstd";

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
        .map(|entry| entry.path().join(SESSION_FILE))
        .filter(|path| path.is_file())
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
        let path = session_dir.join(SESSION_FILE);
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
}
