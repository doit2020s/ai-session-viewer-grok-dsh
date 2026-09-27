use std::fs;
use std::path::Path;
use std::process::Command;

use session_core::fork::{ForkRequest, ForkResult};
use session_core::metadata::validate_session_id;
use session_core::models::session::{SessionsIndex, SessionsIndexFileEntry};
use session_core::parser::jsonl as claude_parser;

/// Quote a string for inclusion inside a single-quoted POSIX shell argument.
/// Each embedded `'` is replaced with `'\''` so the path can never break out
/// of the surrounding quotes (which would otherwise let an attacker-controlled
/// path inject shell commands). Only used on macOS / Linux.
#[cfg(any(target_os = "macos", target_os = "linux"))]
fn posix_single_quote(s: &str) -> String {
    s.replace('\'', "'\\''")
}

fn resume_command(source: &str, session_id: &str) -> Result<String, String> {
    match source {
        "claude" => Ok(format!("claude --resume {session_id}")),
        "codex" => Ok(format!("codex resume {session_id}")),
        "grok" => Ok(format!("grok -r {session_id}")),
        "omp" => Ok(format!("omp --resume {session_id}")),
        _ => Err(format!("Unknown source: {source}")),
    }
}

/// Resolve the Grok executable explicitly on Windows. Desktop apps launched
/// from Explorer do not always inherit the user's refreshed PATH, even when
/// `grok` works in an interactive PowerShell. Prefer GROK_BINARY, then the
/// official per-user install location, and finally let the shell resolve PATH.
#[cfg(target_os = "windows")]
fn grok_command(session_id: &str) -> String {
    let exe = std::env::var("GROK_BINARY")
        .ok()
        .filter(|p| !p.trim().is_empty())
        .or_else(|| {
            let home = std::env::var_os("USERPROFILE")
                .or_else(|| std::env::var_os("HOME"))?;
            let candidate = Path::new(&home).join(".grok").join("bin").join("grok.exe");
            candidate.exists().then(|| candidate.to_string_lossy().into_owned())
        })
        .unwrap_or_else(|| "grok".to_string());
    // The path is local configuration, but quote it for PowerShell/CMD so
    // installs under a path containing spaces work as well.
    if exe.contains(' ') {
        format!("\"{}\" -r {session_id}", exe.replace('"', "\\\""))
    } else {
        format!("{exe} -r {session_id}")
    }
}

/// Start a brand-new native Grok CLI conversation. The CLI receives a fresh
/// UUID and remains responsible for creating every session file and metadata
/// directory, so the resulting conversation has exactly the same format as a
/// session created directly in Grok.
fn grok_new_command(session_id: &str) -> String {
    #[cfg(target_os = "windows")]
    {
        let exe = std::env::var("GROK_BINARY")
            .ok()
            .filter(|p| !p.trim().is_empty())
            .or_else(|| {
                let home = std::env::var_os("USERPROFILE")
                    .or_else(|| std::env::var_os("HOME"))?;
                let candidate = Path::new(&home).join(".grok").join("bin").join("grok.exe");
                candidate.exists().then(|| candidate.to_string_lossy().into_owned())
            })
            .unwrap_or_else(|| "grok".to_string());
        if exe.contains(' ') {
            format!("\"{}\" --session-id {session_id}", exe.replace('"', "\\\""))
        } else {
            format!("{exe} --session-id {session_id}")
        }
    }

    #[cfg(not(target_os = "windows"))]
    {
        format!("grok --session-id {session_id}")
    }
}

#[tauri::command]
pub fn create_grok_session(
    project_path: String,
    shell: Option<String>,
) -> Result<String, String> {
    let canonical = Path::new(&project_path)
        .canonicalize()
        .map_err(|e| format!("Grok 项目路径不可用: {project_path}: {e}"))?;
    if !canonical.is_dir() {
        return Err(format!("Grok 项目路径不是文件夹: {project_path}"));
    }

    // Generate the id in the trusted backend because it is included in the
    // terminal command line. validate_session_id documents and enforces the
    // same shell-safety boundary used by resume_session.
    let session_id = uuid::Uuid::new_v4().to_string();
    validate_session_id(&session_id)?;
    let cli_cmd = grok_new_command(&session_id);
    let canonical = canonical.to_string_lossy().into_owned();
    open_terminal(&canonical, &cli_cmd, shell.as_deref())?;
    Ok(session_id)
}

#[tauri::command]
pub fn resume_session(
    source: String,
    session_id: String,
    project_path: String,
    file_path: Option<String>,
    shell: Option<String>,
) -> Result<(), String> {
    // The session id ends up in a shell command line (e.g. `claude --resume
    // {id}` inside `bash -c '…'`). Reject anything that's not a single safe
    // path component to prevent shell metacharacters (`; & | $() \``)
    // sneaking in via a doctored session file.
    validate_session_id(&session_id)?;

    // Try to derive the correct project path from the session file location
    let project_path = resolve_project_path(&source, &project_path, file_path.as_deref());

    if !Path::new(&project_path).exists() {
        return Err(format!("项目路径不存在: {}", project_path));
    }

    // For Claude sessions, ensure the session is in sessions-index.json
    // so that `claude --resume` can find it
    if source == "claude" {
        if let Some(fp) = &file_path {
            let fp = normalize_path(fp);
            ensure_session_in_index(&session_id, &fp, &project_path);
        }
    }

    let cli_cmd = if source == "grok" {
        #[cfg(target_os = "windows")]
        { grok_command(&session_id) }
        #[cfg(not(target_os = "windows"))]
        { resume_command(&source, &session_id)? }
    } else {
        resume_command(&source, &session_id)?
    };

    open_terminal(&project_path, &cli_cmd, shell.as_deref())
}

#[tauri::command]
pub async fn fork_session(
    source: String,
    original_file_path: String,
    user_msg_uuid: String,
) -> Result<ForkResult, String> {
    session_core::fork::fork_session(ForkRequest {
        source,
        original_file_path,
        user_msg_uuid,
    })
    .await
}

fn open_terminal(project_path: &str, cli_cmd: &str, _shell: Option<&str>) -> Result<(), String> {
    if !Path::new(project_path).exists() {
        return Err(format!("项目路径不存在: {}", project_path));
    }

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        // Spawn a new console window for the shell so we don't need
        // `cmd /c start`, which would re-parse our arguments through cmd.exe
        // and let metacharacters (`&`, `|`, `^`, etc.) in the project path
        // inject commands. Setting `current_dir` natively avoids splicing
        // the path into any shell command line at all.
        const CREATE_NEW_CONSOLE: u32 = 0x0000_0010;

        if _shell == Some("powershell") {
            // PowerShell parses `-Command` as a script block; `cli_cmd` only
            // contains the literal binary + a validated session id, so
            // there's nothing user-controlled left to escape here.
            Command::new("powershell")
                .args(["-NoExit", "-Command", cli_cmd])
                .current_dir(project_path)
                .creation_flags(CREATE_NEW_CONSOLE)
                .spawn()
                .map_err(|e| format!("Failed to open PowerShell: {}", e))?;
        } else {
            // `cmd /k <cli_cmd>` keeps the prompt open after the command
            // finishes. `cli_cmd` is `claude --resume <validated-id>` /
            // `codex resume <validated-id>` — no metacharacters can leak in.
            Command::new("cmd")
                .args(["/k", cli_cmd])
                .current_dir(project_path)
                .creation_flags(CREATE_NEW_CONSOLE)
                .spawn()
                .map_err(|e| format!("Failed to open terminal: {}", e))?;
        }
    }

    #[cfg(target_os = "macos")]
    {
        // Escape both the path and the cli command for inclusion inside
        // single quotes — and then for the *outer* AppleScript double-quoted
        // string. AppleScript treats `\` and `"` specially inside strings.
        let path_q = posix_single_quote(project_path);
        let inner = format!("cd '{}' && {}", path_q, cli_cmd);
        let applescript_inner = inner.replace('\\', "\\\\").replace('"', "\\\"");
        let script = format!(
            "tell application \"Terminal\" to do script \"{}\"",
            applescript_inner
        );
        Command::new("osascript")
            .args(["-e", &script])
            .spawn()
            .map_err(|e| format!("Failed to open terminal: {}", e))?;
    }

    #[cfg(target_os = "linux")]
    {
        use std::os::unix::process::CommandExt;

        // Path is single-quoted in the bash command, so any embedded `'`
        // must be escaped as `'\''` to avoid breaking out of the quotes.
        let path_q = posix_single_quote(project_path);
        let cmd_str = format!("cd '{}' && {}", path_q, cli_cmd);

        let xfce_arg = format!("bash -c '{}'", posix_single_quote(&cmd_str));
        let xterm_arg = format!("bash -c '{}'", posix_single_quote(&cmd_str));
        let terminals: [(&str, &[&str]); 4] = [
            ("gnome-terminal", &["--", "bash", "-c", &cmd_str]),
            ("konsole", &["-e", "bash", "-c", &cmd_str]),
            ("xfce4-terminal", &["-e", &xfce_arg]),
            ("xterm", &["-e", &xterm_arg]),
        ];

        let mut launched = false;
        for (terminal, args) in &terminals {
            if Command::new(terminal)
                .args(*args)
                .process_group(0)
                .spawn()
                .is_ok()
            {
                launched = true;
                break;
            }
        }

        if !launched {
            return Err("No supported terminal emulator found".to_string());
        }
    }

    Ok(())
}

/// Resolve the correct project path for resuming a session.
/// Priority: sessions-index.json original_path > provided project_path
fn resolve_project_path(source: &str, project_path: &str, file_path: Option<&str>) -> String {
    if source == "claude" {
        if let Some(fp) = file_path {
            let fp = normalize_path(fp);
            if let Some(parent) = Path::new(&fp).parent() {
                let index_path = parent.join("sessions-index.json");
                if index_path.exists() {
                    if let Ok(content) = fs::read_to_string(&index_path) {
                        if let Ok(index) = serde_json::from_str::<serde_json::Value>(&content) {
                            if let Some(original) =
                                index.get("originalPath").and_then(|v| v.as_str())
                            {
                                let resolved = normalize_path(original);
                                if Path::new(&resolved).exists() {
                                    return resolved;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    normalize_path(project_path)
}

/// Ensure a session entry exists in sessions-index.json so that
/// `claude --resume {id}` can discover it. Orphan sessions (e.g. from
/// Ctrl+C exits) exist on disk but are missing from the index.
fn ensure_session_in_index(session_id: &str, file_path: &str, project_path: &str) {
    let session_file = Path::new(file_path);
    let parent = match session_file.parent() {
        Some(p) => p,
        None => return,
    };

    let index_path = parent.join("sessions-index.json");

    // Read existing index or create a new one
    let mut index: SessionsIndex = if index_path.exists() {
        match fs::read_to_string(&index_path)
            .ok()
            .and_then(|c| serde_json::from_str(&c).ok())
        {
            Some(idx) => idx,
            None => return, // Can't parse existing index, don't risk corrupting it
        }
    } else {
        SessionsIndex {
            version: Some(1),
            entries: Vec::new(),
            original_path: Some(project_path.to_string()),
        }
    };

    // Already in index — nothing to do
    if index.entries.iter().any(|e| e.session_id == session_id) {
        return;
    }

    // Build an entry from the JSONL file metadata
    let first_prompt = claude_parser::extract_first_prompt(session_file);
    let metadata = claude_parser::extract_session_metadata(session_file);
    let (_, git_branch, cwd) = metadata.unwrap_or((String::new(), None, None));

    let file_meta = fs::metadata(session_file).ok();
    let mtime = file_meta.as_ref().and_then(|m| {
        m.modified()
            .ok()
            .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
    });
    let modified = file_meta.as_ref().and_then(|m| {
        m.modified().ok().map(|t| {
            let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()
        })
    });
    let created = file_meta.as_ref().and_then(|m| {
        m.created().ok().map(|t| {
            let d = t.duration_since(std::time::UNIX_EPOCH).unwrap_or_default();
            chrono::DateTime::from_timestamp(d.as_secs() as i64, 0)
                .map(|dt| dt.to_rfc3339())
                .unwrap_or_default()
        })
    });

    let message_count = count_user_assistant(session_file);

    index.entries.push(SessionsIndexFileEntry {
        session_id: session_id.to_string(),
        full_path: Some(file_path.to_string()),
        file_mtime: mtime,
        first_prompt,
        message_count: Some(message_count),
        created,
        modified,
        git_branch,
        project_path: cwd.or_else(|| Some(project_path.to_string())),
        is_sidechain: Some(false),
    });

    // Write back (atomic: tmp + rename)
    if let Ok(json) = serde_json::to_string_pretty(&index) {
        let tmp_path = index_path.with_extension("json.tmp");
        if fs::write(&tmp_path, &json).is_ok() {
            let _ = fs::rename(&tmp_path, &index_path);
        }
    }
}

fn count_user_assistant(path: &Path) -> u32 {
    use std::io::{BufRead, BufReader};
    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return 0,
    };
    let reader = BufReader::new(file);
    let mut count: u32 = 0;
    for line in reader.lines().map_while(Result::ok) {
        let trimmed = line.trim();
        if trimmed.contains("\"type\":\"user\"") || trimmed.contains("\"type\":\"assistant\"") {
            count += 1;
        }
    }
    count
}

fn normalize_path(path: &str) -> String {
    if cfg!(windows) {
        path.replace('/', "\\")
    } else {
        path.replace('\\', "/")
    }
}

#[cfg(test)]
mod tests {
    use super::{grok_new_command, resume_command};

    #[test]
    fn builds_omp_resume_command() {
        assert_eq!(
            resume_command("omp", "01a066c5-490b-77e1-be17-dd879255a46f").unwrap(),
            "omp --resume 01a066c5-490b-77e1-be17-dd879255a46f"
        );
    }

    #[test]
    fn builds_native_grok_new_session_command() {
        let id = "e81cc406-b872-4b34-9c1a-aefe92364fb0";
        let command = grok_new_command(id);
        assert!(command.ends_with(&format!("--session-id {id}")));
        assert!(!command.contains(" -r "));
    }
}
