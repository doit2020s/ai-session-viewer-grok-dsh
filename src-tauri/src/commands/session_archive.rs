use std::path::PathBuf;

use session_core::session_archive::{self, BackupResult, BackupSessionInput, RestoreResult};

#[tauri::command]
pub async fn backup_sessions(
    source: String,
    project_id: String,
    project_path: Option<String>,
    sessions: Vec<BackupSessionInput>,
    output_path: String,
) -> Result<BackupResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        session_archive::create_backup(
            &source,
            &project_id,
            project_path.as_deref(),
            &sessions,
            &PathBuf::from(output_path),
        )
    })
    .await
    .map_err(|error| format!("备份任务失败：{error}"))?
}

#[tauri::command]
pub async fn restore_session_backup(
    archive_path: String,
    target_source: String,
    target_project_path: Option<String>,
    target_anchor_path: String,
) -> Result<RestoreResult, String> {
    tauri::async_runtime::spawn_blocking(move || {
        session_archive::restore_backup(
            &PathBuf::from(archive_path),
            &target_source,
            target_project_path.as_deref(),
            &target_anchor_path,
        )
    })
    .await
    .map_err(|error| format!("还原任务失败：{error}"))?
}
