import { useCallback, useRef, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { api } from "../../services/api";
import { createSessionFork } from "../../services/sessionFork";
import { getApiBaseUrl, isRemoteNodeActive } from "../../services/nodeConfig";
import { useAppStore } from "../../stores/appStore";

declare const __IS_TAURI__: boolean;

export function useSessionFork(source: string, filePath?: string) {
  const navigate = useNavigate();
  const { projectId } = useParams();
  const terminalShell = useAppStore((state) => state.terminalShell);
  const refresh = useAppStore((state) => state.refreshInBackground);
  const busy = useRef(false);
  const [pendingMessageId, setPendingMessageId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const fork = useCallback(async (messageId: string) => {
    if (!filePath || !projectId || busy.current) return;
    busy.current = true;
    setPendingMessageId(messageId);
    setError(null);
    const nodeUrl = getApiBaseUrl();
    try {
      const { result, warning } = await createSessionFork(
        api, source, filePath, messageId,
        __IS_TAURI__ && !isRemoteNodeActive() && ["claude", "codex", "grok", "omp"].includes(source), terminalShell,
      );
      if (getApiBaseUrl() !== nodeUrl || useAppStore.getState().source !== source) return;
      // Refresh is independent of creation: a refresh failure cannot invite
      // users to create a duplicate. The destination also loads its own data.
      void refresh(true).catch(() => {});
      navigate(`/projects/${encodeURIComponent(result.projectId)}/session/${encodeURIComponent(result.newFilePath)}`, {
        state: { forkResult: result, forkWarning: warning },
      });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      busy.current = false;
      setPendingMessageId(null);
    }
  }, [filePath, navigate, projectId, refresh, source, terminalShell]);

  return { fork, pendingMessageId, error };
}
