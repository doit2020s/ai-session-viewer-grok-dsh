import { useEffect, useRef } from "react";
import { useUpdateStore } from "../stores/updateStore";

declare const __IS_TAURI__: boolean;

export function useUpdateChecker() {
  const hasChecked = useRef(false);
  const {
    detectInstallType,
    loadCurrentVersion,
  } = useUpdateStore();

  useEffect(() => {
    if (!__IS_TAURI__) return;
    if (hasChecked.current) return;
    hasChecked.current = true;

    detectInstallType();
    loadCurrentVersion();
    // This fork carries local DSH, Kiro and Grok integrations that are not in
    // upstream releases. Do not run the upstream updater automatically at
    // startup because accepting it replaces the customized executable.
    // Manual update controls remain available in Settings for an intentional
    // migration.
  }, [detectInstallType, loadCurrentVersion]);
}
