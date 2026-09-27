declare const __IS_TAURI__: boolean;

import type * as TauriApi from "./tauriApi";
import { isRemoteNodeActive } from "./nodeConfig";

type ApiModule = typeof TauriApi;

// Session editing must always use the local Tauri command because it writes the
// user's Grok session files.  Routing it through the remote/web API would make
// the UI appear to save while leaving the local CLI context unchanged.
const LOCAL_ONLY_METHODS = new Set<PropertyKey>([
  "getInstallType",
  "writeExportFile",
  "editMessage",
  "deleteMessage",
  "openSessionFolder",
  "backupSessions",
  "restoreSessionBackup",
  "createGrokSession",
]);

function getApiModule(prop: PropertyKey): Promise<ApiModule> {
  if (__IS_TAURI__ && (LOCAL_ONLY_METHODS.has(prop) || !isRemoteNodeActive())) {
    return import("./tauriApi");
  }
  return import("./webApi") as Promise<ApiModule>;
}

export const api = new Proxy({} as ApiModule, {
  get(_target, prop) {
    return async (...args: unknown[]) => {
      const apiModule = await getApiModule(prop);
      const member = apiModule[prop as keyof ApiModule];

      if (typeof member !== "function") {
        return member;
      }

      return (member as (...innerArgs: unknown[]) => unknown)(...args);
    };
  },
});
