// Squid's own settings: ~/.squid/config.json. Everything in it is optional. For example:
//   { "defaultModel": "qwen3.8-27b-text", "models": { "qwen3.8-27b-text": { "contextLength": 49152 } } }
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const SQUID_HOME = join(homedir(), ".squid");

export type Config = {
  defaultModel?: string; // used when no model is named: bare `squid`, `squid ask`
  defaultMode?: "manual" | "auto" | "plan"; // the mode a chat starts in (Claude Code: defaultMode); manual if unset
  jitTtlMinutes?: number; // idle time before an automatically loaded model is unloaded (LM Studio: 60)
  // Per-model load settings. runtime "squid" runs the model in Squid's own TypeScript engine (step 17).
  models?: Record<string, { contextLength?: number; parallel?: number; runtime?: "llama.cpp" | "squid" }>;
};

export function readConfig(file = join(SQUID_HOME, "config.json")): Config {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return {}; // no file yet: all defaults
  }
}
