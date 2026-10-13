// Squid's own settings: ~/.squid/config.json. Everything in it is optional. For example:
//   { "defaultModel": "qwen3.8-27b-text", "models": { "qwen3.8-27b-text": { "contextLength": 49152 } } }
import { readFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";

export const SQUID_HOME = join(homedir(), ".squid");

export type Config = {
  defaultModel?: string; // used when no model is named: bare `squid`, `squid ask`
  jitTtlMinutes?: number; // idle time before an automatically loaded model is unloaded (LM Studio: 60)
  models?: Record<string, { contextLength?: number; parallel?: number }>; // per-model load settings
};

export function readConfig(file = join(SQUID_HOME, "config.json")): Config {
  try {
    return JSON.parse(readFileSync(file, "utf8"));
  } catch {
    return {}; // no file yet: all defaults
  }
}
