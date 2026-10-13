// Squid's server running in the background, as LM Studio's does: started when a command needs it,
// checked with `squid server status`, stopped with `squid server stop`. A program that runs in the
// background like this is called a daemon.
import { spawn } from "node:child_process";
import { mkdirSync, openSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { SQUID_HOME } from "./config.ts";
import { isAlive } from "./engine.ts";
import { DEFAULT_PORT } from "./server.ts";

const STATE = join(SQUID_HOME, "server.json"); // { pid, port } of the running server
export type ServerInfo = { pid: number; port: number };

// Written by `squid serve` when it starts listening, whether it runs in a window or in the background.
export function recordServer(info: ServerInfo): void {
  mkdirSync(SQUID_HOME, { recursive: true });
  writeFileSync(STATE, JSON.stringify(info));
}

// The running server, if there is one. Both checks matter: the process is alive, and the port
// answers like Squid. A leftover file from a crash or reboot is removed.
export async function serverStatus(): Promise<ServerInfo | undefined> {
  let info: ServerInfo;
  try {
    info = JSON.parse(readFileSync(STATE, "utf8"));
  } catch {
    return undefined;
  }
  if (isAlive(info.pid) && (await answersLikeSquid(info.port))) return info;
  rmSync(STATE, { force: true });
  return undefined;
}

// Starts the server in the background unless it's already running, and waits until it answers.
export async function ensureServer(port = DEFAULT_PORT): Promise<ServerInfo> {
  const running = await serverStatus();
  if (running) return running;
  const cli = fileURLToPath(new URL("./cli.ts", import.meta.url));
  mkdirSync(join(SQUID_HOME, "logs"), { recursive: true });
  const log = openSync(join(SQUID_HOME, "logs", "server.log"), "a");
  // The same program (node + cli.ts) running `squid serve`, detached so it outlives this command.
  const child = spawn(process.execPath, [cli, "serve", "--port", String(port)], { detached: true, stdio: ["ignore", log, log], windowsHide: true });
  child.unref();
  for (let waited = 0; waited < 15_000; waited += 100) {
    const info = await serverStatus();
    if (info) return info;
    if (child.exitCode !== null) break;
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  throw new Error(`the server didn't start; see ${join(SQUID_HOME, "logs", "server.log")}`);
}

export async function stopServer(): Promise<boolean> {
  const info = await serverStatus();
  if (!info) return false;
  process.kill(info.pid);
  rmSync(STATE, { force: true });
  return true;
}

// Squid answers an unknown address with LM Studio's "Unexpected endpoint" message. Another program
// that happened to get the port after a reboot won't (LM Studio would, but it lives on 1234).
async function answersLikeSquid(port: number): Promise<boolean> {
  try {
    const res = await fetch(`http://127.0.0.1:${port}/squid-ping`, { signal: AbortSignal.timeout(1000) });
    return (await res.text()).includes("Unexpected endpoint or method. (GET /squid-ping)");
  } catch {
    return false;
  }
}
