// Starting and stopping the engine. For each loaded model Squid runs one llama.cpp `llama-server`
// in the background on a private port, and writes a small record so `ps` and `unload` can find it later.
import { spawn, execFileSync } from "node:child_process";
import { randomBytes } from "node:crypto";
import { existsSync, mkdirSync, openSync, readdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { createServer, type AddressInfo } from "node:net";
import { homedir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import type { ShelfModel } from "./models.ts";
import { readConfig, SQUID_HOME } from "./config.ts";

// Downloaded by `npm run engine:get` (scripts/get-engine.ps1): llama.cpp b11580, CUDA 13 build.
export const ENGINE = fileURLToPath(new URL("../runtime/llama.cpp/llama-server.exe", import.meta.url));
// How replies name the engine, like LM Studio's "llama.cpp-win-x86_64-nvidia-cuda12-avx2" 2.56.0.
export const RUNTIME = { name: "llama.cpp-win-x86_64-nvidia-cuda13", version: engineVersion() };
const LOADED = join(SQUID_HOME, "loaded"); // one record per running engine
const LOGS = join(SQUID_HOME, "logs");

// What Squid remembers about a running engine, saved as ~/.squid/loaded/<identifier>.json.
export type Engine = {
  identifier: string; // the name requests use: the model key unless you pick another
  key: string;
  pid: number;
  port: number;
  apiKey: string; // a random password only Squid knows, so other programs and web pages can't use the engine
  contextLength: number;
  parallel: number;
  sizeBytes: number;
  architecture: string; // copied from the shelf, so replies can describe the model without rereading it
  quantization?: string;
  log: string;
  loader: number; // the Squid process that started it. pid stays 0 until the engine has been launched
  jit: boolean; // loaded on demand by a request, not by `squid load`
  ttlSeconds?: number; // unload after this long unused (LM Studio gives on-demand models 60 minutes)
  lastUsed: number; // when it last answered, in ms since 1970: the idle timer counts from here
};

export type LoadOptions = { contextLength?: number; parallel?: number; identifier?: string; jit?: boolean; ttlSeconds?: number };
type Sizes = { contextLength?: number; parallel?: number };

// Settings LM Studio saved for a model when you tuned it there: flash attention, KV cache type, parallel slots.
// Squid reads them so a model runs the same way in both. The 27B's file sets q4_0 cache and 1 slot.
export type Tuning = { flashAttention?: boolean; kCacheType?: string; vCacheType?: string; parallel?: number; contextLength?: number };

export function lmstudioTuning(
  model: ShelfModel,
  configDir = join(homedir(), ".lmstudio", ".internal", "user-concrete-model-default-config"),
): Tuning {
  // Laid out like the models folder: <publisher>/<repo folder>/<file>.gguf.json
  const [file, repo, publisher] = model.path.split(/[\\/]/).reverse();
  try {
    const config = JSON.parse(readFileSync(join(configDir, publisher, repo, `${file}.json`), "utf8"));
    const fields: { key: string; value: any }[] = config.load?.fields ?? [];
    const get = (key: string) => fields.find((f) => f.key === key)?.value;
    const cacheType = (setting: { checked?: boolean; value?: string } | undefined) => (setting?.checked ? setting.value : undefined);
    return {
      flashAttention: get("llm.load.llama.flashAttention"),
      kCacheType: cacheType(get("llm.load.llama.kCacheQuantizationType")),
      vCacheType: cacheType(get("llm.load.llama.vCacheQuantizationType")),
      parallel: get("llm.load.numParallelSessions"),
      contextLength: get("llm.load.contextLength"),
    };
  } catch {
    return {}; // never tuned in LM Studio
  }
}

// Four layers, the later one wins: LM Studio's defaults (8,192-token context, 4 slots), LM Studio's
// saved tuning for this model, Squid's config file (~/.squid/config.json), then what you typed.
export function settingsFor(tuning: Tuning, configured: Sizes, typed: Sizes): Tuning & { contextLength: number; parallel: number } {
  return {
    ...tuning,
    contextLength: typed.contextLength ?? configured.contextLength ?? tuning.contextLength ?? 8192,
    parallel: typed.parallel ?? configured.parallel ?? tuning.parallel ?? 4,
  };
}

// The command line for llama-server.
export function engineArgs(model: ShelfModel, identifier: string, port: number, apiKey: string, settings: ReturnType<typeof settingsFor>): string[] {
  const args = [
    "--model", model.path,
    "--alias", identifier,
    "--host", "127.0.0.1", "--port", String(port), // reachable from this PC only
    "--api-key", apiKey,
    "--ctx-size", String(settings.contextLength),
    "--parallel", String(settings.parallel), "--kv-unified", // the slots share one context
    "--flash-attn", settings.flashAttention === undefined ? "auto" : settings.flashAttention ? "on" : "off",
    "--no-ui", "--offline", // no web page and no downloads: it only serves this one model
    // No --gpu-layers on purpose: llama.cpp's --fit puts as many layers on the GPU as fit with this
    // much (MiB) to spare, and the rest on the CPU. Overfilling instead makes Windows borrow system
    // RAM: 0.4 tok/s measured. The default 1024 pushed one of the 27B's 64 layers off the GPU on this
    // 16 GB card (30 tok/s); 512 keeps them all there (44 tok/s at a 48K context).
    "--fit-target", "512",
  ];
  if (settings.kCacheType) args.push("--cache-type-k", settings.kCacheType);
  if (settings.vCacheType) args.push("--cache-type-v", settings.vCacheType);
  return args;
}

// Makes sure a model is loaded and returns its engine. If it's already loaded, or another Squid is
// loading it right now, this waits for that one instead of starting a second engine.
export async function load(model: ShelfModel, options: LoadOptions = {}, onWait?: (seconds: number) => void): Promise<Engine> {
  if (!existsSync(ENGINE)) throw new Error("the engine isn't downloaded yet: run `npm run engine:get`");
  const identifier = options.identifier ?? model.key;
  const settings = settingsFor(lmstudioTuning(model), readConfig().models?.[model.key] ?? {}, options);
  mkdirSync(LOADED, { recursive: true });
  mkdirSync(LOGS, { recursive: true });
  const log = join(LOGS, `${fileName(identifier)}.log`);
  const engine: Engine = {
    identifier, key: model.key, pid: 0, port: 0, apiKey: randomBytes(24).toString("hex"),
    contextLength: settings.contextLength, parallel: settings.parallel, sizeBytes: model.sizeBytes,
    architecture: model.architecture, quantization: model.quantization, log,
    loader: process.pid, jit: options.jit ?? false, ttlSeconds: options.ttlSeconds, lastUsed: Date.now(),
  };

  // The record doubles as a lock. Flag "wx" means "create it, but fail if it already exists", and the
  // operating system does that check and the creation as one step. Two loads that start at the same
  // instant can't both get past this line, which is the check-then-act race from step 04 closed.
  try {
    writeFileSync(recordFile(identifier), JSON.stringify(engine, null, 2), { flag: "wx" });
  } catch (err) {
    if ((err as NodeJS.ErrnoException).code !== "EEXIST") throw err;
    running(); // clears the record if it was left behind by a crash
    if (!existsSync(recordFile(identifier))) return load(model, options, onWait);
    return waitUntilReady(identifier, onWait); // loaded, or being loaded by someone else
  }

  try {
    engine.port = await freePort();
  } catch (err) {
    rmSync(recordFile(identifier), { force: true });
    throw err;
  }

  // detached + unref: the engine keeps running after this command exits, like a background service.
  const child = spawn(ENGINE, engineArgs(model, identifier, engine.port, engine.apiKey, settings), {
    detached: true,
    stdio: ["ignore", openSync(log, "w"), openSync(log, "a")],
    windowsHide: true,
  });
  child.unref();
  let exitCode: number | null = null;
  child.on("exit", (code) => (exitCode = code ?? -1));

  engine.pid = child.pid!;
  writeFileSync(recordFile(identifier), JSON.stringify(engine, null, 2));

  // /health answers 503 while the weights stream in from disk, then 200 when the model is ready.
  const started = Date.now();
  while ((await health(engine)) !== "ok") {
    if (exitCode !== null) {
      rmSync(recordFile(identifier), { force: true });
      throw new Error(`the engine stopped while loading (exit code ${exitCode}); see ${log}`);
    }
    if (Date.now() - started > 10 * 60_000) {
      await unload(identifier);
      throw new Error(`still not ready after 10 minutes; see ${log}`);
    }
    onWait?.((Date.now() - started) / 1000);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
  return engine;
}

// Waits for an engine that someone else is loading (or has loaded) to be ready.
async function waitUntilReady(identifier: string, onWait?: (seconds: number) => void): Promise<Engine> {
  const started = Date.now();
  for (;;) {
    const engine = running(true).find((e) => e.identifier === identifier);
    if (!engine) throw new Error(`loading ${identifier} failed elsewhere; try again`);
    if (engine.pid !== 0 && (await health(engine)) === "ok") return engine;
    if (Date.now() - started > 10 * 60_000) throw new Error(`${identifier} still isn't ready after 10 minutes; see ${engine.log}`);
    onWait?.((Date.now() - started) / 1000);
    await new Promise((resolve) => setTimeout(resolve, 250));
  }
}

// Loading on demand, as LM Studio's JIT does. The model gets an idle timer, and any other model that
// was loaded on demand is unloaded first, so only one is ever loaded this way (LM Studio's auto-evict).
export async function loadOnDemand(model: ShelfModel, onWait?: (seconds: number) => void): Promise<Engine> {
  for (const other of running(true)) {
    if (other.jit && other.key !== model.key && other.pid !== 0) await unload(other.identifier);
  }
  return load(model, { jit: true, ttlSeconds: (readConfig().jitTtlMinutes ?? 60) * 60 }, onWait);
}

// Notes that an engine just answered, which restarts its idle timer.
export function touch(engine: Engine): void {
  try {
    const record: Engine = JSON.parse(readFileSync(recordFile(engine.identifier), "utf8"));
    writeFileSync(recordFile(engine.identifier), JSON.stringify({ ...record, lastUsed: Date.now() }, null, 2));
  } catch {
    // unloaded in the meantime: nothing to time
  }
}

// Seconds until an unused engine is unloaded; undefined when it has no timer (loaded with `squid load`).
export function secondsLeft(engine: Engine, now = Date.now()): number | undefined {
  return engine.ttlSeconds === undefined ? undefined : engine.ttlSeconds - (now - engine.lastUsed) / 1000;
}

// Every engine whose record points at a live llama-server. Records left behind by a crash are removed.
// `quick` only asks "is that process ID alive?" and skips asking Windows for the program's name
// (~100 ms). The front door uses it on every request, where the worst case is a "not answering" error.
// `ps` and `unload`, which can end processes, always check the name.
export function running(quick = false): Engine[] {
  if (!existsSync(LOADED)) return [];
  const engines: Engine[] = [];
  for (const name of readdirSync(LOADED)) {
    const engine: Engine = JSON.parse(readFileSync(join(LOADED, name), "utf8"));
    // pid 0 means "being set up": alive as long as the Squid that's loading it is.
    const alive = engine.pid === 0 ? isAlive(engine.loader) : quick ? isAlive(engine.pid) : isEngine(engine.pid);
    if (alive) engines.push(engine);
    else rmSync(join(LOADED, name), { force: true });
  }
  return engines.sort((a, b) => (a.identifier < b.identifier ? -1 : 1));
}

// LOADING while the weights are read, IDLE when ready, GENERATING while any slot is answering (LM Studio's word).
export async function status(engine: Engine): Promise<"LOADING" | "IDLE" | "GENERATING"> {
  if (engine.pid === 0 || (await health(engine)) !== "ok") return "LOADING";
  const res = await fetch(`http://127.0.0.1:${engine.port}/slots`, { headers: { Authorization: `Bearer ${engine.apiKey}` } });
  const slots = (await res.json()) as { is_processing: boolean }[];
  return slots.some((slot) => slot.is_processing) ? "GENERATING" : "IDLE";
}

// Stops the engine and waits until its process is gone, so the GPU memory is really free.
export async function unload(identifier: string): Promise<boolean> {
  const engine = running().find((e) => e.identifier === identifier);
  if (!engine || engine.pid === 0) return false; // not loaded, or still being set up by another Squid
  process.kill(engine.pid);
  for (let waited = 0; isEngine(engine.pid) && waited < 10_000; waited += 100) {
    await new Promise((resolve) => setTimeout(resolve, 100));
  }
  rmSync(recordFile(identifier), { force: true });
  return true;
}

async function health(engine: Engine): Promise<"ok" | "loading" | "down"> {
  try {
    const res = await fetch(`http://127.0.0.1:${engine.port}/health`, { signal: AbortSignal.timeout(2000) });
    return res.ok ? "ok" : "loading";
  } catch {
    return "down"; // not listening yet, or gone
  }
}

// Signal 0 delivers nothing; it only fails if no process has this ID.
export function isAlive(pid: number): boolean {
  try {
    process.kill(pid, 0);
    return true;
  } catch {
    return false;
  }
}

// Is this process ID still a llama-server? Windows reuses IDs, so "some process has this ID" isn't
// enough: without the name check, `unload` after a crash could end an unrelated program.
function isEngine(pid: number): boolean {
  try {
    if (process.platform === "win32") {
      const row = execFileSync("tasklist", ["/FI", `PID eq ${pid}`, "/FO", "CSV", "/NH"], { encoding: "utf8", windowsHide: true });
      return row.toLowerCase().startsWith('"llama-server.exe"');
    }
    return execFileSync("ps", ["-p", String(pid), "-o", "comm="], { encoding: "utf8" }).trim().endsWith("llama-server");
  } catch {
    return false;
  }
}

// Asks the operating system for a port nobody is using: bind to port 0, read which one we got, let it go.
function freePort(): Promise<number> {
  return new Promise((resolve, reject) => {
    const server = createServer();
    server.on("error", reject);
    server.listen(0, "127.0.0.1", () => {
      const { port } = server.address() as AddressInfo;
      server.close(() => resolve(port));
    });
  });
}

// scripts/get-engine.ps1 writes the build tag (b11580) next to the engine.
function engineVersion(): string {
  try {
    return readFileSync(new URL("../runtime/llama.cpp/VERSION", import.meta.url), "utf8").trim();
  } catch {
    return "not downloaded";
  }
}

const fileName = (identifier: string) => encodeURIComponent(identifier); // "qwen/qwen3.5-9b" → "qwen%2Fqwen3.5-9b"
const recordFile = (identifier: string) => join(LOADED, `${fileName(identifier)}.json`);
