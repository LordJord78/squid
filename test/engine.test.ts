import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { ENGINE, engineArgs, forget, lmstudioTuning, load, running, secondsLeft, settingsFor, THINKING_BUDGET, unload, type Engine } from "../src/engine.ts";
import { SQUID_HOME } from "../src/config.ts";
import { listModels, modelRoots, type ShelfModel } from "../src/models.ts";
import { tempDir } from "./helpers.ts";

const fakeModel = (path: string): ShelfModel => ({
  key: "tiny-7b", type: "llm", publisher: "acme", architecture: "llama", sizeBytes: 1, vision: false, toolUse: false, catalog: false, variants: 1, path,
});

test("reads the tuning LM Studio saved for a model", () => {
  const configDir = tempDir();
  mkdirSync(join(configDir, "acme", "Tiny-7B-GGUF"), { recursive: true });
  // The same shape as LM Studio's file for the 27B (fields this test doesn't use are left out)
  writeFileSync(join(configDir, "acme", "Tiny-7B-GGUF", "Tiny-7B-Q4_K_M.gguf.json"), JSON.stringify({
    load: { fields: [
      { key: "llm.load.llama.flashAttention", value: true },
      { key: "llm.load.numParallelSessions", value: 1 },
      { key: "llm.load.llama.kCacheQuantizationType", value: { checked: true, value: "q4_0" } },
      { key: "llm.load.llama.vCacheQuantizationType", value: { checked: false, value: "q8_0" } },
    ] },
  }));
  const tuning = lmstudioTuning(fakeModel(join("D:", "models", "acme", "Tiny-7B-GGUF", "Tiny-7B-Q4_K_M.gguf")), configDir);
  assert.deepEqual(tuning, { flashAttention: true, kCacheType: "q4_0", vCacheType: undefined, parallel: 1, contextLength: undefined },
    "an unticked setting (checked: false) is not applied");
  assert.deepEqual(lmstudioTuning(fakeModel(join("D:", "models", "acme", "Other", "x.gguf")), configDir), {}, "never tuned");
});

test("settings: LM Studio's defaults, its saved tuning, Squid's config, then what you typed", () => {
  assert.deepEqual(settingsFor({}, {}, {}), { contextLength: 8192, parallel: 4, engineArgs: [] });
  assert.deepEqual(settingsFor({ parallel: 1, kCacheType: "q4_0" }, {}, {}), { parallel: 1, kCacheType: "q4_0", contextLength: 8192, engineArgs: [] });
  assert.deepEqual(settingsFor({ parallel: 1 }, { contextLength: 49152 }, {}), { parallel: 1, contextLength: 49152, engineArgs: [] }, "config file beats LM Studio");
  assert.deepEqual(settingsFor({ parallel: 1 }, { contextLength: 49152 }, { contextLength: 4096, parallel: 2 }), { parallel: 2, contextLength: 4096, engineArgs: [] }, "typed beats all");
  assert.deepEqual(settingsFor({}, { reasoningBudget: 8192, engineArgs: ["-cms", "2048"] }, { reasoningBudget: -1 }), { contextLength: 8192, parallel: 4, reasoningBudget: -1, engineArgs: ["-cms", "2048"] });
});

test("a thinking budget and extra flags reach llama-server's command line", () => {
  const args = engineArgs(fakeModel("m.gguf"), "m", 50000, "k", settingsFor({}, { reasoningBudget: 8192, engineArgs: ["--checkpoint-min-step", "2048"] }, {}));
  assert.equal(args[args.indexOf("--reasoning-budget") + 1], "8192");
  assert.match(args[args.indexOf("--reasoning-budget-message") + 1], /Time to act/);
  assert.deepEqual(args.slice(-2), ["--checkpoint-min-step", "2048"]);
  const unlimited = engineArgs(fakeModel("m.gguf"), "m", 50000, "k", settingsFor({}, { reasoningBudget: -1 }, {}));
  assert.ok(!unlimited.includes("--reasoning-budget"), "-1: no budget, llama.cpp's default");
  const plain = engineArgs(fakeModel("m.gguf"), "m", 50000, "k", settingsFor({}, {}, {}));
  assert.equal(plain[plain.indexOf("--reasoning-budget") + 1], String(THINKING_BUDGET), "unset: Squid's default budget");
});

test("unload removes an engine's record only if it's still that engine's", () => {
  // Step 21: while unload waited for an engine to exit, a chat loaded the model again under the same name;
  // removing the record by name alone left the new engine running, invisible to ps and unload.
  const identifier = "squid-forget-test";
  const file = join(SQUID_HOME, "loaded", `${identifier}.json`);
  mkdirSync(dirname(file), { recursive: true });
  try {
    writeFileSync(file, JSON.stringify({ identifier, pid: 222 })); // the newer engine's record
    forget({ identifier, pid: 111 }); // the old one has exited
    assert.ok(existsSync(file), "the newer engine's record stays");
    forget({ identifier, pid: 222 });
    assert.ok(!existsSync(file), "its own record goes");
    forget({ identifier, pid: 222 }); // and no record at all is fine
  } finally {
    rmSync(file, { force: true });
  }
});

test("the idle timer counts from the last answer", () => {
  const engine = { ttlSeconds: 3600, lastUsed: 1_000_000 } as Engine;
  assert.equal(secondsLeft(engine, 1_000_000), 3600);
  assert.equal(secondsLeft(engine, 1_000_000 + 10 * 60_000), 3000, "10 minutes later");
  assert.equal(secondsLeft({ lastUsed: 0 } as Engine), undefined, "loaded with squid load: no timer");
});

test("the engine answers only on this PC, only to Squid, and never downloads", () => {
  const args = engineArgs(fakeModel("m.gguf"), "tiny-7b", 50000, "secret", settingsFor({ kCacheType: "q4_0" }, {}, {}));
  const flag = (name: string) => args[args.indexOf(name) + 1];
  assert.equal(flag("--host"), "127.0.0.1");
  assert.equal(flag("--api-key"), "secret");
  assert.equal(flag("--cache-type-k"), "q4_0");
  assert.equal(flag("--fit-target"), "512");
  assert.ok(args.includes("--offline") && args.includes("--no-ui"));
  assert.ok(!args.includes("--gpu-layers") && !args.includes("-ngl"), "--fit decides how many layers go on the GPU");
});

// Side by side with `lms ps`: load the same model with the same context, and the tables should match.
const ready = existsSync(ENGINE) && existsSync("D:\\LMStudio\\models");
test("load, ps and unload match LM Studio", { skip: !ready && "engine not downloaded or model drive not connected", timeout: 300_000 }, async (t) => {
  const model = listModels(modelRoots()).find((m) => m.key === "qwen/qwen3.5-9b");
  assert.ok(model, "the 9B is on the shelf");
  t.after(() => unload(model.key));

  const engine = await load(model, { contextLength: 8192 });
  assert.ok(running().some((e) => e.identifier === model.key));

  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  const rows = (text: string) => text.split(/\r?\n/).map((line) => line.trimEnd()).filter((line) => /^(IDENTIFIER|qwen\/qwen3\.5-9b)\s/.test(line));
  const golden = readFileSync(new URL("../golden/lmstudio/lms-ps.txt", import.meta.url), "utf8");
  assert.deepEqual(rows(execFileSync(process.execPath, [cli, "ps"], { encoding: "utf8" })), rows(golden));

  // The engine is locked: without Squid's key, the API refuses.
  assert.equal((await fetch(`http://127.0.0.1:${engine.port}/v1/models`)).status, 401);

  assert.equal(await unload(model.key), true);
  assert.ok(!running().some((e) => e.identifier === model.key), "gone from ps");
  assert.equal(await fetch(`http://127.0.0.1:${engine.port}/health`).then(() => "up", () => "down"), "down", "process stopped");
});

// The race from step 04: two loads of the same model at the same instant.
test("two loads at once start one engine", { skip: !ready && "engine not downloaded or model drive not connected", timeout: 300_000 }, async (t) => {
  const model = listModels(modelRoots()).find((m) => m.key === "qwen/qwen3.5-9b")!;
  t.after(() => unload(model.key));
  const [first, second] = await Promise.all([load(model, { contextLength: 8192 }), load(model, { contextLength: 8192 })]);
  assert.equal(first.pid, second.pid, "both got the same engine");
  const engines = execFileSync("tasklist", ["/FI", "IMAGENAME eq llama-server.exe", "/FO", "CSV", "/NH"], { encoding: "utf8" });
  assert.equal(engines.trim().split("\n").filter((row) => row.startsWith('"llama-server.exe"')).length, 1, "one llama-server running");
  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  assert.match(execFileSync(process.execPath, [cli, "load", model.key], { encoding: "utf8" }), /is already loaded/);
});