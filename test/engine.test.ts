import { test } from "node:test";
import assert from "node:assert/strict";
import { execFileSync } from "node:child_process";
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { ENGINE, engineArgs, lmstudioTuning, load, running, settingsFor, unload } from "../src/engine.ts";
import { listModels, modelRoots, type ShelfModel } from "../src/models.ts";
import { tempDir } from "./helpers.ts";

const fakeModel = (path: string): ShelfModel => ({
  key: "tiny-7b", type: "llm", publisher: "acme", architecture: "llama", sizeBytes: 1, vision: false, catalog: false, variants: 1, path,
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

test("settings: LM Studio's defaults, then the saved tuning, then what you typed", () => {
  assert.deepEqual(settingsFor({}, {}), { contextLength: 8192, parallel: 4 });
  assert.deepEqual(settingsFor({ parallel: 1, kCacheType: "q4_0" }, {}), { parallel: 1, kCacheType: "q4_0", contextLength: 8192 });
  assert.deepEqual(settingsFor({ parallel: 1 }, { contextLength: 49152, parallel: 2 }), { parallel: 2, contextLength: 49152 });
});

test("the engine answers only on this PC, only to Squid, and never downloads", () => {
  const args = engineArgs(fakeModel("m.gguf"), "tiny-7b", 50000, "secret", settingsFor({ kCacheType: "q4_0" }, {}));
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
