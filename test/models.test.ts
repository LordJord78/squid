import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync, writeFileSync, mkdirSync, existsSync } from "node:fs";
import { join } from "node:path";
import { listModels, catalogNames, modelRoots } from "../src/models.ts";
import { tempDir, writeGguf } from "./helpers.ts";

// A fake models folder laid out like LM Studio's: <root>/<publisher>/<repo>/<file>.gguf
function fakeShelf() {
  const root = tempDir();
  writeGguf(join(root, "acme", "Tiny-7B-GGUF", "Tiny-7B-Q4_K_M.gguf"),
    { "general.architecture": "llama", "general.size_label": "7B", "general.file_type": 15 }, 5000);
  writeGguf(join(root, "acme", "Tiny-7B-GGUF", "mmproj-Tiny-7B-F16.gguf"),
    { "general.architecture": "clip" }, 1000);
  writeGguf(join(root, "nomic-ai", "embedder-v1-GGUF", "embedder-v1.Q4_K_M.gguf"),
    { "general.architecture": "nomic-bert", "nomic-bert.pooling_type": 1 }, 300);
  return root;
}

test("one model per folder, with its vision add-on counted in", () => {
  const [tiny] = listModels([fakeShelf()], new Map()).filter((m) => m.type === "llm");
  assert.equal(tiny.key, "tiny-7b");
  assert.equal(tiny.params, "7B");
  assert.equal(tiny.vision, true);
  assert.equal(tiny.sizeBytes, 6000, "model 5000 + add-on 1000");
});

test("embedding models are recognised and named like LM Studio names them", () => {
  const embedder = listModels([fakeShelf()], new Map()).find((m) => m.type === "embedding");
  assert.equal(embedder?.key, "text-embedding-embedder-v1");
});

test("a catalog manifest gives a folder its short name, and its own files count toward the size", () => {
  const hub = tempDir();
  mkdirSync(join(hub, "acme", "tiny"), { recursive: true });
  const manifest = JSON.stringify({
    owner: "acme", name: "tiny",
    dependencies: [{ sources: [{ type: "huggingface", user: "acme", repo: "Tiny-7B-GGUF" }] }],
  });
  writeFileSync(join(hub, "acme", "tiny", "manifest.json"), manifest);
  const [tiny] = listModels([fakeShelf()], catalogNames(hub));
  assert.equal(tiny.key, "acme/tiny");
  assert.equal(tiny.catalog, true);
  assert.equal(tiny.sizeBytes, 6000 + Buffer.byteLength(manifest));
});

// Side by side with `lms ls --json`: same models, same facts, same sizes to the byte.
test("matches the real LM Studio shelf", { skip: !existsSync("D:\\LMStudio\\models") && "model drive not connected" }, () => {
  const golden = JSON.parse(readFileSync(new URL("../golden/lmstudio/lms-ls.json", import.meta.url), "utf8"));
  // Entries whose path starts with "<device id>:" live on a linked machine (LM Link), not here.
  const local = golden.filter((m: { path: string }) => !/^[0-9a-f]{32}:/.test(m.path));
  const ours = new Map(listModels(modelRoots()).map((m) => [m.key, m]));
  assert.deepEqual([...ours.keys()].sort(), local.map((m: { modelKey: string }) => m.modelKey).sort());
  for (const real of local) {
    const mine = ours.get(real.modelKey)!;
    assert.equal(mine.type, real.type, real.modelKey);
    assert.equal(mine.architecture, real.architecture, real.modelKey);
    assert.equal(mine.params ?? "", real.paramsString ?? "", real.modelKey);
    assert.equal(mine.sizeBytes, real.sizeBytes, real.modelKey);
    assert.equal(mine.vision, real.vision ?? false, real.modelKey);
  }
});
