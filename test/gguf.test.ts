import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync, readFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { readGgufHeader, describeModel } from "../src/gguf.ts";
import { tempDir, writeGguf } from "./helpers.ts";

function tinyGguf(): string {
  return writeGguf(join(tempDir(), "tiny.gguf"), {
    "general.architecture": "llama",
    "general.name": "Tiny Test",
    "general.file_type": 15,
    "llama.context_length": 4096,
    "llama.block_count": 2n, // stored as u64 on purpose: real files use both widths
    "tokenizer.ggml.add_bos_token": true,
    "tokenizer.ggml.tokens": Array.from({ length: 100 }, (_, i) => `tok${i}`),
  });
}

test("reads every metadata type from a hand-built file", () => {
  const h = readGgufHeader(tinyGguf());
  assert.equal(h.version, 3);
  assert.equal(h.metadata.get("general.name"), "Tiny Test");
  assert.equal(h.metadata.get("tokenizer.ggml.add_bos_token"), true);
  assert.equal(h.metadata.get("llama.block_count"), 2n);
  // The 100-word vocabulary is walked past, not kept, but its length is known.
  assert.deepEqual(h.metadata.get("tokenizer.ggml.tokens"), { arrayOf: 8, length: 100 });
  assert.equal(h.headerBytes, h.fileBytes, "a file with no tensors is all header");
});

test("turns metadata into the facts LM Studio shows", () => {
  const m = describeModel(readGgufHeader(tinyGguf()));
  assert.equal(m.architecture, "llama");
  assert.equal(m.quantization, "Q4_K_M");
  assert.equal(m.contextLength, 4096);
  assert.equal(m.layers, 2);
  assert.equal(m.embedding, false);
});

test("refuses a file that isn't GGUF", () => {
  const path = join(tempDir(), "fake.gguf");
  writeFileSync(path, "PK\u0003\u0004 this is a zip file");
  assert.throws(() => readGgufHeader(path), /not a GGUF file/);
});

// Side by side with the real LM Studio: same model file, same answers.
const qwen9b = "D:\\LMStudio\\models\\lmstudio-community\\Qwen3.5-9B-GGUF\\Qwen3.5-9B-Q4_K_M.gguf";
test("matches what LM Studio reported for qwen3.5-9b", { skip: !existsSync(qwen9b) && "model drive not connected" }, () => {
  const golden = JSON.parse(readFileSync(new URL("../golden/lmstudio/api-v0-models-loaded.json", import.meta.url), "utf8"));
  const lmStudio = golden.data.find((m: { id: string }) => m.id === "qwen/qwen3.5-9b");
  const ours = describeModel(readGgufHeader(qwen9b));
  assert.equal(ours.architecture, lmStudio.arch);
  assert.equal(ours.quantization, lmStudio.quantization);
  assert.equal(ours.contextLength, lmStudio.max_context_length);
});
