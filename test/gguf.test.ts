import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, writeFileSync, readFileSync, existsSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { readGgufHeader, describeModel } from "../src/gguf.ts";

// Build a tiny GGUF file by hand, byte by byte, so the reader has something exact to check against.
const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const u64 = (n: number) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s, "utf8")]);
const kv = (key: string, type: number, value: Buffer) => Buffer.concat([str(key), u32(type), value]);

function tinyGguf(): string {
  const vocab = Array.from({ length: 100 }, (_, i) => `tok${i}`);
  const entries = [
    kv("general.architecture", 8, str("llama")),
    kv("general.name", 8, str("Tiny Test")),
    kv("general.file_type", 4, u32(15)),
    kv("llama.context_length", 4, u32(4096)),
    kv("llama.block_count", 10, u64(2)), // stored as u64 on purpose: real files use both widths
    kv("tokenizer.ggml.add_bos_token", 7, Buffer.from([1])),
    kv("tokenizer.ggml.tokens", 9, Buffer.concat([u32(8), u64(vocab.length), ...vocab.map(str)])),
  ];
  const file = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(0), u64(entries.length), ...entries]);
  const path = join(mkdtempSync(join(tmpdir(), "squid-")), "tiny.gguf");
  writeFileSync(path, file);
  return path;
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
});

test("refuses a file that isn't GGUF", () => {
  const path = join(mkdtempSync(join(tmpdir(), "squid-")), "fake.gguf");
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
