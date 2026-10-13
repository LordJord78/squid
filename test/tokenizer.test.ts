import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, globSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { BYTE_CHAR, decode, encode, loadVocab, piece } from "../src/tokenizer.ts";
import { VOCABS } from "../golden/capture-tokens.ts";
import { tempDir, writeGguf } from "./helpers.ts";

type Case = { text: string; special?: false; ids: number[] };
const golden = (name: string): Case[] => JSON.parse(readFileSync(new URL(`../golden/llamacpp/tokens-${name}.json`, import.meta.url), "utf8")).cases;

for (const [name, path] of Object.entries(VOCABS)) {
  const missing = !existsSync(path) && `${path} isn't on this PC`;

  test(`${name}: every text becomes exactly the token ids llama.cpp gives`, { skip: missing }, () => {
    const vocab = loadVocab(path);
    const wrong = golden(name).filter((c) => JSON.stringify(encode(vocab, c.text, { special: c.special ?? true })) !== JSON.stringify(c.ids));
    assert.deepEqual(wrong.map((c) => c.text), [], "these texts were split differently from llama.cpp");
  });

  test(`${name}: decoding the ids gives back the exact text`, { skip: missing }, () => {
    const vocab = loadVocab(path);
    for (const c of golden(name)) assert.equal(decode(vocab, c.ids), c.text);
  });
}

test("your 9B and 27B read the same 248,320 tokens", { skip: !existsSync(VOCABS["qwen3.5"]) && "the 9B isn't on this PC" }, () => {
  const big = globSync("D:/LMStudio/models/**/Qwen3.8-27B*.gguf")[0];
  if (!big) return;
  const [nine, twentySeven] = [loadVocab(VOCABS["qwen3.5"]), loadVocab(big)];
  assert.equal(nine.tokens.length, 248_320);
  assert.deepEqual(nine.tokens, twentySeven.tokens);
  assert.deepEqual([...nine.ranks.keys()], [...twentySeven.ranks.keys()], "and the same merge rules, in the same order");
});

test("GPT-2's byte characters: a space is Ġ, a newline Ċ, and printable bytes stay themselves", () => {
  assert.equal(BYTE_CHAR[32], "Ġ");
  assert.equal(BYTE_CHAR[10], "Ċ");
  assert.equal(BYTE_CHAR["A".charCodeAt(0)], "A");
  assert.equal(new Set(BYTE_CHAR).size, 256, "every byte has its own character");
});

test("a token that's only part of a character is shown as bytes", { skip: !existsSync(VOCABS["qwen3.5"]) && "the 9B isn't on this PC" }, () => {
  const vocab = loadVocab(VOCABS["qwen3.5"]);
  const ids = encode(vocab, "🇯🇵");
  assert.equal(decode(vocab, ids), "🇯🇵", "together they're the flag");
  assert.ok(ids.map((id) => piece(vocab, id)).some((p) => /^(<[0-9A-F]{2}>)+$/.test(p)), "alone, some are just bytes");
});

// A made-up vocabulary of six tokens shows the one rule BPE runs on: the earliest merge rule wins.
function tinyVocab(merges: string[]) {
  const path = join(tempDir(), "tiny.gguf");
  writeGguf(path, { "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "gpt2", "tokenizer.ggml.tokens": ["a", "b", "c", "ab", "bc"], "tokenizer.ggml.merges": merges });
  return loadVocab(path);
}
test("BPE: the merge rule that comes first wins, so the same letters can become different tokens", () => {
  const abFirst = tinyVocab(["a b", "b c"]);
  const bcFirst = tinyVocab(["b c", "a b"]);
  assert.deepEqual(encode(abFirst, "abc").map((id) => abFirst.tokens[id]), ["ab", "c"]);
  assert.deepEqual(encode(bcFirst, "abc").map((id) => bcFirst.tokens[id]), ["a", "bc"]);
});

test("vocabularies Squid can't read yet are refused with a reason", () => {
  const spm = join(tempDir(), "spm.gguf");
  writeGguf(spm, { "tokenizer.ggml.model": "llama", "tokenizer.ggml.tokens": ["a"] });
  assert.throws(() => loadVocab(spm), /byte-level BPE \("gpt2"\) vocabularies; this model's is "llama"/);
  const odd = join(tempDir(), "odd.gguf");
  writeGguf(odd, { "tokenizer.ggml.model": "gpt2", "tokenizer.ggml.pre": "klingon", "tokenizer.ggml.tokens": ["a"] });
  assert.throws(() => loadVocab(odd), /doesn't know the "klingon" pre-tokenizer/);
});
