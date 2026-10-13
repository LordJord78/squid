import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { feed, forward, HALF, loadModel, logprobs, newState, rope, top, type Model } from "../src/transformer.ts";
import { SMOLLM } from "../golden/capture-forward.ts";
import { tempDir, writeGguf } from "./helpers.ts";

type Case = { text: string; promptIds: number[]; generated: number[]; steps: { id: number; logprob: number }[][] };
const missing = !existsSync(SMOLLM) && "SmolLM2-135M isn't on this PC";

// Runs every golden prompt through Squid's engine. At each step: Squid's top two tokens, and for each of
// llama.cpp's top 5, the log-probability llama.cpp gave it and the one Squid gives it. Then the token
// llama.cpp chose is fed in, so both engines always continue from the same text.
type Compared = { at: string; ours: number[]; pairs: { id: number; real: number; ours: number }[] };
let compared: Compared[] | undefined;
function compare(model: Model): Compared[] {
  if (compared) return compared;
  compared = [];
  const cases: Case[] = JSON.parse(readFileSync(new URL("../golden/llamacpp/forward-smollm2.json", import.meta.url), "utf8")).cases;
  for (const c of cases) {
    const state = newState(model, 64);
    let logits = feed(model, state, c.promptIds);
    c.steps.forEach((theirs, i) => {
      const lp = logprobs(logits);
      compared!.push({
        at: `${JSON.stringify(c.text)} step ${i}`,
        ours: top(logits, 2).map((t) => t.id),
        pairs: theirs.map((t) => ({ id: t.id, real: t.logprob, ours: lp[t.id] })),
      });
      logits = forward(model, state, c.generated[i]);
    });
  }
  return compared;
}

let model: Model | undefined;
const smollm = () => (model ??= loadModel(SMOLLM));

test("the same next token as llama.cpp, except on near-ties, where it's the same two tokens swapped", { skip: missing }, () => {
  for (const { ours, pairs, at } of compare(smollm())) {
    if (ours[0] === pairs[0].id) continue;
    const margin = pairs[0].real - pairs[1].real;
    assert.ok(margin < 0.2, `${at}: llama.cpp was sure (margin ${margin.toFixed(3)}) and Squid chose differently`);
    assert.deepEqual(new Set(ours), new Set([pairs[0].id, pairs[1].id]), `${at}: not the same two candidates`);
  }
});

test("probabilities as close to llama.cpp's CPU as llama.cpp's own GPU build is (average gap 0.090, worst 0.52)", { skip: missing }, () => {
  const gaps = compare(smollm()).flatMap(({ pairs }) => pairs.map((p) => Math.abs(p.ours - p.real)));
  const average = gaps.reduce((a, b) => a + b, 0) / gaps.length;
  assert.ok(average < 0.15, `average log-probability gap ${average.toFixed(3)}`);
  assert.ok(Math.max(...gaps) < 0.6, `worst gap ${Math.max(...gaps).toFixed(3)}`);
});

test("SmolLM2's shape, read from its file", { skip: missing }, () => {
  const { config, layers } = smollm();
  assert.deepEqual(
    [config.dim, config.layers, config.heads, config.kvHeads, config.headDim, config.ffn, config.vocab, config.ropeBase],
    [576, 30, 9, 3, 64, 1536, 49152, 100000],
  );
  assert.equal(layers.length, 30);
  assert.equal(smollm().output, smollm().embed, "tied: the token table also scores the next token");
});

test("the F16 table decodes all 65,536 values exactly as JavaScript's own Float16Array", () => {
  const real = new (globalThis as any).Float16Array(new Uint16Array(65536).map((_, i) => i).buffer);
  for (let i = 0; i < 65536; i++) if (!Number.isNaN(real[i])) assert.equal(HALF[i], real[i], `bits 0x${i.toString(16)}`);
  assert.equal(HALF[0x3c00], 1);
  assert.equal(HALF[0xc000], -2);
  assert.equal(HALF[0x7bff], 65504, "the largest F16");
});

test("RoPE: position 0 changes nothing, and turning never changes a vector's length", () => {
  const v = Float32Array.from({ length: 64 }, (_, i) => Math.sin(i + 1));
  const at0 = Float32Array.from(v);
  rope(at0, 0, 64, 10000);
  assert.deepEqual(at0, v);
  const length = (x: Float32Array) => Math.hypot(...x);
  const at7 = Float32Array.from(v);
  rope(at7, 7, 64, 10000);
  assert.ok(Math.abs(length(at7) - length(v)) < 1e-5);
  assert.notDeepEqual(at7, v);
});

test("top() gives log-probabilities: they turn back into probabilities that add up to 1", () => {
  const all = top(Float32Array.from([2, 1, 0, -1]), 4);
  assert.deepEqual(all.map((t) => t.id), [0, 1, 2, 3]);
  // 32-bit floats keep about 7 significant digits, so "adds up to 1" means to within a millionth.
  assert.ok(Math.abs(all.reduce((sum, t) => sum + Math.exp(t.logprob), 0) - 1) < 1e-6);
});

test("models it can't run are refused with a reason", () => {
  const path = join(tempDir(), "qwen.gguf");
  writeGguf(path, { "general.architecture": "qwen35" });
  assert.throws(() => loadModel(path), /runs Llama-style models; this one is "qwen35"/);
});
