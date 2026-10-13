import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { candidates, DEFAULTS, mt19937, pick, type Sampling } from "../src/sampler.ts";

// Recorded by `npm run capture:sampling` from llama.cpp's own server, on the CPU.
const golden = JSON.parse(readFileSync(new URL("../golden/llamacpp/sampling-smollm2.json", import.meta.url), "utf8"));
const settings = (s: any): Sampling => ({ temperature: s.temperature ?? DEFAULTS.temperature, topK: s.top_k ?? DEFAULTS.topK, topP: s.top_p ?? DEFAULTS.topP, minP: s.min_p ?? DEFAULTS.minP });

test("the Mersenne Twister gives the number the C++ standard checks for", () => {
  const { next } = mt19937(5489); // std::mt19937's default seed
  let n = 0;
  for (let i = 0; i < 10_000; i++) n = next();
  assert.equal(n, 4123659995, "the 10,000th number");
});

test("each filter, on five made-up scores", () => {
  const logits = Float32Array.from([2, 1, 0, -1, -3]);
  const none = { topK: 0, topP: 1, minP: 0, temperature: 1 };
  const ids = (s: Partial<Sampling>) => candidates(logits, { ...none, ...s }).map((c) => c.id);
  assert.deepEqual(ids({}), [0, 1, 2, 3, 4], "no filters: all five, best first");
  assert.deepEqual(ids({ topK: 2 }), [0, 1]);
  assert.deepEqual(ids({ topP: 0.7 }), [0, 1], "64% + 24% = 88%, enough for 70%");
  assert.deepEqual(ids({ minP: 0.3 }), [0, 1], "token 2 is only e⁻² = 14% as likely as token 0");
  assert.deepEqual(ids({ minP: 0.3, temperature: 5 }), [0, 1], "temperature comes last, so it can't bring back what min-p dropped");
  assert.deepEqual(ids({ temperature: 0 }), [0], "0 = always the favourite");
  const chances = (temperature: number) => candidates(logits, { ...none, topK: 2, temperature }).map((c) => +c.p.toFixed(3));
  assert.deepEqual(chances(1), [0.731, 0.269], "e² : e¹");
  assert.deepEqual(chances(0.5), [0.881, 0.119], "a low temperature favours the favourite");
  assert.deepEqual(chances(2), [0.622, 0.378], "a high one evens them out");
});

test("the dice: a random number from 0 to 1 lands in one token's share of the line", () => {
  const list = [{ id: 7, p: 0.5 }, { id: 8, p: 0.3 }, { id: 9, p: 0.2 }];
  assert.equal(pick(list, () => 0.1), 7);
  assert.equal(pick(list, () => 0.6), 8, "0.5 to 0.8 is token 8's");
  assert.equal(pick(list, () => 0.95), 9);
});

test("the same candidates as llama.cpp, from the same scores", () => {
  for (const c of golden.chains) {
    // llama.cpp's own top 50 log-probabilities. They're its scores minus one constant, which no filter
    // notices; tokens outside the top 50 never survive top-k 40.
    const logits = new Float32Array(49152).fill(-Infinity);
    for (const t of golden.raw[c.text]) logits[t.id] = t.logprob;
    const ours = candidates(logits, settings(c.settings));
    const where = `${JSON.stringify(c.text)} ${JSON.stringify(c.settings)}`;
    assert.deepEqual(ours.map((o) => o.id), c.candidates.map((t: any) => t.id), `${where}: the same tokens, in the same order`);
    ours.forEach((o, i) => assert.ok(Math.abs(o.p - c.candidates[i].p) < 1e-5, `${where}: chances ${o.p} vs ${c.candidates[i].p}`));
  }
  assert.equal(golden.chains.length, 21);
});

test("the same dice as llama.cpp: with its candidates and its seed, every pick is the same", () => {
  let picks = 0;
  for (const c of golden.seeded) {
    const { random } = mt19937(c.seed);
    c.steps.forEach((list: { id: number; p: number }[], i: number) => {
      assert.equal(pick(list, random), c.tokens[i], `${JSON.stringify(c.text)} seed ${c.seed}, token ${i}`);
      picks++;
    });
  }
  assert.equal(picks, 216, "3 prompts × 3 seeds × 24 tokens");
});
