// Sampling: how the next token is chosen from the model's scores. Always taking the likeliest one
// (step 16's "greedy") gives flat, repetitive text, so engines roll weighted dice instead, after first
// throwing out the unlikely tokens. This is llama.cpp's way, filter for filter and in its order (its
// src/llama-sampler.cpp), down to its random number generator: with the same seed and the same scores,
// Squid picks the same tokens.
import { highest } from "./transformer.ts";

export type Sampling = { temperature: number; topK: number; topP: number; minP: number; seed?: number };

// llama-server's defaults (tools/server/README.md), used for whatever a request doesn't set.
export const DEFAULTS: Sampling = { temperature: 0.8, topK: 40, topP: 0.95, minP: 0.05 };

export type Candidate = { id: number; logit: number; p: number };

// llama.cpp's chain: top-k → top-p → min-p → temperature. Returns the tokens still in the running,
// likeliest first, each with its chance of being picked.
export function candidates(logits: Float32Array, s: Sampling): Candidate[] {
  // top-k: keep the k highest scores (0 = keep all 49,152, sorted).
  const ids = s.topK > 0 ? highest(logits, s.topK) : [...logits.keys()].sort((a, b) => logits[b] - logits[a]);
  let list = ids.map((id) => ({ id, logit: logits[id], p: 0 }));

  // top-p ("nucleus"): keep the fewest tokens whose chances add up to p. The chances are worked out
  // among the survivors of top-k, not the whole vocabulary.
  if (s.topP < 1) {
    softmax(list, 1);
    let total = 0;
    const keep = list.findIndex((c) => (total += c.p) >= s.topP);
    if (keep !== -1) list = list.slice(0, keep + 1);
  }

  // min-p: drop tokens less than min_p times as likely as the top one. In scores (logs), "p_i ≥
  // min_p × p_top" is "logit_i ≥ logit_top + ln(min_p)".
  if (s.minP > 0) {
    const floor = list[0].logit + Math.log(s.minP);
    list = list.filter((c, i) => i === 0 || c.logit >= floor);
  }

  // Temperature, last: divide the scores by it. Below 1 the favourite gets more likely, above 1 the
  // chances even out. 0 means "always the favourite".
  if (s.temperature <= 0) list = list.slice(0, 1);
  softmax(list, s.temperature > 0 ? s.temperature : 1);
  return list;
}

function softmax(list: Candidate[], temperature: number): void {
  const max = list[0].logit / temperature;
  let total = 0;
  for (const c of list) total += c.p = Math.exp(c.logit / temperature - max);
  for (const c of list) c.p /= total;
}

// The roll of the dice: one random number r between 0 and 1, then walk down the list adding up the
// chances until the total reaches r. A token with a 47% chance owns 47% of the line from 0 to 1.
export function pick(list: { id: number; p: number }[], random: () => number): number {
  const r = random();
  let total = 0;
  for (const c of list) {
    total += c.p;
    if (total >= r) return c.id;
  }
  return list[list.length - 1].id; // rounding left the total a hair under r
}

// The Mersenne Twister (MT19937), the generator llama.cpp uses (C++'s std::mt19937). It isn't random
// at all: 624 numbers of state, stirred by a fixed recipe. The seed sets the starting state, so the same
// seed always gives the same numbers. `next` gives whole numbers (0 to 2³²−1); `random` gives a
// fraction from two of them, as C++'s uniform_real_distribution<double> does.
export function mt19937(seed: number) {
  const state = new Uint32Array(624);
  state[0] = seed;
  for (let i = 1; i < 624; i++) {
    const prev = state[i - 1] ^ (state[i - 1] >>> 30);
    state[i] = Math.imul(1812433253, prev) + i; // Math.imul: 32-bit multiplication, as in C++
  }
  let at = 624;
  const next = (): number => {
    if (at === 624) {
      // Every 624 numbers, stir the whole state ("twist").
      for (let i = 0; i < 624; i++) {
        const y = (state[i] & 0x80000000) | (state[(i + 1) % 624] & 0x7fffffff);
        state[i] = state[(i + 397) % 624] ^ (y >>> 1) ^ (y & 1 ? 0x9908b0df : 0);
      }
      at = 0;
    }
    // Then scramble the bits of the number on its way out ("tempering").
    let y = state[at++];
    y ^= y >>> 11;
    y ^= (y << 7) & 0x9d2c5680;
    y ^= (y << 15) & 0xefc60000;
    y ^= y >>> 18;
    return y >>> 0;
  };
  const random = (): number => {
    const low = next();
    const high = next();
    return (low + high * 2 ** 32) / 2 ** 64;
  };
  return { next, random };
}
