// A transformer, the program inside every chat model, written out plainly so each step can be read.
// It runs SmolLM2-135M (Llama's design: 30 layers) straight from its GGUF file, on the CPU: one token
// in, a score for every possible next token out (the "logits"). llama.cpp does the same arithmetic on
// the GPU, far faster; here every number is computed in TypeScript.
import { readFileSync } from "node:fs";
import { readGgufFile, TENSOR_TYPES, type GgufFile } from "./gguf.ts";
import { loadVocab, type Vocab } from "./tokenizer.ts";

// --- Numbers as the file stores them ---

// F16 (half precision): 1 sign bit, 5 exponent bits, 10 fraction bits. All 65,536 possible values are
// turned into ordinary floats once, so reading one later is a table lookup.
export const HALF = new Float32Array(65536);
for (let bits = 0; bits < 65536; bits++) {
  const sign = bits & 0x8000 ? -1 : 1;
  const exponent = (bits >> 10) & 0x1f;
  const fraction = bits & 0x3ff;
  HALF[bits] =
    exponent === 0 ? sign * 2 ** -14 * (fraction / 1024) // tiny numbers close to 0 ("subnormal")
    : exponent === 31 ? (fraction ? NaN : sign * Infinity)
    : sign * 2 ** (exponent - 15) * (1 + fraction / 1024);
}

// A weight table ready for multiplying. Q8_0 stores numbers in blocks of 32: a 2-byte F16 scale, then
// 32 signed bytes (-128..127); each number is scale × byte. 34 bytes for 32 numbers, about a quarter of
// plain floats. The scales are unpacked into floats once; the bytes are kept as they are.
type Matrix = { rows: number; cols: number; scales: Float32Array; quants: Int8Array };

type Layer = {
  attnNorm: Float32Array; q: Matrix; k: Matrix; v: Matrix; out: Matrix; // attention
  ffnNorm: Float32Array; gate: Matrix; up: Matrix; down: Matrix; // feed-forward
};
export type Config = { dim: number; layers: number; heads: number; kvHeads: number; headDim: number; ffn: number; eps: number; ropeBase: number; context: number; vocab: number };
export type Model = { config: Config; embed: Matrix; layers: Layer[]; outputNorm: Float32Array; output: Matrix; vocab: Vocab };

export function loadModel(path: string): Model {
  const file = readGgufFile(path);
  const arch = file.metadata.get("general.architecture");
  if (arch !== "llama") throw new Error(`Squid's engine runs Llama-style models; this one is "${arch}"`);
  const n = (key: string, fallback?: number) => Number(file.metadata.get(`llama.${key}`) ?? fallback);
  const dim = n("embedding_length");
  const heads = n("attention.head_count");
  const config: Config = {
    dim, heads,
    layers: n("block_count"),
    kvHeads: n("attention.head_count_kv", heads),
    headDim: dim / heads,
    ffn: n("feed_forward_length"),
    eps: n("attention.layer_norm_rms_epsilon", 1e-5),
    ropeBase: n("rope.freq_base", 10000),
    context: n("context_length"),
    vocab: 0,
  };
  const bytes = readFileSync(path); // the whole file: 145 MB for this model
  const get = tensorReader(file, bytes);
  const embed = get.matrix("token_embd.weight");
  config.vocab = embed.rows;
  const layers = Array.from({ length: config.layers }, (_, i): Layer => ({
    attnNorm: get.vector(`blk.${i}.attn_norm.weight`),
    q: get.matrix(`blk.${i}.attn_q.weight`), k: get.matrix(`blk.${i}.attn_k.weight`),
    v: get.matrix(`blk.${i}.attn_v.weight`), out: get.matrix(`blk.${i}.attn_output.weight`),
    ffnNorm: get.vector(`blk.${i}.ffn_norm.weight`),
    gate: get.matrix(`blk.${i}.ffn_gate.weight`), up: get.matrix(`blk.${i}.ffn_up.weight`), down: get.matrix(`blk.${i}.ffn_down.weight`),
  }));
  // Small models reuse the token table to score the next token ("tied embeddings"); SmolLM2 does.
  const output = get.has("output.weight") ? get.matrix("output.weight") : embed;
  return { config, embed, layers, outputNorm: get.vector("output_norm.weight"), output, vocab: loadVocab(path) };
}

// Finds tensors by name and unpacks them. Only the two formats this model uses: Q8_0 tables and F32
// vectors. Anything else is refused with its name rather than read wrongly.
function tensorReader(file: GgufFile, bytes: Buffer) {
  const signed = new Int8Array(bytes.buffer, bytes.byteOffset, bytes.length); // the same bytes, read as -128..127
  const find = (name: string) => {
    const tensor = file.tensors.find((t) => t.name === name);
    if (!tensor) throw new Error(`the model file has no tensor called ${name}`);
    return { ...tensor, at: file.dataStart + tensor.offset };
  };
  return {
    has: (name: string) => file.tensors.some((t) => t.name === name),
    vector(name: string): Float32Array {
      const t = find(name);
      if (t.type !== 0) throw new Error(`${name} is ${TENSOR_TYPES[t.type] ?? t.type}; Squid's engine reads vectors as F32`);
      const out = new Float32Array(t.shape[0]);
      for (let i = 0; i < out.length; i++) out[i] = bytes.readFloatLE(t.at + i * 4);
      return out;
    },
    matrix(name: string): Matrix {
      const t = find(name);
      if (t.type !== 8) throw new Error(`${name} is ${TENSOR_TYPES[t.type] ?? t.type}; Squid's engine reads weight tables as Q8_0`);
      const [cols, rows] = t.shape; // shape[0] = numbers per row
      const blocks = (rows * cols) / 32;
      const scales = new Float32Array(blocks);
      const quants = new Int8Array(rows * cols);
      for (let b = 0; b < blocks; b++) {
        const at = t.at + b * 34;
        scales[b] = HALF[bytes.readUInt16LE(at)];
        for (let j = 0; j < 32; j++) quants[b * 32 + j] = signed[at + 2 + j];
      }
      return { rows, cols, scales, quants };
    },
  };
}

// --- The building blocks ---

// Matrix × vector: each output number is one row of the table "dotted" with the input (multiply pairs,
// add them up). This is nearly all the work a model does: 134 million multiply-adds per token here.
function matvec(m: Matrix, x: Float32Array, out: Float32Array): void {
  const { rows, cols, scales, quants } = m;
  for (let r = 0, at = 0, block = 0; r < rows; r++) {
    let sum = 0;
    for (let c = 0; c < cols; c += 32, block++) {
      let part = 0;
      for (let j = 0; j < 32; j++) part += quants[at + j] * x[c + j];
      sum += scales[block] * part; // the block's scale once, not 32 times
      at += 32;
    }
    out[r] = sum;
  }
}

// RMS norm: scale the vector so its typical size (root mean square) is 1, then by learned weights.
// It keeps the numbers from growing or shrinking as they pass through 30 layers.
function rmsNorm(x: Float32Array, weight: Float32Array, eps: number, out: Float32Array): void {
  let squares = 0;
  for (let i = 0; i < x.length; i++) squares += x[i] * x[i];
  const scale = 1 / Math.sqrt(squares / x.length + eps);
  for (let i = 0; i < x.length; i++) out[i] = x[i] * scale * weight[i];
}

// RoPE: a token's position is written into its query and key by rotating each pair of numbers
// (0,1), (2,3), … through an angle that grows with the position, each pair at its own speed. Two vectors
// rotated this way compare (dot product) according to how far apart their tokens are.
export function rope(v: Float32Array, pos: number, headDim: number, base: number): void {
  for (let head = 0; head < v.length; head += headDim) {
    for (let i = 0; i < headDim; i += 2) {
      const angle = pos * base ** (-i / headDim);
      const cos = Math.cos(angle);
      const sin = Math.sin(angle);
      const a = v[head + i];
      const b = v[head + i + 1];
      v[head + i] = a * cos - b * sin;
      v[head + i + 1] = a * sin + b * cos;
    }
  }
}

// Softmax: turns any scores into weights that are positive and add up to 1. The largest is subtracted
// first so exp() never overflows; it doesn't change the result.
function softmax(x: Float32Array, n: number): void {
  let max = -Infinity;
  for (let i = 0; i < n; i++) max = Math.max(max, x[i]);
  let total = 0;
  for (let i = 0; i < n; i++) total += x[i] = Math.exp(x[i] - max);
  for (let i = 0; i < n; i++) x[i] /= total;
}

const silu = (x: number) => x / (1 + Math.exp(-x)); // the feed-forward's "gate": ~0 for negatives, ~x for positives

// --- Running it ---

// What the model remembers of the text so far: every layer's keys and values for every earlier position
// (the "KV cache"). Without it, each new token would mean running the whole text through again.
export type State = { pos: number; size: number; keys: Float32Array[]; values: Float32Array[] };
export function newState(model: Model, size = Math.min(model.config.context, 2048)): State {
  const kvDim = model.config.kvHeads * model.config.headDim;
  const room = () => Array.from({ length: model.config.layers }, () => new Float32Array(size * kvDim));
  return { pos: 0, size, keys: room(), values: room() };
}

// One token through the whole model, at the next position. Returns the logits: one score per word in
// the vocabulary for what comes next (higher = more likely).
export function forward(model: Model, state: State, token: number): Float32Array {
  const { dim, heads, kvHeads, headDim, ffn, eps, ropeBase } = model.config;
  const pos = state.pos;
  if (pos >= state.size) throw new Error(`the text is longer than the ${state.size} tokens set aside for it`);
  const kvDim = kvHeads * headDim;
  const group = heads / kvHeads; // 3 query heads share each key/value head ("grouped-query attention")

  // 1. The token becomes its row of the embedding table: 576 numbers.
  const x = new Float32Array(dim);
  for (let i = 0, block = (token * dim) / 32; i < dim; i += 32, block++) {
    for (let j = 0; j < 32; j++) x[i + j] = model.embed.scales[block] * model.embed.quants[token * dim + i + j];
  }
  const h = new Float32Array(dim), q = new Float32Array(dim), k = new Float32Array(kvDim), v = new Float32Array(kvDim);
  const mixed = new Float32Array(dim), back = new Float32Array(dim), gate = new Float32Array(ffn), up = new Float32Array(ffn);
  const scores = new Float32Array(pos + 1);

  for (let l = 0; l < model.layers.length; l++) {
    const layer = model.layers[l];
    const keys = state.keys[l];
    const values = state.values[l];

    // 2. Attention: this token looks back at every earlier one (and itself).
    rmsNorm(x, layer.attnNorm, eps, h);
    matvec(layer.q, h, q); // what it's looking for
    matvec(layer.k, h, k); // what it offers to later tokens
    matvec(layer.v, h, v); // what it passes on if chosen
    rope(q, pos, headDim, ropeBase);
    rope(k, pos, headDim, ropeBase);
    keys.set(k, pos * kvDim); // remembered for the tokens that come after
    values.set(v, pos * kvDim);
    for (let head = 0; head < heads; head++) {
      const qAt = head * headDim;
      const kvAt = Math.floor(head / group) * headDim;
      // How well each earlier token's key matches this query, scaled by 1/√64, then made into weights.
      for (let t = 0; t <= pos; t++) {
        let dot = 0;
        for (let i = 0; i < headDim; i++) dot += q[qAt + i] * keys[t * kvDim + kvAt + i];
        scores[t] = dot / Math.sqrt(headDim);
      }
      softmax(scores, pos + 1);
      // The head's answer: the earlier tokens' values, mixed by those weights.
      for (let i = 0; i < headDim; i++) {
        let sum = 0;
        for (let t = 0; t <= pos; t++) sum += scores[t] * values[t * kvDim + kvAt + i];
        mixed[qAt + i] = sum;
      }
    }
    matvec(layer.out, mixed, back);
    for (let i = 0; i < dim; i++) x[i] += back[i]; // added to what was there (a "residual" connection)

    // 3. Feed-forward: each token on its own, through a wider layer (1,536) and back, gated by SiLU.
    rmsNorm(x, layer.ffnNorm, eps, h);
    matvec(layer.gate, h, gate);
    matvec(layer.up, h, up);
    for (let i = 0; i < ffn; i++) gate[i] = silu(gate[i]) * up[i];
    matvec(layer.down, gate, back);
    for (let i = 0; i < dim; i++) x[i] += back[i];
  }

  // 4. A last norm, then a score for every token in the vocabulary.
  rmsNorm(x, model.outputNorm, eps, h);
  const logits = new Float32Array(model.config.vocab);
  matvec(model.output, h, logits);
  state.pos++;
  return logits;
}

// Feeds a list of tokens (a prompt) and returns the logits after the last one.
export function feed(model: Model, state: State, tokens: number[]): Float32Array {
  let logits: Float32Array = new Float32Array(0);
  for (const token of tokens) logits = forward(model, state, token);
  return logits;
}

// Every token's log-probability: log(softmax(logits)), worked out without ever making the huge exp()
// values (the "log-sum-exp" trick). A logprob of −0.69 is 50%, −2.3 is 10%, 0 would be certain.
export function logprobs(logits: Float32Array): Float32Array {
  let max = -Infinity;
  for (const l of logits) max = Math.max(max, l);
  let total = 0;
  for (const l of logits) total += Math.exp(l - max);
  const logTotal = max + Math.log(total);
  return logits.map((l) => l - logTotal);
}

// The k most likely next tokens, most likely first.
export function top(logits: Float32Array, k = 5): { id: number; logprob: number }[] {
  const lp = logprobs(logits);
  return highest(lp, k).map((id) => ({ id, logprob: lp[id] }));
}

// The positions of the k highest scores, highest first. One pass, keeping a short list in order, rather
// than sorting all 49,152 (step 17's sampler uses it for top-k, once per token).
export function highest(scores: Float32Array, k: number): number[] {
  const best: number[] = [];
  for (let id = 0; id < scores.length; id++) {
    if (best.length === k && scores[id] <= scores[best[k - 1]]) continue; // not better than the last on the list
    let at = best.length;
    while (at > 0 && scores[best[at - 1]] < scores[id]) at--;
    best.splice(at, 0, id);
    if (best.length > k) best.pop();
  }
  return best;
}
