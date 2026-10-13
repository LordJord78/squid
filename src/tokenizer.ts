// Text → token ids, exactly as llama.cpp does it, for the byte-level BPE ("gpt2") vocabularies your
// models use: Qwen 3.5 / 3.8 (248,320 tokens) and SmolLM2 (49,152). A model never sees letters, only
// these numbers. Four stages:
//   1. special tokens ("<|im_start|>") are taken out whole;
//   2. the rest is cut into word-like pieces by the model family's pattern (the pre-tokenizer);
//   3. each piece's bytes become visible characters (GPT-2's trick);
//   4. neighbouring characters are merged into longer tokens, most common pairs first (BPE).
// Everything it needs is in the model file's header: the vocabulary and the merge rules (step 01).
import { readGgufHeader, type GgufArray } from "./gguf.ts";

// tokenizer.ggml.token_type: what kind of token each one is (llama.h, LLAMA_TOKEN_TYPE_*).
const NORMAL = 1;
const UNKNOWN = 2;
const CONTROL = 3; // chat markers like <|im_start|>: only special when asked for
const USER_DEFINED = 4; // added words like <think>: always kept whole

export type Vocab = {
  tokens: string[]; // id → the token's text (normal tokens are written in GPT-2's byte characters)
  types: number[];
  ids: Map<string, number>; // text → id
  ranks: Map<string, number>; // "left right" → when that merge rule comes (0 = first)
  split: RegExp[]; // the pre-tokenizer: patterns applied one after another
  specials: string[]; // control, user-defined and unknown tokens, longest first
  bos?: number;
  addBos: boolean;
  cache: Map<string, number[]>; // pieces already worked out (the same words come up again and again)
};

// How each model family cuts text into pieces before BPE. From llama.cpp b11580 (src/llama-vocab.cpp),
// which took them from each model's tokenizer.json. \p{L} = any letter, \p{N} = any digit or number,
// \p{M} = a combining mark (an accent written as its own character, a Hindi vowel sign).
const GPT2 = /'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)/gu;
const PRE_TOKENIZERS: Record<string, RegExp[]> = {
  gpt2: [GPT2],
  smollm: [/\p{N}/gu, GPT2], // first every digit on its own, then GPT-2's pattern on what's between
  qwen2: [/(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu],
  // Qwen 3.5's adds \p{M}, so a letter keeps its combining marks: "नमस्ते" stays one piece.
  qwen35: [/(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu],
};

export function loadVocab(path: string): Vocab {
  const header = readGgufHeader(path, ["tokenizer.ggml.tokens", "tokenizer.ggml.token_type", "tokenizer.ggml.merges"]);
  const meta = header.metadata;
  const model = meta.get("tokenizer.ggml.model");
  if (model !== "gpt2") throw new Error(`Squid's tokenizer reads byte-level BPE ("gpt2") vocabularies; this model's is "${model}"`);
  const pre = String(meta.get("tokenizer.ggml.pre") ?? "gpt2");
  const split = PRE_TOKENIZERS[pre];
  if (!split) throw new Error(`Squid doesn't know the "${pre}" pre-tokenizer yet (it knows ${Object.keys(PRE_TOKENIZERS).join(", ")})`);
  const list = (key: string) => (meta.get(key) as GgufArray | undefined)?.items;
  const tokens = list("tokenizer.ggml.tokens") as string[];
  const types = (list("tokenizer.ggml.token_type") as number[] | undefined) ?? tokens.map(() => NORMAL);
  const merges = (list("tokenizer.ggml.merges") as string[] | undefined) ?? [];
  const bos = meta.get("tokenizer.ggml.bos_token_id");
  return {
    tokens,
    types,
    ids: new Map(tokens.map((text, id) => [text, id])),
    ranks: new Map(merges.map((rule, rank) => [rule, rank])),
    split,
    specials: tokens.filter((text, id) => text && [CONTROL, USER_DEFINED, UNKNOWN].includes(types[id])).sort((a, b) => b.length - a.length),
    bos: typeof bos === "number" ? bos : undefined,
    addBos: meta.get("tokenizer.ggml.add_bos_token") === true,
    cache: new Map(),
  };
}

// `special`: whether chat markers like <|im_start|> in the text count as markers (llama.cpp's
// parse_special, on by default) or as ordinary characters. `bos`: put the start-of-text token first.
export function encode(vocab: Vocab, text: string, options: { special?: boolean; bos?: boolean } = {}): number[] {
  const ids: number[] = [];
  if ((options.bos ?? vocab.addBos) && vocab.bos !== undefined) ids.push(vocab.bos);
  for (const part of specialSplit(vocab, text, options.special ?? true)) {
    if (typeof part === "number") ids.push(part);
    else for (const piece of preTokenize(part, vocab.split)) ids.push(...bpe(vocab, piece));
  }
  return ids;
}

// 1. Special tokens are found first and kept whole, each one id. Longest first, so a long marker wins
// over a shorter token hiding inside it. Control tokens count only when `special` is on; user-defined
// ones always do (as in llama.cpp).
function specialSplit(vocab: Vocab, text: string, special: boolean): (string | number)[] {
  let parts: (string | number)[] = [text];
  for (const token of vocab.specials) {
    const id = vocab.ids.get(token)!;
    if (!special && vocab.types[id] !== USER_DEFINED) continue;
    parts = parts.flatMap((part) => {
      if (typeof part !== "string" || !part.includes(token)) return [part];
      return part.split(token).flatMap((between, i) => [...(i > 0 ? [id] : []), ...(between ? [between] : [])]);
    });
  }
  return parts;
}

// 2. The pre-tokenizer. Each pattern cuts every piece further: what it matches becomes a piece, and so
// does any text between two matches. A token can never cross one of these cuts.
function preTokenize(text: string, patterns: RegExp[]): string[] {
  let pieces = [text];
  for (const pattern of patterns) {
    pieces = pieces.flatMap((piece) => {
      const out: string[] = [];
      let at = 0;
      for (const match of piece.matchAll(pattern)) {
        if (match.index > at) out.push(piece.slice(at, match.index));
        out.push(match[0]);
        at = match.index + match[0].length;
      }
      if (at < piece.length) out.push(piece.slice(at));
      return out;
    });
  }
  return pieces;
}

// 3. GPT-2's trick: each of the 256 byte values gets its own visible character. Printable ones keep
// their own; the rest (space, newline, control bytes) are moved up past 255: a space becomes "Ġ", a
// newline "Ċ". So the whole vocabulary is plain text, and any bytes at all, even half an emoji, can be
// spelled with it. Nothing is ever "unknown".
export const BYTE_CHAR: string[] = [];
const CHAR_BYTE = new Map<string, number>();
for (let byte = 0, next = 256; byte < 256; byte++) {
  const visible = (byte >= 33 && byte <= 126) || (byte >= 161 && byte <= 172) || (byte >= 174 && byte <= 255);
  BYTE_CHAR[byte] = String.fromCodePoint(visible ? byte : next++);
  CHAR_BYTE.set(BYTE_CHAR[byte], byte);
}

// 4. Byte-pair encoding. Start from single characters; join the neighbouring pair whose merge rule comes
// earliest (the rules were learned in order, the most common pairs in the training text first); repeat
// until no rule applies. What's left are the tokens.
function bpe(vocab: Vocab, piece: string): number[] {
  const known = vocab.cache.get(piece);
  if (known) return known;
  const parts = Array.from(Buffer.from(piece, "utf8"), (byte) => BYTE_CHAR[byte]);
  for (;;) {
    let best = -1;
    let bestRank = Infinity;
    for (let i = 0; i + 1 < parts.length; i++) {
      const rank = vocab.ranks.get(`${parts[i]} ${parts[i + 1]}`);
      if (rank !== undefined && rank < bestRank) {
        best = i; // the leftmost pair wins a tie (the same pair twice: "aaa")
        bestRank = rank;
      }
    }
    if (best === -1) break;
    parts.splice(best, 2, parts[best] + parts[best + 1]);
  }
  // A result missing from the vocabulary (a complete one has them all) is spelled byte by byte.
  const ids = parts.flatMap((part) => vocab.ids.get(part) ?? [...part].map((char) => vocab.ids.get(char)!));
  if (vocab.cache.size > 50_000) vocab.cache.clear(); // a long-running server mustn't grow forever
  vocab.cache.set(piece, ids);
  return ids;
}

// Ids back to text. Each token's characters go back to bytes, and all the bytes are read as UTF-8 at
// the end, because one character can be split across two tokens (an emoji is 4 bytes).
// `special: false` leaves chat markers out of the text.
export function decode(vocab: Vocab, ids: number[], options: { special?: boolean } = {}): string {
  return Buffer.concat(ids.map((id) => tokenBytes(vocab, id, options.special ?? true))).toString("utf8");
}

function tokenBytes(vocab: Vocab, id: number, special: boolean): Buffer {
  const text = vocab.tokens[id];
  if (text === undefined) throw new Error(`there's no token ${id} (the vocabulary has ${vocab.tokens.length})`);
  const type = vocab.types[id];
  if (type === NORMAL) return Buffer.from([...text].map((char) => CHAR_BYTE.get(char) ?? 0x3f));
  if (type === CONTROL && !special) return Buffer.alloc(0);
  return Buffer.from(text, "utf8"); // markers and added words are stored as they're written
}

// One token as readable text, for showing: its text if it's whole characters, or its bytes in hex
// when it's only part of one ("<F0><9F>" is the first half of an emoji).
export function piece(vocab: Vocab, id: number): string {
  const bytes = tokenBytes(vocab, id, true);
  try {
    return new TextDecoder("utf-8", { fatal: true }).decode(bytes);
  } catch {
    return [...bytes].map((b) => `<${b.toString(16).toUpperCase().padStart(2, "0")}>`).join("");
  }
}
