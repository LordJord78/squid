// Reads the front of a GGUF model file: the metadata that describes the model.
// The weights behind it (gigabytes) are never loaded. Format: ggml docs/gguf.md.
import { openSync, readSync, closeSync, fstatSync } from "node:fs";

// Every metadata value is tagged with one of these type numbers.
const T = { U8: 0, I8: 1, U16: 2, I16: 3, U32: 4, I32: 5, F32: 6, BOOL: 7, STRING: 8, ARRAY: 9, U64: 10, I64: 11, F64: 12 };

export type GgufArray = { arrayOf: number; length: number; items?: GgufValue[] };
export type GgufValue = number | bigint | boolean | string | GgufArray;
export type GgufHeader = { version: number; tensorCount: number; metadata: Map<string, GgufValue>; headerBytes: number; fileBytes: number };

// Arrays longer than this (like a 150,000-word vocabulary) are walked past but not kept, unless asked for
// by name (the tokenizer, step 15, asks for the vocabulary and its merge rules).
const KEEP_ARRAYS_UP_TO = 32;

// Hands out the file's bytes in order, reading 1 MiB at a time from disk as it goes.
class ByteReader {
  fd: number;
  buf = Buffer.alloc(0);
  at = 0; // next unread byte inside buf
  fileAt = 0; // how far into the file buf reaches
  consumed = 0; // bytes handed out so far = position in the file

  constructor(fd: number) {
    this.fd = fd;
  }

  // Advance past n bytes and return where they start in buf.
  take(n: number): number {
    const left = this.buf.length - this.at;
    if (left < n) {
      const chunk = Buffer.alloc(Math.max(n - left, 1 << 20));
      const got = readSync(this.fd, chunk, 0, chunk.length, this.fileAt);
      if (got < n - left) throw new Error("file ends in the middle of the header");
      this.fileAt += got;
      this.buf = Buffer.concat([this.buf.subarray(this.at), chunk.subarray(0, got)]);
      this.at = 0;
    }
    const start = this.at;
    this.at += n;
    this.consumed += n;
    return start;
  }

  // Take n bytes, then decode them. take() may swap in a new buffer,
  // so this.buf must only be looked at after it returns.
  read<V>(n: number, decode: (buf: Buffer, at: number) => V): V {
    const at = this.take(n);
    return decode(this.buf, at);
  }

  u8() { return this.read(1, (b, at) => b.readUInt8(at)); }
  u32() { return this.read(4, (b, at) => b.readUInt32LE(at)); }
  u64() { return this.read(8, (b, at) => b.readBigUInt64LE(at)); }

  // A GGUF string is its byte length (u64) followed by that many UTF-8 bytes.
  string(keep = true): string {
    const length = Number(this.u64());
    return this.read(length, (b, at) => (keep ? b.toString("utf8", at, at + length) : ""));
  }
}

function readValue(r: ByteReader, type: number, keep = true, whole = false): GgufValue {
  switch (type) {
    case T.U8: return r.u8();
    case T.I8: return r.read(1, (b, at) => b.readInt8(at));
    case T.U16: return r.read(2, (b, at) => b.readUInt16LE(at));
    case T.I16: return r.read(2, (b, at) => b.readInt16LE(at));
    case T.U32: return r.u32();
    case T.I32: return r.read(4, (b, at) => b.readInt32LE(at));
    case T.F32: return r.read(4, (b, at) => b.readFloatLE(at));
    case T.BOOL: return r.u8() !== 0;
    case T.STRING: return r.string(keep);
    case T.U64: return r.u64();
    case T.I64: return r.read(8, (b, at) => b.readBigInt64LE(at));
    case T.F64: return r.read(8, (b, at) => b.readDoubleLE(at));
    case T.ARRAY: {
      const arrayOf = r.u32();
      const length = Number(r.u64());
      const keepItems = keep && (whole || length <= KEEP_ARRAYS_UP_TO);
      const items: GgufValue[] = [];
      for (let i = 0; i < length; i++) {
        const item = readValue(r, arrayOf, keepItems);
        if (keepItems) items.push(item);
      }
      return keepItems ? { arrayOf, length, items } : { arrayOf, length };
    }
    default: throw new Error(`unknown metadata value type ${type}`);
  }
}

// `arrays`: keys whose arrays are kept whole, however long.
export function readGgufHeader(path: string, arrays: string[] = []): GgufHeader {
  const fd = openSync(path, "r");
  try {
    const r = new ByteReader(fd);
    const magic = r.u32();
    if (magic !== 0x46554747) throw new Error(`not a GGUF file: ${path} (starts with 0x${magic.toString(16)}, expected "GGUF")`);
    const version = r.u32();
    if (version < 2) throw new Error(`GGUF version ${version} is too old (counts were 32-bit before v2)`);
    const tensorCount = Number(r.u64());
    const kvCount = Number(r.u64());
    const metadata = new Map<string, GgufValue>();
    for (let i = 0; i < kvCount; i++) {
      const key = r.string();
      metadata.set(key, readValue(r, r.u32(), true, arrays.includes(key)));
    }
    return { version, tensorCount, metadata, headerBytes: r.consumed, fileBytes: fstatSync(fd).size };
  } finally {
    closeSync(fd);
  }
}

// general.file_type → the quantisation name people use (llama.h, LLAMA_FTYPE_*).
const FILE_TYPES: Record<number, string> = {
  0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1", 10: "Q2_K",
  11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M", 16: "Q5_K_S", 17: "Q5_K_M",
  18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "Q2_K_S", 22: "IQ3_XS", 23: "IQ3_XXS", 24: "IQ1_S",
  25: "IQ4_NL", 26: "IQ3_S", 27: "IQ3_M", 28: "IQ2_S", 29: "IQ2_M", 30: "IQ4_XS", 31: "IQ1_M",
  32: "BF16", 36: "TQ1_0", 37: "TQ2_0", 38: "MXFP4_MOE",
};

// The handful of facts LM Studio shows about a model, pulled out of the metadata.
export function describeModel(h: GgufHeader) {
  const get = (key: string) => h.metadata.get(key);
  const num = (key: string) => {
    const v = get(key);
    return typeof v === "bigint" ? Number(v) : typeof v === "number" ? v : undefined;
  };
  const arch = String(get("general.architecture"));
  const fileType = num("general.file_type");
  return {
    name: get("general.name") as string | undefined,
    architecture: arch,
    sizeLabel: get("general.size_label") as string | undefined,
    quantization: fileType === undefined ? undefined : (FILE_TYPES[fileType] ?? `type ${fileType}`),
    contextLength: num(`${arch}.context_length`),
    layers: num(`${arch}.block_count`),
    // Embedding models turn text into one vector, so they say how to pool token outputs; chat models don't.
    embedding: h.metadata.has(`${arch}.pooling_type`),
    hasChatTemplate: typeof get("tokenizer.chat_template") === "string",
    tensorCount: h.tensorCount,
    headerBytes: h.headerBytes,
    fileBytes: h.fileBytes,
  };
}
