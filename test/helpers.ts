// Builds small GGUF files by hand for tests: just a header, no weights.
import { mkdtempSync, writeFileSync, mkdirSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, dirname } from "node:path";

const u32 = (n: number) => { const b = Buffer.alloc(4); b.writeUInt32LE(n); return b; };
const u64 = (n: number | bigint) => { const b = Buffer.alloc(8); b.writeBigUInt64LE(BigInt(n)); return b; };
const str = (s: string) => Buffer.concat([u64(Buffer.byteLength(s)), Buffer.from(s, "utf8")]);

type MetaValue = string | number | bigint | boolean | string[];

// string → STRING (8), number → UINT32 (4), bigint → UINT64 (10), boolean → BOOL (7), string[] → ARRAY of STRING (9)
function encode(value: MetaValue): [number, Buffer] {
  if (typeof value === "string") return [8, str(value)];
  if (typeof value === "number") return [4, u32(value)];
  if (typeof value === "bigint") return [10, u64(value)];
  if (typeof value === "boolean") return [7, Buffer.from([value ? 1 : 0])];
  return [9, Buffer.concat([u32(8), u64(value.length), ...value.map(str)])];
}

export function ggufBytes(meta: Record<string, MetaValue>, padTo = 0): Buffer {
  const entries = Object.entries(meta).map(([key, value]) => {
    const [type, bytes] = encode(value);
    return Buffer.concat([str(key), u32(type), bytes]);
  });
  const header = Buffer.concat([Buffer.from("GGUF"), u32(3), u64(0), u64(entries.length), ...entries]);
  // padTo stands in for weights, so folder sizes can be checked
  return header.length >= padTo ? header : Buffer.concat([header, Buffer.alloc(padTo - header.length)]);
}

export function tempDir(): string {
  return mkdtempSync(join(tmpdir(), "squid-"));
}

export function writeGguf(path: string, meta: Record<string, MetaValue>, padTo = 0): string {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, ggufBytes(meta, padTo));
  return path;
}
