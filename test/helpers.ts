// Builds small GGUF files by hand for tests: just a header, no weights. And a scripted stand-in model.
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
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

// A stand-in for Squid's server that follows a script: each chat request gets the next step, either a
// tool call or an answer, streamed in LM Studio's format with the arguments cut into 5-character pieces.
// It also answers the model list, as a loaded model "fake" with an 8,192-token context.
export type Step = { tool: [string, object] } | { say: string };
export async function scriptedServer(script: Step[]) {
  const requests: any[] = [];
  const server = createServer(async (req, res) => {
    if (req.url === "/api/v0/models") return res.end(JSON.stringify({ data: [{ id: "fake", loaded_context_length: 8192 }] }));
    let body = "";
    for await (const chunk of req) body += chunk;
    requests.push(JSON.parse(body));
    const step = script[Math.min(requests.length - 1, script.length - 1)];
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    const send = (delta: object, finish: string | null = null) =>
      res.write(`data: ${JSON.stringify({ model: "fake", choices: [{ index: 0, delta, logprobs: null, finish_reason: finish }] })}\n\n`);
    send({ reasoning_content: `plan ${requests.length}` });
    if ("tool" in step) {
      const [name, args] = step.tool;
      send({ tool_calls: [{ index: 0, id: `call${requests.length}`, type: "function", function: { name, arguments: "" } }] });
      const text = JSON.stringify(args);
      for (let i = 0; i < text.length; i += 5) send({ tool_calls: [{ index: 0, type: "function", function: { arguments: text.slice(i, i + 5) } }] });
      send({}, "tool_calls");
    } else {
      send({ content: step.say });
      send({}, "stop");
    }
    res.end("data: [DONE]\n\n");
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { base: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, requests, close: () => server.close() };
}

export function writeGguf(path: string, meta: Record<string, MetaValue>, padTo = 0): string {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, ggufBytes(meta, padTo));
  return path;
}
