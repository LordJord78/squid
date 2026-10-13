import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { sseData } from "../src/sse.ts";

// Feeds bytes in pieces of `size`, the way a network might deliver them.
async function* inPieces(bytes: Uint8Array, size: number) {
  for (let i = 0; i < bytes.length; i += size) yield bytes.subarray(i, i + size);
}
async function collect(source: AsyncIterable<Uint8Array>) {
  const events: string[] = [];
  for await (const data of sseData(source)) events.push(data);
  return events;
}

test("events come out whole however the bytes are cut", async () => {
  const bytes = readFileSync(new URL("../golden/lmstudio/chat-stream.sse", import.meta.url));
  const whole = await collect(inPieces(bytes, bytes.length));
  assert.equal(whole.at(-1), "[DONE]");
  for (const size of [1, 2, 3, 7, 64, 1000]) {
    assert.deepEqual(await collect(inPieces(bytes, size)), whole, `${size}-byte pieces`);
  }
});

test("a character split between two pieces stays whole", async () => {
  // 👋 is 4 bytes in UTF-8; 1-byte pieces cut it three times.
  const bytes = new TextEncoder().encode('data: {"content":"Hi 👋"}\n\ndata: [DONE]\n\n');
  assert.deepEqual(await collect(inPieces(bytes, 1)), ['{"content":"Hi 👋"}', "[DONE]"]);
});

test("Windows line endings work, even when \\r and \\n arrive separately", async () => {
  const bytes = new TextEncoder().encode("data: one\r\n\r\ndata: two\r\n\r\n");
  assert.deepEqual(await collect(inPieces(bytes, 1)), ["one", "two"]);
});
