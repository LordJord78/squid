// The golden files are recordings of the real LM Studio (golden/capture-lmstudio.ps1).
// These tests pin down the shapes Squid has to reproduce, so later steps can compare against them.
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";

const golden = (name: string) => readFileSync(new URL(`../golden/lmstudio/${name}`, import.meta.url), "utf8");

// An SSE stream is lines of `data: <json>`, ending with `data: [DONE]`.
function sseEvents(text: string): string[] {
  return text.split(/\r?\n/).filter((line) => line.startsWith("data: ")).map((line) => line.slice(6));
}

test("non-streamed reply keeps thinking and answer apart", () => {
  const reply = JSON.parse(golden("chat-nonstream.json"));
  const message = reply.choices[0].message;
  assert.equal(reply.object, "chat.completion");
  assert.equal(message.content, "pong");
  assert.ok(message.reasoning_content.length > 0);
  assert.equal(reply.choices[0].finish_reason, "stop");
});

test("streamed reply: thinking first, then the answer, then [DONE]", () => {
  const events = sseEvents(golden("chat-stream.sse"));
  assert.equal(events.at(-1), "[DONE]");
  const chunks = events.slice(0, -1).map((e) => JSON.parse(e));
  const kinds = chunks.map((c) => Object.keys(c.choices[0].delta).filter((k) => k !== "role")[0] ?? "end");
  const firstAnswer = kinds.indexOf("content");
  assert.ok(kinds.lastIndexOf("reasoning_content") < firstAnswer, "all thinking arrives before the answer");
  assert.equal(chunks.at(-1).choices[0].finish_reason, "stop");
});

test("streamed tool call: name first, arguments after, finish_reason tool_calls", () => {
  const chunks = sseEvents(golden("chat-tools-stream.sse")).slice(0, -1).map((e) => JSON.parse(e));
  const calls = chunks.flatMap((c) => c.choices[0].delta.tool_calls ?? []);
  assert.equal(calls[0].function.name, "read_file");
  const args = calls.map((c: { function: { arguments?: string } }) => c.function.arguments ?? "").join("");
  assert.deepEqual(JSON.parse(args), { path: "README.md" });
  assert.equal(chunks.at(-1).choices[0].finish_reason, "tool_calls");
});

test("LM Studio's own API reports speed and the engine it runs on", () => {
  const reply = JSON.parse(golden("api-v0-chat.json"));
  assert.ok(reply.stats.tokens_per_second > 0);
  assert.match(reply.runtime.name, /^llama\.cpp-/);
});
