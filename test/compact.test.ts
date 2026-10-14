import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { startConversation, takeTurn } from "../src/agent.ts";
import { estimate, isSummary, latestRound, limitFor, SUMMARY_HEADER, summaryPrompt, trim } from "../src/compact.ts";
import type { Message, Output } from "../src/reply.ts";
import { newSession } from "../src/tools.ts";
import { withoutNote } from "../src/undo.ts";
import { scriptedServer, tempDir } from "./helpers.ts";

function collector() {
  const seen = { notes: "" };
  const out: Output = { answer: () => {}, thinking: () => {}, note: (t) => (seen.notes += t) };
  return { seen, out };
}
const call = (id: string, name: string, args: object) => ({ id, type: "function" as const, function: { name, arguments: JSON.stringify(args) } });
const big = (n: number) => Array.from({ length: n }, (_, i) => `line ${i}: some text that takes up room`).join("\n");

test("trimming stubs old file contents, cuts long results and drops old thinking; your words and the latest round stay", () => {
  const messages: Message[] = [
    { role: "system", content: "You are Squid." },
    { role: "user", content: "Build it." },
    { role: "assistant", content: "", reasoning_content: "Long thoughts…", tool_calls: [call("1", "write_file", { path: "a.py", content: big(60) })] },
    { role: "tool", tool_call_id: "1", content: "Wrote a.py (60 lines)" },
    { role: "assistant", content: "", tool_calls: [call("2", "read_file", { path: "b.py" })] },
    { role: "tool", tool_call_id: "2", content: big(200) },
    { role: "user", content: big(80) }, // your own words are never cut, however long
    { role: "assistant", content: "", reasoning_content: "Now c.", tool_calls: [call("3", "write_file", { path: "c.py", content: big(60) })] },
    { role: "tool", tool_call_id: "3", content: "Wrote c.py (60 lines)" },
  ];
  const userText = messages[6].content;
  const latest = latestRound(messages);
  assert.equal(latest, 7, "the latest round: the last reply with tool calls, and its results");
  trim(messages, latest);

  const first = JSON.parse(messages[2].tool_calls![0].function.arguments);
  assert.equal(first.path, "a.py", "the path stays, so it's clear which file was written");
  assert.equal(first.content, "[Squid: 60 lines of a.py went here. They were removed from the conversation to save memory; the file was written and is unchanged on disk. read_file shows it.]");
  assert.equal(messages[2].reasoning_content, undefined, "old thinking is gone");
  assert.ok(messages[5].content.length < 800 && /\[Squid: the other \d+ characters of this result were removed/.test(messages[5].content));
  assert.equal(messages[6].content, userText);
  assert.equal(JSON.parse(messages[7].tool_calls![0].function.arguments).content, big(60), "the latest round is untouched");
  assert.equal(messages[7].reasoning_content, "Now c.");
  assert.deepEqual(messages.filter((m) => m.role === "tool").map((m) => m.tool_call_id), ["1", "2", "3"], "every result still answers its call");
  assert.equal(latestRound(messages.slice(0, 7)), 7, "ending on your message: no round in progress");
});

test("the estimate is exact up to the last reply, and a guess (3 characters a token) for what came after", () => {
  const session = newSession(tempDir());
  const messages: Message[] = [{ role: "user", content: "x".repeat(3000) }];
  assert.equal(estimate(messages, session), 1000, "nothing measured yet: all guessed");
  session.measured = { tokens: 700, chars: 3000 }; // the model's own count was lower
  messages.push({ role: "tool", tool_call_id: "1", content: "y".repeat(3000) });
  assert.equal(estimate(messages, session), 1700);
});

// The scripted server says its model "fake" has an 8,192-token memory.
test("past compactAt, the agent trims before asking: no summary needed, and files must be read again before editing", async () => {
  const root = tempDir();
  const fake = await scriptedServer([{ tool: ["write_file", { path: "big.txt", content: big(300) }] }, { say: "Written." }, { say: "Next done." }]);
  const session = newSession(root, undefined, undefined, { mode: "auto", compactAt: 0.3 }); // 0.3 × 8,192 = 2,458 tokens
  const { seen, out } = collector();
  try {
    const history = startConversation(root);
    const first = await takeTurn(fake.base, undefined, history, "Write big.txt.", undefined, session, undefined, out);
    assert.equal(first.error, undefined);
    assert.ok(session.seen.size > 0, "it may edit what it just wrote");
    await takeTurn(fake.base, first.model, history, "Next.", undefined, session, undefined, out);

    assert.equal(fake.requests.length, 3, "no summary request: trimming made enough room");
    assert.match(seen.notes, /trimmed old file contents, tool results and thinking: ~\d+(\.\d)?K → ~\d+(\.\d)?K tokens/);
    const written = fake.requests[2].messages.find((m: Message) => m.tool_calls)!.tool_calls![0];
    assert.match(JSON.parse(written.function.arguments).content, /^\[Squid: 300 lines of big\.txt went here\./);
    assert.equal(session.seen.size, 0, "after a trim, a file must be read again before it's edited");
    assert.equal(readFileSync(join(root, "big.txt"), "utf8"), big(300), "the file itself is untouched");
  } finally {
    fake.close();
  }
});

test("when trimming isn't enough, the model summarises (no tools, no thinking, capped) and the summary replaces the conversation", async () => {
  const root = tempDir();
  // A long answer can't be trimmed (it's the model's own words), so only a summary makes room.
  const fake = await scriptedServer([{ say: big(300) }, { say: "SUMMARY: we talked about lines." }, { say: "Done." }]);
  const session = newSession(root, undefined, undefined, { compactAt: 0.3 });
  const { seen, out } = collector();
  try {
    const history = startConversation(root);
    const first = await takeTurn(fake.base, undefined, history, "Talk.", undefined, session, undefined, out);
    await takeTurn(fake.base, first.model, history, "Next.", undefined, session, undefined, out);

    const asked = fake.requests[1];
    assert.equal(asked.tools, undefined, "no tools: it can only write");
    assert.equal(asked.max_tokens, 2048);
    assert.deepEqual(asked.chat_template_kwargs, { enable_thinking: false });
    assert.equal(asked.messages.at(-1).content, summaryPrompt());
    assert.deepEqual(history.map((m) => m.role), ["system", "user", "user", "assistant"]);
    assert.ok(isSummary(history[1]));
    assert.equal(withoutNote(history[1].content), "SUMMARY: we talked about lines.");
    assert.equal(history[2].content, "Next.", "your message survives word for word");
    assert.match(fake.requests[2].messages[1].content, new RegExp(`^${SUMMARY_HEADER.replace(/[()]/g, "\\$&")}`), "the model reads the summary first");
    assert.match(seen.notes, /conversation compacted into a summary/);
  } finally {
    fake.close();
  }
});

test("an answer cut off by a full memory is healed: compact, then ask for that round again", async () => {
  const root = tempDir();
  const fake = await scriptedServer([
    { tool: ["write_file", { path: "a.txt", content: "A\n" }] },
    { tool: ["write_file", { path: "b.txt", content: big(50) }], cut: "length" }, // the memory runs out mid-call
    { say: "SUMMARY: a.txt is written; b.txt is next." },
    { tool: ["write_file", { path: "b.txt", content: "B\n" }] },
    { say: "Both written." },
  ]);
  const session = newSession(root, undefined, undefined, { mode: "auto" });
  const { seen, out } = collector();
  try {
    const history = startConversation(root);
    const turn = await takeTurn(fake.base, undefined, history, "Write a.txt and b.txt.", undefined, session, undefined, out);
    assert.equal(turn.error, undefined, "the turn finished");
    assert.equal(readFileSync(join(root, "b.txt"), "utf8"), "B\n");
    assert.match(seen.notes, /that answer ran out of memory: compacting, then asking again/);
    assert.ok(isSummary(history[1]));
    assert.match(history[1].content, /cut off because your memory filled up[\s\S]*at most one or two files per reply/);
    assert.equal(history[2].content, "Write a.txt and b.txt.");
    assert.equal(fake.requests[3].messages.filter((m: Message) => m.role === "tool").length, 1, "the latest round (a.txt) stayed for the retry");
  } finally {
    fake.close();
  }
});

test("a second cut-off in the same turn isn't retried forever: the finished work is kept and you're told", async () => {
  const root = tempDir();
  const cut = { tool: ["write_file", { path: "b.txt", content: big(50) }] as [string, object], cut: "length" as const };
  const fake = await scriptedServer([{ tool: ["write_file", { path: "a.txt", content: "A\n" }] }, cut, { say: "SUMMARY." }, cut]);
  const session = newSession(root, undefined, undefined, { mode: "auto" });
  try {
    const history = startConversation(root);
    const turn = await takeTurn(fake.base, undefined, history, "Write both.", undefined, session, undefined, collector().out);
    assert.equal(fake.requests.length, 4, "asked once more, not again and again");
    assert.equal(turn.kept, 1, "the a.txt round, which the summary kept as the latest round");
    assert.ok(existsSync(join(root, "a.txt")) && !existsSync(join(root, "b.txt")));
  } finally {
    fake.close();
  }
});

test("the memory size is asked for again until the model is loaded", async () => {
  // The first request of a chat is what loads the model; before it, the server can't say how big its
  // memory is. Caching that "unknown" switched automatic compaction off for a whole session.
  const fake = await scriptedServer([{ say: "Hi." }]);
  const session = newSession(tempDir());
  try {
    assert.equal(await limitFor(fake.base, "fake", session), undefined, "not loaded yet");
    await takeTurn(fake.base, "fake", startConversation(session.root), "Hello.", undefined, session, undefined, collector().out);
    assert.equal(await limitFor(fake.base, "fake", session), 8192, "asked again, and now known");
  } finally {
    fake.close();
  }
});
