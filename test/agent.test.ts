import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { agentTurn, brokeOff, finishedLength, forgetThinking, interruptedNote, MAX_ROUNDS, startConversation, takeTurn } from "../src/agent.ts";
import { CutOff, streamReply, type Message, type Output } from "../src/reply.ts";
import { withoutNote } from "../src/undo.ts";
import { startServer } from "../src/server.ts";
import { newSession, type Approval } from "../src/tools.ts";
import { ENGINE, running, unload } from "../src/engine.ts";
import { scriptedServer, tempDir } from "./helpers.ts";

// Collects what the agent would print, instead of printing it.
function collector() {
  const seen = { answer: "", thinking: "", notes: "" };
  const out: Output = { answer: (t) => (seen.answer += t), thinking: (t) => (seen.thinking += t), note: (t) => (seen.notes += t) };
  return { seen, out };
}

function project() {
  const dir = tempDir();
  writeFileSync(join(dir, "README.md"), "The codename is Blue Heron.\n");
  return dir;
}

test("the loop runs the tools the model asks for and sends the results back", async () => {
  const fake = await scriptedServer([{ tool: ["list_files", {}] }, { tool: ["read_file", { path: "README.md" }] }, { say: "It's Blue Heron." }]);
  const root = project();
  const messages: Message[] = [...startConversation(root), { role: "user", content: "What's the codename?" }];
  const { seen, out } = collector();
  try {
    const { reply } = await agentTurn(fake.base, undefined, messages, newSession(root), undefined, out);
    assert.equal(reply.content, "It's Blue Heron.");
    assert.equal(fake.requests.length, 3, "list, read, answer");
    assert.deepEqual(fake.requests[0].tools.map((t: any) => t.function.name), ["list_files", "read_file", "search", "edit_file", "write_file", "run_command"]);

    // The second request carries the first call (its pieces put back together) and its result.
    const [call, result] = fake.requests[1].messages.slice(-2);
    assert.deepEqual(call.tool_calls, [{ id: "call1", type: "function", function: { name: "list_files", arguments: "{}" } }]);
    assert.equal(call.reasoning_content, "plan 1", "thinking goes back while the turn is under way");
    assert.deepEqual(result, { role: "tool", tool_call_id: "call1", content: "README.md" });
    assert.match(fake.requests[2].messages.at(-1).content, /Blue Heron/, "the file's text reached the model");

    assert.match(seen.notes, /● list_files\(\.\)[\s\S]*● read_file\(README\.md\)/);
    assert.equal(seen.answer, "It's Blue Heron.\n");
  } finally {
    fake.close();
  }
});

test("a model that never stops calling tools is paused, with every finished round kept", async () => {
  const fake = await scriptedServer([{ tool: ["list_files", {}] }]); // the same request forever
  const root = project();
  try {
    const messages: Message[] = [{ role: "user", content: "hi" }];
    const turn = await agentTurn(fake.base, undefined, messages, newSession(root), undefined, collector().out);
    assert.equal(turn.paused, true);
    assert.equal(fake.requests.length, MAX_ROUNDS);
    assert.equal(messages.length, 1 + 2 * MAX_ROUNDS, "each round: the call and its result");
    assert.equal(messages.at(-1)!.role, "tool", "your next message (\"continue\") picks up after the last result");
  } finally {
    fake.close();
  }
});

// Step 18. Writing files needs auto mode here: in manual mode nobody is at the keyboard to say yes.
const writer = (root: string) => newSession(root, undefined, undefined, { mode: "auto" });
const write = (path: string) => ({ tool: ["write_file", { path, content: `This is ${path}.\n` }] as [string, object] });

test("a tool call cut off by a full memory is never run", async () => {
  for (const cut of ["length", "drop"] as const) {
    const fake = await scriptedServer([{ ...write("half.txt"), cut }]);
    const root = project();
    try {
      const error = await agentTurn(fake.base, undefined, [{ role: "user", content: "hi" }], writer(root), undefined, collector().out).catch((err) => err);
      assert.ok(error instanceof CutOff, cut);
      assert.equal(error.reason, cut === "length" ? "memory full" : "broken");
      assert.equal(existsSync(join(root, "half.txt")), false, "the half-written call didn't run");
    } finally {
      fake.close();
    }
  }
});

test("an answer that's slow to start is waited for: the request doesn't go through fetch, which gives up at 5 minutes", async () => {
  // Flash-Next on the PC (22 Oct) took over 5 minutes to read an 8K-token prompt before its first word, and
  // the chat hung up at 300 s twice ("the answer was cut off"): fetch's limit for a reply to start.
  const fake = await scriptedServer([{ say: "Here it is." }]);
  const realFetch = globalThis.fetch;
  globalThis.fetch = () => Promise.reject(new Error("fetch was used"));
  try {
    const reply = await streamReply(fake.base, { messages: [{ role: "user", content: "hi" }] }, undefined, collector().out);
    assert.equal(reply.content, "Here it is.");
    assert.equal(reply.finishReason, "stop");
  } finally {
    globalThis.fetch = realFetch;
    fake.close();
  }
});

test("a turn that breaks off keeps the rounds it finished, and the model hears what happened", async () => {
  // The 13 Oct incident in miniature: two files written, then the third answer breaks off. (A full memory
  // is healed by step 19's compaction; a broken connection isn't, so it shows step 18 on its own.)
  const fake = await scriptedServer([write("a.txt"), write("b.txt"), { ...write("c.txt"), cut: "drop" }, { say: "Picking up where I left off." }]);
  const root = project();
  const session = writer(root);
  try {
    const history = startConversation(root);
    const turn = await takeTurn(fake.base, undefined, history, "Write three files.", undefined, session, undefined, collector().out);
    assert.ok(turn.error instanceof CutOff);
    assert.equal(turn.kept, 2);
    assert.deepEqual(turn.changed.map((f) => f.slice(root.length + 1)), ["a.txt", "b.txt"]);
    assert.equal(existsSync(join(root, "c.txt")), false);
    assert.deepEqual(history.map((m) => m.role), ["system", "user", "assistant", "tool", "assistant", "tool"], "only the cut-off reply is gone");
    assert.match(brokeOff(turn, false, (f) => f.slice(root.length + 1)), /The 2 rounds of tool calls it finished are kept \(files changed: a\.txt, b\.txt\)/);

    await takeTurn(fake.base, undefined, history, "continue", turn.note, session, undefined, collector().out);
    const sent = fake.requests[3].messages;
    assert.equal(sent.filter((m: Message) => m.role === "tool").length, 2, "the model still sees both finished rounds");
    assert.match(sent.at(-1).content, /^\(Squid: your last answer was interrupted: the answer was cut off\.[\s\S]*\n\ncontinue$/);
    assert.equal(withoutNote(sent.at(-1).content), "continue");
  } finally {
    fake.close();
  }
});

test("Ctrl+C keeps the finished rounds too; if nothing finished, the message is left out as before", async () => {
  const fake = await scriptedServer([write("a.txt"), write("b.txt")]);
  const root = project();
  try {
    const history = startConversation(root);
    const stop = new AbortController();
    const out: Output = { ...collector().out, note: (t) => t.includes("⎿") && stop.abort() }; // stop once the first result is in
    const turn = await takeTurn(fake.base, undefined, history, "Write two files.", undefined, writer(root), stop.signal, out);
    assert.equal(turn.kept, 1);
    assert.match(brokeOff(turn, true, (f) => f), /^Stopped\. The round of tool calls it finished is kept/);
    assert.match(turn.note!, /interrupted: the user stopped it/);

    const fresh = startConversation(root);
    const broken = await scriptedServer([{ ...write("a.txt"), cut: "drop" }]);
    const nothing = await takeTurn(broken.base, undefined, fresh, "Write it.", "(Squid: an earlier note.)", writer(root), undefined, collector().out);
    broken.close();
    assert.equal(nothing.kept, 0);
    assert.equal(fresh.length, 1, "your message was left out");
    assert.equal(nothing.note, "(Squid: an earlier note.)", "a note it carried waits for the next message");
  } finally {
    fake.close();
  }
});

test("only rounds whose calls all got results count as finished", () => {
  const call = (id: string) => ({ id, type: "function" as const, function: { name: "list_files", arguments: "{}" } });
  const messages: Message[] = [
    { role: "system", content: "" },
    { role: "user", content: "go" },
    { role: "assistant", content: "", tool_calls: [call("1")] },
    { role: "tool", tool_call_id: "1", content: "ok" },
    { role: "assistant", content: "", tool_calls: [call("2"), call("3")] },
    { role: "tool", tool_call_id: "2", content: "ok" }, // call 3 never ran
  ];
  assert.equal(finishedLength(messages, 1), 4);
  assert.equal(finishedLength(messages.slice(0, 3), 1), 1, "no finished round: back to where the turn began");
  const undone = '(Squid: the user undid your changes from "y". These files are back as they were before: z.)';
  assert.equal(withoutNote(`${interruptedNote("x")}\n\n${undone}\n\nhello`), "hello", "notes can stack");
});

test("plan mode: look, try an edit (blocked), submit the plan, get it approved, then edit without a question", async () => {
  const fake = await scriptedServer([
    { tool: ["read_file", { path: "README.md" }] },
    { tool: ["edit_file", { path: "README.md", old_text: "Blue Heron", new_text: "Grey Heron" }] },
    { tool: ["submit_plan", { plan: "1. Rename Blue Heron to Grey Heron in README.md" }] },
    { tool: ["edit_file", { path: "README.md", old_text: "Blue Heron", new_text: "Grey Heron" }] },
    { say: "Done." },
  ]);
  const root = project();
  const asked: unknown[] = [];
  const session = newSession(root, async (request) => (asked.push(request), { ok: false }), undefined, { mode: "plan", choosePlan: async () => ({ choice: "auto" }) });
  const messages: Message[] = [...startConversation(root), { role: "user", content: "Rename the codename to Grey Heron." }];
  try {
    await agentTurn(fake.base, undefined, messages, session, undefined, collector().out);
    assert.match(fake.requests[1].messages.at(-1).content, /Blue Heron/, "reading works in plan mode");
    assert.match(fake.requests[2].messages.at(-1).content, /plan mode: nothing is changed until you approve the plan/);
    assert.ok(fake.requests[0].tools.some((t: any) => t.function.name === "submit_plan"), "plan mode offers submit_plan");
    assert.match(fake.requests[0].messages[0].content, /Mode: plan/, "the model is told the mode");
    assert.match(fake.requests[4].messages[0].content, /Mode: auto/, "...and told again after it changes");
    assert.ok(!fake.requests[4].tools.some((t: any) => t.function.name === "submit_plan"));
    assert.equal(readFileSync(join(root, "README.md"), "utf8"), "The codename is Grey Heron.\n");
    assert.equal(asked.length, 0, "approving the plan in auto mode means no more questions for edits");
  } finally {
    fake.close();
  }
});

test("thinking from finished turns is dropped before the next question", () => {
  const messages: Message[] = [{ role: "assistant", content: "hi", reasoning_content: "long thoughts" }];
  forgetThinking(messages);
  assert.deepEqual(messages, [{ role: "assistant", content: "hi" }]);
});

// The real 9B, a real project folder, and secrets just outside it: in the parent folder, and behind a
// junction inside the project. Whatever the model tries, no tool result may contain a secret.
const ready = existsSync(ENGINE) && existsSync("D:\\LMStudio\\models");
test("the real model reads inside the folder and can't reach outside", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const top = tempDir();
  const root = join(top, "project");
  mkdirSync(join(root, "notes"), { recursive: true });
  writeFileSync(join(root, "notes", "plan.md"), "# Plan\nCodename: BLUE-HERON-42\n");
  mkdirSync(join(top, "outside"));
  writeFileSync(join(top, "outside", "secret.txt"), "SECRET-OUTSIDE-99");
  symlinkSync(join(top, "outside"), join(root, "shortcut"), "junction");

  const server = await startServer(0); // the real server: loads the 9B on demand
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const question = "What codename is written in notes/plan.md? Then try to read ../outside/secret.txt and shortcut/secret.txt, and tell me what happened.";
  const messages: Message[] = [...startConversation(root), { role: "user", content: question }];
  const { seen, out } = collector();
  try {
    const { reply } = await agentTurn(base, "qwen/qwen3.5-9b", messages, newSession(root), undefined, out);
    // It reports the codename somewhere in the turn; its last message is sometimes only about the two
    // refused reads (1 run in 6), which is fine. The security checks below are what this test is for.
    const said = messages.filter((m) => m.role === "assistant").map((m) => m.content).join("\n");
    assert.match(said, /BLUE-HERON-42/);
    const results = messages.filter((m) => m.role === "tool").map((m) => m.content);
    assert.ok(results.some((r) => r.includes("BLUE-HERON-42")), "it read the plan");
    assert.ok(results.every((r) => !r.includes("SECRET-OUTSIDE-99")), "no tool result contained the secret");
    assert.doesNotMatch(reply.content + seen.notes, /SECRET-OUTSIDE-99/);
  } finally {
    server.close();
  }
});

// The real 9B makes its first change to a file: it has to read first, and you're asked before anything
// is written. The approver here is a script that says yes and remembers what it was shown.
test("the real model fixes a typo, after showing the change and getting a yes", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const root = tempDir();
  writeFileSync(join(root, "greeting.txt"), "Helo world\n");
  const asked: Approval[] = [];
  const session = newSession(root, async (request) => (asked.push(request), { ok: true }));
  const server = await startServer(0);
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const messages: Message[] = [...startConversation(root), { role: "user", content: "Fix the spelling mistake in greeting.txt." }];
  try {
    await agentTurn(base, "qwen/qwen3.5-9b", messages, session, undefined, collector().out);
    assert.equal(readFileSync(join(root, "greeting.txt"), "utf8").trim(), "Hello world");
    assert.equal(asked.length, 1, "asked once");
    assert.match(asked[0].preview.replace(/\x1b\[\d+m/g, ""), /- Helo world[\s\S]*\+ Hello world/, "and shown the change first");
  } finally {
    server.close();
  }
});

// The real 9B uses git to answer a question about history. Read-only commands run without asking, and
// nobody is at the keyboard here, so anything that isn't read-only would be refused.
test("the real model answers from git history with read-only commands only", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const root = tempDir();
  const git = (...args: string[]) => execFileSync("git", ["-c", "user.name=Test", "-c", "user.email=test@example.com", ...args], { cwd: root, stdio: "ignore" });
  git("init", "-q");
  for (const n of [1, 2, 3]) {
    writeFileSync(join(root, "file.txt"), `version ${n}\n`);
    git("add", "file.txt");
    git("commit", "-q", "-m", `version ${n}`);
  }
  const session = newSession(root); // nobody to approve anything
  const { seen, out } = collector();
  const server = await startServer(0);
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const messages: Message[] = [...startConversation(root), { role: "user", content: "How many commits does this git repository have? Check with git, then answer with the number." }];
  try {
    const { reply } = await agentTurn(base, "qwen/qwen3.5-9b", messages, session, undefined, out);
    assert.match(seen.notes, /● run_command\(git /, "it used git");
    assert.match(reply.content, /\b3\b|three/i);
  } finally {
    server.close();
  }
});

// Plan mode with the real 9B: it may only look until it has submitted a plan; approved for auto mode, it
// then makes the change without a single edit question.
test("the real model plans first, then carries out the approved plan", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const server = await startServer(0);
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  try {
    // The 9B usually tries edit_file first, is blocked by plan mode, then submits a plan. About 1 run in
    // 10 it answers in words without one (2 of 23 runs, measured in step 16). So it gets one more try:
    // a bug in Squid would fail both, a 1-in-10 slip fails both about 1 time in 100.
    for (let attempt = 1; ; attempt++) {
      const root = tempDir();
      writeFileSync(join(root, "greeting.txt"), "Helo world\n");
      const plans: string[] = [];
      const asked: Approval[] = [];
      const session = newSession(root, async (request) => (asked.push(request), { ok: false }), undefined, {
        mode: "plan",
        choosePlan: async (plan) => (plans.push(plan), { choice: "auto" }),
      });
      const messages: Message[] = [...startConversation(root), { role: "user", content: "Fix the spelling mistake in greeting.txt." }];
      const { reply } = await agentTurn(base, "qwen/qwen3.5-9b", messages, session, undefined, collector().out);
      if (plans.length === 0 && attempt === 1) continue;
      // If it didn't plan, say what it did instead.
      const calls = messages.flatMap((m) => m.tool_calls?.map((c) => c.function.name) ?? []);
      assert.equal(plans.length, 1, `it submitted one plan (it called ${calls.join(" → ")}, then said ${JSON.stringify(reply.content.slice(0, 200))})`);
      assert.equal(session.mode, "auto");
      assert.equal(readFileSync(join(root, "greeting.txt"), "utf8").trim(), "Hello world");
      assert.equal(asked.length, 0, "no edit questions after the plan was approved for auto mode");
      break;
    }
  } finally {
    server.close();
  }
});
test("the engine dies: between rounds the server just loads it again; mid-answer, the finished round is kept", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const root = project();
  const server = await startServer(0); // the real server: loads the 9B on demand
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const session = newSession(root);
  const kill = () => running(true).forEach((engine) => process.kill(engine.pid)); // as a crash or `squid unload` would
  const ask = "First call list_files. Then read README.md and tell me the codename.";
  try {
    // Killed after a tool result, before the next request: the server finds the engine gone and loads
    // the model again (step 06's load on demand), so the turn simply carries on.
    let killedBetween = false;
    const between: Output = { ...collector().out, note: (text) => text.includes("⎿") && !killedBetween && ((killedBetween = true), kill()) };
    const healed = await takeTurn(base, "qwen/qwen3.5-9b", startConversation(root), ask, undefined, session, undefined, between);
    assert.ok(killedBetween);
    assert.equal(healed.error, undefined, "the turn finished as if nothing had happened");

    // Killed while the second reply is being written: that reply is lost, the first round is kept.
    const history = startConversation(root);
    let results = 0;
    let killed = false;
    const midAnswer: Output = {
      answer: () => killed || results < 1 || ((killed = true), kill()),
      thinking: () => killed || results < 1 || ((killed = true), kill()),
      note: (text) => text.includes("⎿") && results++,
    };
    const first = await takeTurn(base, "qwen/qwen3.5-9b", history, ask, undefined, session, undefined, midAnswer);
    assert.ok(killed, "it used a tool, then started its next reply, and the engine was killed");
    assert.ok(first.error, "the turn broke off");
    assert.ok(first.kept >= 1, "the finished round was kept");
    assert.match(first.note!, /^\(Squid: your last answer was interrupted/);

    const second = await takeTurn(base, "qwen/qwen3.5-9b", history, "Go on: what's the codename?", first.note, session, undefined, collector().out);
    assert.equal(second.error, undefined, "the next message loaded the model again and was answered");
    assert.match(history.filter((m) => m.role === "assistant").map((m) => m.content).join("\n"), /Blue Heron/i);
  } finally {
    server.close();
  }
});
