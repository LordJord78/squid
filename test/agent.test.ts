import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer } from "node:http";
import { existsSync, mkdirSync, readFileSync, symlinkSync, writeFileSync } from "node:fs";
import { execFileSync } from "node:child_process";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { agentTurn, forgetThinking, MAX_ROUNDS, startConversation } from "../src/agent.ts";
import type { Message, Output } from "../src/reply.ts";
import { startServer } from "../src/server.ts";
import { newSession, type Approval } from "../src/tools.ts";
import { ENGINE, unload } from "../src/engine.ts";
import { tempDir } from "./helpers.ts";

// A stand-in for Squid's server that follows a script: each request gets the next step, either a tool
// call or an answer, streamed in LM Studio's format with the arguments cut into 5-character pieces.
type Step = { tool: [string, object] } | { say: string };
async function scriptedServer(script: Step[]) {
  const requests: any[] = [];
  const server = createServer(async (req, res) => {
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

test("a model that never stops calling tools is stopped", async () => {
  const fake = await scriptedServer([{ tool: ["list_files", {}] }]); // the same request forever
  const root = project();
  try {
    await assert.rejects(agentTurn(fake.base, undefined, [{ role: "user", content: "hi" }], newSession(root), undefined, collector().out), /stopped after 25 rounds/);
    assert.equal(fake.requests.length, MAX_ROUNDS);
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
    assert.match(reply.content, /BLUE-HERON-42/);
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