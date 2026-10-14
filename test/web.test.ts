import { test } from "node:test";
import assert from "node:assert/strict";
import { request } from "node:http";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { startWeb, type Event, type Web } from "../src/web.ts";
import { sseData } from "../src/sse.ts";
import { startServer } from "../src/server.ts";
import { ENGINE, unload } from "../src/engine.ts";
import type { Mode } from "../src/rules.ts";
import { scriptedServer, tempDir, type Step } from "./helpers.ts";

type Ask = Extract<Event, { type: "ask" }>;

// What the page's script does: every request carries the key, and POSTs are JSON.
function client(web: Web) {
  const call = (path: string, body?: object, signal?: AbortSignal) =>
    fetch(`http://127.0.0.1:${web.port}${path}`, {
      method: body ? "POST" : "GET",
      headers: { Authorization: `Bearer ${web.token}`, ...(body && { "Content-Type": "application/json" }) },
      body: body && JSON.stringify(body),
      signal,
    });
  // Sends a message and reads the turn's events, answering each question the way `answer` says.
  async function say(text: string, answer: (event: Event) => string = () => "n") {
    const res = await call("/api/send", { text });
    assert.equal(res.status, 200);
    const events: Event[] = [];
    for await (const data of sseData(res.body!)) {
      const event = JSON.parse(data) as Event;
      events.push(event);
      if (event.type === "ask" || event.type === "plan") await call("/api/answer", { id: event.id, answer: answer(event) });
    }
    return events;
  }
  return { call, say };
}

// The page for a fresh folder (its README says Blue Heron), talking to a scripted model.
async function page(script: Step[], mode: Mode = "manual") {
  const fake = await scriptedServer(script);
  const root = tempDir();
  writeFileSync(join(root, "README.md"), "The codename is Blue Heron.\n");
  const web = await startWeb({ base: fake.base, root, mode });
  return {
    ...client(web), fake, root, web,
    readme: () => readFileSync(join(root, "README.md"), "utf8"),
    close: () => (web.close(), fake.close()),
  };
}
const RENAME: Step[] = [
  { tool: ["read_file", { path: "README.md" }] },
  { tool: ["edit_file", { path: "README.md", old_text: "Blue Heron", new_text: "Grey Heron" }] },
  { say: "Done." },
];

// A request with headers as an attacker's page would make the browser send them. (fetch() won't let a
// script set Host, so this uses node:http.)
function raw(port: number, path: string, headers: Record<string, string>, method = "GET"): Promise<number> {
  return new Promise((resolve, reject) => {
    const req = request({ host: "127.0.0.1", port, path, method, headers }, (res) => (res.resume(), resolve(res.statusCode!)));
    req.on("error", reject);
    req.end(method === "POST" ? "{}" : undefined);
  });
}

test("the page itself is served to this PC; everything else needs the key from the link", async () => {
  const p = await page([{ say: "hi" }]);
  try {
    const home = `http://127.0.0.1:${p.web.port}`;
    const html = await fetch(`${home}/`);
    assert.equal(html.status, 200);
    assert.match(html.headers.get("content-security-policy")!, /script-src 'self'.*frame-ancestors 'none'/);
    assert.equal(html.headers.get("referrer-policy"), "no-referrer");
    assert.equal((await fetch(`${home}/page.js`)).headers.get("content-type"), "text/javascript; charset=utf-8");
    assert.equal((await fetch(`${home}/api/state`)).status, 401, "no key");
    assert.equal((await fetch(`${home}/api/state`, { headers: { Authorization: `Bearer ${"0".repeat(48)}` } })).status, 401, "a wrong key");
    const state = await (await p.call("/api/state")).json();
    assert.deepEqual([state.root, state.mode, state.busy, state.messages], [p.root, "manual", false, []]);
    assert.match(p.web.url, /^http:\/\/127\.0\.0\.1:\d+\/#token=[0-9a-f]{48}$/, "the key is after #, so no server ever receives the link's key");
  } finally {
    p.close();
  }
});

test("other websites, other names for this PC and plain forms are refused, even with the key", async () => {
  const p = await page([{ say: "hi" }]);
  try {
    const port = p.web.port;
    const key = { Authorization: `Bearer ${p.web.token}` };
    const here = `127.0.0.1:${port}`;
    assert.equal(await raw(port, "/api/state", { ...key, Host: "evil.example" }), 403, "DNS rebinding: another site's name pointed at 127.0.0.1");
    assert.equal(await raw(port, "/api/state", { ...key, Host: here, Origin: "http://evil.example" }), 403, "a page on another website");
    assert.equal(await raw(port, "/api/state", { ...key, Host: here, Origin: "http://127.0.0.1:1234" }), 403, "another port is another website");
    assert.equal(await raw(port, "/api/send", { ...key, Host: here, "Content-Type": "text/plain" }, "POST"), 415, "a plain form can't send JSON");
    assert.equal(await raw(port, "/api/state", { ...key, Host: `localhost:${port}`, Origin: `http://localhost:${port}` }), 200, "the page itself is fine");
  } finally {
    p.close();
  }
});

test("a turn streams to the page, and an edit waits for the page's Yes", async () => {
  const p = await page(RENAME);
  try {
    const events = await p.say("Rename it to Grey Heron.", () => "y");
    const ask = events.find((e): e is Ask => e.type === "ask")!;
    assert.equal(ask.title, "Edit README.md");
    assert.match(ask.preview, /- The codename is Blue Heron\.[\s\S]*\+ The codename is Grey Heron\./);
    assert.doesNotMatch(ask.preview, /\x1b/, "no terminal colour codes");
    assert.equal(p.readme(), "The codename is Grey Heron.\n");
    // In the order it happened: thinking, the read, the edit (with its question), then the answer.
    const order = events.map((e) => (e.type === "note" ? e.text.trim().split("\n")[0] : e.type));
    assert.deepEqual(order.filter((x) => x !== "thinking"), ["● read_file(README.md)", "⎿ 2 lines", "● edit_file(README.md)", "ask", "⎿ 1 line", "answer", "answer", "done"]);
    assert.deepEqual(events.at(-1), { type: "done", model: "fake", used: 0, limit: 8192, mode: "manual" });
    const state = await (await p.call("/api/state")).json();
    assert.deepEqual(state.messages, [{ role: "user", text: "Rename it to Grey Heron." }, { role: "assistant", text: "Done." }], "a reloaded page shows the conversation");
  } finally {
    p.close();
  }
});

test("a No with instructions from the page goes back to the model; nothing is changed", async () => {
  const p = await page(RENAME);
  try {
    await p.say("Rename it.", () => "call it Grey Heron instead");
    assert.equal(p.readme(), "The codename is Blue Heron.\n");
    assert.match(p.fake.requests[2].messages.at(-1).content, /declined: Edit README\.md\. Their instructions: call it Grey Heron instead/);
  } finally {
    p.close();
  }
});

test("closing the tab while a question is open counts as No and ends the turn; the next message works", async () => {
  const p = await page([...RENAME.slice(0, 2), { say: "Hello again." }]);
  try {
    const tab = new AbortController();
    const res = await p.call("/api/send", { text: "Rename it." }, tab.signal);
    for await (const data of sseData(res.body!)) if (JSON.parse(data).type === "ask") break;
    tab.abort(); // the tab is gone
    for (let i = 0; i < 100 && (await (await p.call("/api/state")).json()).busy; i++) await new Promise((r) => setTimeout(r, 20));
    assert.equal(p.readme(), "The codename is Blue Heron.\n", "nobody clicked Yes");
    // Both rounds finished (a read, and an edit refused when the tab closed), so they're kept (step 18).
    assert.deepEqual((await (await p.call("/api/state")).json()).messages, [{ role: "user", text: "Rename it." }], "the finished rounds were kept");
    const events = await p.say("Hi");
    assert.equal(p.fake.requests.length, 3, "nothing more was asked of the model after the tab closed");
    assert.match(p.fake.requests[2].messages.at(-1).content, /^\(Squid: your last answer was interrupted: the user stopped it\.[\s\S]*\n\nHi$/);
    assert.ok(events.some((e) => e.type === "answer" && e.text === "Hello again."));
  } finally {
    p.close();
  }
});

test("/compact from the page summarises; after a reload the page shows a note where the conversation was", async () => {
  const p = await page([{ say: "Hello." }, { say: "SUMMARY: we said hello." }]);
  try {
    assert.ok((await p.say("/compact")).some((e) => e.type === "note" && /Nothing to compact yet/.test(e.text)));
    await p.say("Hi");
    const events = await p.say("/compact greetings");
    assert.equal(p.fake.requests.length, 2, "the summary was the only request; /compact itself never reached the model as a message");
    assert.match(p.fake.requests[1].messages.at(-1).content, /Pay special attention to: greetings/);
    assert.ok(events.some((e) => e.type === "note" && /compacted into a summary/.test(e.text)));
    assert.ok(events.some((e) => e.type === "done"));
    assert.deepEqual((await (await p.call("/api/state")).json()).messages, [{ role: "note", text: "The conversation before this point was compacted into a summary." }]);
  } finally {
    p.close();
  }
});

test("one turn at a time: while it's answering, sending, changing mode and starting over wait", async () => {
  const p = await page(RENAME);
  try {
    const res = await p.call("/api/send", { text: "Rename it." });
    for await (const data of sseData(res.body!)) {
      const event = JSON.parse(data) as Event;
      if (event.type !== "ask") continue;
      assert.equal((await p.call("/api/send", { text: "and another thing" })).status, 409);
      assert.equal((await p.call("/api/mode", { mode: "auto" })).status, 409);
      assert.equal((await p.call("/api/clear", {})).status, 409);
      assert.equal((await p.call("/api/answer", { id: "not-a-question", answer: "y" })).status, 404, "only open questions can be answered");
      await p.call("/api/answer", { id: event.id, answer: "n" });
    }
    assert.equal((await p.call("/api/mode", { mode: "auto" })).status, 200, "after the turn, it's fine");
  } finally {
    p.close();
  }
});

test("plan mode in the page: the plan is approved with 1, then the edit goes ahead without a question", async () => {
  const p = await page([RENAME[0], { tool: ["submit_plan", { plan: "1. Rename Blue Heron to Grey Heron in README.md" }] }, RENAME[1], { say: "Done." }], "plan");
  try {
    const asked: string[] = [];
    const events = await p.say("Rename it to Grey Heron.", (e) => (asked.push(e.type), "1"));
    assert.deepEqual(asked, ["plan"], "one question: the plan");
    assert.equal(p.readme(), "The codename is Grey Heron.\n");
    assert.deepEqual(events.at(-1), { type: "done", model: "fake", used: 0, limit: 8192, mode: "auto" }, "the page learns the new mode");
  } finally {
    p.close();
  }
});

test("Undo from the page: see what it would do, confirm it, and the model hears about it", async () => {
  const p = await page([{ tool: ["write_file", { path: "notes.txt", content: "hello" }] }, { say: "Created." }, { say: "OK." }], "auto");
  try {
    await p.say("Create notes.txt");
    assert.ok(existsSync(join(p.root, "notes.txt")));
    const plan = await (await p.call("/api/undo")).json();
    assert.deepEqual([plan.files, plan.label], [1, "Create notes.txt"]);
    assert.match(plan.text, /notes\.txt: delete it \(the agent created it\)/);
    assert.equal((await p.call("/api/undo", { label: "a different turn" })).status, 409, "only the undo you were shown");
    assert.deepEqual(await (await p.call("/api/undo", { label: plan.label })).json(), { undone: 1 });
    assert.equal(existsSync(join(p.root, "notes.txt")), false);
    await p.say("What now?");
    assert.match(p.fake.requests.at(-1).messages.at(-1).content, /^\(Squid: the user undid your changes from "Create notes\.txt"\.[^]*\)\n\nWhat now\?$/);
    const shown = (await (await p.call("/api/state")).json()).messages.map((m: any) => m.text);
    assert.deepEqual(shown, ["Create notes.txt", "Created.", "What now?", "OK."], "the note is for the model, not for the page");
  } finally {
    p.close();
  }
});

test("the page puts what the model says on screen as text, never as HTML", () => {
  // A file the model reads could hold "<img src=x onerror=…>". These are the ways a script turns text
  // into HTML or code; page.js uses none of them (it uses textContent). The policy header is the
  // second lock: even if one slipped in, inline scripts don't run.
  const script = readFileSync(new URL("../src/page/page.js", import.meta.url), "utf8");
  assert.doesNotMatch(script, /innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|new Function|setAttribute\(\s*["']on/);
  const html = readFileSync(new URL("../src/page/index.html", import.meta.url), "utf8");
  assert.deepEqual(html.match(/<script[^>]*>/g), ['<script src="/page.js">'], "one script, from a file, none inline");
});

// The real 9B, through the page: in manual mode it has to ask the page before creating a file.
const ready = existsSync(ENGINE) && existsSync("D:\\LMStudio\\models");
test("the real 9B through the page: it asks before creating a file, and the page's Yes lets it", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  t.after(() => unload("qwen/qwen3.5-9b"));
  const server = await startServer(0); // the real server: loads the 9B on demand
  const root = tempDir();
  const web = await startWeb({ base: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, root, mode: "manual", model: "qwen/qwen3.5-9b" });
  try {
    const asked: Ask[] = [];
    const events = await client(web).say("Create a file named notes.txt containing the word hello.", (e) => (e.type === "ask" && asked.push(e), "y"));
    assert.ok(asked.some((a) => /^(Create|Overwrite) notes\.txt$/.test(a.title)), "the page was asked first");
    assert.match(readFileSync(join(root, "notes.txt"), "utf8"), /hello/i);
    const done = events.at(-1)!;
    assert.equal(done.type, "done");
    assert.ok(done.type === "done" && done.used > 0 && done.limit! > 0, "the meter has numbers");
  } finally {
    web.close();
    server.close();
  }
});
