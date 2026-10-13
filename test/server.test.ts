import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { execFileSync } from "node:child_process";
import { fileURLToPath } from "node:url";
import { existsSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { startServer, type Sources } from "../src/server.ts";
import { serverStatus, ensureServer, stopServer } from "../src/daemon.ts";
import { ENGINE, load, running, unload, type Engine } from "../src/engine.ts";
import { listModels, modelRoots, type ShelfModel } from "../src/models.ts";

const golden = (path: string) => JSON.parse(readFileSync(new URL(`../golden/${path}`, import.meta.url), "utf8"));
const portOf = (server: Server) => (server.address() as AddressInfo).port;

// The structure of a JSON value with the values blanked out: {a: 1, b: [{c: "x"}]} → {a: "number", b: [{c: "string"}]}.
// Two replies with the same shape have the same fields, nested the same way, whatever the model said.
function shape(value: unknown): unknown {
  if (Array.isArray(value)) return value.length ? [shape(value[0])] : [];
  if (value === null) return "null";
  if (typeof value === "object") return Object.fromEntries(Object.entries(value).map(([k, v]) => [k, shape(v)]));
  return typeof value;
}

// A stand-in for llama-server: answers with the recorded raw replies, and only with the right key.
// Streams are sent in 100-byte pieces, so events arrive cut at random places, as over a network.
async function fakeEngine(): Promise<{ server: Server; engine: Engine }> {
  const server = createServer(async (req, res) => {
    if (req.headers.authorization !== "Bearer test-key") return res.writeHead(401).end();
    let body = "";
    for await (const chunk of req) body += chunk;
    if (req.url === "/tokenize") return res.end(JSON.stringify({ tokens: new Array(129).fill(0) }));
    const request = JSON.parse(body);
    if (!request.stream) return res.end(JSON.stringify(golden("llamacpp/chat-nonstream.json")));
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    const recording = request.tools ? "chat-tools-stream.sse" : "chat-stream-usage.sse";
    const bytes = readFileSync(new URL(`../golden/llamacpp/${recording}`, import.meta.url));
    for (let i = 0; i < bytes.length; i += 100) res.write(bytes.subarray(i, i + 100));
    res.end();
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  return { server, engine: engineRecord(portOf(server)) };
}

// The record `squid load` would have written for an engine on this port.
const engineRecord = (port: number): Engine => ({
  identifier: "qwen/qwen3.5-9b", key: "qwen/qwen3.5-9b", pid: process.pid, port, apiKey: "test-key",
  contextLength: 8192, parallel: 4, sizeBytes: 1, architecture: "qwen35", quantization: "Q4_K_M", log: "",
  loader: process.pid, jit: false, lastUsed: Date.now(),
});

// No sources: the real shelf and engines, loading on demand.
async function withServer(sources: Sources | undefined, run: (base: string) => Promise<void>) {
  const server = await startServer(0, sources);
  try {
    await run(`http://127.0.0.1:${portOf(server)}`);
  } finally {
    server.close();
  }
}

const post = (url: string, body: string, type = "application/json") => fetch(url, { method: "POST", headers: { "Content-Type": type }, body });
const nothing: Sources = { shelf: () => [], engines: () => [] };

test("unknown routes get LM Studio's answer: 200 and the same message", () =>
  withServer(nothing, async (base) => {
    const res = await fetch(`${base}/v1/does-not-exist`);
    assert.equal(res.status, 200);
    assert.equal(await res.text(), readFileSync(new URL("../golden/lmstudio/v1-models-bad-route.json", import.meta.url), "utf8").trim());
  }));

test("bad requests get LM Studio's errors", () =>
  withServer(nothing, async (base) => {
    const url = `${base}/v1/chat/completions`;
    const broken = await post(url, "{not json");
    assert.equal(broken.status, 400);
    assert.equal((await broken.json()).error.code, "invalid_json");
    const noMessages = await post(url, '{"model":"x"}');
    assert.deepEqual([noMessages.status, await noMessages.json()], [400, { error: "'messages' field is required" }]);
    assert.equal((await post(url, "{}", "text/plain")).status, 415);
  }));

test("only requests addressed to localhost, and only from Squid's own pages", () =>
  withServer(nothing, async (base) => {
    // fetch won't let a script change Host, so this one goes through node:http directly
    const { request } = await import("node:http");
    const status = await new Promise((resolve) =>
      request(`${base}/v1/models`, { headers: { Host: "evil.example" } }, (res) => resolve(res.statusCode)).end());
    assert.equal(status, 403, "DNS rebinding: wrong Host");
    assert.equal((await fetch(`${base}/v1/models`, { headers: { Origin: "https://evil.example" } })).status, 403, "another website");
    assert.equal((await fetch(`${base}/v1/models`, { headers: { Origin: base } })).status, 200, "Squid's own page");
  }));

test("a chat reply is rebuilt in LM Studio's shape", async () => {
  const { server, engine } = await fakeEngine();
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const ask = JSON.stringify({ model: "qwen/qwen3.5-9b", messages: [{ role: "user", content: "Reply with exactly one word: pong" }] });
      const v1 = await (await post(`${base}/v1/chat/completions`, ask)).json();
      assert.deepEqual(shape(v1), shape(golden("lmstudio/chat-nonstream.json")), "same fields as LM Studio's /v1 reply");
      assert.deepEqual(Object.keys(v1), Object.keys(golden("lmstudio/chat-nonstream.json")), "in the same order");
      assert.equal(v1.choices[0].message.content, "pong");
      assert.equal(v1.usage.completion_tokens_details.reasoning_tokens, 129, "counted with the engine's tokenizer");
      assert.match(v1.id, /^chatcmpl-[a-z0-9]{22}$/);

      const v0 = await (await post(`${base}/api/v0/chat/completions`, ask)).json();
      assert.deepEqual(shape(v0), shape(golden("lmstudio/api-v0-chat.json")), "same fields as LM Studio's /api/v0 reply");
      assert.equal(v0.stats.stop_reason, "eosFound");
      assert.deepEqual(v0.model_info, { arch: "qwen35", quant: "Q4_K_M", format: "gguf", context_length: 8192 });
    });
  } finally {
    server.close();
  }
});

// Splits a recorded stream into its events: the JSON chunks, with "[DONE]" checked and removed.
function sseEvents(text: string) {
  const events = text.split("\n\n").filter(Boolean).map((event) => event.replace(/^data: /, ""));
  assert.equal(events.at(-1), "[DONE]");
  return events.slice(0, -1).map((data) => JSON.parse(data));
}
// What each chunk carries: "thinking", "answer", "end" (finish_reason) or "usage".
const kinds = (chunks: any[]) =>
  chunks.map((c) => (c.usage ? "usage" : c.choices[0].finish_reason ? "end" : c.choices[0].delta.content ? "answer" : "thinking"));
const squeeze = (list: string[]) => list.filter((kind, i) => kind !== list[i - 1]); // thinking,thinking,answer → thinking,answer

test("a streamed reply is passed on event by event, in LM Studio's shape", async () => {
  const { server, engine } = await fakeEngine();
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const ask = { model: "qwen/qwen3.5-9b", messages: [{ role: "user", content: "Reply with exactly one word: pong" }], stream: true, stream_options: { include_usage: true } };
      const res = await post(`${base}/v1/chat/completions`, JSON.stringify(ask));
      assert.equal(res.headers.get("content-type"), "text/event-stream");
      const text = await res.text();
      assert.match(text, /^(data: [^\n]+\n\n)+$/, "every event is one data: line and a blank line");

      const ours = sseEvents(text);
      const lmstudio = sseEvents(readFileSync(new URL("../golden/lmstudio/chat-stream-usage.sse", import.meta.url), "utf8"));
      assert.deepEqual(squeeze(kinds(ours)), ["thinking", "answer", "end", "usage"]);
      assert.deepEqual(squeeze(kinds(ours)), squeeze(kinds(lmstudio)), "same order as LM Studio");
      // Same fields as LM Studio's chunk of the same kind; the first one also carries the role.
      for (const kind of ["thinking", "answer", "end", "usage"]) {
        assert.deepEqual(shape(ours.find((c, i) => kinds(ours)[i] === kind)), shape(lmstudio.find((c, i) => kinds(lmstudio)[i] === kind)), kind);
      }
      assert.deepEqual(Object.keys(ours[0].choices[0].delta), ["role", "reasoning_content"], "role rides on the first real event");
      assert.equal(new Set(ours.map((c) => c.id)).size, 1, "one id for the whole reply");
      assert.equal(ours.map((c) => c.choices[0]?.delta.content ?? "").join(""), "pong");
      assert.equal(ours.at(-1).usage.completion_tokens_details.reasoning_tokens, 129);
    });
  } finally {
    server.close();
  }
});

test("a streamed tool call is reshaped into LM Studio's three parts", async () => {
  const { server, engine } = await fakeEngine();
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const ask = { model: "qwen/qwen3.5-9b", messages: [{ role: "user", content: "Use the read_file tool to open README.md." }], tools: [{ type: "function" }], stream: true };
      const ours = sseEvents(await (await post(`${base}/v1/chat/completions`, JSON.stringify(ask))).text());
      const lmstudio = sseEvents(readFileSync(new URL("../golden/lmstudio/chat-tools-stream.sse", import.meta.url), "utf8"));
      const toolEvents = (chunks: any[]) => chunks.filter((c) => c.choices[0]?.delta.tool_calls);
      const [oursNamed, ...oursArgs] = toolEvents(ours);
      const [theirsNamed, theirsArgs] = toolEvents(lmstudio);
      // 1. the name, with empty arguments
      assert.deepEqual(shape(oursNamed), shape(theirsNamed));
      assert.equal(oursNamed.choices[0].delta.tool_calls[0].function.arguments, "");
      // 2. the arguments in pieces, each marked type "function"
      for (const piece of oursArgs) assert.deepEqual(shape(piece), shape(theirsArgs));
      const args = oursArgs.map((c: any) => c.choices[0].delta.tool_calls[0].function.arguments).join("");
      assert.deepEqual(JSON.parse(args), { path: "README.md" });
      // 3. an empty event that says why it stopped
      assert.equal(ours.at(-1).choices[0].finish_reason, "tool_calls");
      assert.deepEqual(ours.at(-1).choices[0].delta, {});
    });
  } finally {
    server.close();
  }
});

test("a stray </think> in an answer is removed, even when it arrives in pieces", async () => {
  // An engine that sends each answer as the pieces given, streamed or whole.
  const pieces: Record<string, string[]> = {
    quirk: ["</", "think>", "\n\n", "pur", "ple"],
    repeat: ["purple\n", "</thi", "nk>", "\n", "\n", "purple"], // what the 9B really sends, 8 times in 20
    normal: ["\n\n", "<b>", "Hello", "</b>", " <", "3"],
  };
  const engine = createServer(async (req, res) => {
    let body = "";
    for await (const chunk of req) body += chunk;
    const request = JSON.parse(body);
    const parts = pieces[request.messages[0].content];
    if (!request.stream) {
      return res.end(JSON.stringify({ created: 1, choices: [{ index: 0, message: { role: "assistant", content: parts.join("") }, finish_reason: "stop" }], usage: { prompt_tokens: 1, completion_tokens: 1, total_tokens: 2 } }));
    }
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    for (const content of parts) res.write(`data: ${JSON.stringify({ created: 1, choices: [{ index: 0, delta: { content } }] })}\n\n`);
    res.end(`data: ${JSON.stringify({ created: 1, choices: [{ index: 0, delta: {}, finish_reason: "stop" }] })}\n\ndata: [DONE]\n\n`);
  });
  await new Promise<void>((resolve) => engine.listen(0, "127.0.0.1", resolve));
  try {
    await withServer({ shelf: () => [], engines: () => [engineRecord(portOf(engine))] }, async (base) => {
      const ask = (content: string, stream: boolean) => post(`${base}/v1/chat/completions`, JSON.stringify({ model: "qwen/qwen3.5-9b", messages: [{ role: "user", content }], stream }));
      const streamed = async (content: string) => sseEvents(await (await ask(content, true)).text()).map((c) => c.choices[0].delta.content ?? "").join("");
      assert.equal(await streamed("quirk"), "purple");
      assert.equal(await streamed("repeat"), "purple\npurple", "the tag goes; text already sent stays");
      assert.equal(await streamed("normal"), "\n\n<b>Hello</b> <3", "a real answer is passed on untouched");
      assert.equal((await (await ask("quirk", false)).json()).choices[0].message.content, "purple");
    });
  } finally {
    engine.close();
  }
});

test("hanging up mid-answer stops the engine", async () => {
  let engineHungUp = false;
  // An engine that never finishes: one event every 20 ms until its caller goes away.
  const slow = createServer((req, res) => {
    res.writeHead(200, { "Content-Type": "text/event-stream" });
    const timer = setInterval(() => res.write('data: {"created":1,"choices":[{"index":0,"delta":{"content":"la "}}]}\n\n'), 20);
    req.on("close", () => {
      clearInterval(timer);
      engineHungUp = true;
    });
  });
  await new Promise<void>((resolve) => slow.listen(0, "127.0.0.1", resolve));
  const engine = engineRecord(portOf(slow));
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const stop = new AbortController();
      const res = await fetch(`${base}/v1/chat/completions`, {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: stop.signal,
        body: JSON.stringify({ model: engine.identifier, messages: [], stream: true }),
      });
      const reader = res.body!.getReader();
      await reader.read(); // the first words arrive...
      stop.abort(); // ...and the app hangs up
      for (let waited = 0; !engineHungUp && waited < 2000; waited += 20) await new Promise((r) => setTimeout(r, 20));
      assert.ok(engineHungUp, "the front door hung up on the engine too");
    });
  } finally {
    slow.closeAllConnections();
    slow.close();
  }
});

test("an unknown model is refused, not swapped for another", async () => {
  const { server, engine } = await fakeEngine();
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const res = await post(`${base}/v1/chat/completions`, JSON.stringify({ model: "nope", messages: [] }));
      assert.equal(res.status, 404);
      assert.equal((await res.json()).error.code, "model_not_found");
    });
  } finally {
    server.close();
  }
});

test("a model that isn't loaded is loaded on demand, once, however many requests ask at the same time", async () => {
  const { server, engine } = await fakeEngine();
  const shelfModel = { key: "qwen/qwen3.5-9b", type: "llm" } as ShelfModel;
  let loads = 0;
  let loaded: Engine[] = [];
  const sources: Sources = {
    shelf: () => [shelfModel],
    engines: () => loaded,
    load: async () => {
      loads++;
      await new Promise((resolve) => setTimeout(resolve, 100)); // loading takes a while
      loaded = [engine];
      return engine;
    },
    config: () => ({ defaultModel: "qwen/qwen3.5-9b" }),
  };
  try {
    await withServer(sources, async (base) => {
      const ask = (model?: string) => post(`${base}/v1/chat/completions`, JSON.stringify({ model, messages: [{ role: "user", content: "hi" }] }));
      const replies = await Promise.all([ask("qwen/qwen3.5-9b"), ask("qwen/qwen3.5-9b"), ask("qwen/qwen3.5-9b")]);
      assert.deepEqual(replies.map((r) => r.status), [200, 200, 200]);
      assert.equal(loads, 1, "three requests, one load");
      loaded = [];
      assert.equal((await ask()).status, 200, "no model named: the default model from the config file");
      assert.equal(loads, 2);
    });
  } finally {
    server.close();
  }
});

// Side by side with LM Studio's recordings, using your real models.
const drive = existsSync("D:\\LMStudio\\models");
const byId = (list: { id: string }[]) => Object.fromEntries(list.map((m) => [m.id, m]));

test("model lists match LM Studio's, field for field", { skip: !drive && "model drive not connected" }, () =>
  withServer({ shelf: () => listModels(modelRoots()), engines: () => [] }, async (base) => {
    const v1 = await (await fetch(`${base}/v1/models`)).json();
    assert.deepEqual(byId(v1.data), byId(golden("lmstudio/v1-models.json").data));
    const v0 = await (await fetch(`${base}/api/v0/models`)).json();
    assert.deepEqual(byId(v0.data), byId(golden("lmstudio/api-v0-models.json").data));
  }));

test("with the 9B loaded: model list and a real reply match LM Studio's", { skip: !(drive && existsSync(ENGINE)) && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  const model = listModels(modelRoots()).find((m) => m.key === "qwen/qwen3.5-9b") as ShelfModel;
  t.after(() => unload(model.key));
  await load(model, { contextLength: 8192 });
  await withServer({ shelf: () => listModels(modelRoots()), engines: running }, async (base) => {
    const v0 = await (await fetch(`${base}/api/v0/models`)).json();
    assert.deepEqual(byId(v0.data), byId(golden("lmstudio/api-v0-models-loaded.json").data), "loaded state and context");
    const ask = JSON.stringify({ model: model.key, messages: [{ role: "user", content: "Reply with exactly one word: pong" }], temperature: 0, max_tokens: 2000 });
    const reply = await (await post(`${base}/api/v0/chat/completions`, ask)).json();
    assert.deepEqual(shape(reply), shape(golden("lmstudio/api-v0-chat.json")));
    assert.equal(reply.choices[0].message.content, "pong");
    // completion = thinking + "<think>", "</think>" and blank-line tokens + the answer, as in LM Studio's 133 = 129 + 4
    assert.equal(reply.usage.completion_tokens, reply.usage.completion_tokens_details.reasoning_tokens + 4);

    // Streamed, through the same front door: same order of events as LM Studio's, same answer.
    const streamed = await (await post(`${base}/v1/chat/completions`, JSON.stringify({ ...JSON.parse(ask), stream: true }))).text();
    const lmstudio = readFileSync(new URL("../golden/lmstudio/chat-stream.sse", import.meta.url), "utf8");
    assert.deepEqual(squeeze(kinds(sseEvents(streamed))), squeeze(kinds(sseEvents(lmstudio))));
    assert.equal(sseEvents(streamed).map((c) => c.choices[0].delta.content ?? "").join(""), "pong");
  });

  // `squid ask` end to end, as a separate program: only the answer reaches stdout. It starts the
  // background server if it isn't running; stop it again afterwards if this test started it.
  if (!(await serverStatus())) t.after(() => stopServer());
  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  const stdout = execFileSync(process.execPath, [cli, "ask", "Reply with exactly one word: pong", "--model", model.key], { encoding: "utf8", stdio: ["ignore", "pipe", "ignore"] });
  assert.match(stdout.trim(), /^pong$/i);
});

test("a request for an unloaded model loads it, with LM Studio's idle timer", { skip: !(drive && existsSync(ENGINE)) && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  const key = "qwen/qwen3.5-9b";
  await unload(key);
  t.after(() => unload(key));
  await withServer(undefined, async (base) => {
    const res = await post(`${base}/v1/chat/completions`, JSON.stringify({ model: key, messages: [{ role: "user", content: "Reply with exactly one word: pong" }], max_tokens: 2000 }));
    assert.equal(res.status, 200);
    assert.match((await res.json()).choices[0].message.content, /pong/i);
  });
  const engine = running().find((e) => e.identifier === key)!;
  assert.equal(engine.jit, true);
  assert.equal(engine.ttlSeconds, 3600);
  // LM Studio's `lms ps` showed "60m / 1h" for a model it had just loaded on demand.
  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  assert.match(execFileSync(process.execPath, [cli, "ps"], { encoding: "utf8" }), /Local\s+60m \/ 1h/);
});

test("the server starts in the background, reports its status, and stops", { timeout: 60_000 }, async (t) => {
  if (await serverStatus()) return t.skip("a Squid server is already running; leaving it alone");
  const info = await ensureServer();
  assert.deepEqual(await serverStatus(), info);
  assert.equal((await fetch(`http://127.0.0.1:${info.port}/v1/does-not-exist`)).status, 200, "it answers like LM Studio");
  assert.deepEqual(await ensureServer(), info, "a second start finds the running one");
  assert.equal(await stopServer(), true);
  assert.equal(await serverStatus(), undefined);
});