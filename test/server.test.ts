import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import { existsSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { startServer, type Sources } from "../src/server.ts";
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

// A stand-in for llama-server: answers with the recorded raw reply, and only with the right key.
async function fakeEngine(): Promise<{ server: Server; engine: Engine }> {
  const server = createServer((req, res) => {
    if (req.headers.authorization !== "Bearer test-key") return res.writeHead(401).end();
    res.setHeader("Content-Type", "application/json");
    if (req.url === "/tokenize") return res.end(JSON.stringify({ tokens: new Array(129).fill(0) }));
    res.end(JSON.stringify(golden("llamacpp/chat-nonstream.json")));
  });
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve));
  const engine: Engine = {
    identifier: "qwen/qwen3.5-9b", key: "qwen/qwen3.5-9b", pid: 0, port: portOf(server), apiKey: "test-key",
    contextLength: 8192, parallel: 4, sizeBytes: 1, architecture: "qwen35", quantization: "Q4_K_M", log: "",
  };
  return { server, engine };
}

async function withServer(sources: Sources, run: (base: string) => Promise<void>) {
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

test("a model that isn't loaded is refused, not swapped for another", async () => {
  const { server, engine } = await fakeEngine();
  try {
    await withServer({ shelf: () => [], engines: () => [engine] }, async (base) => {
      const res = await post(`${base}/v1/chat/completions`, JSON.stringify({ model: "nope", messages: [] }));
      assert.equal(res.status, 404);
      assert.equal((await res.json()).error.code, "model_not_loaded");
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
  });
});
