import { test } from "node:test";
import assert from "node:assert/strict";
import { existsSync, readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { chatPrompt } from "../src/runtime.ts";
import { readGgufHeader } from "../src/gguf.ts";
import { encode, loadVocab } from "../src/tokenizer.ts";
import { load, running, status, unload } from "../src/engine.ts";
import { listModels, modelRoots } from "../src/models.ts";
import { startServer } from "../src/server.ts";
import { sseData } from "../src/sse.ts";
import { SMOLLM } from "../golden/capture-forward.ts";

const golden = JSON.parse(readFileSync(new URL("../golden/llamacpp/sampling-smollm2.json", import.meta.url), "utf8"));
const missing = !existsSync(SMOLLM) && "SmolLM2-135M isn't on this PC";

test("the chat template gives llama.cpp's prompt, character for character", { skip: missing }, () => {
  const template = String(readGgufHeader(SMOLLM).metadata.get("tokenizer.chat_template"));
  for (const t of golden.templates) assert.equal(chatPrompt(template, t.messages), t.prompt);
  assert.match(golden.templates[0].prompt, /named SmolLM/, "no system message: the template adds its own");
});

// The real LM Studio (golden/lmstudio/chat-smollm2.json, `npm run capture:lmstudio-smol`) reported how many
// tokens each conversation took: Squid's template and tokenizer give the same count, every time.
test("the same prompt tokens as the real LM Studio", { skip: missing }, () => {
  const template = String(readGgufHeader(SMOLLM).metadata.get("tokenizer.chat_template"));
  const vocab = loadVocab(SMOLLM);
  const lmstudio = JSON.parse(readFileSync(new URL("../golden/lmstudio/chat-smollm2.json", import.meta.url), "utf8")).chats;
  for (const c of lmstudio) assert.equal(encode(vocab, chatPrompt(template, c.messages)).length, c.usage.prompt_tokens);
  assert.equal(lmstudio.length, 12);
});

test("a template Squid can't follow is refused", () => {
  assert.throws(() => chatPrompt("{% for m in messages %}[INST] {{ m.content }} [/INST]{% endfor %}", []), /ChatML/);
});

// The whole path: Squid's runtime loaded like any engine, asked through Squid's front door.
test("Squid's runtime behind Squid's front door", { skip: missing, timeout: 300_000 }, async (t) => {
  const model = listModels(modelRoots()).find((m) => m.key === "smollm2-135m-instruct")!;
  const identifier = "squid-runtime-test";
  t.after(() => unload(identifier));
  const engine = await load(model, { runtime: "squid", identifier, contextLength: 1024 });
  assert.equal(engine.parallel, 1, "one slot");
  const server = await startServer(0, { shelf: () => [model], engines: () => running(true) });
  t.after(() => server.close());
  const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
  const chat = (body: object) => fetch(`${base}/api/v0/chat/completions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ model: identifier, ...body }) });

  // llama.cpp's recorded chats: the same number of prompt tokens every time (template + tokenizer), and
  // at temperature 0 on "Another one?" the same answer. (The others split at near-ties, as in step 16.)
  for (const c of golden.chats.filter((c: any) => c.settings.temperature === 0)) {
    const reply = await (await chat({ messages: c.messages, max_tokens: 40, temperature: 0 })).json();
    assert.equal(reply.usage.prompt_tokens, c.usage.prompt_tokens);
    if (c.content === "Green.") assert.equal(reply.choices[0].message.content, "Green.");
    assert.equal(reply.runtime.name, "squid-typescript-cpu", "LM Studio's /api/v0 names the runtime");
  }

  // Streamed, with a running ps check: the runtime still answers /slots while it computes.
  // Greedy, on a question whose answer runs past 12 tokens, so the count is always 12.
  const streaming = chat({ messages: golden.chats[2].messages, max_tokens: 12, temperature: 0, stream: true, stream_options: { include_usage: true } });
  await new Promise((resolve) => setTimeout(resolve, 600));
  assert.equal(await status(engine), "GENERATING");
  const events = [];
  for await (const data of sseData((await streaming).body!)) events.push(data);
  assert.equal(events.at(-1), "[DONE]");
  assert.equal(JSON.parse(events.at(-2)!).usage.completion_tokens, 12, "the usage event comes last");

  // The prompt cache: a follow-up only reads what's new. Asked of the runtime directly, to see its timings.
  const direct = (messages: object[]) =>
    fetch(`http://127.0.0.1:${engine.port}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json", Authorization: `Bearer ${engine.apiKey}` },
      body: JSON.stringify({ messages, max_tokens: 1, temperature: 0 }),
    }).then((r) => r.json());
  const first = await direct([{ role: "user", content: "Name a colour." }]);
  const second = await direct([{ role: "user", content: "Name a colour." }, { role: "assistant", content: "Blue." }, { role: "user", content: "Another one?" }]);
  assert.equal(second.timings.cache_n, first.usage.prompt_tokens, "the whole first prompt came from the cache");
  assert.equal(second.timings.prompt_n, second.usage.prompt_tokens - first.usage.prompt_tokens, "only the new turn was read");
  const again = await direct([{ role: "user", content: "Name a colour." }]);
  assert.equal(again.timings.prompt_n, 1, "the same prompt again: only its last token is read, for its scores");

  // Hanging up stops the work: the runtime is idle again long before 200 tokens would be done.
  const reader = (await chat({ messages: golden.chats[2].messages, max_tokens: 200, stream: true })).body!.getReader();
  await reader.read(); // the answer has started
  await reader.cancel(); // hang up
  await new Promise((resolve) => setTimeout(resolve, 1500));
  assert.equal(await status(engine), "IDLE");

  assert.equal(await unload(identifier), true);
  assert.equal(await fetch(`http://127.0.0.1:${engine.port}/health`).then(() => "up", () => "down"), "down", "the runtime has left");
  assert.ok(!running().some((e) => e.identifier === identifier));
});
