// Squid's own runtime: the TypeScript engine of steps 15–17 behind the same door as llama-server. It
// takes llama-server's command line and answers the requests Squid sends to llama-server (/health,
// /slots, /tokenize, /v1/chat/completions), so nothing else in Squid can tell them apart: ps, unload, the
// idle timer, the API, the chat and the browser page all work unchanged. LM Studio does the same with its
// runtimes (`lms runtime ls` lists a dozen llama.cpp builds): one app, swappable engines.
//   squid load smol --runtime squid
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomBytes, randomInt } from "node:crypto";
import { parseArgs } from "node:util";
import { readGgufHeader } from "./gguf.ts";
import { encode, tokenBytes } from "./tokenizer.ts";
import { forward, loadModel, newState, type Model, type State } from "./transformer.ts";
import { candidates, DEFAULTS, mt19937, pick, type Sampling } from "./sampler.ts";
import { sseEvent } from "./sse.ts";
import { keyMatches } from "./web.ts";

type Message = { role: string; content: unknown };

// The chat template turns a conversation into the one long text the model was trained on. A model
// carries it as a small program in the Jinja language (tokenizer.chat_template), and llama.cpp runs that
// program. Squid doesn't run Jinja: it knows one layout, ChatML (SmolLM2 and Qwen use it), and reads the
// template only for the system message it adds when a conversation has none.
export function chatPrompt(template: string, messages: Message[]): string {
  if (!template.includes("<|im_start|>")) throw new Error("Squid's runtime only knows the ChatML chat template");
  const fallback = /<\|im_start\|>system\n([^'"]*?)<\|im_end\|>/.exec(template)?.[1];
  const turns = messages[0]?.role !== "system" && fallback ? [{ role: "system", content: fallback }, ...messages] : messages;
  return turns.map((m) => `<|im_start|>${m.role}\n${text(m.content)}<|im_end|>\n`).join("") + "<|im_start|>assistant\n";
}

// A message's content is usually text, but the OpenAI format also allows a list of parts.
const text = (content: unknown): string =>
  typeof content === "string" ? content : Array.isArray(content) ? content.map((part) => part?.text ?? "").join("") : "";

// The one "slot": the model, its KV cache, and which tokens that cache holds right now.
type Slot = { model: Model; template: string; state: State; cached: number[]; busy: boolean };

type Job = { prompt: number[]; sampling: Sampling; limit: number; onText: (text: string) => void; cancelled: () => boolean };
type Done = { finish: "stop" | "length"; promptTokens: number; reused: number; completionTokens: number; promptMs: number; predictedMs: number };

// One answer: read the prompt, then choose tokens one at a time until the model ends its turn, the
// limit is reached, or the caller hangs up.
async function generate(slot: Slot, job: Job): Promise<Done> {
  const { model, state } = slot;
  const started = performance.now();
  // The prompt cache: the KV cache still holds the previous request. Keep the part this prompt begins
  // with and read only the rest; in a chat, that's little more than the newest message. The last token
  // is always read again, because its scores are what the first new token is chosen from.
  let reused = 0;
  while (reused < job.prompt.length - 1 && slot.cached[reused] === job.prompt[reused]) reused++;
  state.pos = reused; // forgetting is just moving back: later positions get written over
  slot.cached.length = reused;
  let logits: Float32Array = new Float32Array(0);
  for (const token of job.prompt.slice(reused)) {
    if (job.cancelled()) break;
    logits = forward(model, state, token);
    slot.cached.push(token);
    await breathe();
  }
  const read = performance.now();

  const random = mt19937(job.sampling.seed ?? randomInt(2 ** 32)).random;
  const decoder = new TextDecoder(); // holds on to a character split across tokens until it's whole
  let produced = 0;
  let finish: Done["finish"] = "length";
  while (produced < job.limit && !job.cancelled()) {
    const token = pick(candidates(logits, job.sampling), random);
    produced++; // llama.cpp counts the end token too
    if (model.vocab.ends.has(token)) {
      finish = "stop";
      break;
    }
    const piece = decoder.decode(tokenBytes(model.vocab, token, false), { stream: true });
    if (piece) job.onText(piece);
    if (produced === job.limit || state.pos === state.size) break; // no room for another
    logits = forward(model, state, token);
    slot.cached.push(token);
    await breathe();
  }
  const rest = decoder.decode(); // an unfinished character at the very end
  if (rest) job.onText(rest);
  return { finish, promptTokens: job.prompt.length, reused, completionTokens: produced, promptMs: read - started, predictedMs: performance.now() - read };
}

// The arithmetic blocks Node's event loop. A pause after every token lets the server get a word in
// meanwhile: answer /health and /slots, or notice a caller that hung up.
const breathe = () => new Promise((resolve) => setImmediate(resolve));

// One slot, so requests take turns: one that arrives while another is answered waits, as llama-server
// queues requests when all its slots are busy.
let queue: Promise<unknown> = Promise.resolve();
function inTurn<T>(slot: Slot, job: () => Promise<T>): Promise<T> {
  const turn = queue.then(async () => {
    slot.busy = true;
    try {
      return await job();
    } finally {
      slot.busy = false;
    }
  });
  queue = turn.catch(() => {});
  return turn;
}

// A chat reply in llama-server's shape, streamed or whole. Squid's front door reshapes it for LM Studio's
// format exactly as it does llama-server's.
async function chat(slot: Slot, alias: string, body: any, res: ServerResponse) {
  if (!Array.isArray(body.messages)) return send(res, 400, { error: { code: 400, message: "'messages' is required", type: "invalid_request_error" } });
  const prompt = encode(slot.model.vocab, chatPrompt(slot.template, body.messages));
  if (prompt.length >= slot.state.size) {
    return send(res, 400, { error: { code: 400, message: "the request exceeds the available context size, try increasing it", type: "exceed_context_size_error" } });
  }
  const sampling: Sampling = {
    temperature: body.temperature ?? DEFAULTS.temperature,
    topK: body.top_k ?? DEFAULTS.topK,
    topP: body.top_p ?? DEFAULTS.topP,
    minP: body.min_p ?? DEFAULTS.minP,
    seed: body.seed >= 0 ? body.seed : undefined, // -1, llama.cpp's default, means a new seed every time
  };
  const max = body.max_tokens ?? body.n_predict ?? -1;
  let gone = false;
  res.on("close", () => (gone = !res.writableFinished)); // closed before the answer was finished: the caller left
  const job = { prompt, sampling, limit: max < 0 ? Infinity : max, cancelled: () => gone };
  const head = { id: `chatcmpl-${randomBytes(12).toString("hex")}`, created: Math.floor(Date.now() / 1000), model: alias };

  if (body.stream) {
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
    const event = (choices: object[], extra = {}) => res.write(sseEvent(JSON.stringify({ ...head, object: "chat.completion.chunk", choices, ...extra })));
    event([{ index: 0, delta: { role: "assistant", content: null }, finish_reason: null }]);
    const done = await inTurn(slot, () => generate(slot, { ...job, onText: (text) => event([{ index: 0, delta: { content: text }, finish_reason: null }]) }));
    event([{ index: 0, delta: {}, finish_reason: done.finish }]);
    if (body.stream_options?.include_usage) event([], { usage: usage(done), timings: timings(done) });
    logLine(done, gone);
    return res.end(sseEvent("[DONE]"));
  }
  let content = "";
  const done = await inTurn(slot, () => generate(slot, { ...job, onText: (text) => (content += text) }));
  logLine(done, gone);
  send(res, 200, { ...head, object: "chat.completion", choices: [{ index: 0, message: { role: "assistant", content }, finish_reason: done.finish }], usage: usage(done), timings: timings(done) });
}

const usage = (d: Done) => ({
  prompt_tokens: d.promptTokens,
  completion_tokens: d.completionTokens,
  total_tokens: d.promptTokens + d.completionTokens,
  prompt_tokens_details: { cached_tokens: d.reused },
});
const timings = (d: Done) => ({
  cache_n: d.reused,
  prompt_n: d.promptTokens - d.reused,
  prompt_ms: d.promptMs,
  predicted_n: d.completionTokens,
  predicted_ms: d.predictedMs,
  predicted_per_second: d.completionTokens / (d.predictedMs / 1000),
});
const logLine = (d: Done, left: boolean) =>
  console.log(`prompt ${d.promptTokens} tokens (${d.reused} from the cache) in ${(d.promptMs / 1000).toFixed(1)} s; ${d.completionTokens} new in ${(d.predictedMs / 1000).toFixed(1)} s; ${left ? "the caller left" : d.finish}`);

function send(res: ServerResponse, status: number, body: unknown) {
  if (res.headersSent) return res.destroy(); // failed mid-stream: cut it off, so it can't pass as finished
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(JSON.stringify(body));
}

async function readJson(req: IncomingMessage): Promise<any> {
  let body = "";
  for await (const chunk of req) body += chunk;
  return JSON.parse(body);
}

if (import.meta.main) {
  // llama-server's command line. Its GPU and slot settings (--fit-target, --parallel, --flash-attn…)
  // mean nothing to a one-slot CPU engine and are ignored.
  const { values } = parseArgs({
    strict: false,
    options: { model: { type: "string" }, alias: { type: "string" }, port: { type: "string" }, "api-key": { type: "string" }, "ctx-size": { type: "string" } },
  });
  const path = String(values.model);
  const alias = String(values.alias ?? path);
  const key = String(values["api-key"]);
  let slot: Slot | undefined; // undefined while the model loads

  const server = createServer(async (req, res) => {
    try {
      const route = `${req.method} ${new URL(req.url ?? "/", "http://localhost").pathname}`;
      // Like llama-server: /health is open to all; every other route needs Squid's key.
      if (route === "GET /health") return slot ? send(res, 200, { status: "ok" }) : send(res, 503, { error: { code: 503, message: "Loading model", type: "unavailable_error" } });
      if (!keyMatches(req.headers.authorization, key)) return send(res, 401, { error: { code: 401, message: "Invalid API Key", type: "authentication_error" } });
      if (!slot) return send(res, 503, { error: { code: 503, message: "Loading model", type: "unavailable_error" } });
      switch (route) {
        case "GET /slots":
          return send(res, 200, [{ id: 0, is_processing: slot.busy }]);
        case "POST /tokenize":
          return send(res, 200, { tokens: encode(slot.model.vocab, String((await readJson(req)).content ?? ""), { bos: false }) });
        case "POST /v1/chat/completions":
          return await chat(slot, alias, await readJson(req), res);
        case "POST /shutdown":
          // Squid's own route: `squid unload` asks the runtime to leave (engine.ts explains why).
          res.on("finish", () => process.exit(0));
          return send(res, 200, { status: "stopping" });
        default:
          return send(res, 404, { error: { code: 404, message: "File Not Found", type: "not_found_error" } });
      }
    } catch (err) {
      send(res, 500, { error: { code: 500, message: (err as Error).message, type: "server_error" } });
    }
  });

  server.listen(Number(values.port), "127.0.0.1", () => {
    const started = performance.now();
    const model = loadModel(path);
    const template = String(readGgufHeader(path).metadata.get("tokenizer.chat_template") ?? "");
    chatPrompt(template, []); // refuse a template Squid can't follow now, not at the first message
    const size = Math.min(Number(values["ctx-size"] ?? 4096), model.config.context);
    slot = { model, template, state: newState(model, size), cached: [], busy: false };
    console.log(`Squid's runtime: ${alias} loaded in ${Math.round(performance.now() - started)} ms, ${size}-token context, on 127.0.0.1:${values.port}`);
  });
}
