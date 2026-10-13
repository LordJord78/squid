// The front door: one local server, like LM Studio's on port 1234, that speaks the OpenAI API and
// LM Studio's own /api/v0. For each chat it finds the model's engine (step 03), passes the request on,
// and rebuilds the engine's reply in LM Studio's shape.
import { createServer, request, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { randomBytes } from "node:crypto";
import { listModels, type ShelfModel } from "./models.ts";
import { running, RUNTIME, type Engine } from "./engine.ts";
import { sseData, sseEvent } from "./sse.ts";

export const DEFAULT_PORT = 1235; // next door to LM Studio's 1234, so both can run at once
const MAX_BODY_BYTES = 16 * 1024 * 1024;

// Where the server gets its facts. Tests pass in fakes; normally it's the real shelf and engines.
export type Sources = { shelf: () => ShelfModel[]; engines: () => Engine[] };
const REAL: Sources = { shelf: () => listModels(), engines: () => running(true) };

export function startServer(port = DEFAULT_PORT, sources = REAL): Promise<Server> {
  const server = createServer((req, res) => {
    handle(req, res, sources).catch((err: Error) => fail(res, 500, { error: { message: err.message, type: "server_error" } }));
  });
  return new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", () => resolve(server)); // this PC only, never 0.0.0.0
  });
}

async function handle(req: IncomingMessage, res: ServerResponse, sources: Sources) {
  // DNS rebinding: a web page can point its own domain at 127.0.0.1, but the browser still sends
  // "Host: that-domain". Only answer requests addressed to this PC by name.
  if (!isLoopbackHost(req.headers.host)) return fail(res, 403, { error: "Squid only answers requests addressed to localhost" });
  // Browsers add Origin to requests a web page makes. Only pages served by Squid itself may call it.
  const origin = req.headers.origin;
  if (origin !== undefined && origin !== `http://${req.headers.host}`) return fail(res, 403, { error: "requests from other websites are refused" });

  const path = new URL(req.url ?? "/", "http://localhost").pathname;
  switch (`${req.method} ${path}`) {
    case "GET /v1/models":
      return send(res, 200, { data: catalog(sources).map(({ id }) => ({ id, object: "model", owned_by: "organization_owner" })), object: "list" });
    case "GET /api/v0/models":
      return send(res, 200, { data: catalog(sources).map(describe), object: "list" });
    case "POST /v1/chat/completions":
      return chat(req, res, sources, false);
    case "POST /api/v0/chat/completions":
      return chat(req, res, sources, true);
    default:
      // Observed: LM Studio answers unknown routes with 200 OK and this message, not 404.
      return fail(res, 200, { error: `Unexpected endpoint or method. (${req.method} ${path})` });
  }
}

// Every model on the shelf, the loaded ones first as LM Studio lists them, with its engine if it has one.
type Entry = { id: string; model: ShelfModel; engine?: Engine };
function catalog(sources: Sources): Entry[] {
  const engines = sources.engines();
  const entries = sources.shelf().map((model) => {
    const engine = engines.find((e) => e.key === model.key);
    return { id: engine?.identifier ?? model.key, model, engine };
  });
  return [...entries.filter((e) => e.engine), ...entries.filter((e) => !e.engine)];
}

// One row of /api/v0/models, field for field as LM Studio writes it.
function describe({ id, model, engine }: Entry) {
  return {
    id,
    object: "model",
    type: model.type === "embedding" ? "embeddings" : model.vision ? "vlm" : "llm",
    publisher: model.publisher,
    arch: model.architecture,
    compatibility_type: "gguf",
    quantization: model.quantization,
    state: engine ? "loaded" : "not-loaded",
    max_context_length: model.contextLength,
    ...(engine && { loaded_context_length: engine.contextLength }),
    ...(model.toolUse && { capabilities: ["tool_use"] }),
  };
}

async function chat(req: IncomingMessage, res: ServerResponse, sources: Sources, lmstudioExtras: boolean) {
  // Observed: LM Studio refuses bodies that aren't JSON (415). That also blocks plain web forms.
  if (!String(req.headers["content-type"] ?? "").startsWith("application/json")) {
    return fail(res, 415, { error: "Content-Type must be application/json" });
  }
  let body;
  try {
    body = JSON.parse(await readBody(req));
  } catch (err) {
    if (err instanceof TooLarge) return fail(res, 413, { error: "request body is larger than 16 MB" });
    return fail(res, 400, { error: { message: "Invalid body: failed to parse JSON value. Please check the value to ensure it is valid JSON", type: "invalid_request", code: "invalid_json" } });
  }
  if (!Array.isArray(body.messages)) return fail(res, 400, { error: "'messages' field is required" });

  // The model asked for, or the only one loaded if none was named. LM Studio would load a model here
  // (and answers an unknown name with a different model); Squid says what's wrong instead.
  const engines = sources.engines();
  const engine = engines.find((e) => e.identifier === body.model) ?? (body.model === undefined && engines.length === 1 ? engines[0] : undefined);
  if (!engine) {
    const message = body.model === undefined ? "No model named, and not exactly one model loaded. See `squid ps`." : `Model "${body.model}" isn't loaded. Load it with: squid load ${body.model}`;
    return fail(res, 404, { error: { message, type: "invalid_request", code: "model_not_loaded" } });
  }
  // Observed: LM Studio streams /api/v0 exactly like /v1, without the extra stats.
  if (body.stream) return streamChat(res, engine, body);

  let raw;
  try {
    raw = await toEngine(engine, "/v1/chat/completions", { ...body, model: engine.identifier, stream: false }, res);
  } catch {
    return fail(res, 502, { error: { message: `the engine for ${engine.identifier} isn't answering (see \`squid ps\`)`, type: "server_error" } });
  }
  if (raw.status !== 200) return fail(res, raw.status, raw.json); // the engine's own error, e.g. prompt too long

  const r = raw.json;
  // llama.cpp doesn't report thinking tokens; LM Studio does. Counting the thinking text's tokens gives
  // LM Studio's number exactly (129 for the recorded "pong" reply).
  const thinking = r.choices.map((c: any) => c.message.reasoning_content ?? "").join("");
  const reasoningTokens = thinking ? (await toEngine(engine, "/tokenize", { content: thinking })).json.tokens.length : 0;

  send(res, 200, {
    id: `chatcmpl-${randomId()}`,
    object: "chat.completion",
    created: r.created,
    model: engine.identifier,
    choices: r.choices.map((c: any) => ({
      index: c.index,
      message: {
        role: "assistant",
        content: c.message.content ?? "",
        ...(c.message.reasoning_content !== undefined && { reasoning_content: c.message.reasoning_content }),
        tool_calls: (c.message.tool_calls ?? []).map((t: any) => ({ type: t.type, id: t.id, function: t.function })),
      },
      logprobs: c.logprobs ?? null,
      finish_reason: c.finish_reason,
    })),
    usage: {
      prompt_tokens: r.usage.prompt_tokens,
      completion_tokens: r.usage.completion_tokens,
      total_tokens: r.usage.total_tokens,
      completion_tokens_details: { reasoning_tokens: reasoningTokens },
    },
    // /v1 sends empty stats; LM Studio's own /api/v0 adds speed, the model and the engine.
    ...(lmstudioExtras
      ? {
          stats: {
            tokens_per_second: r.timings.predicted_per_second,
            time_to_first_token: r.timings.prompt_ms / 1000, // reading the prompt is what you wait for
            generation_time: r.timings.predicted_ms / 1000,
            stop_reason: STOP_REASONS[r.choices[0].finish_reason] ?? r.choices[0].finish_reason,
          },
          model_info: { arch: engine.architecture, quant: engine.quantization, format: "gguf", context_length: engine.contextLength },
          runtime: { ...RUNTIME, supported_formats: ["gguf"] },
        }
      : { stats: {}, system_fingerprint: engine.identifier }),
  });
}

// OpenAI's finish_reason → LM Studio's stop_reason. The first two are observed; "toolCalls" is the
// name LM Studio's SDK documents.
const STOP_REASONS: Record<string, string> = { stop: "eosFound", length: "maxPredictedTokensReached", tool_calls: "toolCalls" };

// Streaming: pass the engine's events on one at a time as they arrive, rewritten in LM Studio's shape.
async function streamChat(res: ServerResponse, engine: Engine, body: any) {
  let incoming;
  try {
    incoming = await engineRequest(engine, "/v1/chat/completions", { ...body, model: engine.identifier, stream: true }, res);
  } catch {
    return fail(res, 502, { error: { message: `the engine for ${engine.identifier} isn't answering (see \`squid ps\`)`, type: "server_error" } });
  }
  if (incoming.statusCode !== 200) return fail(res, incoming.statusCode ?? 502, JSON.parse(await readAll(incoming))); // refused before streaming began

  // LM Studio's stream headers (observed). no-cache: nothing between here and the app may hold events back.
  res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache", Connection: "keep-alive" });
  const id = `chatcmpl-${randomId()}`;
  let thinking = ""; // collected to count thinking tokens for the usage event
  let roleSent = false;
  try {
    for await (const data of sseData(incoming)) {
      if (data === "[DONE]") break;
      const chunk = JSON.parse(data);
      const head = { id, object: "chat.completion.chunk", created: chunk.created, model: engine.identifier, system_fingerprint: engine.identifier };

      // The usage event, sent last when the app asks (stream_options.include_usage), with LM Studio's thinking count.
      if (chunk.usage && !chunk.choices.length) {
        const reasoningTokens = thinking ? (await toEngine(engine, "/tokenize", { content: thinking })).json.tokens.length : 0;
        const { prompt_tokens, completion_tokens, total_tokens } = chunk.usage;
        res.write(sseEvent(JSON.stringify({ ...head, choices: [], usage: { prompt_tokens, completion_tokens, total_tokens, completion_tokens_details: { reasoning_tokens: reasoningTokens } } })));
        continue;
      }
      const choices = [];
      for (const c of chunk.choices) {
        // Drop empty fields and the role: llama.cpp opens with an event that says only {role, content: null}.
        // LM Studio instead puts the role on the first event that has words in it.
        const delta: Record<string, any> = Object.fromEntries(Object.entries(c.delta ?? {}).filter(([key, value]) => value !== null && key !== "role"));
        thinking += delta.reasoning_content ?? "";
        if (!Object.keys(delta).length && c.finish_reason == null) continue; // nothing to pass on yet
        choices.push({ index: c.index, delta: roleSent ? delta : { role: "assistant", ...delta }, logprobs: c.logprobs ?? null, finish_reason: c.finish_reason ?? null });
        roleSent = true;
      }
      if (choices.length) res.write(sseEvent(JSON.stringify({ ...head, choices })));
    }
  } catch {
    // The engine went away mid-answer. Cut the connection rather than send [DONE], so the app can tell
    // a broken answer from a finished one.
    return res.destroy();
  }
  res.end(sseEvent("[DONE]"));
}

// Opens a request to an engine; resolves as soon as its reply starts arriving. node:http rather than
// fetch: fetch gives up when no reply has started after 5 minutes, and a long non-streamed answer from
// the 27B can take longer than that.
function engineRequest(engine: Engine, path: string, body: unknown, caller?: ServerResponse): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const outgoing = request(
      { host: "127.0.0.1", port: engine.port, path, method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${engine.apiKey}` } },
      resolve,
    );
    outgoing.on("error", reject);
    // If the caller hangs up first, hang up on the engine too: llama.cpp then stops generating,
    // so the GPU doesn't keep working on an answer nobody will read.
    caller?.on("close", () => caller.writableEnded || outgoing.destroy());
    outgoing.end(JSON.stringify(body));
  });
}

// A whole (non-streamed) JSON reply from an engine.
async function toEngine(engine: Engine, path: string, body: unknown, caller?: ServerResponse): Promise<{ status: number; json: any }> {
  const incoming = await engineRequest(engine, path, body, caller);
  return { status: incoming.statusCode ?? 502, json: JSON.parse(await readAll(incoming)) };
}

async function readAll(incoming: IncomingMessage): Promise<string> {
  incoming.setEncoding("utf8"); // decode as text, keeping characters split between chunks whole
  let text = "";
  for await (const chunk of incoming) text += chunk;
  return text;
}

class TooLarge extends Error {}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new TooLarge();
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}

function isLoopbackHost(host: string | undefined): boolean {
  try {
    return ["127.0.0.1", "localhost", "[::1]"].includes(new URL(`http://${host}`).hostname);
  } catch {
    return false; // no Host header, or not a valid one
  }
}

// LM Studio pretty-prints replies and sends errors on one line; Squid does the same.
function send(res: ServerResponse, status: number, body: unknown, pretty = true) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8" });
  res.end(pretty ? JSON.stringify(body, null, 2) : JSON.stringify(body));
}
const fail = (res: ServerResponse, status: number, body: unknown) => send(res, status, body, false);

// LM Studio's reply IDs are about 22 random lowercase letters and digits.
const randomId = () => Array.from(randomBytes(22), (b) => "abcdefghijklmnopqrstuvwxyz0123456789"[b % 36]).join("");
