// The client side of a chat request: send the conversation, show the reply as it streams in, and
// return all of it. Used by `squid ask`, the chat, and the agent loop.
import { request, type IncomingMessage } from "node:http";
import { sseData } from "./sse.ts";
import type { ToolCall } from "./tools.ts";

export type Message = {
  role: "system" | "user" | "assistant" | "tool";
  content: string;
  reasoning_content?: string; // the model's thinking, kept while it works through tool calls
  tool_calls?: { id: string; type: "function"; function: { name: string; arguments: string } }[];
  tool_call_id?: string; // on a tool result: which call it answers
};
export type Reply = {
  model: string;
  content: string;
  reasoning: string;
  toolCalls: ToolCall[];
  finishReason?: string;
  usage?: any;
  secondsToFirstWord?: number;
  tokensPerSecond?: number;
};

// Messages for you (thinking, tool use, stats, meters) go to stderr, dimmed; the answer goes to stdout.
// Both show on screen, but `squid ask "…" > answer.txt` saves just the answer.
export const dim = (text: string) => (process.stderr.isTTY ? `\x1b[2m${text}\x1b[0m` : text);
export type Output = { answer(text: string): void; thinking(text: string): void; note(text: string): void };
export const terminal: Output = {
  answer: (text) => process.stdout.write(text),
  thinking: (text) => process.stderr.write(dim(text)),
  note: (text) => process.stderr.write(text),
};

// An answer that didn't finish. Either the model's memory (its context) filled up mid-answer, or the
// engine or the connection broke. What it wrote so far can't be trusted (a tool call may be half
// written), so none of it is used.
export class CutOff extends Error {
  readonly reason: "memory full" | "broken";
  constructor(reason: "memory full" | "broken", message = reason === "memory full" ? MEMORY_FULL : BROKEN) {
    super(message);
    this.reason = reason;
  }
}
const MEMORY_FULL = "The model's memory (its context) filled up in the middle of the answer.";
const BROKEN = "The answer was cut off before it finished (the engine stopped, or the connection broke).";

// Sends a chat and shows the reply as it streams in. Returns all of it: the answer, the thinking, and
// any tool calls, which arrive in pieces (the name first, then the arguments a few characters at a
// time) and are put back together by their index.
export async function streamReply(
  base: string,
  // max_tokens caps the reply's length; chat_template_kwargs reach the model's chat template (step 19's
  // summary turns the 27B's thinking off with { enable_thinking: false }).
  body: { model?: string; messages: Message[]; tools?: object[]; max_tokens?: number; chat_template_kwargs?: object },
  signal?: AbortSignal,
  out: Output = terminal,
): Promise<Reply> {
  const started = performance.now();
  const res = await post(`${base}/v1/chat/completions`, { ...body, stream: true, stream_options: { include_usage: true } }, signal).catch((err) => {
    if (signal?.aborted) throw err;
    throw new CutOff("broken"); // the connection broke before the answer began
  });
  if (res.statusCode !== 200) {
    const refusal = JSON.parse(await readAll(res));
    // llama.cpp refuses a conversation that no longer fits: {"error": {"type": "exceed_context_size_error", ...}}
    if (refusal.error?.type === "exceed_context_size_error") throw new CutOff("memory full", "The conversation no longer fits in the model's memory (its context).");
    throw new Error(errorText(refusal));
  }

  const reply: Reply = { model: body.model ?? "", content: "", reasoning: "", toolCalls: [] };
  let firstWord: number | undefined;
  let thinking = false;
  try {
    for await (const data of sseData(res)) {
      if (data === "[DONE]") break;
      const chunk = JSON.parse(data);
      reply.model = chunk.model; // which model answered (the server may have picked it)
      if (chunk.usage) reply.usage = chunk.usage;
      const choice = chunk.choices[0];
      if (!choice) continue;
      if (choice.finish_reason) reply.finishReason = choice.finish_reason;
      const delta = choice.delta ?? {};
      if ((delta.reasoning_content || delta.content) && firstWord === undefined) firstWord = performance.now();
      if (delta.reasoning_content) {
        out.thinking(delta.reasoning_content);
        reply.reasoning += delta.reasoning_content;
        thinking = true;
      }
      if ((delta.content || delta.tool_calls) && thinking) {
        out.thinking("\n"); // a line between the thinking and what follows
        thinking = false;
      }
      if (delta.content) {
        out.answer(delta.content);
        reply.content += delta.content;
      }
      for (const piece of delta.tool_calls ?? []) {
        const call = (reply.toolCalls[piece.index] ??= { id: "", name: "", arguments: "" });
        if (piece.id) call.id = piece.id;
        if (piece.function?.name) call.name += piece.function.name;
        if (piece.function?.arguments) call.arguments += piece.function.arguments;
      }
    }
  } catch (err) {
    if (signal?.aborted) throw err; // you stopped it (Ctrl+C): that's not a breakdown
    throw new CutOff("broken");
  } finally {
    if (reply.content) out.answer("\n");
  }
  // A finished answer always says why it ended. No reason means the stream just stopped.
  if (!reply.finishReason) throw new CutOff("broken");
  // "length": it ran out of room. Words already shown can stay, but a tool call may be half written.
  if (reply.finishReason === "length" && reply.toolCalls.length) throw new CutOff("memory full");
  reply.toolCalls = reply.toolCalls.filter(Boolean);
  if (firstWord !== undefined) {
    reply.secondsToFirstWord = (firstWord - started) / 1000;
    if (reply.usage) reply.tokensPerSecond = reply.usage.completion_tokens / ((performance.now() - firstWord) / 1000);
  }
  return reply;
}

// Sends a request and resolves as soon as the reply starts arriving. node:http rather than fetch, as in
// Squid's server: fetch gives up when no reply has started after 5 minutes, and a big model can take longer
// than that to read a long prompt (Flash-Next on the PC, 22 Oct: an 8,330-token prompt, cut off at 300 s).
function post(url: string, body: unknown, signal?: AbortSignal): Promise<IncomingMessage> {
  return new Promise((resolve, reject) => {
    const outgoing = request(url, { method: "POST", headers: { "Content-Type": "application/json" }, signal }, resolve);
    outgoing.on("error", reject);
    outgoing.end(JSON.stringify(body));
  });
}

async function readAll(res: IncomingMessage): Promise<string> {
  res.setEncoding("utf8");
  let text = "";
  for await (const chunk of res) text += chunk;
  return text;
}

// An exact model name, or the one whose name contains what was typed (as `squid load` accepts): `--model
// smol`, `/model 9b`.
export async function pickModel(base: string, query: string): Promise<string> {
  const ids: string[] = (await (await fetch(`${base}/v1/models`)).json()).data.map((m: any) => m.id);
  if (ids.includes(query)) return query;
  const matches = ids.filter((id) => id.includes(query.toLowerCase()));
  if (matches.length === 1) return matches[0];
  throw new Error(matches.length ? `"${query}" matches ${matches.join(", ")}` : `no model matches "${query}" (see squid ls)`);
}

// The model's context size, as the server reports it for a loaded model (/api/v0/models). The chat and
// the browser page use it for the "how full is the model's memory" meter.
export async function contextLimit(base: string, model: string): Promise<number | undefined> {
  const list = await (await fetch(`${base}/api/v0/models`)).json();
  return list.data.find((m: any) => m.id === model)?.loaded_context_length;
}

// Errors come as {"error": "text"} or {"error": {"message": "text"}}, as LM Studio sends them.
export const errorText = (body: any) => (typeof body.error === "string" ? body.error : body.error?.message ?? JSON.stringify(body));
