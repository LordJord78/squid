// The client side of a chat request: send the conversation, show the reply as it streams in, and
// return all of it. Used by `squid ask`, the chat, and the agent loop.
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

// Sends a chat and shows the reply as it streams in. Returns all of it: the answer, the thinking, and
// any tool calls, which arrive in pieces (the name first, then the arguments a few characters at a
// time) and are put back together by their index.
export async function streamReply(
  base: string,
  body: { model?: string; messages: Message[]; tools?: object[] },
  signal?: AbortSignal,
  out: Output = terminal,
): Promise<Reply> {
  const started = performance.now();
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
    signal,
  });
  if (!res.ok) throw new Error(errorText(await res.json()));

  const reply: Reply = { model: body.model ?? "", content: "", reasoning: "", toolCalls: [] };
  let firstWord: number | undefined;
  let thinking = false;
  for await (const data of sseData(res.body!)) {
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
  if (reply.content) out.answer("\n");
  reply.toolCalls = reply.toolCalls.filter(Boolean);
  if (firstWord !== undefined) {
    reply.secondsToFirstWord = (firstWord - started) / 1000;
    if (reply.usage) reply.tokensPerSecond = reply.usage.completion_tokens / ((performance.now() - firstWord) / 1000);
  }
  return reply;
}

// The model's context size, as the server reports it for a loaded model (/api/v0/models). The chat and
// the browser page use it for the "how full is the model's memory" meter.
export async function contextLimit(base: string, model: string): Promise<number | undefined> {
  const list = await (await fetch(`${base}/api/v0/models`)).json();
  return list.data.find((m: any) => m.id === model)?.loaded_context_length;
}

// Errors come as {"error": "text"} or {"error": {"message": "text"}}, as LM Studio sends them.
export const errorText = (body: any) => (typeof body.error === "string" ? body.error : body.error?.message ?? JSON.stringify(body));
