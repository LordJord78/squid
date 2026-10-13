// `squid ask "…"`: one question, with the answer printed as it's written, like `lms chat -p`.
// streamReply() is shared with the chat (step 07).
import { ensureServer } from "./daemon.ts";
import { sseData } from "./sse.ts";

export type AskOptions = { model?: string; system?: string; stats?: boolean };
export type Message = { role: "system" | "user" | "assistant"; content: string };
export type Reply = { model: string; content: string; usage?: any; secondsToFirstWord?: number; tokensPerSecond?: number };

// Messages for you (thinking, stats, meters) go to stderr, dimmed; the answer goes to stdout.
// Both show on screen, but `squid ask "…" > answer.txt` saves just the answer.
export const dim = (text: string) => (process.stderr.isTTY ? `\x1b[2m${text}\x1b[0m` : text);

// With no model named, the server uses the one loaded, or loads your default model.
export async function ask(prompt: string, options: AskOptions): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`; // starts it in the background if needed
  // Text piped in (`cat notes.md | squid ask "Summarise this"`) comes along with the question, as
  // `lms chat -p` documents.
  const piped = process.stdin.isTTY ? "" : (await readStdin()).trim();
  const messages: Message[] = [
    ...(options.system ? [{ role: "system" as const, content: options.system }] : []),
    { role: "user", content: piped ? `${prompt}\n\n${piped}` : prompt },
  ];
  const reply = await streamReply(base, { model: options.model, messages });
  // Like `lms chat --stats`, measured here at the receiving end.
  if (options.stats && reply.usage && reply.tokensPerSecond !== undefined) {
    const thinking = reply.usage.completion_tokens_details?.reasoning_tokens ?? 0;
    process.stderr.write(dim(`${reply.tokensPerSecond.toFixed(1)} tok/s · ${reply.usage.completion_tokens} tokens (${thinking} thinking) · first word after ${reply.secondsToFirstWord!.toFixed(2)} s\n`));
  }
}

// Sends a chat and prints the reply as it streams in: thinking dimmed on stderr, the answer on stdout.
// Returns the whole answer, so a conversation can remember it.
export async function streamReply(base: string, body: { model?: string; messages: Message[] }, signal?: AbortSignal): Promise<Reply> {
  const started = performance.now();
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...body, stream: true, stream_options: { include_usage: true } }),
    signal,
  });
  if (!res.ok) throw new Error(errorText(await res.json()));

  const reply: Reply = { model: body.model ?? "", content: "" };
  let firstWord: number | undefined;
  let thinking = false;
  for await (const data of sseData(res.body!)) {
    if (data === "[DONE]") break;
    const chunk = JSON.parse(data);
    reply.model = chunk.model; // which model answered (the server may have picked it)
    if (chunk.usage) reply.usage = chunk.usage;
    const delta = chunk.choices[0]?.delta ?? {};
    if ((delta.reasoning_content || delta.content) && firstWord === undefined) firstWord = performance.now();
    if (delta.reasoning_content) {
      process.stderr.write(dim(delta.reasoning_content));
      thinking = true;
    }
    if (delta.content) {
      if (thinking) process.stderr.write("\n"); // a line between the thinking and the answer
      thinking = false;
      process.stdout.write(delta.content);
      reply.content += delta.content;
    }
  }
  process.stdout.write("\n");
  if (firstWord !== undefined) {
    reply.secondsToFirstWord = (firstWord - started) / 1000;
    if (reply.usage) reply.tokensPerSecond = reply.usage.completion_tokens / ((performance.now() - firstWord) / 1000);
  }
  return reply;
}

// Errors come as {"error": "text"} or {"error": {"message": "text"}}, as LM Studio sends them.
export const errorText = (body: any) => (typeof body.error === "string" ? body.error : body.error?.message ?? JSON.stringify(body));

async function readStdin(): Promise<string> {
  process.stdin.setEncoding("utf8"); // decode the stream as a whole, so no character is cut in two (step 05)
  let text = "";
  for await (const chunk of process.stdin) text += chunk;
  return text;
}
