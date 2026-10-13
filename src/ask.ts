// `squid ask "…"`: one question, with the answer printed as it's written, like `lms chat -p`.
import { ensureServer } from "./daemon.ts";
import { sseData } from "./sse.ts";

export type AskOptions = { model?: string; system?: string; stats?: boolean };

// With no model named, the server uses the one loaded, or loads your default model.
export async function ask(prompt: string, options: AskOptions): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`; // starts it in the background if needed
  const messages = [...(options.system ? [{ role: "system", content: options.system }] : []), { role: "user", content: prompt }];
  const started = performance.now();
  const res = await fetch(`${base}/v1/chat/completions`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ model: options.model, messages, stream: true, stream_options: { include_usage: true } }),
  });
  if (!res.ok) throw new Error(errorText(await res.json()));

  // Thinking goes to stderr, dimmed: you see it on screen, but `squid ask "…" > answer.txt`
  // (which only captures stdout) saves just the answer.
  const dim = (text: string) => (process.stderr.isTTY ? `\x1b[2m${text}\x1b[0m` : text);
  let firstWord: number | undefined;
  let thought = false;
  let usage: any;
  for await (const data of sseData(res.body!)) {
    if (data === "[DONE]") break;
    const chunk = JSON.parse(data);
    if (chunk.usage) usage = chunk.usage;
    const delta = chunk.choices[0]?.delta ?? {};
    if ((delta.reasoning_content || delta.content) && firstWord === undefined) firstWord = performance.now();
    if (delta.reasoning_content) {
      process.stderr.write(dim(delta.reasoning_content));
      thought = true;
    }
    if (delta.content) {
      if (thought) process.stderr.write("\n"); // a line between the thinking and the answer
      thought = false;
      process.stdout.write(delta.content);
    }
  }
  process.stdout.write("\n");

  // Like `lms chat --stats`, measured here at the receiving end.
  if (options.stats && usage && firstWord !== undefined) {
    const seconds = (performance.now() - firstWord) / 1000;
    const thinking = usage.completion_tokens_details?.reasoning_tokens ?? 0;
    process.stderr.write(dim(`${(usage.completion_tokens / seconds).toFixed(1)} tok/s · ${usage.completion_tokens} tokens (${thinking} thinking) · first word after ${((firstWord - started) / 1000).toFixed(2)} s\n`));
  }
}

// Errors come as {"error": "text"} or {"error": {"message": "text"}}, as LM Studio sends them.
const errorText = (body: any) => (typeof body.error === "string" ? body.error : body.error?.message ?? JSON.stringify(body));
