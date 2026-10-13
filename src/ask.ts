// `squid ask "…"`: one question, with the answer printed as it's written, like `lms chat -p`.
import type { AddressInfo } from "node:net";
import { startServer, DEFAULT_PORT } from "./server.ts";
import { sseData } from "./sse.ts";

export type AskOptions = { model?: string; system?: string; stats?: boolean };

export async function ask(prompt: string, options: AskOptions): Promise<void> {
  const door = await frontDoor();
  try {
    const messages = [...(options.system ? [{ role: "system", content: options.system }] : []), { role: "user", content: prompt }];
    const started = performance.now();
    const res = await fetch(`${door.base}/v1/chat/completions`, {
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
  } finally {
    door.close();
  }
}

// The shared front door if `squid serve` is running. Otherwise a private one on a free port, just for
// this question, closed afterwards. (Step 06 makes `squid` start the shared one by itself.)
async function frontDoor(): Promise<{ base: string; close: () => void }> {
  const shared = `http://127.0.0.1:${DEFAULT_PORT}`;
  try {
    await fetch(`${shared}/v1/models`, { signal: AbortSignal.timeout(2000) });
    return { base: shared, close: () => {} };
  } catch {
    const server = await startServer(0);
    const close = () => {
      server.close();
      server.closeAllConnections(); // don't wait for idle keep-alive connections before exiting
    };
    return { base: `http://127.0.0.1:${(server.address() as AddressInfo).port}`, close };
  }
}

// Errors come as {"error": "text"} or {"error": {"message": "text"}}, as LM Studio sends them.
const errorText = (body: any) => (typeof body.error === "string" ? body.error : body.error?.message ?? JSON.stringify(body));
