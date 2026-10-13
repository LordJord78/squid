// `squid` and `squid chat`: a conversation in the terminal, like `lms chat`.
// The model remembers nothing between requests. Every message is sent together with the whole
// conversation so far; that list of messages *is* the chat's memory.
import { createInterface } from "node:readline";
import { ensureServer } from "./daemon.ts";
import { dim, streamReply, type Message } from "./ask.ts";

// What a typed line asks for: nothing, a message for the model, or a /command for Squid.
export type Line = { kind: "empty" } | { kind: "say"; text: string } | { kind: "command"; name: string; arg: string };

export function parseLine(line: string): Line {
  const text = line.trim();
  if (!text) return { kind: "empty" };
  if (!text.startsWith("/")) return { kind: "say", text };
  const name = text.slice(1).split(/\s/)[0];
  return { kind: "command", name: name.toLowerCase(), arg: text.slice(1 + name.length).trim() };
}

// How full the model's memory is, drawn as a bar: ▓▓░░░░░░░░░░░░░░░░░░ 4,812 / 49,152 tokens (10%)
export function meter(used: number, limit: number, width = 20): string {
  const share = Math.min(1, used / limit);
  const filled = Math.round(share * width);
  return `${"▓".repeat(filled)}${"░".repeat(width - filled)} ${used.toLocaleString("en-US")} / ${limit.toLocaleString("en-US")} tokens (${Math.round(share * 100)}%)`;
}

const HELP = `/clear            forget the conversation and start a new one
/model [name]     show or switch the model (it loads if needed)
/system [text]    show or set instructions the model follows from the next message
/exit             leave (so does Ctrl+D, or Ctrl+C at an empty prompt)
Ctrl+C while it's answering stops the answer.`;

export async function chat(options: { model?: string; system?: string }): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`;
  let model = options.model; // undefined: the server picks the loaded or default model
  let system = options.system;
  let history: Message[] = [];
  let used = 0; // tokens the conversation takes up, from the last reply's usage count
  let answering: AbortController | undefined;

  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "› " });
  // Piped-in input ends early (all lines arrive at once), and the reader closes while answers are still
  // being written. The lines already read are still handed over; only the prompt must not be shown.
  let open = true;
  rl.on("close", () => (open = false));
  const prompt = () => open && rl.prompt();
  // Ctrl+C: stop the answer being written, or, at the prompt, leave.
  rl.on("SIGINT", () => (answering ? answering.abort() : rl.close()));
  process.stderr.write(dim(`Chatting with ${model ?? "your default model"}. /help for commands.\n`));
  prompt();

  // `for await` hands over typed lines one at a time and holds on to any that arrive while an answer
  // is still being written (pasted text, or lines piped in), so nothing typed is lost.
  for await (const line of rl) {
    const input = parseLine(line);
    if (input.kind === "command") {
      if (input.name === "exit" || input.name === "quit") break;
      else if (input.name === "help") process.stderr.write(dim(`${HELP}\n`));
      else if (input.name === "clear") {
        history = [];
        used = 0;
        process.stderr.write(dim("Started a new conversation.\n"));
      } else if (input.name === "model") {
        if (input.arg) {
          model = await pickModel(base, input.arg).catch((err: Error) => (process.stderr.write(`${err.message}\n`), model));
        }
        process.stderr.write(dim(`Model: ${model ?? "your default model"}\n`));
      } else if (input.name === "system") {
        if (input.arg) system = input.arg;
        process.stderr.write(dim(system ? `System: ${system}\n` : "No system instructions.\n"));
      } else process.stderr.write(dim(`Unknown command /${input.name}. /help lists them.\n`));
    } else if (input.kind === "say") {
      history.push({ role: "user", content: input.text });
      answering = new AbortController();
      try {
        const messages: Message[] = [...(system ? [{ role: "system" as const, content: system }] : []), ...history];
        const reply = await streamReply(base, { model, messages }, answering.signal);
        history.push({ role: "assistant", content: reply.content });
        model = reply.model;
        used = reply.usage?.total_tokens ?? used;
        const limit = await contextLimit(base, model);
        if (limit) process.stderr.write(dim(`${meter(used, limit)}${used / limit > 0.8 ? "  · nearly full: /clear starts fresh" : ""}\n`));
      } catch (err) {
        history.pop(); // the question wasn't answered, so it isn't part of the conversation
        process.stderr.write(answering.signal.aborted ? dim("\n(stopped; that message was left out of the conversation)\n") : `${(err as Error).message}\n`);
      } finally {
        answering = undefined;
      }
    }
    prompt();
  }
  rl.close();
}

// The model's context size, as the server reports it for a loaded model (/api/v0/models).
async function contextLimit(base: string, model: string): Promise<number | undefined> {
  const list = await (await fetch(`${base}/api/v0/models`)).json();
  return list.data.find((m: any) => m.id === model)?.loaded_context_length;
}

// An exact model name, or the one whose name contains what was typed (as `squid load` accepts).
async function pickModel(base: string, query: string): Promise<string> {
  const ids: string[] = (await (await fetch(`${base}/v1/models`)).json()).data.map((m: any) => m.id);
  if (ids.includes(query)) return query;
  const matches = ids.filter((id) => id.includes(query.toLowerCase()));
  if (matches.length === 1) return matches[0];
  throw new Error(matches.length ? `"${query}" matches ${matches.join(", ")}` : `no model matches "${query}" (see squid ls)`);
}
