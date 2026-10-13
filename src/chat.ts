// `squid` and `squid chat`: a conversation in the terminal, like `lms chat`.
// The model remembers nothing between requests. Every message is sent together with the whole
// conversation so far; that list of messages *is* the chat's memory.
import { createInterface } from "node:readline";
import { ensureServer } from "./daemon.ts";
import { dim, type Message } from "./reply.ts";
import { agentTurn, forgetThinking, startConversation } from "./agent.ts";
import { newSession, nobodyToAsk } from "./tools.ts";
import { keyboardApprover, keyboardPlanChooser, questionOn } from "./approve.ts";
import { MODES, type Mode } from "./rules.ts";
import { readConfig } from "./config.ts";

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
/mode [name]      show or switch the mode: manual, auto or plan (or press Shift+Tab)
/system [text]    show or set instructions the model follows from the next message
/exit             leave (so does Ctrl+D, or Ctrl+C at an empty prompt)
Ctrl+C while it's answering stops the answer.`;

// The mode is always on screen, in the prompt itself (Claude Code shows it under the input box).
export const promptFor = (mode: Mode) => (mode === "manual" ? "› " : mode === "auto" ? `${dim("⏵⏵ auto")} › ` : `${dim("⏸ plan")} › `);
export const nextMode = (mode: Mode): Mode => MODES[(MODES.indexOf(mode) + 1) % MODES.length]; // manual → auto → plan → manual
const MODE_HELP: Record<Mode, string> = {
  manual: "manual: every change and every command that isn't read-only is shown to you first",
  auto: "auto: edits inside this folder go ahead without asking; commands still ask",
  plan: "plan: it only looks, then proposes a plan for you to approve",
};

export async function chat(options: { model?: string; system?: string; mode?: Mode }): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`;
  const root = process.cwd(); // the folder the model may look at: wherever you typed `squid`
  let model = options.model; // undefined: the server picks the loaded or default model
  let system = options.system;
  let history: Message[] = startConversation(root, system); // [0] is the system message
  let used = 0; // tokens the conversation takes up, from the last reply's usage count
  let answering: AbortController | undefined;

  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "› " });
  // Piped-in input ends early (all lines arrive at once), and the reader closes while answers are still
  // being written. The lines already read are still handed over; only the prompt must not be shown.
  let open = true;
  rl.on("close", () => (open = false));
  // Approvals and plan choices are asked on the same readline. Piped-in input has nobody to answer, so
  // changes are refused and plans aren't approved.
  let cancelQuestion: (() => void) | undefined;
  const question = questionOn(rl, (cancel) => (cancelQuestion = cancel));
  const keyboard = process.stdin.isTTY;
  const session = newSession(root, keyboard ? keyboardApprover(question) : nobodyToAsk, (text) => process.stderr.write(text), {
    mode: options.mode ?? readConfig().defaultMode ?? "manual",
    choosePlan: keyboard ? keyboardPlanChooser(question) : undefined,
  }); // lasts the whole chat
  // The prompt is redrawn with the current mode each time, since a turn can change it (an approved plan).
  const prompt = () => {
    if (!open) return;
    rl.setPrompt(promptFor(session.mode));
    rl.prompt();
  };
  // Ctrl+C: answer "no" to an open question, or stop the answer being written, or, at the prompt, leave.
  rl.on("SIGINT", () => (cancelQuestion ? cancelQuestion() : answering ? answering.abort() : rl.close()));
  // Shift+Tab (it arrives as the key "tab" with shift held) cycles the mode while you're typing, and the
  // prompt is redrawn in place, keeping what you've typed so far.
  process.stdin.on("keypress", (_text, key) => {
    if (key?.name !== "tab" || !key.shift || answering || cancelQuestion) return;
    session.mode = nextMode(session.mode);
    rl.setPrompt(promptFor(session.mode));
    rl.prompt(true);
  });
  process.stderr.write(dim(`Chatting with ${model ?? "your default model"} in ${root}. Mode ${MODE_HELP[session.mode]}. Shift+Tab changes mode; /help for commands.\n`));
  prompt();

  // `for await` hands over typed lines one at a time and holds on to any that arrive while an answer
  // is still being written (pasted text, or lines piped in), so nothing typed is lost.
  for await (const line of rl) {
    const input = parseLine(line);
    if (input.kind === "command") {
      if (input.name === "exit" || input.name === "quit") break;
      else if (input.name === "help") process.stderr.write(dim(`${HELP}\n`));
      else if (input.name === "clear") {
        history = startConversation(root, system); // a fresh conversation still starts with its instructions
        used = 0;
        process.stderr.write(dim("Started a new conversation.\n"));
      } else if (input.name === "model") {
        if (input.arg) {
          model = await pickModel(base, input.arg).catch((err: Error) => (process.stderr.write(`${err.message}\n`), model));
        }
        process.stderr.write(dim(`Model: ${model ?? "your default model"}\n`));
      } else if (input.name === "mode") {
        const wanted = input.arg.toLowerCase();
        if (MODES.includes(wanted as Mode)) session.mode = wanted as Mode;
        else if (wanted) process.stderr.write(`There's no mode called ${input.arg}: manual, auto or plan.\n`);
        process.stderr.write(dim(`Mode ${MODE_HELP[session.mode]}\n`));
      } else if (input.name === "system") {
        if (input.arg) {
          system = input.arg;
          history[0] = startConversation(root, system)[0];
        }
        process.stderr.write(dim(system ? `System: ${system}\n` : "No system instructions.\n"));
      } else process.stderr.write(dim(`Unknown command /${input.name}. /help lists them.\n`));
    } else if (input.kind === "say") {
      forgetThinking(history);
      const before = history.length;
      history.push({ role: "user", content: input.text });
      answering = new AbortController();
      try {
        // The agent loop adds the answer, plus any tool calls and their results, to the history.
        const turn = await agentTurn(base, model, history, session, answering.signal);
        model = turn.model;
        used = turn.reply.usage?.total_tokens ?? used;
        const limit = await contextLimit(base, model);
        if (limit) process.stderr.write(dim(`${meter(used, limit)}${used / limit > 0.8 ? "  · nearly full: /clear starts fresh" : ""}\n`));
      } catch (err) {
        history.length = before; // the question wasn't answered: drop it, and any tool calls it started
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
