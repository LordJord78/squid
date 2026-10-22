// `squid` and `squid chat`: a conversation in the terminal, like `lms chat`.
// The model remembers nothing between requests. Every message is sent together with the whole
// conversation so far; that list of messages *is* the chat's memory.
import { createInterface } from "node:readline";
import { reachServer } from "./daemon.ts";
import { contextLimit, dim, pickModel, terminal, type Message } from "./reply.ts";
import { compact } from "./compact.ts";
import { brokeOff, MAX_ROUNDS, startConversation, takeTurn } from "./agent.ts";
import { relative } from "node:path";
import { newSession, nobodyToAsk, writeAtomically, type Session } from "./tools.ts";
import { applyUndo, describeUndo, planUndo, undoNote, withNote } from "./undo.ts";
import { keyboardApprover, keyboardPlanChooser, questionOn, type OpenQuestion } from "./approve.ts";
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

// The line after each answer: how full the memory is, and how fast the model wrote (step 20). A model that
// suddenly writes at half its speed usually shares the GPU with something new: a game, a video, a browser.
export function status(used: number, limit: number, compactAt: number, speed?: number, fastest = 0): string {
  const parts = [meter(used, limit)];
  if (used / limit > compactAt) parts.push("compacts before the next answer");
  if (speed) parts.push(`${Math.round(speed)} tok/s`);
  if (speed && speed < fastest / 2) parts.push(`slower than earlier (${Math.round(fastest)} tok/s): is another program using the GPU?`);
  return parts.join("  · ");
}

const HELP = `/clear            forget the conversation and start a new one
/model [name]     show or switch the model (it loads if needed)
/mode [name]      show or switch the mode: manual, auto or plan (or press Shift+Tab, at any time)
/system [text]    show or set instructions the model follows from the next message
/compact [focus]  make room in the model's memory: trim, then summarise (it happens by itself when 65% full)
/undo             put back the files the agent changed for your last message (again: further back)
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
  // Looked up again before each message: if the background server has stopped (closed, crashed, or
  // `squid server stop`), it is started again rather than leaving the chat stuck (step 21).
  const server = async () => `http://127.0.0.1:${(await reachServer()).port}`;
  let base = await server();
  const root = process.cwd(); // the folder the model may look at: wherever you typed `squid`
  let model = options.model && (await pickModel(base, options.model)); // undefined: the server picks the loaded or default model
  let system = options.system;
  let history: Message[] = startConversation(root, system); // [0] is the system message
  let used = 0; // tokens the conversation takes up, from the last reply's usage count
  let fastest = 0; // the best tokens per second so far in this chat
  let answering: AbortController | undefined;
  let pendingNote: string | undefined; // told to the model with your next message (after an /undo)

  const rl = createInterface({ input: process.stdin, output: process.stdout, prompt: "› " });
  // Piped-in input ends early (all lines arrive at once), and the reader closes while answers are still
  // being written. The lines already read are still handed over; only the prompt must not be shown.
  let open = true;
  rl.on("close", () => (open = false));
  // Approvals and plan choices are asked on the same readline. Piped-in input has nobody to answer, so
  // changes are refused and plans aren't approved.
  let openQuestion: OpenQuestion | undefined;
  const question = questionOn(rl, (open) => (openQuestion = open));
  const keyboard = process.stdin.isTTY;
  const session = newSession(root, keyboard ? keyboardApprover(question) : nobodyToAsk, (text) => process.stderr.write(text), {
    mode: options.mode ?? readConfig().defaultMode ?? "manual",
    compactAt: readConfig().compactAt,
    choosePlan: keyboard ? keyboardPlanChooser(question) : undefined,
  }); // lasts the whole chat
  // The prompt is redrawn with the current mode each time, since a turn can change it (an approved plan).
  const prompt = () => {
    if (!open) return;
    rl.setPrompt(promptFor(session.mode));
    rl.prompt();
  };
  // Ctrl+C: answer "no" to an open question, or stop the answer being written, or, at the prompt, leave.
  rl.on("SIGINT", () => (openQuestion ? openQuestion.cancel() : answering ? answering.abort() : rl.close()));
  // Shift+Tab (it arrives as the key "tab" with shift held) moves to the next mode at any time, as in
  // Claude Code. A question with its own Shift+Tab answer gets that instead: an edit's "yes, and switch to
  // auto mode", a plan's "go ahead in auto mode". While you're typing, the prompt is redrawn in place,
  // keeping what you've typed so far; while it's answering or asking, the new mode gets a line of its own.
  process.stdin.on("keypress", (_text, key) => {
    if (key?.name !== "tab" || !key.shift) return;
    if (openQuestion?.shiftTab) return openQuestion.shiftTab();
    session.mode = nextMode(session.mode);
    if (openQuestion) {
      process.stderr.write(`\r\x1b[K${dim(`Mode ${MODE_HELP[session.mode]}`)}\n`);
      rl.prompt(true); // the question again, with what you've typed so far
    } else if (answering) process.stderr.write(dim(`\nMode ${MODE_HELP[session.mode]}, from its next step\n`));
    else {
      rl.setPrompt(promptFor(session.mode));
      rl.prompt(true);
    }
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
      } else if (input.name === "undo") {
        const done = await undoLast(session, root, keyboard ? question : undefined);
        pendingNote = withNote([pendingNote, done]); // the model hears about it with your next message
      } else if (input.name === "mode") {
        const wanted = input.arg.toLowerCase();
        if (MODES.includes(wanted as Mode)) session.mode = wanted as Mode;
        else if (wanted) process.stderr.write(`There's no mode called ${input.arg}: manual, auto or plan.\n`);
        process.stderr.write(dim(`Mode ${MODE_HELP[session.mode]}\n`));
      } else if (input.name === "compact") {
        // Claude Code's /compact: the summary replaces the conversation. An optional focus steers it.
        if (history.length < 3) process.stderr.write(dim("Nothing to compact yet.\n"));
        else {
          answering = new AbortController(); // Ctrl+C stops the summary, as it stops an answer
          await compact(base, model, history, session, terminal, { force: true, focus: input.arg || undefined, signal: answering.signal })
            .then((done) => (used = done.after))
            .catch((err: Error) => process.stderr.write(answering?.signal.aborted ? dim("\n(stopped: trimmed, but not summarised)\n") : `${err.message}\n`))
            .finally(() => (answering = undefined));
        }
      } else if (input.name === "system") {
        if (input.arg) {
          system = input.arg;
          history[0] = startConversation(root, system)[0];
        }
        process.stderr.write(dim(system ? `System: ${system}\n` : "No system instructions.\n"));
      } else process.stderr.write(dim(`Unknown command /${input.name}. /help lists them.\n`));
    } else if (input.kind === "say") {
      answering = new AbortController();
      try {
        base = await server().catch(() => base); // if it can't start, the request below says so
        // The agent loop adds the answer, plus any tool calls and their results, to the history.
        const turn = await takeTurn(base, model, history, input.text, pendingNote, session, answering.signal);
        pendingNote = turn.note;
        if (turn.error) {
          const stopped = answering.signal.aborted; // Ctrl+C: said quietly, on a line of its own
          const told = brokeOff(turn, stopped, (file) => relative(root, file) || file);
          process.stderr.write(stopped ? dim(`\n${told}\n`) : `${told}\n`);
        } else {
          model = turn.model;
          used = turn.reply?.usage?.total_tokens ?? used;
          if (turn.paused) process.stderr.write(dim(`(paused after ${MAX_ROUNDS} rounds of tool calls without an answer; say "continue" to carry on)\n`));
          const limit = model && (await contextLimit(base, model));
          const speed = turn.reply?.tokensPerSecond;
          if (limit) process.stderr.write(dim(`${status(used, limit, session.compactAt, speed, fastest)}\n`));
          fastest = Math.max(fastest, speed ?? 0);
        }
      } finally {
        answering = undefined;
      }
    }
    prompt();
  }
  rl.close();
}

// /undo: shows what would be put back, asks (at a keyboard), and does it. Returns a note for the model,
// whose picture of those files is now out of date, or undefined if nothing was undone.
async function undoLast(session: Session, root: string, question?: (prompt: string) => Promise<string | undefined>): Promise<string | undefined> {
  const plan = planUndo(session.checkpoints);
  if (!plan) {
    process.stderr.write(dim("Nothing to undo: the agent hasn't changed any files in this chat. (Changes made by commands can't be undone here; use git.)\n"));
    return undefined;
  }
  const name = (file: string) => relative(root, file) || file;
  process.stderr.write(`${describeUndo(plan, name, process.stderr.isTTY)}\n`);
  if (!plan.restore.length) return undefined;
  if (question && !/^(y|yes)$/i.test(((await question("Undo these? y = yes · n = no › ")) ?? "").trim())) {
    process.stderr.write(dim("Nothing undone.\n"));
    return undefined;
  }
  applyUndo(session.checkpoints, plan, writeAtomically);
  process.stderr.write(dim(`Undid ${plan.restore.length} file${plan.restore.length === 1 ? "" : "s"}.\n`));
  return undoNote(plan.checkpoint.label, plan.restore.map((s) => name(s.file)));
}
