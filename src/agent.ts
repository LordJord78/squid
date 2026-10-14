// The agent loop, the heart of Claude Code: send the conversation; if the model asks for tools, run
// them and send back what they returned; repeat until the model answers in words.
import { describeCall, runTool, toolsFor, type Session } from "./tools.ts";
import { loadRulebook, type Mode } from "./rules.ts";
import { startCheckpoint, withoutNote } from "./undo.ts";
import { CutOff, dim, streamReply, terminal, type Message, type Output, type Reply } from "./reply.ts";

// A model that keeps calling tools without ever answering may be stuck. The AI Lab's benchmark saw the
// 30B model burn all 30 of its turns that way. After this many rounds the turn pauses and you decide.
export const MAX_ROUNDS = 25;

// The first message of every conversation: where the model is and what it can do.
export function startConversation(root: string, extra?: string): Message[] {
  const prompt = [
    `You are Squid, an assistant running on this computer, working in the folder ${root}.`,
    "You can look at its files with list_files, read_file and search, change them with edit_file and write_file, and run programs with run_command (one at a time: there is no shell, so no &&, | or >). Paths are relative to that folder; nothing outside it can be opened.",
    "Every change and every command that isn't read-only is shown to the user, who approves or declines it. If they decline, don't try the same change again; follow their instructions if they gave any.",
    "Some files are off limits (secrets, holdout data, and anything the user's rules deny). If a tool says so, don't try to get around it.",
    "When a question is about the project, look before you answer, and say which files you read. Be concise.",
  ].join("\n");
  return [{ role: "system", content: extra ? `${prompt}\n\n${extra}` : prompt }];
}

// What each mode means, told to the model in the first (system) message. It's put there, rather than in
// a message of its own, because some models' chat templates refuse a system message anywhere but first.
const MODE_NOTE: Record<Mode, string> = {
  manual: "Mode: manual. Changes, and commands that aren't read-only, are shown to the user to approve first.",
  auto: "Mode: auto. Edits inside this folder go ahead without asking; commands that aren't read-only still need approval.",
  plan: "Mode: plan. Investigate first (read files, read-only commands), then call submit_plan with a clear numbered plan. Don't change anything: edits are blocked until the user approves the plan.",
};
function withMode(messages: Message[], mode: Mode): Message[] {
  const [first, ...rest] = messages;
  if (first?.role !== "system") return [{ role: "system", content: MODE_NOTE[mode] }, ...messages];
  return [{ ...first, content: `${first.content}\n\n${MODE_NOTE[mode]}` }, ...rest];
}

// One turn of the conversation. The model may call tools again and again (up to MAX_ROUNDS) before it
// answers. Every call and result is added to `messages`, so later turns remember what was looked at.
// The session (tools.ts) carries the folder, the rules, who approves changes, and what has been read.
export async function agentTurn(
  base: string,
  model: string | undefined,
  messages: Message[],
  session: Session,
  signal?: AbortSignal,
  out: Output = terminal,
): Promise<{ model: string; reply: Reply; paused?: boolean }> {
  session.book = loadRulebook(session.root); // rules edited mid-chat apply from the next question (step 09)
  session.book.allow.push(...session.remembered); // plus commands you said "don't ask again" for (step 12)
  session.signal = signal; // Ctrl+C also stops a command that's running (step 11)
  // Each of your messages starts a checkpoint: /undo puts back what this turn changes (step 13).
  startCheckpoint(session.checkpoints, withoutNote([...messages].reverse().find((m) => m.role === "user")?.content ?? ""));
  let reply: Reply | undefined;
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    // The mode can change during a turn (an approved plan), so the tools and the note are fresh each round.
    reply = await streamReply(base, { model, messages: withMode(messages, session.mode), tools: toolsFor(session.mode) }, signal, out);
    model = reply.model;
    messages.push({
      role: "assistant",
      content: reply.content,
      // The thinking is sent back while the model is still working through tools, so it can follow
      // its own plan from one call to the next.
      ...(reply.reasoning ? { reasoning_content: reply.reasoning } : {}),
      ...(reply.toolCalls.length
        ? { tool_calls: reply.toolCalls.map((c) => ({ id: c.id, type: "function" as const, function: { name: c.name, arguments: c.arguments } })) }
        : {}),
    });
    if (!reply.toolCalls.length) return { model, reply }; // it answered in words: the turn is over

    for (const call of reply.toolCalls) {
      out.note(`● ${describeCall(call)}\n`);
      const result = await runTool(session, call); // may stop here to ask you (step 10)
      out.note(`${dim(`  ⎿ ${summarise(result)}`)}\n`);
      // A tool result is a message of its own, linked to the call by its id.
      messages.push({ role: "tool", tool_call_id: call.id, content: result });
    }
  }
  // Every round finished, so nothing is lost: the conversation ends on tool results, and your next
  // message ("continue") picks up from there.
  return { model: model!, reply: reply!, paused: true };
}

// One of your messages, start to finish: the chat and the browser page both use this. Your message is
// added to the history, and the agent loop adds its tool calls, their results and its answer.
// If the turn breaks off (Ctrl+C, a cut-off answer, an error), the rounds that finished are kept: their
// tool calls really ran, and files really changed, so the model must not forget them. Only the
// unfinished reply is dropped. Until 13 Oct the whole turn was dropped here, and a 27B that had just
// written 17 files forgot all of them.
export type Outcome = {
  model: string | undefined;
  reply?: Reply; // the last reply, when the turn ended normally or paused
  paused?: boolean; // MAX_ROUNDS rounds of tool calls, no answer yet
  error?: Error; // why it broke off; Ctrl+C is an AbortError
  kept: number; // rounds of tool calls kept from a turn that broke off (0: your message was left out too)
  changed: string[]; // files the agent changed during the turn
  note?: string; // to tell the model, in front of your next message
};

export async function takeTurn(
  base: string,
  model: string | undefined,
  history: Message[],
  text: string,
  note: string | undefined,
  session: Session,
  signal?: AbortSignal,
  out: Output = terminal,
): Promise<Outcome> {
  forgetThinking(history);
  const before = history.length;
  history.push({ role: "user", content: note ? `${note}\n\n${text}` : text });
  const changed = () => [...(session.checkpoints.at(-1)?.snapshots.values() ?? [])].filter((s) => s.after !== null).map((s) => s.file);
  try {
    const turn = await agentTurn(base, model, history, session, signal, out);
    return { ...turn, kept: 0, changed: changed() };
  } catch (err) {
    history.length = finishedLength(history, before);
    const kept = history.slice(before).filter((m) => m.role === "assistant").length;
    // Nothing finished: your message is left out, and a note it carried waits for the next one.
    if (!kept) return { model, error: err as Error, kept, changed: changed(), note };
    return { model, error: err as Error, kept, changed: changed(), note: interruptedNote(why(err as Error, signal)) };
  }
}

// How much of a turn to keep: up to the end of the last round whose tool calls all got their results.
// A reply that was still being written, or whose calls didn't all run, is cut away. `before` is where
// the turn started; if no round finished, that's where the history ends again.
export function finishedLength(messages: Message[], before: number): number {
  let end = before;
  let i = before + 1; // after your message
  while (messages[i]?.role === "assistant" && messages[i].tool_calls?.length) {
    const calls = messages[i].tool_calls!;
    const results = messages.slice(i + 1, i + 1 + calls.length);
    if (results.length < calls.length || results.some((r, k) => r.role !== "tool" || r.tool_call_id !== calls[k].id)) break;
    i += 1 + calls.length;
    end = i;
  }
  return end;
}

// What you're told when a turn breaks off, in the terminal and on the page. `name` shortens paths.
export function brokeOff(turn: Outcome, stopped: boolean, name: (file: string) => string): string {
  const what = stopped ? "Stopped." : turn.error?.message ?? "";
  const full = turn.error instanceof CutOff && turn.error.reason === "memory full";
  if (!turn.kept) return `${what} That message was left out of the conversation.${full ? " /clear starts a fresh one." : ""}`;
  const files = turn.changed.map(name);
  const shown = files.length > 5 ? `${files.slice(0, 5).join(", ")} and ${files.length - 5} more` : files.join(", ");
  const rounds = turn.kept === 1 ? "The round of tool calls it finished is" : `The ${turn.kept} rounds of tool calls it finished are`;
  return [
    what,
    `${rounds} kept${files.length ? ` (files changed: ${shown})` : ""}; only the unfinished reply was left out.`,
    `Say "continue" to carry on.${full ? " If that doesn't fit either, /clear starts a fresh conversation." : ""}`,
  ].join(" ");
}

function why(err: Error, signal?: AbortSignal): string {
  if (signal?.aborted) return "the user stopped it";
  if (err instanceof CutOff) return err.reason === "memory full" ? "your memory (the context) filled up" : "the answer was cut off";
  return `an error (${err.message})`;
}

// Told to the model in front of your next message, like the undo note (undo.ts). It starts the same way,
// so withoutNote() can take it off again when the message is shown or used as a label.
export const interruptedNote = (reason: string) =>
  `(Squid: your last answer was interrupted: ${reason}. The tool calls above did run, and their results are real; the reply you were writing was not saved.)`;

// Thinking from finished turns isn't needed again, and it takes up context. Dropped before each new question.
export function forgetThinking(messages: Message[]): void {
  for (const message of messages) delete message.reasoning_content;
}

// One line about a tool's result for the screen: "180 lines", or the error.
function summarise(result: string): string {
  if (result.startsWith("Error: ")) return result.slice(7);
  const lines = result.split("\n").length;
  return `${lines} line${lines === 1 ? "" : "s"}`;
}
