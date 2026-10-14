// Compaction (step 19): the model's memory, its context, is a fixed size (49,152 tokens for the 27B
// here), and every round of a long task adds to it. When it gets too full, Squid makes room before the
// next request rather than letting an answer run into the wall (step 18 tells that story). Claude Code
// does the same: it compacts automatically, and /compact does it on request.
//
// Two stages, cheapest first:
//   1. Trim: no model needed, done in a moment. What gets cut can be got back: a file the agent wrote is
//      on disk (a stub says so), an old tool result can be read again, and old thinking isn't needed.
//   2. Summarise: if trimming wasn't enough, the model writes a summary of the conversation (thinking
//      off, so it's quick) and that replaces everything before the latest round.
import { contextLimit, dim, streamReply, type Message, type Output } from "./reply.ts";
import type { Session } from "./tools.ts";

export const COMPACT_AT = 0.65; // compact when the memory is this full (Squid's default; config: compactAt)
const ROOMY = 0.45; // after trimming, this full or less is room enough; otherwise summarise too
const LONG_RESULT = 1_500; // characters: older tool results longer than this are cut down
const LONG_TEXT = 400; // characters: file contents in older write_file / edit_file calls longer than this become stubs

// Everything in the messages that the model reads, in characters.
export const charsOf = (messages: Message[]) =>
  messages.reduce((n, m) => n + m.content.length + (m.reasoning_content?.length ?? 0) + (m.tool_calls ?? []).reduce((k, c) => k + c.function.arguments.length, 0), 0);

// How many tokens the conversation takes up now. The last reply's usage is exact for everything up to
// it; whatever was added since (tool results, your message) is guessed at 3 characters per token, a little
// more than Qwen's tokenizer really uses, so the guess errs on the side of compacting early.
export function estimate(messages: Message[], session: Session): number {
  const chars = charsOf(messages);
  if (!session.measured) return Math.ceil(chars / 3);
  return Math.max(Math.ceil(chars / 4), Math.ceil(session.measured.tokens + (chars - session.measured.chars) / 3));
}

// Called by the agent loop after each reply, with the conversation as it now stands.
export function measured(messages: Message[], session: Session, totalTokens: number | undefined): void {
  if (totalTokens) session.measured = { tokens: totalTokens, chars: charsOf(messages) };
}

// The model's memory size, asked once per model (switching models with /model asks again).
export async function limitFor(base: string, model: string | undefined, session: Session): Promise<number | undefined> {
  if (!model) return undefined;
  if (session.limit?.model !== model) session.limit = { model, tokens: await contextLimit(base, model).catch(() => undefined) };
  return session.limit.tokens;
}

// Called before every round. Compacts if the memory is fuller than `compactAt`.
export async function compactIfFull(base: string, model: string | undefined, messages: Message[], session: Session, out: Output, signal?: AbortSignal): Promise<void> {
  const limit = await limitFor(base, model, session);
  if (!limit || estimate(messages, session) <= limit * session.compactAt) return;
  // What compaction must keep (the system message, your message and the latest round) can be big on its
  // own: one large file just read. If there's little else, compacting frees nothing, so don't.
  const latest = latestRound(messages);
  const request = messages.findLast((m) => m.role === "user");
  const removable = messages.slice(0, latest).filter((m) => m.role !== "system" && m !== request);
  if (charsOf(removable) / 3 < limit * 0.1) return;
  try {
    await compact(base, model, messages, session, out, { inTurn: true, signal });
  } catch (err) {
    if (signal?.aborted) throw err;
    // The trim has already happened; the summary didn't. The next request may well fit anyway.
    out.note(dim(`(couldn't summarise the conversation: ${(err as Error).message} Carrying on.)\n`));
  }
}

export type Compacted = { before: number; after: number; summarised: boolean };

// Makes room. `inTurn`: called in the middle of answering your message, which must survive word for word
// (it's the task), along with the latest round, whose results the model is about to read. Otherwise
// (/compact between turns) everything can go into the summary. `force` summarises even if trimming was
// enough; `note` is added to the summary for the model to read.
export async function compact(
  base: string,
  model: string | undefined,
  messages: Message[],
  session: Session,
  out: Output,
  options: { inTurn?: boolean; force?: boolean; focus?: string; note?: string; signal?: AbortSignal } = {},
): Promise<Compacted> {
  const limit = await limitFor(base, model, session);
  const before = estimate(messages, session);
  const latest = latestRound(messages);
  trim(messages, latest);
  session.seen.clear(); // the model no longer has those files' text in front of it: it must read before it edits
  session.measured = undefined; // the next reply measures again
  let after = estimate(messages, session);
  const full = limit ? ` ${Math.round((100 * before) / limit)}% full` : "";
  if (after < before) out.note(dim(`(memory${full}: trimmed old file contents, tool results and thinking: ~${k(before)} → ~${k(after)} tokens)\n`));
  if (!options.force && limit && after <= limit * ROOMY) return { before, after, summarised: false };

  out.note(dim("(summarising the conversation to make more room; this takes a moment…)\n"));
  const summary = await summarise(base, model, messages, options.focus, options.signal);
  const request = options.inTurn ? messages.findLast((m) => m.role === "user") : undefined;
  const kept: Message[] = [
    ...(messages[0]?.role === "system" ? [messages[0]] : []), // who Squid is, the folder, the rules
    { role: "user", content: `${SUMMARY_HEADER}\n\n${summary}${options.note ? `\n\n${options.note}` : ""}` },
    ...(request ? [request] : []),
    ...(options.inTurn ? messages.slice(latest) : []),
  ];
  messages.splice(0, messages.length, ...kept); // the same array, changed in place: the chat and the page hold it
  after = estimate(messages, session);
  out.note(dim(`(conversation compacted into a summary: ~${k(before)} → ~${k(after)} tokens)\n`));
  return { before, after, summarised: true };
}

// The summary goes in front of everything else as a message of its own. It starts like Squid's other
// notes, so withoutNote() leaves just the summary, and the page can tell it apart (isSummary).
export const SUMMARY_HEADER = "(Squid: the conversation so far was compacted to free memory. This is your own summary of it.)";
export const isSummary = (m: Message) => m.role === "user" && m.content.startsWith(SUMMARY_HEADER);

// Where the latest round starts: the last reply with tool calls, if only its results follow it. If the
// conversation ends any other way (your message, an answer), there's no round in progress.
export function latestRound(messages: Message[]): number {
  let i = messages.length;
  while (i > 0 && messages[i - 1].role === "tool") i--;
  return i < messages.length && messages[i - 1]?.tool_calls?.length ? i - 1 : messages.length;
}

// Stage 1. Changes the messages before `upTo` in place. Your own messages and the model's answers are
// never cut; tool call ids stay paired with their results, and every call's arguments stay valid JSON.
export function trim(messages: Message[], upTo: number): void {
  for (const m of messages.slice(0, upTo)) {
    if (m.role === "assistant") {
      delete m.reasoning_content;
      for (const call of m.tool_calls ?? []) call.function.arguments = stubbed(call.function.name, call.function.arguments);
    }
    if (m.role === "tool" && m.content.length > LONG_RESULT) {
      m.content = `${m.content.slice(0, 600)}\n… [Squid: the other ${m.content.length - 600} characters of this result were removed from the conversation to save memory. Run the tool again if you need them.]`;
    }
  }
}

// A write_file or edit_file call with its file text replaced by a note. The path stays, so the
// conversation still says which file was changed. The note has to be unmistakable: tried on the 9B, a
// plain "[12 lines, removed to free memory]" was taken for the text it had written, and it planned to
// "fix" the file. So the note names the file, says it's from Squid, and the system message explains.
function stubbed(name: string, json: string): string {
  if (name !== "write_file" && name !== "edit_file") return json;
  let args: Record<string, unknown>;
  try {
    args = JSON.parse(json);
  } catch {
    return json;
  }
  let changed = false;
  for (const field of ["content", "old_text", "new_text"]) {
    const text = args[field];
    if (typeof text === "string" && text.length > LONG_TEXT) {
      const lines = text.split("\n").length;
      args[field] = `[Squid: ${lines} lines of ${String(args.path)} went here. They were removed from the conversation to save memory; the ${name === "write_file" ? "file was written" : "edit was made"} and is unchanged on disk. read_file shows it.]`;
      changed = true;
    }
  }
  return changed ? JSON.stringify(args) : json;
}

// For the system message: what those notes mean.
export const NOTE_EXPLAINED =
  'Text in square brackets that starts "[Squid:" is a note from Squid, not file content or tool output. To save memory, Squid removes old file text and long tool results from the conversation; the files on disk are unchanged.';

// Stage 2. One request with no tools, thinking switched off (the 27B's chat template takes enable_thinking)
// and a cap on the reply's length, so the summary comes back in about a minute.
async function summarise(base: string, model: string | undefined, messages: Message[], focus?: string, signal?: AbortSignal): Promise<string> {
  const quiet: Output = { answer: () => {}, thinking: () => {}, note: () => {} };
  const asked: Message[] = [...messages.map(({ reasoning_content, ...m }) => m), { role: "user", content: summaryPrompt(focus) }];
  const reply = await streamReply(base, { model, messages: asked, max_tokens: 2048, chat_template_kwargs: { enable_thinking: false } }, signal, quiet);
  if (!reply.content.trim()) throw new Error("The model wrote an empty summary, so the conversation wasn't compacted.");
  return reply.content.trim();
}

export const summaryPrompt = (focus?: string) =>
  [
    "Stop the task for a moment. Your memory is nearly full, so write a summary of this whole conversation that will replace it. You'll continue from the summary alone, so write it for yourself, and be specific: names, paths, numbers.",
    "1. What the user asked for, in their own words where possible, and any instructions or preferences they gave.",
    "2. Every file created, changed or read, with one line each on what it holds or what changed.",
    "3. Decisions made and why, and anything the user declined or asked you not to do.",
    "4. Where the work stands: what's done, what's half done, what's left, and the very next step.",
    "5. Errors met, and how they were dealt with.",
    focus ? `Pay special attention to: ${focus}` : "",
    "Write the summary only, under 700 words. Don't call any tools.",
  ].filter(Boolean).join("\n");

const k = (tokens: number) => `${(tokens / 1000).toFixed(tokens < 10_000 ? 1 : 0)}K`;
