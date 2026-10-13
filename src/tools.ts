// The agent's tools: what the model may do in the folder you started Squid in. It can look (list, read,
// search), change files (edit, write; step 10) and run programs (step 11), and every change and every
// command that isn't read-only is shown to you first.
// Every path must stay inside that folder (step 08) and pass the rulebook (step 09).
import { existsSync, lstatSync, mkdirSync, readdirSync, readFileSync, realpathSync, renameSync, statSync, writeFileSync } from "node:fs";
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path";
import { decide, loadRulebook, type Rulebook } from "./rules.ts";
import { renderDiff } from "./diff.ts";
import { CommandError, describeRun, insideFolder, isReadOnly, isShell, pathArguments, removesCriticalPath, resolveCommand, runCommand, splitCommand, type Command } from "./shell.ts";

export type ToolCall = { id: string; name: string; arguments: string }; // arguments: JSON text, as the model wrote it

// The tools as the model sees them: a name, what it's for, and its inputs as a JSON Schema. The server
// passes this list into the model's chat template, so the descriptions are written for the model.
const tool = (name: string, description: string, properties: Record<string, object>, required: string[] = []) =>
  ({ type: "function", function: { name, description, parameters: { type: "object", properties, required } } });
const PATH = { type: "string", description: 'Relative to the project folder, e.g. "src" or "src/cli.ts". Leave out for the project folder itself.' };

export const TOOLS = [
  tool("list_files", "List the files and folders in a folder of the project. Folder names end with /.", { path: PATH }),
  tool("read_file", "Read a text file from the project. Lines come numbered. Long files come in parts; use offset for the next part.",
    { path: PATH, offset: { type: "integer", description: "First line to read (1 = the start)" }, limit: { type: "integer", description: "How many lines (default 400)" } },
    ["path"]),
  tool("search", "Find the lines that contain some text, in every text file under a folder. Plain text, not a pattern; upper and lower case match each other.",
    { text: { type: "string", description: "The text to look for" }, path: PATH },
    ["text"]),
  tool("edit_file", "Change part of a text file: old_text is replaced by new_text. old_text must match the file exactly, spaces and indentation included (copy it without the line numbers), and appear only once unless replace_all is true. Read the file first. The user sees the change and approves or declines it.",
    { path: PATH, old_text: { type: "string" }, new_text: { type: "string" }, replace_all: { type: "boolean", description: "Replace every occurrence (default false)" } },
    ["path", "old_text", "new_text"]),
  tool("write_file", "Create a new text file, or replace a whole file (read an existing file first). The user sees the change and approves or declines it.",
    { path: PATH, content: { type: "string", description: "The whole new content of the file" } },
    ["path", "content"]),
  tool("run_command", "Run one program in the project folder and get its output, e.g. `git status`, `npm test`, `node --version`. There is no shell: &&, ||, ;, |, >, < and $VAR don't work, so run commands one at a time. Read-only commands (git status, diff, log, show; version checks) run straight away; anything else is shown to the user to approve.",
    { command: { type: "string" }, timeout_seconds: { type: "integer", description: "Stop it after this many seconds (default 120, at most 600)" } },
    ["command"]),
];

// A problem with a request (wrong path, not allowed, declined). It goes back to the model as the tool's
// result, so it can correct itself, instead of stopping the conversation.
export class ToolError extends Error {}

// What you're asked to approve, and your answer: yes, or no with optional instructions for the model.
export type Approval = { title: string; reason: string; preview: string };
export type Answer = { ok: boolean; feedback?: string };
export type Approver = (request: Approval) => Promise<Answer>;
// With nobody at the keyboard (input piped in, a script), anything that needs approval is refused.
export const nobodyToAsk: Approver = async () => ({ ok: false, feedback: "nobody was at the keyboard to approve it" });

// One working session: the folder, its rules, who approves, and which files the model has read, with
// each file's modification time at that moment. Edits need a fresh read, so the model can't change a
// file it hasn't looked at, or overwrite changes it hasn't seen.
export type Session = {
  root: string;
  book: Rulebook;
  approve: Approver;
  seen: Map<string, number>;
  show: (text: string) => void;
  signal?: AbortSignal; // Ctrl+C: stops a running command too (step 11)
};
export function newSession(root: string, approve: Approver = nobodyToAsk, show: (text: string) => void = () => {}): Session {
  return { root, book: loadRulebook(root), approve, seen: new Map(), show };
}

// Runs one tool call and returns what the model gets to see, always as text.
export async function runTool(session: Session, call: Pick<ToolCall, "name" | "arguments">): Promise<string> {
  let args;
  try {
    args = JSON.parse(call.arguments || "{}");
  } catch {
    return `Error: the arguments weren't valid JSON: ${call.arguments}`;
  }
  const text = (value: unknown) => (value === undefined || value === null ? "" : String(value));
  try {
    const result =
      call.name === "list_files" ? await listFiles(session, args.path ?? ".")
      : call.name === "read_file" ? await readFile(session, text(args.path), args.offset ?? undefined, args.limit ?? undefined) // null → default
      : call.name === "search" ? await search(session, text(args.text), args.path ?? ".")
      : call.name === "edit_file" ? await editFile(session, text(args.path), text(args.old_text), text(args.new_text), args.replace_all === true)
      : call.name === "write_file" ? await writeFile(session, text(args.path), text(args.content))
      : call.name === "run_command" ? await runCommandTool(session, text(args.command), args.timeout_seconds ?? undefined)
      : `Error: there's no tool called ${call.name}. The tools are ${TOOLS.map((t) => t.function.name).join(", ")}.`;
    // Whatever a tool returns is added to the conversation, and the context window is finite (step 07).
    return result.length > 24_000 ? `${result.slice(0, 24_000)}\n… (cut: ${result.length - 24_000} more characters)` : result;
  } catch (err) {
    if (err instanceof ToolError) return `Error: ${err.message}`;
    throw err;
  }
}

// --- the fence (step 08) ---

const isInside = (base: string, real: string) => {
  const rel = relative(base, real); // "src\cli.ts" inside; "..\x" or "D:\x" outside
  return !(rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel));
};

// The real location of an existing path the model asked for, if it's inside the project. Links (symbolic
// links, Windows junctions) are followed *before* the check, because a link inside the folder can point
// anywhere: C:\project\docs could lead to C:\Users\Jorda\.ssh.
export function inside(root: string, requested: string): string {
  const base = realpathSync.native(root);
  let real;
  try {
    real = realpathSync.native(resolve(base, requested)); // resolve() handles "..", "C:\…" and "/…"
  } catch {
    throw new ToolError(`${requested} doesn't exist`);
  }
  if (!isInside(base, real)) throw new ToolError(`${requested} is outside the project folder, so it's off limits`);
  return real;
}

// Where a write would land. The file may not exist yet, so the check uses the nearest folder that does
// exist: its real location (links followed) must be inside the project. And the file itself mustn't be
// a link: writing to it would change whatever it points at (Claude Code refuses the same).
function writeTarget(root: string, requested: string): string {
  const base = realpathSync.native(root);
  const full = resolve(base, requested);
  if (existsSync(full) && lstatSync(full).isSymbolicLink()) throw new ToolError(`${requested} is a link; change the file it points to instead`);
  let dir = dirname(full);
  while (!existsSync(dir)) dir = dirname(dir);
  const real = join(realpathSync.native(dir), relative(dir, full));
  if (!isInside(base, real)) throw new ToolError(`${requested} is outside the project folder, so it's off limits`);
  return real;
}

// --- permission (steps 09 and 10) ---

// Asks the rulebook; on "ask", asks you. Unless the answer is yes, throws, so the model hears why.
// An edit that a rule allows still has its change shown, just without the question.
async function permit(session: Session, tool: "Read" | "Edit", requested: string, real: string, title: string, preview = ""): Promise<void> {
  const verdict = decide({ tool, path: resolve(realpathSync.native(session.root), requested), real }, session.book);
  if (verdict.outcome === "deny") throw new ToolError(`${requested}: ${verdict.reason}`);
  if (verdict.outcome === "allow") {
    if (preview) session.show(`${title} (${verdict.reason})\n${preview}\n`);
    return;
  }
  const answer = await session.approve({ title, reason: verdict.reason, preview });
  if (!answer.ok) throw new ToolError(`the user declined: ${title}${answer.feedback ? `. Their instructions: ${answer.feedback}` : ""}`);
}
const readable = (path: string, book: Rulebook) => decide({ tool: "Read", path, real: path }, book).outcome === "allow";

// --- looking ---

async function listFiles(session: Session, path: string): Promise<string> {
  const dir = inside(session.root, path);
  await permit(session, "Read", path, dir, `List ${path}`);
  if (!statSync(dir).isDirectory()) throw new ToolError(`${path} is a file, not a folder; use read_file`);
  const entries = readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  const names = entries.slice(0, 500).map((e) => {
    const name = e.isSymbolicLink() ? `${e.name} (link)` : e.isDirectory() ? `${e.name}/` : e.name;
    return readable(join(dir, e.name), session.book) ? name : `${name} (off limits)`; // so the model doesn't try
  });
  if (entries.length > 500) names.push(`… and ${entries.length - 500} more`);
  return names.join("\n") || "(empty folder)";
}

async function readFile(session: Session, path: string, offset = 1, limit = 400): Promise<string> {
  const file = inside(session.root, path);
  await permit(session, "Read", path, file, `Read ${path}`);
  const info = statSync(file);
  if (info.isDirectory()) throw new ToolError(`${path} is a folder; use list_files`);
  if (info.size > 2_000_000) throw new ToolError(`${path} is ${(info.size / 1e6).toFixed(1)} MB, too big to read; use search`);
  const bytes = readFileSync(file);
  if (bytes.subarray(0, 8000).includes(0)) throw new ToolError(`${path} isn't a text file`); // a zero byte means binary
  session.seen.set(file, info.mtimeMs); // the model has now seen this version of the file
  const lines = bytes.toString("utf8").split(/\r?\n/);
  const start = Math.max(1, Math.floor(offset));
  const shown = lines.slice(start - 1, start - 1 + Math.min(Math.max(1, Math.floor(limit)), 2000));
  // Numbered like `cat -n`, so the model can point at lines.
  const numbered = shown.map((line, i) => `${String(start + i).padStart(5)}\t${line.length > 2000 ? `${line.slice(0, 2000)}…` : line}`);
  const end = start + shown.length - 1;
  if (end < lines.length) numbered.push(`(lines ${start}-${end} of ${lines.length}; read on with offset ${end + 1})`);
  return numbered.join("\n") || "(empty file)";
}

// Plain text, not a regular expression, on purpose: JavaScript's regex engine can take hours on some
// patterns ("catastrophic backtracking"), and the pattern would come from the model.
const SKIP = new Set([".git", "node_modules"]);
async function search(session: Session, text: string, path: string): Promise<string> {
  if (!text) throw new ToolError("text is empty");
  const base = realpathSync.native(session.root);
  const start = inside(session.root, path);
  await permit(session, "Read", path, start, `Search ${path}`);
  const needle = text.toLowerCase();
  const hits: string[] = [];
  const searchFile = (file: string) => {
    let bytes;
    try {
      if (statSync(file).size > 1_000_000) return;
      bytes = readFileSync(file);
    } catch {
      return; // unreadable (locked, no permission): skip it
    }
    if (bytes.subarray(0, 8000).includes(0)) return; // binary
    bytes.toString("utf8").split(/\r?\n/).forEach((line, i) => {
      if (hits.length < 100 && line.toLowerCase().includes(needle)) hits.push(`${relative(base, file)}:${i + 1}: ${line.trim().slice(0, 200)}`);
    });
  };
  const visit = (dir: string) => {
    let entries;
    try {
      entries = readdirSync(dir, { withFileTypes: true });
    } catch {
      return; // a folder we may not open (the AI Lab's locked holdout folder broke a recursive listing once)
    }
    for (const entry of entries) {
      if (hits.length >= 100) return;
      // Links are never followed while walking: one could lead out of the project.
      if (entry.isSymbolicLink() || SKIP.has(entry.name)) continue;
      const full = join(dir, entry.name);
      if (!readable(full, session.book)) continue; // off limits by the rulebook: skipped without a word
      if (entry.isDirectory()) visit(full);
      else if (entry.isFile()) searchFile(full);
    }
  };
  if (statSync(start).isDirectory()) visit(start);
  else searchFile(start);
  if (!hits.length) return `No lines contain "${text}".`;
  return hits.join("\n") + (hits.length >= 100 ? "\n(stopped at 100 matches; search a smaller folder or longer text)" : "");
}

// --- changing (step 10) ---

// An edit needs a fresh read: the model must have read this file in this session, and nobody (you, an
// editor, another program) may have changed it since. Otherwise it would edit from a stale picture.
function mustHaveRead(session: Session, file: string, path: string): void {
  const readAt = session.seen.get(file);
  if (readAt === undefined) throw new ToolError(`read ${path} with read_file before changing it`);
  if (statSync(file).mtimeMs !== readAt) throw new ToolError(`${path} has changed since you read it; read it again first`);
}

async function editFile(session: Session, path: string, oldText: string, newText: string, replaceAll: boolean): Promise<string> {
  const file = inside(session.root, path);
  if (statSync(file).isDirectory()) throw new ToolError(`${path} is a folder`);
  mustHaveRead(session, file, path);
  const before = readFileSync(file, "utf8");
  // Files written on Windows often end their lines with \r\n, while the model writes \n. Match the file.
  const eol = before.includes("\r\n") ? "\r\n" : "\n";
  const [find, put] = [oldText, newText].map((t) => t.replace(/\r?\n/g, eol));
  if (!find) throw new ToolError("old_text is empty; to create a file use write_file");
  const count = before.split(find).length - 1;
  if (count === 0) throw new ToolError(`old_text wasn't found in ${path}; it must match the file exactly, spaces and indentation included`);
  if (count > 1 && !replaceAll) throw new ToolError(`old_text appears ${count} times in ${path}; include more of the surrounding lines so it appears once, or set replace_all`);
  // split/join for every occurrence; for one, replace() with a function, because a plain string
  // replacement treats "$&" and "$1" in the new text as special codes and would mangle them.
  const after = replaceAll ? before.split(find).join(put) : before.replace(find, () => put);
  await permit(session, "Edit", path, file, `Edit ${path}`, preview(before, after));
  writeAtomically(file, after);
  session.seen.set(file, statSync(file).mtimeMs); // it may edit again without re-reading its own change
  return `Edited ${path}: ${replaceAll ? count : 1} replacement${replaceAll && count > 1 ? "s" : ""}.`;
}

async function writeFile(session: Session, path: string, content: string): Promise<string> {
  const file = writeTarget(session.root, path);
  const exists = existsSync(file);
  if (exists && statSync(file).isDirectory()) throw new ToolError(`${path} is a folder`);
  if (exists) mustHaveRead(session, file, path);
  const before = exists ? readFileSync(file, "utf8") : "";
  const after = exists && before.includes("\r\n") ? content.replace(/\r?\n/g, "\r\n") : content;
  await permit(session, "Edit", path, file, `${exists ? "Overwrite" : "Create"} ${path}`, preview(before, after));
  mkdirSync(dirname(file), { recursive: true });
  writeAtomically(file, after);
  session.seen.set(file, statSync(file).mtimeMs);
  const lines = after.split(/\r?\n/).length;
  return `${exists ? "Rewrote" : "Created"} ${path} (${lines} line${lines === 1 ? "" : "s"}).`;
}

// --- running commands (step 11) ---

async function runCommandTool(session: Session, line: string, timeout = 120): Promise<string> {
  let command: Command;
  try {
    command = resolveCommand(splitCommand(line), session.root);
  } catch (err) {
    if (err instanceof CommandError) throw new ToolError(err.message);
    throw err;
  }
  const root = realpathSync.native(session.root);
  const critical = removesCriticalPath(command, root);
  if (critical) throw new ToolError(`${command.text}: removing ${critical} is never allowed`);
  // A path in the arguments counts as reading that path: a secret is refused, as if read_file asked.
  const paths = pathArguments(command, root);
  for (const path of paths.local) {
    const verdict = decide({ tool: "Read", path, real: path }, session.book);
    if (verdict.outcome === "deny") throw new ToolError(`${command.text}: ${verdict.reason}`);
  }
  const outside = paths.local.filter((path) => !insideFolder(root, path));
  const readOnly = isReadOnly(command) && !outside.length && !paths.network.length;
  const verdict = decide({ tool: "Bash", command: command.text, readOnly }, session.book);
  if (verdict.outcome === "deny") throw new ToolError(`${command.text}: ${verdict.reason}`);
  // A shell can run anything at all, so it always asks, whatever an allow rule says.
  if (verdict.outcome === "ask" || isShell(command)) {
    const reason = isShell(command) ? `${command.name} is a shell: it can run anything, so it always needs your OK`
      : paths.network.length ? `${paths.network[0]} is a network path: opening it can send your Windows login to that server`
      : outside.length ? `it mentions ${outside[0]}, outside the project`
      : verdict.reason;
    const answer = await session.approve({ title: `Run ${command.text}`, reason, preview: `in ${root}\n> ${command.text}\n(program: ${command.program})` });
    if (!answer.ok) throw new ToolError(`the user declined: Run ${command.text}${answer.feedback ? `. Their instructions: ${answer.feedback}` : ""}`);
  }
  const seconds = Math.min(Math.max(1, Math.floor(timeout)), 600);
  const ran = await runCommand(command, root, seconds, session.signal);
  // A command that isn't read-only may have changed files (a formatter, a code generator): the model's
  // picture of every file is now possibly stale, so edits need a fresh read again (step 10).
  if (!readOnly) session.seen.clear();
  return describeRun(ran, seconds);
}

// The change as a coloured diff, at most 80 lines of it.
function preview(before: string, after: string): string {
  const lines = renderDiff(before, after, { color: process.stderr.isTTY }).split("\n");
  return lines.length > 80 ? [...lines.slice(0, 80), `   … ${lines.length - 80} more lines`].join("\n") : lines.join("\n");
}

// Writes to a temporary file beside the real one, then renames it into place. A rename is all or
// nothing, so the file is never left half-written, even if Squid crashes in the middle.
function writeAtomically(file: string, text: string): void {
  const temp = `${file}.squid-${process.pid}.tmp`;
  writeFileSync(temp, text);
  renameSync(temp, file);
}

// How a call looks on screen: read_file(src/cli.ts), search("ensureServer" in src).
export function describeCall(call: Pick<ToolCall, "name" | "arguments">): string {
  let args: any = {};
  try {
    args = JSON.parse(call.arguments || "{}");
  } catch {
    // shown as-is below
  }
  if (call.name === "search") return `search("${args.text ?? ""}"${args.path ? ` in ${args.path}` : ""})`;
  if (["list_files", "read_file", "edit_file", "write_file"].includes(call.name)) return `${call.name}(${args.path ?? "."})`;
  if (call.name === "run_command") return `run_command(${args.command ?? ""})`;
  return `${call.name}(${call.arguments})`;
}
