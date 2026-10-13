// The agent's tools: what the model may do in the folder you started Squid in. In this step they only
// look (list, read, search); nothing can change a file yet. Every path must stay inside that folder.
import { readdirSync, readFileSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, join, relative, resolve, sep } from "node:path";

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
];

// A problem the model caused (wrong path, not a text file). It goes back to the model as the tool's
// result, so it can correct itself, instead of stopping the conversation.
export class ToolError extends Error {}

// Runs one tool call and returns what the model gets to see, always as text.
export function runTool(root: string, call: Pick<ToolCall, "name" | "arguments">): string {
  let args;
  try {
    args = JSON.parse(call.arguments || "{}");
  } catch {
    return `Error: the arguments weren't valid JSON: ${call.arguments}`;
  }
  try {
    const result =
      call.name === "list_files" ? listFiles(root, args.path ?? ".")
      : call.name === "read_file" ? readFile(root, String(args.path ?? ""), args.offset ?? undefined, args.limit ?? undefined) // null → default
      : call.name === "search" ? search(root, String(args.text ?? ""), args.path ?? ".")
      : `Error: there's no tool called ${call.name}. The tools are list_files, read_file and search.`;
    // Whatever a tool returns is added to the conversation, and the context window is finite (step 07).
    return result.length > 24_000 ? `${result.slice(0, 24_000)}\n… (cut: ${result.length - 24_000} more characters)` : result;
  } catch (err) {
    if (err instanceof ToolError) return `Error: ${err.message}`;
    throw err;
  }
}

// The real location of a path the model asked for, if it's inside the project; otherwise an error.
// Links (symbolic links, Windows junctions) are followed *before* the check, because a link inside the
// folder can point anywhere: C:\project\docs could lead to C:\Users\Jorda\.ssh.
export function inside(root: string, requested: string): string {
  const base = realpathSync.native(root);
  let real;
  try {
    real = realpathSync.native(resolve(base, requested)); // resolve() handles "..", "C:\…" and "/…"
  } catch {
    throw new ToolError(`${requested} doesn't exist`);
  }
  const rel = relative(base, real); // "src\cli.ts" inside; "..\x" or "D:\x" outside
  if (rel === ".." || rel.startsWith(`..${sep}`) || isAbsolute(rel)) throw new ToolError(`${requested} is outside the project folder, so it's off limits`);
  return real;
}

function listFiles(root: string, path: string): string {
  const dir = inside(root, path);
  if (!statSync(dir).isDirectory()) throw new ToolError(`${path} is a file, not a folder; use read_file`);
  const entries = readdirSync(dir, { withFileTypes: true }).sort((a, b) => a.name.localeCompare(b.name));
  const names = entries.slice(0, 500).map((e) => (e.isSymbolicLink() ? `${e.name} (link)` : e.isDirectory() ? `${e.name}/` : e.name));
  if (entries.length > 500) names.push(`… and ${entries.length - 500} more`);
  return names.join("\n") || "(empty folder)";
}

function readFile(root: string, path: string, offset = 1, limit = 400): string {
  const file = inside(root, path);
  const info = statSync(file);
  if (info.isDirectory()) throw new ToolError(`${path} is a folder; use list_files`);
  if (info.size > 2_000_000) throw new ToolError(`${path} is ${(info.size / 1e6).toFixed(1)} MB, too big to read; use search`);
  const bytes = readFileSync(file);
  if (bytes.subarray(0, 8000).includes(0)) throw new ToolError(`${path} isn't a text file`); // a zero byte means binary
  const lines = bytes.toString("utf8").split(/\r?\n/);
  const start = Math.max(1, Math.floor(offset));
  const shown = lines.slice(start - 1, start - 1 + Math.min(Math.max(1, Math.floor(limit)), 2000));
  // Numbered like `cat -n`, so the model can point at lines (and, from step 10, edit them).
  const numbered = shown.map((line, i) => `${String(start + i).padStart(5)}\t${line.length > 2000 ? `${line.slice(0, 2000)}…` : line}`);
  const end = start + shown.length - 1;
  if (end < lines.length) numbered.push(`(lines ${start}-${end} of ${lines.length}; read on with offset ${end + 1})`);
  return numbered.join("\n") || "(empty file)";
}

// Plain text, not a regular expression, on purpose: JavaScript's regex engine can take hours on some
// patterns ("catastrophic backtracking"), and the pattern would come from the model.
const SKIP = new Set([".git", "node_modules"]);
function search(root: string, text: string, path: string): string {
  if (!text) throw new ToolError("text is empty");
  const base = realpathSync.native(root);
  const start = inside(root, path);
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
      if (entry.isDirectory()) visit(full);
      else if (entry.isFile()) searchFile(full);
    }
  };
  if (statSync(start).isDirectory()) visit(start);
  else searchFile(start);
  if (!hits.length) return `No lines contain "${text}".`;
  return hits.join("\n") + (hits.length >= 100 ? "\n(stopped at 100 matches; search a smaller folder or longer text)" : "");
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
  if (call.name === "list_files" || call.name === "read_file") return `${call.name}(${args.path ?? "."})`;
  return `${call.name}(${call.arguments})`;
}
