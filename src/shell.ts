// Running commands without a shell. A shell (cmd, PowerShell, bash) reads a command line and acts on
// &&, |, >, $VAR and more, so "git status && del /s *" is two commands, and a rule that allows the first
// can be tricked into running the second. Squid never starts a shell: it splits the line into a program
// and its arguments and starts that one program directly. Shell syntax then has no meaning, and Squid
// refuses it outright rather than pass "&&" to a program as a plain word.
import { spawn, execFileSync } from "node:child_process";
import { existsSync, realpathSync, statSync } from "node:fs";
import { basename, delimiter, dirname, extname, join, parse, resolve, sep } from "node:path";
import { homedir } from "node:os";

export type Command = {
  program: string; // the file that will run, e.g. C:\Program Files\Git\cmd\git.exe
  args: string[];
  name: string; // what rules match: the program's own name, lowercase, without .exe ("git")
  text: string; // "git log --oneline": the name plus arguments, as rules see it
};
export class CommandError extends Error {}

// --- 1. splitting the line ---

// Words separated by spaces; "double" or 'single' quotes keep spaces (and anything else) inside one word.
// Backslashes are ordinary characters: on Windows they're how paths are written.
export function splitCommand(line: string): string[] {
  if (line.length > 10_000) throw new CommandError("the command is longer than 10,000 characters");
  const words: string[] = [];
  let word = "";
  let quote: string | null = null;
  let inWord = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (quote) {
      if (c === quote) quote = null;
      else word += c;
    } else if (c === '"' || c === "'") {
      quote = c;
      inWord = true;
    } else if (c === " " || c === "\t") {
      if (inWord) words.push(word);
      word = "";
      inWord = false;
    } else {
      // Outside quotes, these mean something only to a shell: chaining, pipes, redirects, substitution.
      if ("&|;<>`\n\r".includes(c) || (c === "$" && line[i + 1] === "(")) {
        throw new CommandError(`"${c === "\n" ? "a new line" : c}" is shell syntax, and Squid runs one program without a shell: no &&, ||, ;, |, <, > or $( ). Run each command on its own, and use the file tools instead of redirects`);
      }
      word += c;
      inWord = true;
    }
  }
  if (quote) throw new CommandError(`a ${quote} quote is never closed`);
  if (inWord) words.push(word);
  if (!words.length) throw new CommandError("the command is empty");
  return words;
}

// --- 2. finding the program ---

// Programs that run whatever you hand them. Starting one is starting a shell again, so they always need
// your OK, and no allow rule can skip that (as Claude Code treats exec wrappers like `watch`).
const SHELLS = new Set(["cmd", "powershell", "pwsh", "bash", "sh", "zsh", "wsl", "wscript", "cscript", "mshta", "rundll32", "start"]);
export const isShell = (command: Command) => SHELLS.has(command.name);

// npm and npx are batch files on Windows (npm.cmd), which can only run through cmd.exe. They're really
// Node scripts, so Squid runs those scripts with Node directly.
function nodeScript(name: string): string | undefined {
  if (name !== "npm" && name !== "npx") return undefined;
  const script = join(dirname(process.execPath), "node_modules", "npm", "bin", `${name}-cli.js`);
  return existsSync(script) ? script : undefined;
}

// Turns the typed words into the program that will really run. Looks through PATH the way Windows does,
// but runs only real programs (.exe, .com).
export function resolveCommand(words: string[], root: string): Command {
  const [typed, ...rest] = words;
  const script = nodeScript(typed.toLowerCase());
  let program: string;
  let args = rest;
  if (script) {
    program = process.execPath;
    args = [script, ...rest];
  } else {
    program = findProgram(typed, root);
    const extension = extname(program).toLowerCase();
    if (extension === ".bat" || extension === ".cmd") {
      // Windows can only run a batch file by handing the whole line to cmd.exe, which reads the arguments
      // again with its own rules: a crafted argument can then run another command ("BatBadBut",
      // CVE-2024-27980). Node refuses to do it without a shell since 2024, and so does Squid.
      throw new CommandError(`${basename(program)} is a batch file, which Windows can only run through cmd.exe; Squid doesn't start shells`);
    }
    if (extension !== ".exe" && extension !== ".com") throw new CommandError(`${basename(program)} isn't a program Squid can run (only .exe and .com)`);
  }
  const name = (script ? typed : parse(program).name).toLowerCase();
  return { program, args, name, text: [name, ...rest].map(quoteIfNeeded).join(" ") };
}

function findProgram(typed: string, root: string): string {
  const extensions = (process.env.PATHEXT ?? ".COM;.EXE;.BAT;.CMD").split(";").filter(Boolean);
  const candidates = (base: string) => (extname(base) ? [base] : extensions.map((ext) => base + ext.toLowerCase()));
  if (typed.includes("/") || typed.includes("\\")) {
    // A path: relative ones start from the project folder.
    for (const file of candidates(resolve(root, typed))) if (isFile(file)) return file;
    throw new CommandError(`${typed} wasn't found`);
  }
  // Only PATH, never the current folder first. cmd.exe does look there first, so a downloaded project
  // that ships its own git.exe would have it run by a plain `git status`.
  for (const dir of (process.env.PATH ?? "").split(delimiter).filter(Boolean)) {
    for (const file of candidates(join(dir, typed))) if (isFile(file)) return file;
  }
  throw new CommandError(`${typed} wasn't found on PATH`);
}
const isFile = (path: string) => {
  try {
    return statSync(path).isFile();
  } catch {
    return false;
  }
};
const quoteIfNeeded = (word: string) => (/[\s"]/.test(word) || word === "" ? `"${word.replace(/"/g, '\\"')}"` : word);

// --- 3. what's read-only ---

// Commands that only look, run without asking (Claude Code has a similar built-in set). Squid's set is
// small: Windows has no ls or cat programs, and the file tools already read, list and search.
// For git only the subcommand as the very first word counts: `git -c core.fsmonitor=<program> status`
// would run a program, so a global option in front makes it not read-only.
const READ_ONLY_GIT = new Set(["status", "diff", "log", "show", "blame", "ls-files", "rev-parse", "rev-list"]);
const GIT_UNSAFE_FLAG = /^(--output|--ext-diff|--textconv|--exec|--upload-pack|--receive-pack|-o$)/; // write files or run programs
const VERSION = new Set(["--version", "-v", "-V", "version"]);

export function isReadOnly(command: Command): boolean {
  const [first, ...more] = command.args;
  if (command.name === "git") {
    if (first === "--version" && !more.length) return true;
    return READ_ONLY_GIT.has(first) && !more.some((arg) => GIT_UNSAFE_FLAG.test(arg));
  }
  if (["node", "npm", "npx", "python", "py", "uv", "deno", "bun", "tsc"].includes(command.name)) {
    return command.args.length === (command.name === "npm" || command.name === "npx" ? 2 : 1) && VERSION.has(command.args.at(-1)!); // npm's args start with its script
  }
  return command.name === "where" && command.args.length > 0 && command.args.every((arg) => !arg.startsWith("/"));
}

// --- 4. arguments that are paths ---

// Every argument that names an existing file or folder, as its real path (links followed, as in step 08). A read-only command that
// mentions a path outside the project isn't read-only *here*: `git diff --no-index C:\Users\…` reads
// anything. Network paths (\\server\share) can send your Windows login to that server, so they're
// listed separately and always need your OK.
export function pathArguments(command: Command, root: string): { local: string[]; network: string[] } {
  const local: string[] = [];
  const network: string[] = [];
  for (const arg of command.args) {
    const value = arg.includes("=") && arg.startsWith("-") ? arg.slice(arg.indexOf("=") + 1) : arg; // --file=x
    if (/^(\\\\|\/\/)/.test(value)) network.push(value);
    else if (value && existsSync(resolve(root, value))) local.push(realpathSync.native(resolve(root, value)));
  }
  return { local, network };
}

// Folders whose removal would be a disaster: drive roots and the folders right under them (C:\Windows),
// your home folder, and the project folder and every folder above it. Claude Code never auto-approves
// rm on these; Squid refuses them outright.
const REMOVERS = new Set(["rm", "rmdir", "del", "erase", "rd", "remove-item"]);
export function removesCriticalPath(command: Command, root: string): string | undefined {
  if (!REMOVERS.has(command.name)) return undefined;
  const critical = new Set<string>([normal(homedir())]);
  for (let dir = resolve(root); ; dir = dirname(dir)) {
    critical.add(normal(dir));
    if (dirname(dir) === dir) break; // reached the drive root
  }
  for (const arg of command.args) {
    if (arg.startsWith("-") || arg.startsWith("/")) continue; // options
    const target = normal(resolve(root, arg));
    const depth = target.split(sep).filter(Boolean).length; // "c:" = 1, "c:\windows" = 2
    if (critical.has(target) || depth <= 2) return target;
  }
  return undefined;
}
const normal = (path: string) => resolve(path).replace(/[\\/]+$/, "").toLowerCase();

// --- 5. running it ---

// Environment variables whose names suggest a secret are not passed on to the program.
const SECRET_NAME = /(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)/i;
export function safeEnvironment(): NodeJS.ProcessEnv {
  const env = Object.fromEntries(Object.entries(process.env).filter(([name]) => !SECRET_NAME.test(name)));
  // No pager, no colour codes, no password prompts: the program's output goes to the model, and
  // nobody can answer a prompt it waits for.
  return { ...env, GIT_PAGER: "cat", PAGER: "cat", GIT_TERMINAL_PROMPT: "0", NO_COLOR: "1", FORCE_COLOR: "0" };
}

export type Ran = { code: number | null; stdout: string; stderr: string; timedOut: boolean; cancelled: boolean };

// Starts the program (shell: false is the whole point), collects its output, and stops it after
// `seconds`. On Windows a killed program's own child processes keep running, so the whole tree is ended.
export function runCommand(command: Command, cwd: string, seconds: number, signal?: AbortSignal): Promise<Ran> {
  return new Promise((done) => {
    const child = spawn(command.program, command.args, { cwd, shell: false, windowsHide: true, env: safeEnvironment(), stdio: ["ignore", "pipe", "pipe"] });
    let stdout = "";
    let stderr = "";
    const keep = (text: string, more: Buffer) => (text.length < 200_000 ? text + more.toString("utf8") : text);
    child.stdout.on("data", (chunk) => (stdout = keep(stdout, chunk)));
    child.stderr.on("data", (chunk) => (stderr = keep(stderr, chunk)));
    let timedOut = false;
    let cancelled = false;
    const stopTree = () => {
      try {
        execFileSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true, stdio: "ignore" });
      } catch {
        child.kill();
      }
    };
    const timer = setTimeout(() => ((timedOut = true), stopTree()), seconds * 1000);
    const onAbort = () => ((cancelled = true), stopTree());
    signal?.addEventListener("abort", onAbort);
    child.on("error", (err) => {
      clearTimeout(timer);
      done({ code: null, stdout, stderr: `${stderr}${err.message}`, timedOut, cancelled });
    });
    child.on("close", (code) => {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
      done({ code, stdout, stderr, timedOut, cancelled });
    });
  });
}

// What the model is told: the exit code, then the output (cut if long).
export function describeRun(ran: Ran, seconds: number): string {
  const cut = (text: string) => (text.length > 12_000 ? `${text.slice(0, 6_000)}\n… (${text.length - 12_000} characters cut) …\n${text.slice(-6_000)}` : text);
  const head = ran.timedOut ? `stopped: still running after ${seconds} s` : ran.cancelled ? "stopped by the user" : `exit code ${ran.code}`;
  const parts = [head];
  if (ran.stdout.trim()) parts.push(cut(ran.stdout.trimEnd()));
  if (ran.stderr.trim()) parts.push(`[stderr]\n${cut(ran.stderr.trimEnd())}`);
  return parts.join("\n");
}

export const insideFolder = (root: string, path: string) => {
  const base = normal(root);
  const target = normal(path);
  return target === base || target.startsWith(base + sep);
};
