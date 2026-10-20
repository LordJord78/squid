// The rulebook: before any tool runs, Squid decides allow, ask or deny, the way Claude Code's permission
// rules are documented to work. Rules look like `Read(./.env)`, `Edit(src/**)` or `Bash(npm run *)`.
// Everything here is pure logic, so it's tested against the docs' own examples.
import { readFileSync, realpathSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { SQUID_HOME } from "./config.ts";

export type Tool = "Read" | "Edit" | "Bash";
// Claude Code's modes (step 12): manual asks before every change; auto lets edits inside the folder go
// ahead (Claude Code's acceptEdits); plan only looks, until you approve a plan.
export type Mode = "manual" | "auto" | "plan";
export const MODES: Mode[] = ["manual", "auto", "plan"]; // the order Shift+Tab cycles through
export type Kind = "deny" | "ask" | "allow";
export type Rule = { text: string; tool: string; specifier?: string };
// What the agent wants to do. For a file, `path` is what it asked for and `real` is where that leads
// after links are followed (step 08); they differ when a link is involved.
export type Request = { tool: Tool; path?: string; real?: string; command?: string; readOnly?: boolean }; // readOnly: a command that only looks (step 11)
export type Decision = { outcome: Kind; reason: string };
export type Rulebook = { deny: Rule[]; ask: Rule[]; allow: Rule[]; root: string; home?: string };

// Squid's own hard rules, in every mode, whatever any rule says: secrets, and Jordan's research holdout
// data (an agent that sees the holdout set has spoiled the test; see the AI Lab). "//**/" means anywhere
// on disk, not just inside the project, so they also catch where a link really leads.
export const HARD_DENY = [
  "*holdout*", // any file or folder with "holdout" in its name
  ".env", ".env.*", // environment files: API keys, passwords
  ".ssh", ".aws", ".gnupg", // key folders (a folder pattern covers everything inside)
  "id_rsa", "id_ecdsa", "id_ed25519", "id_dsa", // private SSH keys
  "*.pem", "*.key", "*.p12", "*.pfx", "*.kdbx", // certificates, keys, password databases
  ".git-credentials", ".netrc", ".npmrc", // saved logins and tokens
].map((name) => `//**/${name}`);

// Claude Code's protected paths (documented): writes to them always ask, and allow rules can't pre-approve them.
const PROTECTED_DIRS = [".git", ".config/git", ".vscode", ".idea", ".husky", ".cargo", ".devcontainer", ".yarn", ".mvn", ".claude", ".squid"];
const PROTECTED_FILES = [
  ".gitconfig", ".gitmodules",
  ".bashrc", ".bash_profile", ".bash_login", ".bash_aliases", ".bash_logout", ".zshrc", ".zprofile", ".zshenv", ".zlogin", ".zlogout", ".profile", ".envrc",
  ".npmrc", ".yarnrc", ".yarnrc.yml", ".pnp.cjs", ".pnp.loader.mjs", ".pnpmfile.cjs", "bunfig.toml", ".bunfig.toml",
  ".bazelrc", ".bazelversion", ".bazeliskrc",
  ".pre-commit-config.yaml", "lefthook.yml", "lefthook.yaml", ".lefthook.yml", ".lefthook.yaml",
  "gradle-wrapper.properties", "maven-wrapper.properties", ".devcontainer.json",
  ".ripgreprc", "pyrightconfig.json", ".mcp.json", ".claude.json",
];
export const PROTECTED = [...PROTECTED_DIRS, ...PROTECTED_FILES].map((name) => `//**/${name}`);

// `Tool` or `Tool(specifier)`. Parentheses inside the specifier are literal, so it runs to the last ")".
export function parseRule(text: string): Rule {
  const open = text.indexOf("(");
  if (open === -1 || !text.endsWith(")")) return { text, tool: text.trim() };
  const specifier = text.slice(open + 1, -1).trim();
  return { text, tool: text.slice(0, open).trim(), specifier: specifier === "*" ? undefined : specifier }; // Bash(*) = Bash
}

// The decision, in Claude Code's documented order. The first step that matches decides:
//   Squid's hard rules → deny rules → plan mode (edits) → protected paths (edits) → ask rules →
//   allow rules → the mode's default.
// "Rule specificity doesn't change the order": a broad deny beats a narrow allow.
export function decide(request: Request, book: Rulebook, mode: Mode = "manual"): Decision {
  const paths = [request.path, request.real].filter((p): p is string => p !== undefined);
  if (request.tool !== "Bash") {
    const secret = paths.find((p) => HARD_DENY.some((pattern) => pathMatches(pattern, p, book, "deny")));
    if (secret) return { outcome: "deny", reason: `Squid never opens ${shortName(secret)} (secrets and holdout data are off limits in every mode)` };
  }
  // Deny and ask rules count if EITHER the requested path or where it leads matches: a link to a denied
  // file is denied. Allow rules need BOTH to match, so a link can't carry an allowed path somewhere else.
  const deny = book.deny.find((rule) => matches(rule, request, book, "deny", "either"));
  if (deny) return { outcome: "deny", reason: `denied by rule ${deny.text}` };
  // In plan mode nothing changes until you approve the plan, whatever the rules allow (documented).
  if (mode === "plan" && request.tool === "Edit") return { outcome: "deny", reason: "plan mode: nothing is changed until you approve the plan (submit_plan)" };
  if (request.tool === "Edit" && paths.some((p) => PROTECTED.some((pattern) => pathMatches(pattern, p, book, "deny")))) {
    return { outcome: "ask", reason: "protected path: changes here always need your OK" };
  }
  const ask = book.ask.find((rule) => matches(rule, request, book, "ask", "either"));
  if (ask) return { outcome: "ask", reason: `rule ${ask.text} says ask` };
  const allow = book.allow.find((rule) => matches(rule, request, book, "allow", "both"));
  if (allow) return { outcome: "allow", reason: `allowed by rule ${allow.text}` };
  // No rule matched: the mode's default. Reads and read-only commands run in every mode. Edits ask in
  // manual mode and go ahead in auto mode (the folder fence and protected paths still apply). Other
  // commands ask in every mode.
  if (request.tool === "Read") return { outcome: "allow", reason: "reading is allowed" };
  if (request.tool === "Bash" && request.readOnly) return { outcome: "allow", reason: "a read-only command" };
  if (request.tool === "Edit" && mode === "auto") return { outcome: "allow", reason: "auto mode: edits inside this folder go ahead" };
  return { outcome: "ask", reason: request.tool === "Edit" ? "manual mode: changes need your OK" : "commands that aren't read-only need your OK" };
}

function matches(rule: Rule, request: Request, book: Rulebook, kind: Kind, need: "either" | "both"): boolean {
  if (rule.tool !== request.tool) return false;
  if (rule.specifier === undefined) return true; // a bare tool name matches every use
  if (request.tool === "Bash") return commandMatches(rule.specifier, request.command ?? "");
  const paths = [request.path, request.real].filter((p): p is string => p !== undefined);
  const hit = (p: string) => pathMatches(rule.specifier!, p, book, kind);
  return need === "both" ? paths.every(hit) : paths.some(hit);
}

// --- Bash rules: the whole command text, with * standing in for any text (spaces included) ---
export function commandMatches(pattern: string, command: string): boolean {
  if (pattern.endsWith(":*")) pattern = `${pattern.slice(0, -2)} *`; // `ls:*` is another way to write `ls *`
  const regex = new RegExp(`^${pattern.split("*").map(escapeRegex).join(".*")}$`, "s");
  if (regex.test(command)) return true;
  // A trailing " *" that is the rule's only wildcard also matches the bare command: `ls *` matches `ls`.
  const onlyTrailing = pattern.endsWith(" *") && pattern.indexOf("*") === pattern.length - 1;
  return onlyTrailing && command === pattern.slice(0, -2);
}

// --- Read and Edit rules: gitignore-style patterns over the full path ---
// Windows paths are compared in POSIX form, as the docs say: C:\Users\alice → /c/Users/alice.
// Squid also ignores upper/lower case on Windows and macOS, whose disks treat .ENV and .env as the
// same file by default.
export const IGNORE_CASE = process.platform === "win32" || process.platform === "darwin";
export function posix(path: string): string {
  const slashed = path.replace(/\\/g, "/");
  const drive = /^([A-Za-z]):(\/|$)/.exec(slashed);
  const result = drive ? `/${drive[1].toLowerCase()}${slashed.slice(2)}` : slashed;
  return IGNORE_CASE ? result.toLowerCase() : result;
}

export function pathMatches(pattern: string, path: string, book: Rulebook, kind: Kind): boolean {
  const target = posix(path);
  const home = posix(book.home ?? homedir());
  const root = posix(book.root);
  let base: string; // the folder the pattern is anchored at
  let rest: string;
  let anyDepth = false;
  if (pattern.startsWith("//")) [base, rest] = ["", pattern.slice(1)]; // //path: from the filesystem root
  else if (pattern.startsWith("~/")) [base, rest] = [home, pattern.slice(1)]; // ~/path: from your home folder
  else if (pattern.startsWith("/")) [base, rest] = [root, pattern]; // /path: from the project folder
  else {
    rest = `/${pattern.replace(/^\.\//, "")}`; // path or ./path: from the current folder
    base = root;
    // gitignore: a pattern with no "/" inside it (`.env`, `*.log`) matches at any depth. Claude Code adds:
    // a single folder like `src/**` matches at any depth in deny and ask rules, but only at the top in allow.
    const inner = rest.slice(1).replace(/\/$/, "");
    const singleFolder = /^[^/]+\/\*\*$/.test(inner);
    anyDepth = (!pattern.startsWith("./") && !inner.includes("/")) || (singleFolder && kind !== "allow");
  }
  if (IGNORE_CASE) rest = rest.toLowerCase();
  // Matching a folder covers everything inside it, as in .gitignore: hence the optional "/…" at the end.
  const source = `^${escapeRegex(base)}${anyDepth ? "(?:/.*)?" : ""}${globToRegex(rest)}(?:/.*)?$`;
  // Compiled once and remembered: `search` asks about every file it visits.
  let regex = compiled.get(source);
  if (!regex) compiled.set(source, (regex = new RegExp(source, "s")));
  return regex.test(target);
}
const compiled = new Map<string, RegExp>();

// gitignore wildcards: * = anything inside one folder name, ** = across folders, ? = one character.
function globToRegex(glob: string): string {
  let out = "";
  for (let i = 0; i < glob.length; i++) {
    const c = glob[i];
    if (glob.startsWith("/**/", i)) { out += "(?:/.*)?/"; i += 3; }
    else if (glob.startsWith("/**", i) && i + 3 === glob.length) { out += "(?:/.*)?"; i += 2; }
    else if (glob.startsWith("**", i)) { out += ".*"; i += 1; }
    else if (c === "*") out += "[^/]*";
    else if (c === "?") out += "[^/]";
    else out += escapeRegex(c);
  }
  return out;
}

const escapeRegex = (text: string) => text.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
const shortName = (path: string) => path.replace(/\\/g, "/").split("/").slice(-2).join("/");

// The rules in force for a project: yours from ~/.squid/config.json, plus the project's own
// .squid/settings.json. A project may add deny and ask rules, but its allow rules are ignored: a
// downloaded repo shouldn't be able to pre-approve things for your agent. (Claude Code handles the same
// risk by asking whether you trust the folder.)
export function loadRulebook(root: string, files = { user: join(SQUID_HOME, "config.json"), project: join(root, ".squid", "settings.json") }): Rulebook {
  const read = (file: string) => {
    try {
      return JSON.parse(readFileSync(file, "utf8")).permissions ?? {};
    } catch {
      return {};
    }
  };
  const user = read(files.user);
  const project = read(files.project);
  const list = (...lists: (string[] | undefined)[]) => lists.flatMap((l) => l ?? []).map(parseRule);
  // Anchored where the folder really is, like the paths it's checked against: on a Mac /var is a link to
  // /private/var, and a rule anchored at the link's side would never match.
  return { deny: list(user.deny, project.deny), ask: list(user.ask, project.ask), allow: list(user.allow), root: realpathSync.native(root) };
}

