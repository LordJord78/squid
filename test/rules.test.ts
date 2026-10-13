// Most of these cases are the examples in Claude Code's permission docs (code.claude.com/docs/en/permissions),
// turned into tests: the docs are the recording we compare against in this step.
import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { commandMatches, decide, loadRulebook, parseRule, posix, type Request, type Rulebook } from "../src/rules.ts";
import { tempDir } from "./helpers.ts";

const ROOT = "C:\\proj";
const book = (rules: { deny?: string[]; ask?: string[]; allow?: string[] } = {}): Rulebook => ({
  deny: (rules.deny ?? []).map(parseRule),
  ask: (rules.ask ?? []).map(parseRule),
  allow: (rules.allow ?? []).map(parseRule),
  root: ROOT,
  home: "C:\\Users\\alice",
});
const at = (relative: string) => `${ROOT}\\${relative.replace(/\//g, "\\")}`;
const read = (path: string, real = path): Request => ({ tool: "Read", path, real });
const edit = (path: string, real = path): Request => ({ tool: "Edit", path, real });
const bash = (command: string): Request => ({ tool: "Bash", command });
const outcome = (request: Request, b: Rulebook) => decide(request, b).outcome;

test("rules are Tool or Tool(specifier); parentheses inside are literal", () => {
  assert.deepEqual(parseRule("Bash(npm run build)"), { text: "Bash(npm run build)", tool: "Bash", specifier: "npm run build" });
  assert.equal(parseRule("Edit(./Finance (2024)/**)").specifier, "./Finance (2024)/**");
  assert.equal(parseRule("Bash(*)").specifier, undefined, "Bash(*) is the same as Bash");
  assert.deepEqual(parseRule("Read"), { text: "Read", tool: "Read" });
});

test("deny, then ask, then allow: a broad deny beats a narrow allow", () => {
  const b = book({ deny: ["Bash(aws *)"], ask: ["Bash(git push *)"], allow: ["Bash(aws s3 ls)", "Bash(git push origin main)", "Bash(npm run *)"] });
  assert.equal(outcome(bash("aws s3 ls"), b), "deny", "an allow rule can't carve an exception out of a deny rule");
  assert.equal(outcome(bash("git push origin main"), b), "ask", "a matching ask rule prompts even when an allow rule matches");
  assert.equal(outcome(bash("npm run build"), b), "allow");
  assert.equal(outcome(bash("npm install"), b), "ask", "no rule: commands ask");
});

test("Bash wildcards, row by row from the docs' table", () => {
  const rows: [string, string[], string[]][] = [
    ["npm run build", ["npm run build"], ["npm run build --watch"]],
    ["npm run *", ["npm run build", "npm run test --watch", "npm run"], ["npm install"]],
    ["git log * main", ["git log --oneline main", "git log -5 main"], ["git log main", "git push origin main"]],
    ["git * main", ["git merge main", "git push origin main", "git -c core.fsmonitor=x diff main"], ["git log"]],
    ["* --version", ["node --version", "bash -c 'echo hi' --version"], ["node -v"]],
    ["ls *", ["ls -la", "ls"], ["lsof"]],
    ["ls*", ["ls -la", "lsof"], []],
    ["* --help *", ["npm --help x"], ["npm --help"]],
    ["ls:*", ["ls -la", "ls"], ["lsof"]],
  ];
  for (const [pattern, yes, no] of rows) {
    for (const command of yes) assert.ok(commandMatches(pattern, command), `${pattern} should match ${command}`);
    for (const command of no) assert.ok(!commandMatches(pattern, command), `${pattern} should not match ${command}`);
  }
});

test("src/** reaches nested copies in deny rules but not in allow rules (the docs' depth table)", () => {
  const top = at("src/app.ts");
  const nested = at("vendor/pkg/src/lib.js");
  const allowOnly = (rule: string) => book({ allow: [rule] });
  const askOnly = (rule: string) => book({ ask: [rule] });
  // Edit asks by default, so "allow" shows an allow rule matched and "ask" shows it didn't.
  assert.deepEqual([outcome(edit(top), allowOnly("Edit(src/**)")), outcome(edit(nested), allowOnly("Edit(src/**)"))], ["allow", "ask"]);
  assert.deepEqual([outcome(read(top), askOnly("Read(src/**)")), outcome(read(nested), askOnly("Read(src/**)"))], ["ask", "ask"]);
  assert.deepEqual([outcome(edit(top), allowOnly("Edit(/src/**)")), outcome(edit(nested), allowOnly("Edit(/src/**)"))], ["allow", "ask"]);
  assert.deepEqual([outcome(edit(top), allowOnly("Edit(**/src/**)")), outcome(edit(nested), allowOnly("Edit(**/src/**)"))], ["allow", "allow"]);
});

test("bare names match at any depth under the project; // and ~/ anchor elsewhere", () => {
  const b = book({ ask: ["Read(*.log)", "Read(//**/notes.txt)", "Read(~/Documents/*.pdf)"] });
  assert.equal(outcome(read(at("debug.log")), b), "ask");
  assert.equal(outcome(read(at("deep/down/debug.log")), b), "ask");
  assert.equal(outcome(read("C:\\other\\debug.log"), b), "allow", "a bare name doesn't reach outside the project");
  assert.equal(outcome(read("D:\\anywhere\\notes.txt"), b), "ask", "//**/ means anywhere on any drive");
  assert.equal(outcome(read("C:\\Users\\alice\\Documents\\cv.pdf"), b), "ask");
  assert.equal(outcome(read("C:\\Users\\alice\\Documents\\old\\cv.pdf"), b), "allow", "* stays inside one folder");
});

test("Windows paths are compared in POSIX form, ignoring case", () => {
  assert.equal(posix("C:\\Users\\alice"), process.platform === "win32" ? "/c/users/alice" : "/c/Users/alice");
  assert.equal(outcome(read(at("Secrets/Plan.TXT")), book({ deny: ["Read(secrets/**)"] })), "deny");
});

test("links: deny if either end matches; allow only if both ends match (the docs' example)", () => {
  const b = book({ deny: ["Read(~/private/**)"], allow: ["Edit(./project/**)"] });
  // ./project/key is a link to a file in ~/private: the target matches the deny rule
  assert.equal(outcome(read(at("project/key"), "C:\\Users\\alice\\private\\diary.txt"), b), "deny");
  // an edit through a link inside ./project that leads elsewhere isn't covered by the allow rule
  assert.equal(outcome(edit(at("project/link.txt"), "C:\\elsewhere\\x.txt"), b), "ask");
  assert.equal(outcome(edit(at("project/real.txt")), b), "allow");
});

test("protected paths: edits always ask, even under an allow-everything rule; reads are fine", () => {
  const b = book({ allow: ["Edit"] });
  for (const path of [".git/config", ".vscode/settings.json", "sub/.husky/pre-commit", ".npmrc", ".claude/settings.json", ".gitmodules"]) {
    assert.equal(outcome(edit(at(path)), b), path === ".npmrc" ? "deny" : "ask", path); // .npmrc holds tokens: Squid hard-denies it
  }
  assert.equal(outcome(edit(at("package.json")), b), "allow");
  assert.equal(outcome(read(at(".git/config")), b), "allow");
  assert.equal(outcome(edit(at(".git/config")), book({ deny: ["Edit(.git/**)"] })), "deny", "a deny rule still wins");
});

test("Squid's hard rules: secrets and holdout data, whatever the rules say", () => {
  const b = book({ allow: ["Read", "Edit"] });
  const blocked = [".env", "config/.env.local", ".ENV", "data/es_holdout/2026.parquet", "holdout_results.csv", ".ssh/id_rsa", "certs/server.key", "keys/id_ed25519", "vault.kdbx"];
  for (const path of blocked) assert.equal(outcome(read(at(path)), b), "deny", path);
  assert.equal(outcome(read("C:\\AI-Lab\\opencode-test\\data\\es_holdout\\x.csv"), b), "deny", "anywhere on disk");
  assert.equal(outcome(read(at("notes.md"), at(".env")), b), "deny", "a harmless-looking link to .env");
  for (const path of [".environment.md", "README.md", "keys/id_rsa.pub", "src/holdup.ts"]) assert.equal(outcome(read(at(path)), b), "allow", path);
});

test("with no rule, manual mode's defaults: reads run, edits and commands ask", () => {
  assert.deepEqual([outcome(read(at("a.ts")), book()), outcome(edit(at("a.ts")), book()), outcome(bash("ls"), book())], ["allow", "ask", "ask"]);
});

test("a project can add deny and ask rules, but not allow rules", () => {
  const dir = tempDir();
  mkdirSync(join(dir, ".squid"));
  const project = join(dir, ".squid", "settings.json");
  writeFileSync(project, JSON.stringify({ permissions: { deny: ["Read(docs/private/**)"], ask: ["Read(*.csv)"], allow: ["Bash(curl *)"] } }));
  const user = join(dir, "user.json");
  writeFileSync(user, JSON.stringify({ permissions: { allow: ["Bash(npm test)"] } }));
  const b = loadRulebook(dir, { user, project });
  assert.deepEqual(b.deny.map((r) => r.text), ["Read(docs/private/**)"]);
  assert.deepEqual(b.ask.map((r) => r.text), ["Read(*.csv)"]);
  assert.deepEqual(b.allow.map((r) => r.text), ["Bash(npm test)"], "the project's Bash(curl *) is ignored");
});
