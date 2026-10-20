import { test } from "node:test";
import assert from "node:assert/strict";
import { chmodSync, copyFileSync, mkdirSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import { join } from "node:path";
import { isReadOnly, pathArguments, removesCriticalPath, resolveCommand, safeEnvironment, splitCommand, CommandError, type Command } from "../src/shell.ts";
import { newSession, runTool, type Answer, type Approval, type Session } from "../src/tools.ts";
import { parseRule } from "../src/rules.ts";
import { tempDir } from "./helpers.ts";

// The same checks on Windows and on a Mac, each with its own real programs and paths.
const WINDOWS = process.platform === "win32";
const SHELL_LINE = WINDOWS ? "cmd /c echo hi" : "sh -c 'echo hi'";
const SHELL_NAME = WINDOWS ? "cmd" : "sh";

test("a command line is split into words; quotes keep spaces; backslashes are just characters", () => {
  assert.deepEqual(splitCommand('git commit -m "fix the port"'), ["git", "commit", "-m", "fix the port"]);
  assert.deepEqual(splitCommand("type C:\\Users\\Jorda\\notes.txt"), ["type", "C:\\Users\\Jorda\\notes.txt"]);
  assert.deepEqual(splitCommand(`git log --grep 'a|b' --format="%h && %s"`), ["git", "log", "--grep", "a|b", "--format=%h && %s"], "inside quotes, | and && are text");
});

test("shell syntax is refused: there's no shell to give it meaning", () => {
  for (const line of ["git status && del /s *", "npm test || echo failed", "dir; whoami", "type a.txt | findstr x", "echo hi > out.txt", "sort < in.txt", "echo $(whoami)", "echo `whoami`", "git status\ndel x"]) {
    assert.throws(() => splitCommand(line), CommandError, line);
  }
  assert.throws(() => splitCommand('echo "never closed'), /never closed/);
  assert.throws(() => splitCommand("x".repeat(10_001)), /10,000/);
});

test("the program is found on PATH, and rules see its plain name however it was typed", () => {
  const root = tempDir();
  const git = resolveCommand(["git", "status"], root);
  assert.match(git.program, WINDOWS ? /git\.exe$/i : /\/git$/);
  assert.equal(git.text, "git status");
  const full = resolveCommand([git.program, "log", "-1"], root);
  assert.equal(full.text, "git log -1", "a full path to git is still git to the rules");
  assert.throws(() => resolveCommand(["no-such-program-xyz"], root), /wasn't found on PATH/);
});

test("npm runs through Node directly; other batch files are refused", () => {
  const root = tempDir();
  const npm = resolveCommand(["npm", "--version"], root);
  assert.equal(npm.program, process.execPath);
  assert.match(npm.args[0], /npm-cli\.js$/);
  assert.equal(npm.text, "npm --version");
  if (WINDOWS) {
    writeFileSync(join(root, "build.cmd"), "@echo off\r\necho built\r\n");
    assert.throws(() => resolveCommand([".\\build.cmd"], root), /batch file, which Windows can only run through cmd\.exe/);
  } else {
    writeFileSync(join(root, "notes.txt"), "not a program");
    assert.throws(() => resolveCommand(["./notes.txt"], root), /wasn't found/, "a file you may not execute isn't a program");
  }
});

test("a bare name is never looked up in the project folder: a planted git.exe isn't run", () => {
  const root = tempDir();
  const planted = join(root, WINDOWS ? "git.exe" : "git");
  copyFileSync(process.execPath, planted); // any real program, posing as git
  chmodSync(planted, 0o755);
  const git = resolveCommand(["git", "status"], root);
  assert.ok(!git.program.toLowerCase().startsWith(root.toLowerCase()), `ran ${git.program}`);
  assert.equal(resolveCommand([WINDOWS ? ".\\git.exe" : "./git", "status"], root).program.toLowerCase(), planted.toLowerCase(), "typing a path to it is deliberate, and still needs your OK");
});

test("read-only means looking only, judged word by word", () => {
  const root = tempDir();
  const ro = (line: string) => isReadOnly(resolveCommand(splitCommand(line), root));
  for (const line of ["git status", "git log --oneline -5", "git diff HEAD~1", "git rev-list --count HEAD", "git --version", "node --version", "npm --version", WINDOWS ? "where node" : "which node"]) assert.ok(ro(line), line);
  for (const line of ["git push", "git branch new-feature", "git -c core.fsmonitor=evil.exe status", "git -C .. status", "git diff --output=x.txt", "git log --ext-diff", "node -e 1", "npm test", "npm install"]) assert.ok(!ro(line), line);
});

test("path arguments are found, with links followed; network paths are kept apart", () => {
  const root = tempDir();
  writeFileSync(join(root, "a.txt"), "a");
  const command = resolveCommand(["git", "diff", "--no-index", "a.txt", "\\\\server\\share\\x", "--stat"], root);
  const found = pathArguments(command, root);
  assert.equal(found.local.length, 1);
  assert.match(found.local[0], /a\.txt$/);
  assert.deepEqual(found.network, ["\\\\server\\share\\x"]);
});

test("removing a drive, a top folder, home, or the project and its parents is never allowed", () => {
  const root = join(tempDir(), "project");
  const rm = (...args: string[]): Command => ({ program: "rm.exe", args, name: "rm", text: ["rm", ...args].join(" ") });
  const [top, under] = WINDOWS ? ["C:\\", "C:\\Windows"] : ["/", "/usr"];
  assert.ok(removesCriticalPath(rm("-rf", top), root));
  assert.ok(removesCriticalPath(rm("-rf", under), root));
  assert.ok(removesCriticalPath(rm("-rf", "."), root), "the project folder itself");
  assert.ok(removesCriticalPath(rm("-rf", ".."), root), "its parent");
  assert.ok(removesCriticalPath(rm("-r", homedir()), root));
  assert.equal(removesCriticalPath(rm("-rf", "build"), root), undefined, "a folder inside the project is an ordinary request");
});

test("environment variables that look like secrets aren't passed on", () => {
  process.env.SQUID_TEST_API_KEY = "hunter2";
  const env = safeEnvironment();
  delete process.env.SQUID_TEST_API_KEY;
  assert.equal(env.SQUID_TEST_API_KEY, undefined);
  assert.equal(env.GIT_PAGER, "cat");
  assert.ok(env.PATH ?? env.Path, "everything else stays");
});

// --- through the tool, with real programs ---

function approver(...answers: Answer[]) {
  const asked: Approval[] = [];
  return { asked, approve: async (request: Approval) => (asked.push(request), answers.shift() ?? { ok: false }) };
}
const run = (session: Session, command: string, timeout_seconds?: number) => runTool(session, { name: "run_command", arguments: JSON.stringify({ command, timeout_seconds }) });

test("read-only commands run straight away; their output and exit code go to the model", async () => {
  const root = tempDir();
  const you = approver();
  const session = newSession(root, you.approve);
  assert.match(await run(session, "git --version"), /^exit code 0\ngit version \d/);
  assert.match(await run(session, "git status"), /^exit code 128\n\[stderr\]\nfatal: not a git repository/, "a temp folder isn't a repo");
  assert.equal(you.asked.length, 0);
});

test("anything else asks first, showing which program will really run", async () => {
  const root = tempDir();
  const you = approver({ ok: true }, { ok: false, feedback: "use npm test instead" });
  const session = newSession(root, you.approve);
  process.env.SQUID_TEST_SECRET_TOKEN = "hunter2";
  const result = await run(session, `node -e "console.log(process.env.SQUID_TEST_SECRET_TOKEN ?? 'not passed on')"`);
  delete process.env.SQUID_TEST_SECRET_TOKEN;
  assert.equal(result, "exit code 0\nnot passed on");
  assert.equal(you.asked[0].title, `Run node -e "console.log(process.env.SQUID_TEST_SECRET_TOKEN ?? 'not passed on')"`);
  assert.match(you.asked[0].preview, WINDOWS ? /\(program: .*node\.exe\)/i : /\(program: .*\/node\)/);
  assert.equal(await run(session, "node -e 1"), "Error: the user declined: Run node -e 1. Their instructions: use npm test instead");
});

test("a command that runs too long is stopped, and so is one you cancel", async () => {
  const root = tempDir();
  const session = newSession(root, approver({ ok: true }, { ok: true }).approve);
  assert.match(await run(session, `node -e "setTimeout(() => {}, 60000)"`, 1), /^stopped: still running after 1 s/);
  const stop = new AbortController();
  session.signal = stop.signal;
  setTimeout(() => stop.abort(), 300);
  assert.match(await run(session, `node -e "setTimeout(() => {}, 60000)"`), /^stopped by the user/);
});

test("shells always ask, even under an allow-everything rule; a deny rule refuses without asking", async () => {
  const root = tempDir();
  const you = approver({ ok: false });
  const session = newSession(root, you.approve);
  session.book = { ...session.book, allow: [parseRule("Bash")] };
  assert.match(await run(session, SHELL_LINE), /declined/);
  assert.match(you.asked[0].reason, new RegExp(`${SHELL_NAME} is a shell: it can run anything`));
  session.book = { ...session.book, deny: [parseRule("Bash(curl *)")] };
  assert.match(await run(session, `${WINDOWS ? "C:\\Windows\\System32\\curl.exe" : "/usr/bin/curl"} --version`), /denied by rule Bash\(curl \*\)/, "curl by its full path is still curl");
  assert.equal(you.asked.length, 1);
});

test("on a Mac, a path is not an option, and programs that start other programs count as shells", { skip: WINDOWS && "Windows options start with /" }, async () => {
  const root = join(tempDir(), "project");
  const rm = (...args: string[]): Command => ({ program: "/bin/rm", args, name: "rm", text: ["rm", ...args].join(" ") });
  assert.equal(removesCriticalPath(rm("-rf", "/"), root), "/", "rm -rf / is caught, not skipped as an option");
  assert.ok(removesCriticalPath(rm("-rf", homedir()), root));
  const you = approver({ ok: false });
  const session = newSession(tempDir(), you.approve);
  session.book = { ...session.book, allow: [parseRule("Bash")] };
  assert.match(await run(session, "sudo -n true"), /declined/, "sudo always asks");
  assert.match(you.asked[0].reason, /sudo is a shell: it can run anything/);
  assert.equal(you.asked[0].always, undefined);
});

test("paths in a command follow the rulebook: secrets refused, outside paths asked about", async () => {
  const top = tempDir();
  const root = join(top, "project");
  mkdirSync(root);
  writeFileSync(join(root, ".env"), "API_KEY=x");
  writeFileSync(join(root, "a.txt"), "a");
  writeFileSync(join(top, "outside.txt"), "b");
  const you = approver({ ok: false });
  const session = newSession(root, you.approve);
  assert.match(await run(session, "git diff --no-index .env a.txt"), /Squid never opens/);
  assert.match(await run(session, "git diff --no-index a.txt ../outside.txt"), /declined/);
  assert.match(you.asked[0].reason, /outside\.txt, outside the project/);
  assert.match(await run(session, "git status && del a.txt"), /shell syntax/);
});

test("after a command that may change files, edits need a fresh read", async () => {
  const root = tempDir();
  writeFileSync(join(root, "a.txt"), "one\n");
  const session = newSession(root, approver({ ok: true }, { ok: true }).approve);
  await runTool(session, { name: "read_file", arguments: JSON.stringify({ path: "a.txt" }) });
  await run(session, "node -e 1");
  const edit = await runTool(session, { name: "edit_file", arguments: JSON.stringify({ path: "a.txt", old_text: "one", new_text: "two" }) });
  assert.match(edit, /read a\.txt with read_file before changing it/);
});
