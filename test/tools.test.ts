import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { inside, newSession, runTool, ToolError } from "../src/tools.ts";
import { tempDir } from "./helpers.ts";

// A small project, a secret next to it (outside), and a junction inside that points at the secret's folder.
function fixture() {
  const top = tempDir();
  const project = join(top, "project");
  mkdirSync(join(project, "src"), { recursive: true });
  mkdirSync(join(project, "node_modules", "dep"), { recursive: true });
  writeFileSync(join(project, "README.md"), "# Demo\nThe codename is Blue Heron.\n");
  writeFileSync(join(project, "src", "main.ts"), Array.from({ length: 10 }, (_, i) => `line ${i + 1}`).join("\n"));
  writeFileSync(join(project, "node_modules", "dep", "index.js"), "blue heron in a dependency");
  writeFileSync(join(project, "logo.png"), Buffer.from([0x89, 0x50, 0x4e, 0x47, 0, 0, 0, 0]));
  mkdirSync(join(top, "private"));
  writeFileSync(join(top, "private", "secret.txt"), "blue heron password: hunter2");
  // A junction is Windows' folder link; making one needs no admin rights.
  symlinkSync(join(top, "private"), join(project, "docs"), "junction");
  return { top, project };
}
const call = (name: string, args: object) => ({ name, arguments: JSON.stringify(args) });
const run = (root: string, c: { name: string; arguments: string }) => runTool(newSession(root), c);

test("paths inside the project are allowed", async () => {
  const { project } = fixture();
  assert.match(inside(project, "src/main.ts"), /main\.ts$/);
  assert.match(inside(project, "src/../README.md"), /README\.md$/, ".. that stays inside is fine");
});

test("every way out of the project is refused", async () => {
  const { top, project } = fixture();
  const attempts = ["../private/secret.txt", join(top, "private", "secret.txt"), "docs/secret.txt", "docs", "C:\\Windows\\win.ini"];
  for (const path of attempts) assert.throws(() => inside(project, path), ToolError, path);
  assert.throws(() => inside(project, "no-such-file.txt"), /doesn't exist/);
});

test("read_file numbers the lines and reads in parts", async () => {
  const { project } = fixture();
  const part = await run(project, call("read_file", { path: "src/main.ts", offset: 4, limit: 3 }));
  assert.equal(part, "    4\tline 4\n    5\tline 5\n    6\tline 6\n(lines 4-6 of 10; read on with offset 7)");
  assert.match(await run(project, call("read_file", { path: "logo.png" })), /isn't a text file/);
  assert.match(await run(project, call("read_file", { path: "src" })), /is a folder/);
});

test("list_files marks folders and links", async () => {
  const { project } = fixture();
  assert.equal(await run(project, call("list_files", {})), "docs (link)\nlogo.png\nnode_modules/\nREADME.md\nsrc/");
});

test("search finds text in any case, skips dependencies, and never follows a link out", async () => {
  const { project } = fixture();
  assert.equal(await run(project, call("search", { text: "BLUE HERON" })), "README.md:2: The codename is Blue Heron.");
});

test("mistakes go back to the model as text, so it can correct itself", async () => {
  const { project } = fixture();
  assert.match(await run(project, { name: "read_file", arguments: "{not json" }), /^Error: the arguments weren't valid JSON/);
  assert.match(await run(project, call("delete_everything", {})), /^Error: there's no tool called delete_everything/);
  assert.match(await run(project, call("read_file", { path: "../private/secret.txt" })), /^Error: .*outside the project folder/);
});

test("a huge result is cut, to protect the context window", async () => {
  const { project } = fixture();
  writeFileSync(join(project, "big.txt"), Array.from({ length: 400 }, () => "x".repeat(100)).join("\n"));
  assert.match(await run(project, call("read_file", { path: "big.txt" })), /… \(cut: \d+ more characters\)$/);
});

test("the rulebook guards every tool: read, list and search", async () => {
  const root = tempDir();
  writeFileSync(join(root, ".env"), "API_KEY=blue-heron-secret");
  mkdirSync(join(root, "data", "es_holdout"), { recursive: true });
  writeFileSync(join(root, "data", "es_holdout", "2026.csv"), "blue heron in the holdout");
  mkdirSync(join(root, "docs", "private"), { recursive: true });
  writeFileSync(join(root, "docs", "private", "diary.md"), "blue heron in the diary");
  writeFileSync(join(root, "notes.md"), "blue heron in the notes");
  mkdirSync(join(root, ".squid"));
  writeFileSync(join(root, ".squid", "settings.json"), JSON.stringify({ permissions: { deny: ["Read(docs/private/**)"] } }));

  assert.match(await run(root, call("read_file", { path: ".env" })), /^Error: \.env: Squid never opens/);
  assert.match(await run(root, call("read_file", { path: "docs/private/diary.md" })), /^Error: .*denied by rule Read\(docs\/private\/\*\*\)/);
  assert.match(await run(root, call("list_files", { path: "data" })), /^es_holdout\/ \(off limits\)$/);
  assert.match(await run(root, call("list_files", {})), /^\.env \(off limits\)$/m);
  assert.equal(await run(root, call("search", { text: "blue heron" })), "notes.md:1: blue heron in the notes", "the other three are skipped");
  assert.match(await run(root, call("search", { text: "x", path: "data/es_holdout" })), /^Error: .*never opens/);
});

test("the project's rules still hold when its folder is opened through a link", async () => {
  // On Windows, cd into a junction keeps the junction in the current folder's name, while Squid checks
  // where each file really is: the rules have to be anchored at the real folder too.
  const top = tempDir();
  const project = join(top, "project");
  mkdirSync(join(project, "docs", "private"), { recursive: true });
  writeFileSync(join(project, "docs", "private", "diary.md"), "blue heron in the diary");
  mkdirSync(join(project, ".squid"));
  writeFileSync(join(project, ".squid", "settings.json"), JSON.stringify({ permissions: { deny: ["Read(docs/private/**)"] } }));
  const link = join(top, "shortcut");
  symlinkSync(project, link, "junction");
  assert.match(await run(link, call("read_file", { path: "docs/private/diary.md" })), /^Error: .*denied by rule Read\(docs\/private\/\*\*\)/);
});
