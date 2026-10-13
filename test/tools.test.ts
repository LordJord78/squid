import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, symlinkSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { inside, runTool, ToolError } from "../src/tools.ts";
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

test("paths inside the project are allowed", () => {
  const { project } = fixture();
  assert.match(inside(project, "src/main.ts"), /main\.ts$/);
  assert.match(inside(project, "src/../README.md"), /README\.md$/, ".. that stays inside is fine");
});

test("every way out of the project is refused", () => {
  const { top, project } = fixture();
  const attempts = ["../private/secret.txt", join(top, "private", "secret.txt"), "docs/secret.txt", "docs", "C:\\Windows\\win.ini"];
  for (const path of attempts) assert.throws(() => inside(project, path), ToolError, path);
  assert.throws(() => inside(project, "no-such-file.txt"), /doesn't exist/);
});

test("read_file numbers the lines and reads in parts", () => {
  const { project } = fixture();
  const part = runTool(project, call("read_file", { path: "src/main.ts", offset: 4, limit: 3 }));
  assert.equal(part, "    4\tline 4\n    5\tline 5\n    6\tline 6\n(lines 4-6 of 10; read on with offset 7)");
  assert.match(runTool(project, call("read_file", { path: "logo.png" })), /isn't a text file/);
  assert.match(runTool(project, call("read_file", { path: "src" })), /is a folder/);
});

test("list_files marks folders and links", () => {
  const { project } = fixture();
  assert.equal(runTool(project, call("list_files", {})), "docs (link)\nlogo.png\nnode_modules/\nREADME.md\nsrc/");
});

test("search finds text in any case, skips dependencies, and never follows a link out", () => {
  const { project } = fixture();
  assert.equal(runTool(project, call("search", { text: "BLUE HERON" })), "README.md:2: The codename is Blue Heron.");
});

test("mistakes go back to the model as text, so it can correct itself", () => {
  const { project } = fixture();
  assert.match(runTool(project, { name: "read_file", arguments: "{not json" }), /^Error: the arguments weren't valid JSON/);
  assert.match(runTool(project, call("delete_everything", {})), /^Error: there's no tool called delete_everything/);
  assert.match(runTool(project, call("read_file", { path: "../private/secret.txt" })), /^Error: .*outside the project folder/);
});

test("a huge result is cut, to protect the context window", () => {
  const { project } = fixture();
  writeFileSync(join(project, "big.txt"), Array.from({ length: 400 }, () => "x".repeat(100)).join("\n"));
  assert.match(runTool(project, call("read_file", { path: "big.txt" })), /… \(cut: \d+ more characters\)$/);
});
