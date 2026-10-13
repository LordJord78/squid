import { test } from "node:test";
import assert from "node:assert/strict";
import { mkdirSync, readFileSync, symlinkSync, utimesSync, writeFileSync, existsSync } from "node:fs";
import { join } from "node:path";
import { newSession, runTool, type Answer, type Approval, type Session } from "../src/tools.ts";
import { parseRule } from "../src/rules.ts";
import { tempDir } from "./helpers.ts";

// An approver that answers from a script and remembers what it was shown.
function approver(...answers: Answer[]) {
  const asked: Approval[] = [];
  return { asked, approve: async (request: Approval) => (asked.push(request), answers.shift() ?? { ok: false }) };
}
const call = (name: string, args: object) => ({ name, arguments: JSON.stringify(args) });
const plain = (text: string) => text.replace(/\x1b\[\d+m/g, ""); // without colours

function project() {
  const root = tempDir();
  writeFileSync(join(root, "app.ts"), 'const greeting = "Helo";\nconsole.log(greeting);\n');
  return root;
}

test("an edit needs a read first, then shows the change and waits for your yes", async () => {
  const root = project();
  const you = approver({ ok: true });
  const session = newSession(root, you.approve);
  const fix = call("edit_file", { path: "app.ts", old_text: '"Helo"', new_text: '"Hello"' });
  assert.match(await runTool(session, fix), /^Error: read app.ts with read_file before changing it/);
  await runTool(session, call("read_file", { path: "app.ts" }));
  assert.equal(await runTool(session, fix), "Edited app.ts: 1 replacement.");
  assert.equal(you.asked[0].title, "Edit app.ts");
  assert.equal(plain(you.asked[0].preview), '   1 - const greeting = "Helo";\n   1 + const greeting = "Hello";\n   2   console.log(greeting);\n   3   ');
  assert.equal(readFileSync(join(root, "app.ts"), "utf8"), 'const greeting = "Hello";\nconsole.log(greeting);\n');
});

test("no leaves the file alone, and the model hears your instructions", async () => {
  const root = project();
  const session = newSession(root, approver({ ok: false, feedback: 'make it "Hi" instead' }).approve);
  await runTool(session, call("read_file", { path: "app.ts" }));
  const result = await runTool(session, call("edit_file", { path: "app.ts", old_text: "Helo", new_text: "Hello" }));
  assert.equal(result, 'Error: the user declined: Edit app.ts. Their instructions: make it "Hi" instead');
  assert.match(readFileSync(join(root, "app.ts"), "utf8"), /Helo/);
});

test("a file that changed after it was read must be read again", async () => {
  const root = project();
  const session = newSession(root, approver({ ok: true }).approve);
  await runTool(session, call("read_file", { path: "app.ts" }));
  writeFileSync(join(root, "app.ts"), 'const greeting = "Helo"; // you edited this meanwhile\n');
  utimesSync(join(root, "app.ts"), new Date(), new Date(Date.now() + 5000)); // a clearly different modification time
  assert.match(await runTool(session, call("edit_file", { path: "app.ts", old_text: "Helo", new_text: "Hello" })), /has changed since you read it/);
});

test("old_text must match exactly, once (or every time, when asked)", async () => {
  const root = tempDir();
  writeFileSync(join(root, "list.txt"), "apple\nbanana\napple\n");
  const session = newSession(root, approver({ ok: true }, { ok: true }).approve);
  await runTool(session, call("read_file", { path: "list.txt" }));
  assert.match(await runTool(session, call("edit_file", { path: "list.txt", old_text: "cherry", new_text: "x" })), /wasn't found/);
  assert.match(await runTool(session, call("edit_file", { path: "list.txt", old_text: "apple", new_text: "pear" })), /appears 2 times/);
  assert.equal(await runTool(session, call("edit_file", { path: "list.txt", old_text: "apple", new_text: "pear", replace_all: true })), "Edited list.txt: 2 replacements.");
  assert.equal(readFileSync(join(root, "list.txt"), "utf8"), "pear\nbanana\npear\n");
});

test("Windows line endings survive an edit written with plain \\n", async () => {
  const root = tempDir();
  writeFileSync(join(root, "win.txt"), "one\r\ntwo\r\nthree\r\n");
  const session = newSession(root, approver({ ok: true }).approve);
  await runTool(session, call("read_file", { path: "win.txt" }));
  await runTool(session, call("edit_file", { path: "win.txt", old_text: "one\ntwo", new_text: "one\n1.5\ntwo" }));
  assert.equal(readFileSync(join(root, "win.txt"), "utf8"), "one\r\n1.5\r\ntwo\r\nthree\r\n");
});

test('"$&" in the new text is written as typed, not treated as a replace() code', async () => {
  const root = tempDir();
  writeFileSync(join(root, "price.txt"), "price: TBD\n");
  const session = newSession(root, approver({ ok: true }).approve);
  await runTool(session, call("read_file", { path: "price.txt" }));
  await runTool(session, call("edit_file", { path: "price.txt", old_text: "TBD", new_text: "$& → $5" }));
  assert.equal(readFileSync(join(root, "price.txt"), "utf8"), "price: $& → $5\n");
});

test("write_file creates a file (and its folders) after you approve", async () => {
  const root = tempDir();
  const you = approver({ ok: true });
  const result = await runTool(newSession(root, you.approve), call("write_file", { path: "notes/todo.md", content: "- buy milk\n- learn diffs" }));
  assert.equal(result, "Created notes/todo.md (2 lines).");
  assert.equal(you.asked[0].title, "Create notes/todo.md");
  assert.equal(plain(you.asked[0].preview), "   1 + - buy milk\n   2 + - learn diffs");
  assert.equal(readFileSync(join(root, "notes", "todo.md"), "utf8"), "- buy milk\n- learn diffs");
});

test("the rulebook decides before you're asked", async () => {
  const top = tempDir();
  const root = join(top, "project");
  mkdirSync(join(root, "docs"), { recursive: true });
  mkdirSync(join(top, "outside"));
  symlinkSync(join(top, "outside"), join(root, "exit"), "junction");
  const you = approver({ ok: true });
  const shown: string[] = [];
  const session: Session = { ...newSession(root, you.approve, (text) => shown.push(text)) };
  session.book = { ...session.book, allow: [parseRule("Edit(docs/**)"), parseRule("Edit")] };
  const write = (path: string) => runTool(session, call("write_file", { path, content: "x" }));

  assert.match(await write("docs/a.md"), /^Created/);
  assert.equal(you.asked.length, 0, "an allow rule needs no question...");
  assert.match(shown[0], /Create docs\/a\.md \(allowed by rule Edit\(docs\/\*\*\)\)/, "...but the change is still shown");
  assert.match(await write(".vscode/settings.json"), /^Created/);
  assert.equal(you.asked[0].reason, "protected path: changes here always need your OK", "allow rules can't skip a protected path");
  assert.match(await write(".env"), /Squid never opens/);
  assert.match(await write("../escape.txt"), /outside the project folder/);
  assert.match(await write("exit/escape.txt"), /outside the project folder/, "through a junction");
  assert.equal(existsSync(join(top, "outside", "escape.txt")), false);
  assert.equal(you.asked.length, 1, "refusals never reach you");
});

test("with nobody at the keyboard, changes are refused", async () => {
  const root = project();
  const session = newSession(root); // no approver: as when input is piped in
  await runTool(session, call("read_file", { path: "app.ts" }));
  assert.match(await runTool(session, call("edit_file", { path: "app.ts", old_text: "Helo", new_text: "Hello" })), /nobody was at the keyboard/);
});
