import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync, readFileSync, writeFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { applyUndo, planUndo, remember, startCheckpoint, undoNote, withoutNote, wrote, type Checkpoint } from "../src/undo.ts";
import { newSession, runTool, writeAtomically } from "../src/tools.ts";
import { ENGINE, unload } from "../src/engine.ts";
import { serverStatus, stopServer } from "../src/daemon.ts";
import { tempDir } from "./helpers.ts";

// Simulates the agent writing a file, the way edit_file and write_file record it.
function agentWrites(checkpoints: Checkpoint[], file: string, text: string) {
  remember(checkpoints, file);
  writeFileSync(file, text);
  wrote(checkpoints, file, text);
}

test("undo puts a file back as it was before the turn, however many times the turn changed it", () => {
  const file = join(tempDir(), "a.txt");
  writeFileSync(file, "original");
  const checkpoints: Checkpoint[] = [];
  startCheckpoint(checkpoints, "make it better");
  agentWrites(checkpoints, file, "draft 1");
  agentWrites(checkpoints, file, "draft 2");
  const plan = planUndo(checkpoints)!;
  assert.equal(plan.checkpoint.label, "make it better");
  applyUndo(checkpoints, plan, writeFileSync);
  assert.equal(readFileSync(file, "utf8"), "original", "not draft 1");
});

test("a file the agent created is deleted again", () => {
  const file = join(tempDir(), "new.txt");
  const checkpoints: Checkpoint[] = [];
  startCheckpoint(checkpoints, "add notes");
  agentWrites(checkpoints, file, "hello");
  applyUndo(checkpoints, planUndo(checkpoints)!, writeFileSync);
  assert.equal(existsSync(file), false);
});

test("undo never overwrites a change it didn't make", () => {
  const file = join(tempDir(), "a.txt");
  writeFileSync(file, "original");
  const checkpoints: Checkpoint[] = [];
  startCheckpoint(checkpoints, "edit");
  agentWrites(checkpoints, file, "agent's version");
  writeFileSync(file, "your version, saved afterwards");
  const plan = planUndo(checkpoints)!;
  assert.deepEqual([plan.restore.length, plan.skipped.length], [0, 1]);
  applyUndo(checkpoints, plan, writeFileSync);
  assert.equal(readFileSync(file, "utf8"), "your version, saved afterwards");
});

test("undo goes back turn by turn, skipping turns that changed nothing; then there's nothing left", () => {
  const dir = tempDir();
  const checkpoints: Checkpoint[] = [];
  startCheckpoint(checkpoints, "turn 1");
  agentWrites(checkpoints, join(dir, "one.txt"), "1");
  startCheckpoint(checkpoints, "turn 2 (only a question)");
  startCheckpoint(checkpoints, "turn 3");
  agentWrites(checkpoints, join(dir, "three.txt"), "3");
  assert.equal(planUndo(checkpoints)!.checkpoint.label, "turn 3");
  applyUndo(checkpoints, planUndo(checkpoints)!, writeFileSync);
  assert.equal(planUndo(checkpoints)!.checkpoint.label, "turn 1", "turn 2 changed nothing");
  applyUndo(checkpoints, planUndo(checkpoints)!, writeFileSync);
  assert.equal(planUndo(checkpoints), undefined);
});

test("the note the model gets after an undo doesn't become the next checkpoint's name", () => {
  const note = undoNote("add notes", ["notes.txt", "src/a.ts"]);
  assert.equal(note, `(Squid: the user undid your changes from "add notes". These files are back as they were before: notes.txt, src/a.ts.)`);
  assert.equal(withoutNote(`${note}\n\nnow add a README`), "now add a README");
  assert.equal(withoutNote("(an aside.)\n\nhello"), "(an aside.)\n\nhello", "your own brackets are kept");
});

test("only the latest 100 checkpoints are kept", () => {
  const checkpoints: Checkpoint[] = [];
  for (let i = 1; i <= 105; i++) startCheckpoint(checkpoints, `turn ${i}`);
  assert.equal(checkpoints.length, 100);
  assert.equal(checkpoints[0].label, "turn 6");
});

test("the tools record what they change; after an undo, the model must read the file again", async () => {
  const root = tempDir();
  writeFileSync(join(root, "a.txt"), "one\n");
  const session = newSession(root, undefined, undefined, { mode: "auto" });
  const call = (name: string, args: object) => runTool(session, { name, arguments: JSON.stringify(args) });
  startCheckpoint(session.checkpoints, "change a, create b");
  await call("read_file", { path: "a.txt" });
  await call("edit_file", { path: "a.txt", old_text: "one", new_text: "two" });
  await call("write_file", { path: "b.txt", content: "new" });
  const plan = planUndo(session.checkpoints)!;
  assert.deepEqual(plan.restore.map((s) => [s.file.endsWith("a.txt") ? "a" : "b", s.before]), [["a", "one\n"], ["b", null]]);
  applyUndo(session.checkpoints, plan, writeAtomically);
  assert.equal(readFileSync(join(root, "a.txt"), "utf8"), "one\n");
  assert.equal(existsSync(join(root, "b.txt")), false);
  assert.match(await call("edit_file", { path: "a.txt", old_text: "one", new_text: "three" }), /has changed since you read it/);
});

// The real 9B in a chat, in auto mode (so its write goes ahead), then /undo.
const ready = existsSync(ENGINE) && existsSync("D:\\LMStudio\\models");
test("in a real chat, /undo removes the file the model just created", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  if (!(await serverStatus())) t.after(() => stopServer());
  t.after(() => unload("qwen/qwen3.5-9b"));
  const root = tempDir();
  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  const lines = ["Create a file named notes.txt containing the word hello.", "/undo", "/exit"];
  const run = spawnSync(process.execPath, [cli, "chat", "--model", "qwen/qwen3.5-9b", "--mode", "auto"], { cwd: root, input: lines.join("\n") + "\n", encoding: "utf8", timeout: 280_000 });
  assert.match(run.stderr, /Create notes\.txt \(auto mode/, "the model created it");
  assert.match(run.stderr, /notes\.txt: delete it \(the agent created it\)/);
  assert.match(run.stderr, /Undid 1 file/);
  assert.equal(existsSync(join(root, "notes.txt")), false);
});
