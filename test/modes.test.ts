import { test } from "node:test";
import assert from "node:assert/strict";
import { writeFileSync } from "node:fs";
import { join } from "node:path";
import { createInterface } from "node:readline";
import { PassThrough } from "node:stream";
import { decide, parseRule, type Request, type Rulebook } from "../src/rules.ts";
import { newSession, runTool, toolsFor, type Answer, type Approval, type PlanChoice } from "../src/tools.ts";
import { nextMode, promptFor } from "../src/chat.ts";
import { keyboardApprover, keyboardPlanChooser, questionOn, type OpenQuestion } from "../src/approve.ts";
import { tempDir } from "./helpers.ts";

const book = (allow: string[] = []): Rulebook => ({ deny: [], ask: [], allow: allow.map(parseRule), root: "C:\\proj", home: "C:\\Users\\alice" });
const edit = (path: string): Request => ({ tool: "Edit", path: `C:\\proj\\${path}`, real: `C:\\proj\\${path}` });
const command = (text: string, readOnly = false): Request => ({ tool: "Bash", command: text, readOnly });

test("each mode's defaults: manual asks, auto lets edits through, plan blocks them", () => {
  const outcomes = (mode: "manual" | "auto" | "plan") =>
    [decide(edit("a.ts"), book(), mode), decide(command("git status", true), book(), mode), decide(command("npm test"), book(), mode)].map((d) => d.outcome);
  assert.deepEqual(outcomes("manual"), ["ask", "allow", "ask"]);
  assert.deepEqual(outcomes("auto"), ["allow", "allow", "ask"], "auto = Claude Code's acceptEdits: edits only");
  assert.deepEqual(outcomes("plan"), ["deny", "allow", "ask"]);
});

test("auto mode doesn't open protected paths; plan mode can't be unlocked by an allow rule", () => {
  assert.equal(decide(edit(".git/config"), book(), "auto").outcome, "ask");
  assert.equal(decide(edit("a.ts"), book(["Edit"]), "plan").outcome, "deny");
  assert.equal(decide(edit(".env"), book(["Edit"]), "auto").outcome, "deny", "hard rules hold in every mode");
});

test("Shift+Tab cycles manual → auto → plan → manual, and the prompt always shows the mode", () => {
  assert.deepEqual([nextMode("manual"), nextMode("auto"), nextMode("plan")], ["auto", "plan", "manual"]);
  const plain = (text: string) => text.replace(/\x1b\[\d+m/g, "");
  assert.deepEqual([plain(promptFor("manual")), plain(promptFor("auto")), plain(promptFor("plan"))], ["› ", "⏵⏵ auto › ", "⏸ plan › "]);
});

test("at a question, Shift+Tab gives Claude Code's answer: an edit's yes-and-auto, a plan's go-ahead-in-auto; a command has none", async () => {
  const asked: { prompt: string; shiftTab?: string }[] = [];
  const question = async (prompt: string, shiftTab?: string) => (asked.push({ prompt, shiftTab }), "n");
  const approve = keyboardApprover(question);
  await approve({ title: "Edit a.txt", reason: "edits ask", preview: "", always: "yes, and switch to auto mode", switchesToAuto: true });
  await approve({ title: "Run npm test", reason: "commands ask", preview: "", always: 'yes, and don\'t ask again for exactly "npm test" this session' });
  await keyboardPlanChooser(question)("1. do it");
  assert.deepEqual(asked.map((a) => a.shiftTab), ["a", undefined, "1"]);
  assert.match(asked[0].prompt, /Shift\+Tab/);
  assert.doesNotMatch(asked[1].prompt, /Shift\+Tab/);
  assert.match(asked[2].prompt, /Shift\+Tab/);
});

test("a question answered by Shift+Tab is closed: the next line you type goes to the chat", async () => {
  const input = new PassThrough();
  const rl = createInterface({ input, output: new PassThrough(), terminal: false });
  let open: OpenQuestion | undefined;
  const question = questionOn(rl, (o) => (open = o));
  const current = () => open; // read fresh each time: the question sets it
  const answer = question("Allow? ", "a");
  assert.ok(current()?.shiftTab, "this question has a Shift+Tab answer");
  current()!.shiftTab!();
  assert.equal(await answer, "a");
  assert.equal(current(), undefined, "no question is open any more");
  const line = new Promise((resolve) => rl.once("line", resolve));
  input.write("hello\n");
  assert.equal(await line, "hello");
  void question("Allow? "); // and one without a Shift+Tab answer has none to give
  assert.equal(current()?.shiftTab, undefined);
  rl.close();
});

function approver(...answers: Answer[]) {
  const asked: Approval[] = [];
  return { asked, approve: async (request: Approval) => (asked.push(request), answers.shift() ?? { ok: false }) };
}
const call = (name: string, args: object) => ({ name, arguments: JSON.stringify(args) });

test("in auto mode an edit goes ahead without a question, and is still shown", async () => {
  const root = tempDir();
  writeFileSync(join(root, "a.txt"), "one\n");
  const you = approver();
  const shown: string[] = [];
  const session = newSession(root, you.approve, (text) => shown.push(text), { mode: "auto" });
  await runTool(session, call("read_file", { path: "a.txt" }));
  assert.equal(await runTool(session, call("edit_file", { path: "a.txt", old_text: "one", new_text: "two" })), "Edited a.txt: 1 replacement.");
  assert.equal(you.asked.length, 0);
  assert.match(shown[0], /Edit a\.txt \(auto mode: edits inside this folder go ahead\)/);
});

test('"always" on an edit switches to auto mode; on a command, remembers exactly that command', async () => {
  const root = tempDir();
  writeFileSync(join(root, "a.txt"), "one\n");
  const you = approver({ ok: true, always: true }, { ok: true, always: true }, { ok: false });
  const session = newSession(root, you.approve);
  await runTool(session, call("read_file", { path: "a.txt" }));
  await runTool(session, call("edit_file", { path: "a.txt", old_text: "one", new_text: "two" }));
  assert.equal(session.mode, "auto");
  assert.match(you.asked[0].always!, /switch to auto mode/);
  assert.equal(you.asked[0].switchesToAuto, true, "so Shift+Tab gives that answer");
  await runTool(session, call("edit_file", { path: "a.txt", old_text: "two", new_text: "three" }));
  assert.equal(you.asked.length, 1, "no question for the second edit");

  await runTool(session, call("run_command", { command: "node -e 1" })); // asked, answered "always"
  assert.equal(you.asked[1].switchesToAuto, undefined, "a command's always doesn't change the mode");
  await runTool(session, call("run_command", { command: "node -e 1" })); // remembered: not asked
  assert.equal(you.asked.length, 2);
  assert.match(await runTool(session, call("run_command", { command: "node -e 2" })), /declined/, "a different command asks again");
  assert.equal(you.asked.length, 3);
});

test('"always" is never offered for a shell', async () => {
  const you = approver({ ok: false });
  await runTool(newSession(tempDir(), you.approve), call("run_command", { command: process.platform === "win32" ? "cmd /c echo hi" : "sh -c 'echo hi'" }));
  assert.equal(you.asked[0].always, undefined);
});

test("submit_plan: only in plan mode; approved, the session switches to the chosen mode", async () => {
  const choices: PlanChoice[] = [{ choice: "keep", feedback: "also update the README" }, { choice: "auto" }];
  const plans: string[] = [];
  const session = newSession(tempDir(), undefined, undefined, { mode: "plan", choosePlan: async (plan) => (plans.push(plan), choices.shift()!) });
  assert.ok(toolsFor("plan").some((t) => t.function.name === "submit_plan"));
  assert.ok(!toolsFor("manual").some((t) => t.function.name === "submit_plan"));

  const plan = call("submit_plan", { plan: "1. Fix the typo\n2. Run the tests" });
  assert.equal(await runTool(session, plan), "The user wants you to keep planning: also update the README. Stay in plan mode.");
  assert.equal(session.mode, "plan");
  assert.match(await runTool(session, plan), /approved the plan. You are now in auto mode/);
  assert.equal(session.mode, "auto");
  assert.deepEqual(plans, ["1. Fix the typo\n2. Run the tests", "1. Fix the typo\n2. Run the tests"]);
  assert.match(await runTool(session, plan), /^Error: submit_plan is for plan mode/);
});

test("with nobody at the keyboard, a plan isn't approved and the model is told to answer with it", async () => {
  const session = newSession(tempDir(), undefined, undefined, { mode: "plan" });
  assert.match(await runTool(session, call("submit_plan", { plan: "1. x" })), /give the plan as your final answer/);
  assert.equal(session.mode, "plan");
});
