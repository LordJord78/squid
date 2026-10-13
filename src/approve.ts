// Asking you at the keyboard, Claude Code style: what the agent wants to do, the change it would make,
// then yes, no, or type what it should do instead (that text goes back to the model).
import type { Interface } from "node:readline";
import { dim } from "./reply.ts";
import type { Approver, PlanChooser } from "./tools.ts";

type Question = (prompt: string) => Promise<string | undefined>;
const framed = (text: string) => (text ? `\n${text.split("\n").map((line) => `│ ${line}`).join("\n")}` : "");

// `question` shows a prompt and resolves with the line you type, or undefined if there's no answer
// (the input closed, or Ctrl+C).
export function keyboardApprover(question: Question): Approver {
  return async ({ title, reason, preview, always }) => {
    process.stderr.write(`\n┌ ${title}  ${dim(`(${reason})`)}${framed(preview)}\n`);
    if (always) process.stderr.write(dim(`│ a = ${always}\n`));
    const choices = always ? "y = yes · a = always · n = no" : "y = yes · n = no";
    const typed = ((await question(`└ Allow? ${choices} · or type what to do instead › `)) ?? "").trim();
    if (/^(y|yes)$/i.test(typed)) return { ok: true };
    if (always && /^(a|always)$/i.test(typed)) return { ok: true, always: true };
    if (/^(n|no)?$/i.test(typed)) return { ok: false }; // empty counts as no: the safe default
    return { ok: false, feedback: typed };
  };
}

// Your verdict on a plan (step 12), with Claude Code's three choices. Nothing has been changed yet.
export function keyboardPlanChooser(question: Question): PlanChooser {
  return async (plan) => {
    process.stderr.write(`\n┌ Plan  ${dim("(nothing has been changed yet)")}${framed(plan)}\n`);
    const typed = ((await question("└ 1 = go ahead in auto mode · 2 = go ahead, asking before each change · or type what to change › ")) ?? "").trim();
    if (typed === "1") return { choice: "auto" };
    if (typed === "2") return { choice: "manual" };
    return { choice: "keep", feedback: typed && !/^(n|no|3)$/i.test(typed) ? typed : undefined };
  };
}

// A question on an existing readline (the chat's), cancelled by Ctrl+C. Node gives the next typed line
// to a pending question instead of to the chat loop, so the answer doesn't become a chat message.
// `onCancel` receives a cancel function while the question is open, and undefined once it's answered.
export function questionOn(rl: Interface, onCancel: (cancel: (() => void) | undefined) => void): Question {
  return (prompt) =>
    new Promise((resolve) => {
      const stop = new AbortController();
      const done = (answer: string | undefined) => {
        rl.off("close", closed);
        onCancel(undefined);
        resolve(answer);
      };
      const closed = () => done(undefined);
      rl.once("close", closed);
      onCancel(() => stop.abort()); // the chat's Ctrl+C handler calls this while the question is open
      stop.signal.addEventListener("abort", () => done(undefined));
      rl.question(prompt, { signal: stop.signal }, done);
    });
}
