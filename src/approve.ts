// Asking you at the keyboard, Claude Code style: what the agent wants to do, the change it would make,
// then yes, no, or type what it should do instead (that text goes back to the model).
import type { Interface } from "node:readline";
import { dim } from "./reply.ts";
import type { Answer, Approver, PlanChoice, PlanChooser } from "./tools.ts";

// A question asks for a typed line. Some also have a Shift+Tab answer, as in Claude Code.
type Question = (prompt: string, shiftTab?: string) => Promise<string | undefined>;
const framed = (text: string) => (text ? `\n${text.split("\n").map((line) => `│ ${line}`).join("\n")}` : "");

// What an answer means: y, a (if "always" was offered), n, or anything else = instructions for the model.
// The terminal and the browser page (step 14) both use these, so a "y" means the same in both.
// No answer at all (input closed, Ctrl+C, the browser tab closed) counts as no: the safe default.
export function answerFrom(typed: string | undefined, alwaysOffered: boolean): Answer {
  const text = (typed ?? "").trim();
  if (/^(y|yes)$/i.test(text)) return { ok: true };
  if (alwaysOffered && /^(a|always)$/i.test(text)) return { ok: true, always: true };
  if (/^(n|no)?$/i.test(text)) return { ok: false };
  return { ok: false, feedback: text };
}
export function planChoiceFrom(typed: string | undefined): PlanChoice {
  const text = (typed ?? "").trim();
  if (text === "1") return { choice: "auto" };
  if (text === "2") return { choice: "manual" };
  return { choice: "keep", feedback: text && !/^(n|no|3)$/i.test(text) ? text : undefined };
}

// `question` shows a prompt and resolves with the line you type, or undefined if there's no answer
// (the input closed, or Ctrl+C).
export function keyboardApprover(question: Question): Approver {
  return async ({ title, reason, preview, always, switchesToAuto }) => {
    process.stderr.write(`\n┌ ${title}  ${dim(`(${reason})`)}${framed(preview)}\n`);
    if (always) process.stderr.write(dim(`│ a = ${always}\n`));
    const choices = always ? `y = yes · a${switchesToAuto ? " or Shift+Tab" : ""} = always · n = no` : "y = yes · n = no";
    return answerFrom(await question(`└ Allow? ${choices} · or type what to do instead › `, switchesToAuto ? "a" : undefined), Boolean(always));
  };
}

// Your verdict on a plan (step 12), with Claude Code's three choices. Nothing has been changed yet.
export function keyboardPlanChooser(question: Question): PlanChooser {
  return async (plan) => {
    process.stderr.write(`\n┌ Plan  ${dim("(nothing has been changed yet)")}${framed(plan)}\n`);
    return planChoiceFrom(await question("└ 1 or Shift+Tab = go ahead in auto mode · 2 = go ahead, asking before each change · or type what to change › ", "1"));
  };
}

// A question on an existing readline (the chat's), cancelled by Ctrl+C. Node gives the next typed line
// to a pending question instead of to the chat loop, so the answer doesn't become a chat message.
// `onOpen` receives the open question while there is one (Ctrl+C cancels it; Shift+Tab gives its Shift+Tab
// answer, if it has one), and undefined once it's answered.
export type OpenQuestion = { cancel: () => void; shiftTab?: () => void };
export function questionOn(rl: Interface, onOpen: (open: OpenQuestion | undefined) => void): Question {
  return (prompt, shiftTab) =>
    new Promise((resolve) => {
      const stop = new AbortController();
      let given: string | undefined; // the Shift+Tab answer, once pressed
      const done = (answer: string | undefined) => {
        rl.off("close", closed);
        onOpen(undefined);
        resolve(answer);
      };
      const closed = () => done(undefined);
      rl.once("close", closed);
      // Both end the question through its signal, so Node stops waiting for its line: the next line you
      // type goes to the chat again.
      onOpen({ cancel: () => stop.abort(), shiftTab: shiftTab === undefined ? undefined : () => ((given = shiftTab), stop.abort()) });
      stop.signal.addEventListener("abort", () => done(given));
      rl.question(prompt, { signal: stop.signal }, done);
    });
}
