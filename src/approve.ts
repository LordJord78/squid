// Asking you at the keyboard, Claude Code style: what the agent wants to do, the change it would make,
// then yes, no, or type what it should do instead (that text goes back to the model).
import type { Interface } from "node:readline";
import { dim } from "./reply.ts";
import type { Approver } from "./tools.ts";

// `question` shows a prompt and resolves with the line you type, or undefined if there's no answer
// (the input closed, or Ctrl+C).
export function keyboardApprover(question: (prompt: string) => Promise<string | undefined>): Approver {
  return async ({ title, reason, preview }) => {
    const framed = preview ? `\n${preview.split("\n").map((line) => `│ ${line}`).join("\n")}` : "";
    process.stderr.write(`\n┌ ${title}  ${dim(`(${reason})`)}${framed}\n`);
    const typed = ((await question("└ Allow? y = yes · n = no · or type what to do instead › ")) ?? "").trim();
    if (/^(y|yes)$/i.test(typed)) return { ok: true };
    if (/^(n|no)?$/i.test(typed)) return { ok: false }; // empty counts as no: the safe default
    return { ok: false, feedback: typed };
  };
}

// A question on an existing readline (the chat's), cancelled by Ctrl+C. Node gives the next typed line
// to a pending question instead of to the chat loop, so the answer doesn't become a chat message.
// `onCancel` receives a cancel function while the question is open, and undefined once it's answered.
export function questionOn(rl: Interface, onCancel: (cancel: (() => void) | undefined) => void): (prompt: string) => Promise<string | undefined> {
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
