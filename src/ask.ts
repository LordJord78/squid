// `squid ask "…"`: one question, with the answer printed as it's written, like `lms chat -p`.
// Since step 08 the model can look around the folder first (see agent.ts).
import { ensureServer } from "./daemon.ts";
import { agentTurn, startConversation } from "./agent.ts";
import { createInterface } from "node:readline";
import { dim } from "./reply.ts";
import { newSession, nobodyToAsk } from "./tools.ts";
import { keyboardApprover, keyboardPlanChooser, questionOn } from "./approve.ts";
import type { Mode } from "./rules.ts";

export type AskOptions = { model?: string; system?: string; stats?: boolean; mode?: Mode };

// With no model named, the server uses the one loaded, or loads your default model.
export async function ask(prompt: string, options: AskOptions): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`; // starts it in the background if needed
  // Text piped in (`cat notes.md | squid ask "Summarise this"`) comes along with the question, as
  // `lms chat -p` documents.
  const piped = process.stdin.isTTY ? "" : (await readStdin()).trim();
  const messages = startConversation(process.cwd(), options.system);
  messages.push({ role: "user", content: piped ? `${prompt}\n\n${piped}` : prompt });
  const keyboard = process.stdin.isTTY;
  const session = newSession(process.cwd(), keyboard ? keyboardApprover(askOnce) : nobodyToAsk, (text) => process.stderr.write(text), {
    mode: options.mode ?? "manual",
    choosePlan: keyboard ? keyboardPlanChooser(askOnce) : undefined,
  });
  const { reply } = await agentTurn(base, options.model, messages, session);
  // Like `lms chat --stats`, measured here at the receiving end.
  if (options.stats && reply.usage && reply.tokensPerSecond !== undefined) {
    const thinking = reply.usage.completion_tokens_details?.reasoning_tokens ?? 0;
    process.stderr.write(dim(`${reply.tokensPerSecond.toFixed(1)} tok/s · ${reply.usage.completion_tokens} tokens (${thinking} thinking) · first word after ${reply.secondsToFirstWord!.toFixed(2)} s\n`));
  }
}

// One approval question at the keyboard. The prompt goes to stderr, so `> file` still saves only the
// answer. Ctrl+C closes the question, which counts as "no".
async function askOnce(prompt: string): Promise<string | undefined> {
  const rl = createInterface({ input: process.stdin, output: process.stderr });
  rl.on("SIGINT", () => rl.close());
  try {
    return await questionOn(rl, () => {})(prompt);
  } finally {
    rl.close();
  }
}

async function readStdin(): Promise<string> {
  process.stdin.setEncoding("utf8"); // decode the stream as a whole, so no character is cut in two (step 05)
  let text = "";
  for await (const chunk of process.stdin) text += chunk;
  return text;
}
