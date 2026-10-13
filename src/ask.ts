// `squid ask "…"`: one question, with the answer printed as it's written, like `lms chat -p`.
// Since step 08 the model can look around the folder first (see agent.ts).
import { ensureServer } from "./daemon.ts";
import { agentTurn, startConversation } from "./agent.ts";
import { dim } from "./reply.ts";

export type AskOptions = { model?: string; system?: string; stats?: boolean };

// With no model named, the server uses the one loaded, or loads your default model.
export async function ask(prompt: string, options: AskOptions): Promise<void> {
  const base = `http://127.0.0.1:${(await ensureServer()).port}`; // starts it in the background if needed
  // Text piped in (`cat notes.md | squid ask "Summarise this"`) comes along with the question, as
  // `lms chat -p` documents.
  const piped = process.stdin.isTTY ? "" : (await readStdin()).trim();
  const messages = startConversation(process.cwd(), options.system);
  messages.push({ role: "user", content: piped ? `${prompt}\n\n${piped}` : prompt });
  const { reply } = await agentTurn(base, options.model, messages, process.cwd());
  // Like `lms chat --stats`, measured here at the receiving end.
  if (options.stats && reply.usage && reply.tokensPerSecond !== undefined) {
    const thinking = reply.usage.completion_tokens_details?.reasoning_tokens ?? 0;
    process.stderr.write(dim(`${reply.tokensPerSecond.toFixed(1)} tok/s · ${reply.usage.completion_tokens} tokens (${thinking} thinking) · first word after ${reply.secondsToFirstWord!.toFixed(2)} s\n`));
  }
}

async function readStdin(): Promise<string> {
  process.stdin.setEncoding("utf8"); // decode the stream as a whole, so no character is cut in two (step 05)
  let text = "";
  for await (const chunk of process.stdin) text += chunk;
  return text;
}
