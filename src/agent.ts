// The agent loop, the heart of Claude Code: send the conversation; if the model asks for tools, run
// them and send back what they returned; repeat until the model answers in words.
import { TOOLS, describeCall, runTool, type Session } from "./tools.ts";
import { loadRulebook } from "./rules.ts";
import { dim, streamReply, terminal, type Message, type Output, type Reply } from "./reply.ts";

// A model that keeps calling tools without ever answering is stuck. The AI Lab's benchmark saw the
// 30B model burn all 30 of its turns that way; this stops the loop well before.
export const MAX_ROUNDS = 25;

// The first message of every conversation: where the model is and what it can do.
export function startConversation(root: string, extra?: string): Message[] {
  const prompt = [
    `You are Squid, an assistant running on this computer, working in the folder ${root}.`,
    "You can look at its files with list_files, read_file and search, change them with edit_file and write_file, and run programs with run_command (one at a time: there is no shell, so no &&, | or >). Paths are relative to that folder; nothing outside it can be opened.",
    "Every change and every command that isn't read-only is shown to the user, who approves or declines it. If they decline, don't try the same change again; follow their instructions if they gave any.",
    "Some files are off limits (secrets, holdout data, and anything the user's rules deny). If a tool says so, don't try to get around it.",
    "When a question is about the project, look before you answer, and say which files you read. Be concise.",
  ].join("\n");
  return [{ role: "system", content: extra ? `${prompt}\n\n${extra}` : prompt }];
}

// One turn of the conversation. The model may call tools again and again (up to MAX_ROUNDS) before it
// answers. Every call and result is added to `messages`, so later turns remember what was looked at.
// The session (tools.ts) carries the folder, the rules, who approves changes, and what has been read.
export async function agentTurn(
  base: string,
  model: string | undefined,
  messages: Message[],
  session: Session,
  signal?: AbortSignal,
  out: Output = terminal,
): Promise<{ model: string; reply: Reply }> {
  session.book = loadRulebook(session.root); // rules edited mid-chat apply from the next question (step 09)
  session.signal = signal; // Ctrl+C also stops a command that's running (step 11)
  for (let round = 1; round <= MAX_ROUNDS; round++) {
    const reply = await streamReply(base, { model, messages, tools: TOOLS }, signal, out);
    model = reply.model;
    messages.push({
      role: "assistant",
      content: reply.content,
      // The thinking is sent back while the model is still working through tools, so it can follow
      // its own plan from one call to the next.
      ...(reply.reasoning ? { reasoning_content: reply.reasoning } : {}),
      ...(reply.toolCalls.length
        ? { tool_calls: reply.toolCalls.map((c) => ({ id: c.id, type: "function" as const, function: { name: c.name, arguments: c.arguments } })) }
        : {}),
    });
    if (!reply.toolCalls.length) return { model, reply }; // it answered in words: the turn is over

    for (const call of reply.toolCalls) {
      out.note(`● ${describeCall(call)}\n`);
      const result = await runTool(session, call); // may stop here to ask you (step 10)
      out.note(`${dim(`  ⎿ ${summarise(result)}`)}\n`);
      // A tool result is a message of its own, linked to the call by its id.
      messages.push({ role: "tool", tool_call_id: call.id, content: result });
    }
  }
  throw new Error(`stopped after ${MAX_ROUNDS} rounds of tool calls without an answer`);
}

// Thinking from finished turns isn't needed again, and it takes up context. Dropped before each new question.
export function forgetThinking(messages: Message[]): void {
  for (const message of messages) delete message.reasoning_content;
}

// One line about a tool's result for the screen: "180 lines", or the error.
function summarise(result: string): string {
  if (result.startsWith("Error: ")) return result.slice(7);
  const lines = result.split("\n").length;
  return `${lines} line${lines === 1 ? "" : "s"}`;
}
