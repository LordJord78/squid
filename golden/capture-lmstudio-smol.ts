// Records the real LM Studio chatting with SmolLM2-135M: the same conversations llama.cpp answered in
// capture-sampling.ts, for the side by side with Squid's own runtime. LM Studio must be serving it:
//   lms server start; lms load smollm2-135m-instruct; npm run capture:lmstudio-smol; lms unload --all; lms server stop
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { CONVERSATIONS } from "./capture-sampling.ts";

// Greedy, with LM Studio's own settings and with no repeat penalty; a seed with LM Studio's own settings;
// and a seed with llama-server's defaults spelled out.
const SETTINGS = [{ temperature: 0 }, { temperature: 0, repeat_penalty: 1 }, { seed: 42 }, { seed: 42, temperature: 0.8, top_k: 40, top_p: 0.95, min_p: 0.05, repeat_penalty: 1 }];

const chats = [];
for (const messages of CONVERSATIONS) {
  for (const settings of SETTINGS) {
    const res = await fetch("http://127.0.0.1:1234/api/v0/chat/completions", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model: "smollm2-135m-instruct", messages, max_tokens: 40, ...settings }),
    });
    const reply = await res.json();
    const { content } = reply.choices[0].message;
    chats.push({ messages, settings, content, finish_reason: reply.choices[0].finish_reason, usage: reply.usage, stats: reply.stats, runtime: reply.runtime });
    console.log(`${JSON.stringify(settings)}: ${JSON.stringify(content)}`);
  }
}
const golden = { model: "smollm2-135m-instruct (SmolLM2-135M-Instruct-Q8_0.gguf)", app: "LM Studio 0.4.23", chats };
writeFileSync(fileURLToPath(new URL("./lmstudio/chat-smollm2.json", import.meta.url)), `${JSON.stringify(golden, null, 1)}\n`);
