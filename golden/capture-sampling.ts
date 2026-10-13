// Records how llama.cpp samples SmolLM2-135M, so Squid's sampler (src/sampler.ts) and its own runtime
// (src/runtime.ts) can be checked against it.
//   npm run capture:sampling
// llama-server runs on the CPU with CUDA hidden from it, so it never touches the GPU or your models.
//   raw:       llama.cpp's top 50 scores for each prompt, before any sampling
//   chains:    the candidates left after the sampling chain, with their chances, for several settings
//   seeded:    24 tokens with a fixed seed, and the candidate list at every step
//   templates: conversations rendered by the model's chat template (/apply-template)
//   chats:     whole chat replies, greedy and seeded
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encode, loadVocab } from "../src/tokenizer.ts";
import { SMOLLM } from "./capture-forward.ts";

const SERVER = fileURLToPath(new URL("../runtime/llama.cpp/llama-server.exe", import.meta.url));
const PROMPTS = ["The capital of France is", "Once upon a time, there was a", "def fibonacci(n):"];
// Each one changes a single setting from llama-server's defaults (0.8, 40, 0.95, 0.05).
const SETTINGS = [{}, { temperature: 0.3 }, { temperature: 1.5 }, { top_k: 3 }, { top_p: 0.5 }, { min_p: 0.2 }, { temperature: 0 }];
const SEEDS = [1, 42, 1234];
export const CONVERSATIONS = [
  [{ role: "user", content: "What is the capital of France?" }],
  [{ role: "system", content: "You answer in one short sentence." }, { role: "user", content: "Why is the sky blue?" }],
  [
    { role: "user", content: "Name a colour." },
    { role: "assistant", content: "Blue." },
    { role: "user", content: "Another one?" },
  ],
];

if (import.meta.main) {
  const port = 18084;
  const key = randomBytes(16).toString("hex");
  const server = spawn(SERVER, ["-m", SMOLLM, "-c", "1024", "-ngl", "0", "--port", String(port), "--api-key", key, "--no-webui"], {
    env: { ...process.env, CUDA_VISIBLE_DEVICES: "-1" },
    stdio: "ignore",
  });
  try {
    const base = `http://127.0.0.1:${port}`;
    for (let i = 0; i < 120 && !(await fetch(`${base}/health`).then((r) => r.ok, () => false)); i++) await new Promise((r) => setTimeout(r, 500));
    const post = async (path: string, body: object) =>
      (await fetch(`${base}${path}`, { method: "POST", headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` }, body: JSON.stringify(body) })).json();
    const vocab = loadVocab(SMOLLM);
    const list = (step: any) => step.top_probs.map((t: any) => ({ id: t.id, p: t.prob }));

    // n_probs 50 is more than top-k's 40, so the whole candidate list comes back. `raw` is llama.cpp's
    // own top 50 before any sampling, so Squid's chain can be run on exactly the same scores.
    const chains = [];
    const raw: Record<string, { id: number; logprob: number }[]> = {};
    for (const text of PROMPTS) {
      const scores = await post("/completion", { prompt: encode(vocab, text), n_predict: 1, n_probs: 50, temperature: 0, cache_prompt: false });
      raw[text] = scores.completion_probabilities[0].top_logprobs.map((t: any) => ({ id: t.id, logprob: t.logprob }));
      for (const settings of SETTINGS) {
        const reply = await post("/completion", { prompt: encode(vocab, text), n_predict: 1, n_probs: 50, post_sampling_probs: true, seed: 1, cache_prompt: false, ...settings });
        chains.push({ text, settings, candidates: list(reply.completion_probabilities[0]) });
      }
    }

    const seeded = [];
    for (const text of PROMPTS) {
      for (const seed of SEEDS) {
        const promptIds = encode(vocab, text);
        const reply = await post("/completion", { prompt: promptIds, n_predict: 24, n_probs: 50, post_sampling_probs: true, return_tokens: true, seed, cache_prompt: false });
        seeded.push({ text, seed, promptIds, tokens: reply.tokens, steps: reply.completion_probabilities.map(list) });
        console.log(`seed ${seed}: ${JSON.stringify(text + reply.content)}`);
      }
    }

    const templates = [];
    for (const messages of CONVERSATIONS) templates.push({ messages, prompt: (await post("/apply-template", { messages })).prompt });

    const chats = [];
    for (const messages of CONVERSATIONS) {
      for (const settings of [{ temperature: 0 }, { seed: 42 }]) {
        const reply = await post("/v1/chat/completions", { messages, max_tokens: 40, cache_prompt: false, ...settings });
        chats.push({ messages, settings, content: reply.choices[0].message.content, finish_reason: reply.choices[0].finish_reason, usage: reply.usage });
        console.log(`chat ${JSON.stringify(settings)}: ${JSON.stringify(reply.choices[0].message.content)}`);
      }
    }

    const golden = { model: "SmolLM2-135M-Instruct-Q8_0.gguf", engine: "llama.cpp b11580 llama-server, CPU", raw, chains, seeded, templates, chats };
    writeFileSync(fileURLToPath(new URL("./llamacpp/sampling-smollm2.json", import.meta.url)), `${JSON.stringify(golden, null, 1)}\n`);
  } finally {
    server.kill();
  }
}
