// Records what llama.cpp's own engine predicts for SmolLM2-135M, so Squid's TypeScript transformer
// (src/transformer.ts) can be checked against it, number for number.
//   npm run capture:forward
// llama-server runs on the CPU with CUDA hidden from it, so it never touches the GPU or your models.
// For each prompt: 12 tokens, always the most likely one (temperature 0), with the top 5 candidates and
// their log-probabilities at every step.
import { spawn } from "node:child_process";
import { randomBytes } from "node:crypto";
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { encode, loadVocab } from "../src/tokenizer.ts";

export const SMOLLM = "D:\\LMStudio\\models\\bartowski\\SmolLM2-135M-Instruct-GGUF\\SmolLM2-135M-Instruct-Q8_0.gguf";
const SERVER = fileURLToPath(new URL("../runtime/llama.cpp/llama-server.exe", import.meta.url));
const PROMPTS = [
  "The capital of France is",
  "def fibonacci(n):",
  "Once upon a time, there was a",
  "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n",
];
const STEPS = 12;

if (import.meta.main) {
  const port = 18083;
  const key = randomBytes(16).toString("hex");
  const server = spawn(SERVER, ["-m", SMOLLM, "-c", "512", "-ngl", "0", "--port", String(port), "--api-key", key, "--no-webui"], {
    env: { ...process.env, CUDA_VISIBLE_DEVICES: "-1" },
    stdio: "ignore",
  });
  try {
    const base = `http://127.0.0.1:${port}`;
    for (let i = 0; i < 120 && !(await fetch(`${base}/health`).then((r) => r.ok, () => false)); i++) await new Promise((r) => setTimeout(r, 500));
    const vocab = loadVocab(SMOLLM); // step 15's tokenizer, identical to llama.cpp's
    const cases = [];
    for (const text of PROMPTS) {
      const promptIds = encode(vocab, text);
      const res = await fetch(`${base}/completion`, {
        method: "POST",
        headers: { "Content-Type": "application/json", Authorization: `Bearer ${key}` },
        body: JSON.stringify({ prompt: promptIds, n_predict: STEPS, temperature: 0, n_probs: 5, return_tokens: true, cache_prompt: false }),
      });
      const reply = await res.json();
      cases.push({
        text,
        promptIds,
        generated: reply.tokens,
        steps: reply.completion_probabilities.map((step: any) => step.top_logprobs.map((t: any) => ({ id: t.id, logprob: t.logprob }))),
      });
      console.log(`${JSON.stringify(text)} → ${JSON.stringify(reply.content)}`);
    }
    const golden = { model: "SmolLM2-135M-Instruct-Q8_0.gguf", engine: "llama.cpp b11580 llama-server, CPU", cases };
    writeFileSync(fileURLToPath(new URL("./llamacpp/forward-smollm2.json", import.meta.url)), `${JSON.stringify(golden, null, 1)}\n`);
  } finally {
    server.kill();
  }
}
