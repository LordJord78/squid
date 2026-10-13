// Records how llama.cpp's own tokenizer (llama-tokenize, from the pinned engine) splits a set of texts,
// for two vocabularies, so Squid's tokenizer (src/tokenizer.ts) can be checked against it.
//   npm run capture:tokens
// It reads only the vocabulary from each model file, on the CPU (CUDA is hidden from it): nothing is
// loaded on the GPU, so it's safe while your models are running.
import { execFileSync } from "node:child_process";
import { mkdirSync, mkdtempSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { basename, join } from "node:path";
import { fileURLToPath } from "node:url";

const TOOL = fileURLToPath(new URL("../runtime/llama.cpp/llama-tokenize.exe", import.meta.url));
export const VOCABS: Record<string, string> = {
  "smollm2-135m": "D:\\LMStudio\\models\\bartowski\\SmolLM2-135M-Instruct-GGUF\\SmolLM2-135M-Instruct-Q8_0.gguf",
  "qwen3.5": "D:\\LMStudio\\models\\lmstudio-community\\Qwen3.5-9B-GGUF\\Qwen3.5-9B-Q4_K_M.gguf",
};

// Texts chosen to poke at every rule: spaces, line ends, digits, contractions, code, other scripts,
// combining marks, emoji, special tokens, odd whitespace. `special: false` = markers as plain text.
const CORPUS: { text: string; special?: false }[] = [
  { text: "Hello world" },
  { text: "The quick brown fox jumps over the lazy dog." },
  { text: "  two leading spaces" },
  { text: "trailing spaces   " },
  { text: "a  b   c    d" },
  { text: "line one\nline two\n\n\nline five" },
  { text: "Windows line\r\nending\r\n" },
  { text: "\tindented with a tab\n\t\tand two" },
  { text: "1234567890" },
  { text: "3.14159 and 2,718 and -42" },
  { text: "x=42; y = 1,000,000;" },
  { text: "I'm sure you're right: they'll see it's John's, we've, I'd" },
  { text: "IT'S LOUD, DON'T SHOUT" },
  { text: "don't'n'stuff ''quoted''" },
  { text: "function greet(name) {\n  return 'Hello, ' + name;\n}\nmodule.exports = greet;" },
  { text: "def main():\n    for i in range(10):\n        print(f\"{i:>3}\")\n" },
  { text: '{"model": "qwen3.8-27b-text", "temperature": 0.7, "stream": true}' },
  { text: "https://example.com/path/to?q=1&r=two#frag" },
  { text: "C:\\Users\\Jorda\\learning\\squid\\src\\tokenizer.ts" },
  { text: "....!!!???---___***" },
  { text: "## Title\n- item with **bold** and `code`\n> quote" },
  { text: "ES futures closed at 5,432.25 (+0.8%) on Friday." },
  { text: "supercalifragilisticexpialidocious antidisestablishmentarianism" },
  { text: "café naïve résumé déjà vu" },
  { text: "cafe\u0301 (an accent written as its own mark) vs café" },
  { text: "Привет, мир! Как дела?" },
  { text: "你好，世界！今天天气很好。" },
  { text: "日本語のテキストとカタカナ" },
  { text: "مرحبا بالعالم" },
  { text: "नमस्ते दुनिया" },
  { text: "😀🚀👍🏽 emoji and 🇯🇵 a flag" },
  { text: "Mixed: Hello世界123abc!?" },
  { text: "a\u00a0b (a non-breaking space)" },
  { text: "zero\u200bwidth space" },
  { text: "next\u0085line" },
  { text: "    " },
  { text: "\n\n" },
  { text: "" },
  { text: "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n" },
  { text: "<|im_start|>user\nWhat is 2+2?<|im_end|>\n<|im_start|>assistant\n", special: false },
  { text: "<think>\nLet me think.\n</think>\n\nThe answer is 4." },
  { text: "<think>\nLet me think.\n</think>\n\nThe answer is 4.", special: false },
  { text: "<|endoftext|>" },
  {
    text:
      "Squid is a small rebuild of two programs, made to learn how they work. It reads GGUF model files, " +
      "runs them on the GPU with llama.cpp, and serves them through an OpenAI-style API on 127.0.0.1. " +
      "Its agent can read and edit the folder you start it in, but only through a permission gate with " +
      "manual, auto and plan modes. Tokens are what the model actually reads: about 4 characters each in English.",
  },
];

if (import.meta.main) {
  const dir = mkdtempSync(join(tmpdir(), "squid-tokens-"));
  const out = fileURLToPath(new URL("./llamacpp/", import.meta.url));
  mkdirSync(out, { recursive: true });
  for (const [name, model] of Object.entries(VOCABS)) {
    const cases = CORPUS.map(({ text, special }, i) => {
      const file = join(dir, `case${i}.txt`);
      writeFileSync(file, text, "utf8"); // a file, not an argument, so every byte arrives exactly
      // --no-escape: by default llama-tokenize turns a typed \n or \t into a real newline or tab, which
      // would change texts like C:\new\tab before tokenizing them.
      const args = ["-m", model, "-f", file, "--ids", "--no-escape", "--log-disable", ...(special === false ? ["--no-parse-special"] : [])];
      const ids = JSON.parse(execFileSync(TOOL, args, { encoding: "utf8", env: { ...process.env, CUDA_VISIBLE_DEVICES: "-1" } }).trim());
      return { text, ...(special === false && { special }), ids };
    });
    const golden = { model: basename(model), engine: "llama.cpp b11580 llama-tokenize", cases };
    writeFileSync(join(out, `tokens-${name}.json`), `${JSON.stringify(golden, null, 1)}\n`);
    console.log(`${name}: ${cases.length} texts, ${cases.reduce((n, c) => n + c.ids.length, 0)} tokens`);
  }
}
