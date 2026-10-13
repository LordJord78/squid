# Squid

Your local AI, started with one word from any folder.

Squid is a small rebuild of two programs, made to learn how they work:

- **LM Studio's engine room.** It reads GGUF model files, runs them on the GPU with the open-source
  [llama.cpp](https://github.com/ggml-org/llama.cpp) engine, and serves them through an
  OpenAI-style HTTP API on `127.0.0.1`.
- **Claude Code's agent and safety modes.** The model can read and edit the folder you start it in,
  but only through a permission gate with manual, auto and plan modes.

It ends with a tiny engine of our own in TypeScript, so you can see what llama.cpp does inside: a
tokenizer, a transformer and a sampler, which Squid can run as a runtime in place of llama.cpp (CPU, small
Llama-style models such as SmolLM2-135M).

It is built one small step at a time. Every step is one git commit with an explained walkthrough note
in the Obsidian vault (`65 LEARNING/Squid/`).

## Rules this project follows

- **Behaviour, not code.** LM Studio and Claude Code are proprietary, and their terms forbid reverse
  engineering. Everything here is rebuilt from their public docs and from recordings of the real
  LM Studio server (`golden/`). Nothing is decompiled or copied.
- **Local only.** The server binds to `127.0.0.1`. No cloud. Each engine is locked with a random key
  that only Squid knows.
- **No runtime dependencies.** Node 24 runs the `.ts` files directly. `typescript` and `@types/node`
  are dev-only, for type checking.

## Use it

```
npm run engine:get        # once: download llama.cpp (pinned, checksum-verified) into runtime/
npm link                  # once: puts `squid` on your PATH (undo: npm unlink -g squid)

squid                     # start the server in the background and load your default model
squid chat                # the agent in the terminal, working on this folder
squid web                 # the same in your browser (a private link for this folder; Ctrl+C stops it)
squid ask "…"             # one question, streamed
squid tokenize 27b "…"    # the tokens a model reads for some text (from its own vocabulary)
squid next smol "…"       # Squid's own TypeScript engine on the CPU: the next token's chances, then more
squid load smol --runtime squid   # run a model in Squid's own engine; then chat with it as with any model
squid ps                  # what's loaded (models loaded on demand unload after 60 min unused)
squid unload --all        # free the GPU
squid server stop         # stop the background server
```

Settings live in `~/.squid/config.json`, e.g.
`{"defaultModel": "qwen3.8-27b-text", "models": {"qwen3.8-27b-text": {"contextLength": 49152}}}`.
A model entry can also say `"runtime": "squid"`, so it always loads in Squid's own engine.
The OpenAI-style API is at `http://127.0.0.1:1235/v1` (LM Studio uses 1234).

## Develop it

```
npm test                  # every test
npm run check             # type check
npm run capture:lmstudio  # re-record the real LM Studio into golden/lmstudio/
npm run capture:tokens    # re-record llama.cpp's tokenizer into golden/llamacpp/ (CPU only)
npm run capture:forward   # re-record llama.cpp's next-token predictions for SmolLM2 (CPU only)
npm run capture:sampling  # re-record llama.cpp's sampling, chat template and chats for SmolLM2 (CPU only)
npm run capture:lmstudio-smol  # re-record LM Studio chatting with SmolLM2 (load it in LM Studio first)
```

Replay any step with `git log --oneline` and `git show <sha>`.

## Layout

| Path | What it is |
|---|---|
| `src/` | Squid itself (17 steps; `runtime.ts`, `transformer.ts`, `sampler.ts` and `tokenizer.ts` are the engine of its own) |
| `test/` | `node:test` tests, many comparing Squid against `golden/` |
| `golden/lmstudio/` | Recordings of the real LM Studio: what Squid must match |
| `golden/llamacpp/` | Recordings of llama.cpp itself (raw engine replies, its tokenizer, predictions, sampling) |
| `runtime/` | Downloaded llama.cpp engine (not committed) |
