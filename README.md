# Squid

Your local AI, started with one word from any folder.

Squid is a small rebuild of two programs, made to learn how they work:

- **LM Studio's engine room.** It reads GGUF model files, runs them on the GPU with the open-source
  [llama.cpp](https://github.com/ggml-org/llama.cpp) engine, and serves them through an
  OpenAI-style HTTP API on `127.0.0.1`.
- **Claude Code's agent and safety modes.** The model can read and edit the folder you start it in,
  but only through a permission gate with manual, auto and plan modes.

It ends with a tiny engine of our own in TypeScript, so you can see what llama.cpp does inside.

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
squid ask "…"             # one question, streamed
squid ps                  # what's loaded (models loaded on demand unload after 60 min unused)
squid unload --all        # free the GPU
squid server stop         # stop the background server
```

Settings live in `~/.squid/config.json`, e.g.
`{"defaultModel": "qwen3.8-27b-text", "models": {"qwen3.8-27b-text": {"contextLength": 49152}}}`.
The OpenAI-style API is at `http://127.0.0.1:1235/v1` (LM Studio uses 1234).

## Develop it

```
npm test                  # every test
npm run check             # type check
npm run capture:lmstudio  # re-record the real LM Studio into golden/lmstudio/
```

Replay any step with `git log --oneline` and `git show <sha>`.

## Layout

| Path | What it is |
|---|---|
| `src/` | Squid itself (filled in step by step) |
| `test/` | `node:test` tests, many comparing Squid against `golden/` |
| `golden/lmstudio/` | Recordings of the real LM Studio: what Squid must match |
| `runtime/` | Downloaded llama.cpp engine (not committed) |
