#!/usr/bin/env node
// Squid's command line: `squid <command> [arguments]`.
import { parseArgs } from "node:util";
import { readGgufHeader, describeModel } from "./gguf.ts";
import { listModels, type ShelfModel } from "./models.ts";
import { load, loadOnDemand, running, secondsLeft, status, unload, type Engine, type Runtime } from "./engine.ts";
import { startServer, DEFAULT_PORT } from "./server.ts";
import { ensureServer, reachServer, recordServer, serverStatus, stopServer } from "./daemon.ts";
import { closeArgs, ON_THE_PC, remoteSettings, runOnPc, runSsh } from "./remote.ts";
import { readConfig } from "./config.ts";
import { MODES, type Mode } from "./rules.ts";
import { ask } from "./ask.ts";
import { chat } from "./chat.ts";
import { web } from "./web.ts";
import { decode, encode, loadVocab, piece } from "./tokenizer.ts";
import { feed, forward, loadModel, newState, top } from "./transformer.ts";
import { candidates, DEFAULTS, mt19937, pick } from "./sampler.ts";
import { dim } from "./reply.ts";
import { readFileSync } from "node:fs";
import { randomInt } from "node:crypto";

const [command, ...args] = process.argv.slice(2);

const USAGE = `usage:
  squid                     start your local AI (server + default model) and chat with it
  squid chat [--model NAME] [--system TEXT] [--mode manual|auto|plan]
                            a conversation in the terminal (/help inside; Shift+Tab changes mode)
  squid web [--model NAME] [--mode manual|auto|plan] [--port N] [--no-open]
                            the same, in your browser, for the folder you're in (Ctrl+C stops it)
  squid tokenize <model> "<text>" [--ids] [--no-special] [--file PATH]
                            the tokens a model reads for some text (its own vocabulary, no GPU)
  squid next <model> "<text>" [--steps N] [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]
                            Squid's own TypeScript engine (CPU): the next token's chances, then N more
  squid ls                  list your models
  squid info <model.gguf>   what's inside one model file
  squid load <model> [--context N] [--parallel N] [--identifier NAME] [--runtime squid]
                            start the engine with a model ("27b" is enough if only one name contains it);
                            --runtime squid runs it in Squid's own TypeScript engine instead of llama.cpp
  squid ps                  models that are loaded now
  squid unload <name>|--all stop the engine and free the GPU ("27b" is enough here too)
  squid server start|stop|status
                            the API server (OpenAI-style) in the background, on 127.0.0.1:1235
  squid serve [--port N]    the same server in this window (Ctrl+C stops it)
  squid ask "<question>" [--model NAME] [--system TEXT] [--mode manual|auto|plan] [--stats]
                            one question, answered as it's written

remote mode, with {"remote": {"ssh": "bld"}} in ~/.squid/config.json: the models run on that PC. chat, ask
and web work here, on the folder you're in; ls, ps, info, tokenize, next and unload run on the PC.
  squid win [command]       Squid on the PC itself, for the PC's own folders (chat if no command)`;

// LM Studio shows sizes in decimal units (1 GB = 1,000,000,000 bytes), MB below 1 GB.
const size = (bytes: number) => (bytes < 1e9 ? `${(bytes / 1e6).toFixed(2)} MB` : `${(bytes / 1e9).toFixed(2)} GB`);

// LM Studio prettifies some architecture names in its table. These are the ones seen on this PC
// ("llama" since SmolLM2 arrived in step 15; "qwen35" it leaves as it is).
const ARCH_LABELS: Record<string, string> = { qwen2: "Qwen2", "nomic-bert": "Nomic BERT", llama: "Llama" };

// Problems the user can fix (wrong file, missing argument) print one line, not a stack trace.
function fail(message: string): never {
  console.error(`squid: ${message}`);
  process.exit(1);
}

// Lines up columns: each one is as wide as its longest cell, plus a gap.
function table(heading: string[], rows: string[][]): string {
  const widths = heading.map((h, i) => Math.max(h.length, ...rows.map((r) => r[i].length)) + 4);
  return [heading, ...rows].map((row) => row.map((cell, i) => cell.padEnd(widths[i])).join("").trimEnd()).join("\n");
}

// The exact name, or failing that the one model whose name contains what you typed.
function pickModel(query: string): ShelfModel {
  const shelf = listModels();
  const exact = shelf.find((m) => m.key === query);
  if (exact) return exact;
  const matches = shelf.filter((m) => m.key.includes(query.toLowerCase()));
  if (matches.length === 1) return matches[0];
  if (!matches.length) fail(`no model matches "${query}" (see \`squid ls\`)`);
  fail(`"${query}" matches ${matches.length} models: ${matches.map((m) => m.key).join(", ")}`);
}

// --mode, checked against the three modes.
function modeOption(text: string | undefined): Mode | undefined {
  if (text === undefined) return undefined;
  if (!MODES.includes(text as Mode)) fail(`--mode is manual, auto or plan, not "${text}"`);
  return text as Mode;
}

// "--context 49152" arrives as text; accept only a positive whole number.
function whole(text: string | undefined, flag: string, least = 1): number | undefined {
  if (text === undefined) return undefined;
  const n = Number(text);
  if (!Number.isInteger(n) || n < least) fail(`${flag} needs a whole number, got "${text}"`);
  return n;
}

function decimal(text: string | undefined, flag: string): number | undefined {
  if (text === undefined) return undefined;
  const n = Number(text);
  if (!Number.isFinite(n) || n < 0) fail(`${flag} needs a number of 0 or more, got "${text}"`);
  return n;
}

// "Loading … 12 s", updated in place: \r returns to the start of the line, \x1b[K clears it.
function counter(label: string) {
  const tty = process.stdout.isTTY;
  return {
    tick: (seconds: number) => tty && process.stdout.write(`\r${label}… ${Math.floor(seconds)} s`),
    clear: () => tty && process.stdout.write("\r\x1b[K"),
  };
}

// The idle timer as LM Studio shows it: time left / time to live, e.g. "60m / 1h" (observed).
function ttlText(engine: Engine): string {
  const left = secondsLeft(engine);
  if (left === undefined) return "";
  const total = engine.ttlSeconds!;
  return `${Math.max(0, Math.ceil(left / 60))}m / ${total % 3600 === 0 ? `${total / 3600}h` : `${Math.round(total / 60)}m`}`;
}

function shelfTable(title: string, models: ShelfModel[]): string {
  const rows = models.map((m) => [
    m.catalog ? `${m.key} (${m.variants} variant${m.variants === 1 ? "" : "s"})` : m.key,
    m.params ?? "",
    ARCH_LABELS[m.architecture] ?? m.architecture,
    size(m.sizeBytes),
    "Local",
  ]);
  return table([title, "PARAMS", "ARCH", "SIZE", "DEVICE"], rows);
}

// Remote mode (step 22): commands about the models run on the PC, where the models are.
const remote = remoteSettings();
if (remote && ON_THE_PC.has(command)) process.exit(runOnPc(remote, [command, ...args]));
// `squid load` would start the engine from inside an SSH session, which on Windows holds the session
// open and runs the engine elevated. The PC's server loads a model by itself when it's first asked for.
const NO_REMOTE_LOAD = "in remote mode the PC's server loads a model when it's first used (its context length etc. come from the PC's ~/.squid/config.json)";

switch (command) {
  case undefined: {
    if (remote) {
      // The PC's server through the tunnel; the model loads with the first message.
      try {
        await reachServer();
      } catch (err) {
        fail((err as Error).message);
      }
      console.log(`Squid's server on ${remote.ssh} is here on http://127.0.0.1:${remote.localPort}; your default model loads with the first message.`);
      if (process.stdin.isTTY) await chat({});
      break;
    }
    // One word: the server in the background, then your default model, loaded the way a request would
    // load it, so it unloads itself after an hour unused.
    let server;
    try {
      server = await ensureServer();
    } catch (err) {
      fail((err as Error).message);
    }
    console.log(`Squid's server is running on http://127.0.0.1:${server.port}`);
    const name = readConfig().defaultModel;
    if (!name) {
      console.log(`\nNo default model yet. Add one to ~/.squid/config.json, e.g. {"defaultModel": "qwen3.8-27b-text"}\n\n${USAGE}`);
      break;
    }
    const model = pickModel(name);
    const progress = counter(`Loading ${model.key}`);
    let engine;
    try {
      engine = await loadOnDemand(model, progress.tick);
    } catch (err) {
      progress.clear();
      fail((err as Error).message);
    }
    progress.clear();
    const idle = engine.ttlSeconds ? `; unloads after ${Math.round(engine.ttlSeconds / 60)} min unused` : "";
    console.log(`${engine.identifier} is ready (${engine.contextLength.toLocaleString("en-US")}-token context${idle}).`);
    // At a keyboard, go straight into the chat. From a script (input piped in), booting is the whole job.
    if (process.stdin.isTTY) await chat({ model: engine.identifier });
    break;
  }
  case "chat": {
    const { values } = parseArgs({ args, options: { model: { type: "string", short: "m" }, system: { type: "string", short: "s" }, mode: { type: "string" } } });
    try {
      await chat({ ...values, mode: modeOption(values.mode) });
    } catch (err) {
      fail((err as Error).message);
    }
    break;
  }
  case "web": {
    const { values } = parseArgs({ args, options: { model: { type: "string", short: "m" }, mode: { type: "string" }, port: { type: "string" }, "no-open": { type: "boolean" } } });
    try {
      await web({ model: values.model, mode: modeOption(values.mode), port: whole(values.port, "--port"), open: !values["no-open"] });
    } catch (err) {
      fail((err as NodeJS.ErrnoException).code === "EADDRINUSE" ? `port ${values.port} is already in use` : (err as Error).message);
    }
    break; // the page's server keeps this program running until Ctrl+C
  }
  case "tokenize": {
    const { values, positionals } = parseArgs({
      args,
      allowPositionals: true,
      options: { ids: { type: "boolean" }, "no-special": { type: "boolean" }, file: { type: "string", short: "f" } },
    });
    if (!positionals[0] || (positionals.length < 2 && !values.file)) fail('usage: squid tokenize <model> "<text>" [--ids] [--no-special] [--file PATH]');
    const model = pickModel(positionals[0]);
    let vocab, text;
    try {
      vocab = loadVocab(model.path);
      text = values.file ? readFileSync(values.file, "utf8") : positionals.slice(1).join(" ");
    } catch (err) {
      fail((err as Error).message);
    }
    const ids = encode(vocab, text, { special: !values["no-special"] });
    if (values.ids) {
      console.log(`[${ids.join(", ")}]`); // the same form as llama-tokenize --ids
      break;
    }
    // At a terminal: the text with each token on its own colour, like a tokenizer playground.
    if (process.stdout.isTTY) {
      const COLOURS = [54, 24, 89, 23]; // purple, blue, plum, teal (256-colour backgrounds)
      const shown = ids.map((id, i) => `\x1b[48;5;${COLOURS[i % COLOURS.length]}m\x1b[97m${piece(vocab, id).replace(/\n/g, "↵\n")}\x1b[0m`);
      console.log(`${shown.join("")}\n`);
    }
    console.log(`${ids.length} token${ids.length === 1 ? "" : "s"} (${model.key}, ${vocab.tokens.length.toLocaleString("en-US")}-token vocabulary)`);
    for (const id of ids) console.log(`${String(id).padStart(7)} -> '${piece(vocab, id)}'`); // as llama-tokenize lists them
    break;
  }
  case "next": {
    // Squid's own engine (step 16): every number computed in TypeScript, on the CPU, no llama.cpp.
    const { values, positionals } = parseArgs({
      args,
      allowPositionals: true,
      options: {
        steps: { type: "string" }, temperature: { type: "string" }, "top-k": { type: "string" }, "top-p": { type: "string" },
        "min-p": { type: "string" }, seed: { type: "string" },
      },
    });
    if (positionals.length < 2) fail('usage: squid next <model> "<text>" [--steps N] [--temperature T] [--top-k K] [--top-p P] [--min-p P] [--seed N]');
    const steps = whole(values.steps, "--steps") ?? 12;
    // llama.cpp's defaults unless given; a seed of its own each run unless given (step 17).
    const sampling = {
      temperature: decimal(values.temperature, "--temperature") ?? DEFAULTS.temperature,
      topK: whole(values["top-k"], "--top-k", 0) ?? DEFAULTS.topK,
      topP: decimal(values["top-p"], "--top-p") ?? DEFAULTS.topP,
      minP: decimal(values["min-p"], "--min-p") ?? DEFAULTS.minP,
      seed: whole(values.seed, "--seed", 0) ?? randomInt(2 ** 32),
    };
    const shelf = pickModel(positionals[0]);
    let model;
    try {
      model = loadModel(shelf.path);
    } catch (err) {
      fail((err as Error).message);
    }
    const prompt = positionals.slice(1).join(" ");
    const state = newState(model);
    const started = performance.now();
    let logits = feed(model, state, encode(model.vocab, prompt));
    const promptMs = (performance.now() - started) / state.pos;
    console.log(dim(`${shelf.key} in Squid's TypeScript engine, on the CPU (${state.pos} prompt tokens, ${promptMs.toFixed(0)} ms each)\n`));
    const percent = (logprob: number) => `${(Math.exp(logprob) * 100).toFixed(logprob > -4.6 ? 0 : 1)}%`;
    console.log(`next token:  ${top(logits, 5).map((c) => `${JSON.stringify(piece(model.vocab, c.id))} ${percent(c.logprob)}`).join("   ")}`);
    // What the sampling chain leaves of them, and each one's chance of being picked.
    const left = candidates(logits, sampling);
    const { temperature, topK, topP, minP } = sampling;
    console.log(dim(`after top-k ${topK} → top-p ${topP} → min-p ${minP} → temperature ${temperature}: ${left.length} left`));
    const more = left.length > 5 ? "   …" : "";
    console.log(`             ${left.slice(0, 5).map((c) => `${JSON.stringify(piece(model.vocab, c.id))} ${percent(Math.log(c.p))}`).join("   ")}${more}\n`);
    // Then roll the dice for each token in turn.
    const random = mt19937(sampling.seed).random;
    process.stdout.write(prompt);
    const generated: number[] = [];
    let shown = "";
    const genStart = performance.now();
    for (let i = 0; i < steps; i++) {
      const next = pick(candidates(logits, sampling), random);
      if (model.vocab.ends.has(next)) break;
      generated.push(next);
      const text = decode(model.vocab, generated);
      if (!text.endsWith("�")) { // a character split across tokens is shown once it's whole
        process.stdout.write(text.slice(shown.length));
        shown = text;
      }
      logits = forward(model, state, next);
    }
    const seconds = (performance.now() - genStart) / 1000;
    const again = temperature > 0 ? `; --seed ${sampling.seed} gives this text again` : "";
    console.log(dim(`\n\n(${generated.length} tokens, ${(generated.length / seconds).toFixed(1)} tokens/s${again})`));
    break;
  }
  case "help":
  case "--help":
  case "-h":
    console.log(USAGE);
    break;
  case "ls": {
    const shelf = listModels();
    const total = shelf.reduce((sum, m) => sum + m.sizeBytes, 0);
    console.log(`\nYou have ${shelf.length} models, taking up ${size(total)} of disk space.\n`);
    console.log(shelfTable("LLM", shelf.filter((m) => m.type === "llm")));
    const embedders = shelf.filter((m) => m.type === "embedding");
    if (embedders.length) console.log(`\n${shelfTable("EMBEDDING", embedders)}`);
    break;
  }
  case "info": {
    if (!args[0]) fail("usage: squid info <model.gguf>");
    let header;
    try {
      header = readGgufHeader(args[0]);
    } catch (err) {
      fail((err as Error).message);
    }
    const model = describeModel(header);
    const rows: [string, unknown][] = [
      ["name", model.name],
      ["architecture", model.architecture],
      ["size label", model.sizeLabel],
      ["quantization", model.quantization],
      ["max context", model.contextLength?.toLocaleString("en-US")],
      ["layers", model.layers],
      ["chat template", model.hasChatTemplate ? "yes" : "no"],
      ["tensors", model.tensorCount],
      ["file size", size(model.fileBytes)],
      ["header size", `${(model.headerBytes / 1e6).toFixed(1)} MB (all we read)`],
    ];
    for (const [label, value] of rows) console.log(`${label.padEnd(14)}${value ?? "-"}`);
    break;
  }
  case "load": {
    const { values, positionals } = parseArgs({
      args,
      allowPositionals: true,
      options: { context: { type: "string" }, parallel: { type: "string" }, identifier: { type: "string" }, runtime: { type: "string" } },
    });
    if (remote) fail(NO_REMOTE_LOAD);
    if (!positionals[0]) fail("usage: squid load <model> [--context N] [--parallel N] [--runtime squid]");
    if (values.runtime && values.runtime !== "squid" && values.runtime !== "llama.cpp") fail(`--runtime is "squid" or "llama.cpp", not "${values.runtime}"`);
    const model = pickModel(positionals[0]);
    const progress = counter(`Loading ${model.key}`);
    let engine;
    try {
      engine = await load(
        model,
        {
          contextLength: whole(values.context, "--context"), parallel: whole(values.parallel, "--parallel"), identifier: values.identifier,
          runtime: values.runtime as Runtime | undefined,
        },
        progress.tick,
      );
    } catch (err) {
      progress.clear();
      fail((err as Error).message);
    }
    progress.clear();
    // load() hands back the existing engine when the model was already loaded (by someone else).
    if (engine.loader !== process.pid) console.log(`${engine.identifier} is already loaded.`);
    else console.log(`Loaded ${engine.identifier} (${size(model.sizeBytes)}, ${engine.contextLength.toLocaleString("en-US")}-token context).`);
    break;
  }
  case "ps": {
    const engines = running();
    if (!engines.length) {
      console.log("\nNo models are currently loaded.\n\nTo load a model, run:\n\n    squid load <model>\n");
      break;
    }
    const rows = await Promise.all(
      engines.map(async (e) => [e.identifier, e.key, await status(e), size(e.sizeBytes), String(e.contextLength), String(e.parallel), "Local", ttlText(e)]),
    );
    console.log(`\n${table(["IDENTIFIER", "MODEL", "STATUS", "SIZE", "CONTEXT", "PARALLEL", "DEVICE", "TTL"], rows)}\n`);
    break;
  }
  case "unload": {
    const loaded = running().map((e) => e.identifier);
    if (!args[0]) fail("usage: squid unload <name> | --all");
    // Like load: the exact name, or the one loaded model whose name contains what you typed ("27b").
    const matches = loaded.includes(args[0]) ? [args[0]] : loaded.filter((id) => id.includes(args[0].toLowerCase()));
    if (args[0] !== "--all" && matches.length > 1) fail(`"${args[0]}" matches ${matches.join(", ")}`);
    const names = args[0] === "--all" ? loaded : matches.length ? matches : [args[0]];
    if (!names.length) fail("nothing is loaded");
    for (const name of names) {
      if (await unload(name)) console.log(`Unloaded ${name}.`);
      else fail(`${name} isn't loaded (see \`squid ps\`)`);
    }
    break;
  }
  case "ask": {
    const { values, positionals } = parseArgs({
      args,
      allowPositionals: true,
      options: { model: { type: "string", short: "m" }, system: { type: "string", short: "s" }, stats: { type: "boolean" }, mode: { type: "string" } },
    });
    if (!positionals.length) fail('usage: squid ask "<question>" [--model NAME] [--system TEXT] [--mode manual|auto|plan] [--stats]');
    try {
      await ask(positionals.join(" "), { ...values, mode: modeOption(values.mode) });
    } catch (err) {
      fail((err as Error).message);
    }
    break;
  }
  case "serve": {
    const { values } = parseArgs({ args, options: { port: { type: "string" } } });
    const port = whole(values.port, "--port") ?? DEFAULT_PORT;
    try {
      await startServer(port);
    } catch (err) {
      fail((err as NodeJS.ErrnoException).code === "EADDRINUSE" ? `port ${port} is already in use (is Squid already serving?)` : (err as Error).message);
    }
    recordServer({ pid: process.pid, port }); // so `squid server status` and other commands can find it
    console.log(`Squid is serving on http://127.0.0.1:${port}  (Ctrl+C stops the server; loaded models keep running)`);
    break;
  }
  case "server": {
    // Like `lms server start | stop | status`. "The server is not running." is LM Studio's wording.
    if (remote && ["start", "stop", "status"].includes(args[0])) {
      if (args[0] === "start") {
        try {
          await reachServer();
        } catch (err) {
          fail((err as Error).message);
        }
      }
      process.exitCode = runOnPc(remote, ["server", args[0]]);
      if (args[0] === "stop") runSsh(closeArgs(remote)); // and the tunnel with it
      break;
    }
    if (args[0] === "start") {
      try {
        console.log(`The server is running on port ${(await ensureServer()).port}.`);
      } catch (err) {
        fail((err as Error).message);
      }
    } else if (args[0] === "stop") {
      console.log((await stopServer()) ? "Stopped the server." : "The server is not running.");
    } else if (args[0] === "status") {
      const info = await serverStatus();
      console.log(info ? `The server is running on port ${info.port}.` : "The server is not running.");
    } else {
      fail("usage: squid server start | stop | status");
    }
    break;
  }
  case "win": {
    // Squid on the PC, in its home folder: for working on the PC's own files. Its server is started here
    // first, through the task, so nothing started over SSH has to keep running there.
    if (!remote) fail(`squid win needs remote mode: {"remote": {"ssh": "<host>"}} in ~/.squid/config.json`);
    if (args[0] === "load") fail(NO_REMOTE_LOAD);
    try {
      await reachServer();
    } catch (err) {
      fail((err as Error).message);
    }
    process.exitCode = runOnPc(remote, args.length ? args : ["chat"]);
    break;
  }
  default:
    console.log(USAGE);
    process.exitCode = command ? 1 : 0;
}
