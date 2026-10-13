#!/usr/bin/env node
// Squid's command line: `squid <command> [arguments]`.
import { parseArgs } from "node:util";
import { readGgufHeader, describeModel } from "./gguf.ts";
import { listModels, type ShelfModel } from "./models.ts";
import { load, loadOnDemand, running, secondsLeft, status, unload, type Engine } from "./engine.ts";
import { startServer, DEFAULT_PORT } from "./server.ts";
import { ensureServer, recordServer, serverStatus, stopServer } from "./daemon.ts";
import { readConfig } from "./config.ts";
import { MODES, type Mode } from "./rules.ts";
import { ask } from "./ask.ts";
import { chat } from "./chat.ts";
import { web } from "./web.ts";
import { encode, loadVocab, piece } from "./tokenizer.ts";
import { readFileSync } from "node:fs";

const [command, ...args] = process.argv.slice(2);

const USAGE = `usage:
  squid                     start your local AI (server + default model) and chat with it
  squid chat [--model NAME] [--system TEXT] [--mode manual|auto|plan]
                            a conversation in the terminal (/help inside; Shift+Tab changes mode)
  squid web [--model NAME] [--mode manual|auto|plan] [--port N] [--no-open]
                            the same, in your browser, for the folder you're in (Ctrl+C stops it)
  squid tokenize <model> "<text>" [--ids] [--no-special] [--file PATH]
                            the tokens a model reads for some text (its own vocabulary, no GPU)
  squid ls                  list your models
  squid info <model.gguf>   what's inside one model file
  squid load <model> [--context N] [--parallel N] [--identifier NAME]
                            start the engine with a model ("27b" is enough if only one name contains it)
  squid ps                  models that are loaded now
  squid unload <name>|--all stop the engine and free the GPU
  squid server start|stop|status
                            the API server (OpenAI-style) in the background, on 127.0.0.1:1235
  squid serve [--port N]    the same server in this window (Ctrl+C stops it)
  squid ask "<question>" [--model NAME] [--system TEXT] [--mode manual|auto|plan] [--stats]
                            one question, answered as it's written`;

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
function whole(text: string | undefined, flag: string): number | undefined {
  if (text === undefined) return undefined;
  const n = Number(text);
  if (!Number.isInteger(n) || n < 1) fail(`${flag} needs a whole number, got "${text}"`);
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

switch (command) {
  case undefined: {
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
      options: { context: { type: "string" }, parallel: { type: "string" }, identifier: { type: "string" } },
    });
    if (!positionals[0]) fail("usage: squid load <model> [--context N] [--parallel N]");
    const model = pickModel(positionals[0]);
    const progress = counter(`Loading ${model.key}`);
    let engine;
    try {
      engine = await load(
        model,
        { contextLength: whole(values.context, "--context"), parallel: whole(values.parallel, "--parallel"), identifier: values.identifier },
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
    const names = args[0] === "--all" ? running().map((e) => e.identifier) : args.slice(0, 1);
    if (!names.length) fail(args[0] === "--all" ? "nothing is loaded" : "usage: squid unload <name> | --all");
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
  default:
    console.log(USAGE);
    process.exitCode = command ? 1 : 0;
}
