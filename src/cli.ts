#!/usr/bin/env node
// Squid's command line: `squid <command> [arguments]`.
import { parseArgs } from "node:util";
import { readGgufHeader, describeModel } from "./gguf.ts";
import { listModels, type ShelfModel } from "./models.ts";
import { load, running, status, unload } from "./engine.ts";

const [command, ...args] = process.argv.slice(2);

const USAGE = `usage:
  squid ls                  list your models
  squid info <model.gguf>   what's inside one model file
  squid load <model> [--context N] [--parallel N] [--identifier NAME]
                            start the engine with a model ("27b" is enough if only one name contains it)
  squid ps                  models that are loaded now
  squid unload <name>|--all stop the engine and free the GPU`;

// LM Studio shows sizes in decimal units (1 GB = 1,000,000,000 bytes), MB below 1 GB.
const size = (bytes: number) => (bytes < 1e9 ? `${(bytes / 1e6).toFixed(2)} MB` : `${(bytes / 1e9).toFixed(2)} GB`);

// LM Studio prettifies some architecture names in its table. These two are the ones seen on this PC.
const ARCH_LABELS: Record<string, string> = { qwen2: "Qwen2", "nomic-bert": "Nomic BERT" };

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

// "--context 49152" arrives as text; accept only a positive whole number.
function whole(text: string | undefined, flag: string): number | undefined {
  if (text === undefined) return undefined;
  const n = Number(text);
  if (!Number.isInteger(n) || n < 1) fail(`${flag} needs a whole number, got "${text}"`);
  return n;
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
    const tty = process.stdout.isTTY;
    let engine;
    try {
      engine = await load(
        model,
        { contextLength: whole(values.context, "--context"), parallel: whole(values.parallel, "--parallel"), identifier: values.identifier },
        // \r returns to the start of the line, so the counter updates in place.
        (seconds) => tty && process.stdout.write(`\rLoading ${model.key}… ${Math.floor(seconds)} s`),
      );
    } catch (err) {
      if (tty) process.stdout.write("\n");
      fail((err as Error).message);
    }
    if (tty) process.stdout.write("\r\x1b[K"); // clear the counter line
    console.log(`Loaded ${engine.identifier} (${size(model.sizeBytes)}, ${engine.contextLength.toLocaleString("en-US")}-token context).`);
    break;
  }
  case "ps": {
    const engines = running();
    if (!engines.length) {
      console.log("\nNo models are currently loaded.\n\nTo load a model, run:\n\n    squid load <model>\n");
      break;
    }
    const rows = await Promise.all(
      engines.map(async (e) => [e.identifier, e.key, await status(e), size(e.sizeBytes), String(e.contextLength), String(e.parallel), "Local", ""]),
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
  default:
    console.log(USAGE);
    process.exitCode = command ? 1 : 0;
}
