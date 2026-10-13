#!/usr/bin/env node
// Squid's command line: `squid <command> [arguments]`.
import { readGgufHeader, describeModel } from "./gguf.ts";
import { listModels, type ShelfModel } from "./models.ts";

const [command, ...args] = process.argv.slice(2);

const USAGE = `usage:
  squid ls                  list your models
  squid info <model.gguf>   what's inside one model file`;

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
  default:
    console.log(USAGE);
    process.exitCode = command ? 1 : 0;
}
