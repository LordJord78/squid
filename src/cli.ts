#!/usr/bin/env node
// Squid's command line: `squid <command> [arguments]`.
import { readGgufHeader, describeModel } from "./gguf.ts";

const [command, ...args] = process.argv.slice(2);

// LM Studio shows sizes in decimal gigabytes (1 GB = 1,000,000,000 bytes).
const gb = (bytes: number) => `${(bytes / 1e9).toFixed(2)} GB`;

// Problems the user can fix (wrong file, missing argument) print one line, not a stack trace.
function fail(message: string): never {
  console.error(`squid: ${message}`);
  process.exit(1);
}

switch (command) {
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
      ["file size", gb(model.fileBytes)],
      ["header size", `${(model.headerBytes / 1e6).toFixed(1)} MB (all we read)`],
    ];
    for (const [label, value] of rows) console.log(`${label.padEnd(14)}${value ?? "-"}`);
    break;
  }
  default:
    console.log("usage: squid info <model.gguf>");
    process.exitCode = command ? 1 : 0;
}
