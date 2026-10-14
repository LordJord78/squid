import { test } from "node:test";
import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { meter, parseLine, status } from "../src/chat.ts";
import { ENGINE, unload } from "../src/engine.ts";
import { serverStatus, stopServer } from "../src/daemon.ts";
import { tempDir } from "./helpers.ts";

test("a typed line is a message, a /command, or nothing", () => {
  assert.deepEqual(parseLine("  hello there "), { kind: "say", text: "hello there" });
  assert.deepEqual(parseLine("/model 27b"), { kind: "command", name: "model", arg: "27b" });
  assert.deepEqual(parseLine("/SYSTEM Be brief. Very."), { kind: "command", name: "system", arg: "Be brief. Very." });
  assert.deepEqual(parseLine("/exit"), { kind: "command", name: "exit", arg: "" });
  assert.deepEqual(parseLine("   "), { kind: "empty" });
});

test("the context meter draws how full the model's memory is", () => {
  assert.equal(meter(0, 49152), "░░░░░░░░░░░░░░░░░░░░ 0 / 49,152 tokens (0%)");
  assert.equal(meter(24576, 49152), "▓▓▓▓▓▓▓▓▓▓░░░░░░░░░░ 24,576 / 49,152 tokens (50%)");
  assert.equal(meter(60000, 49152), "▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓ 60,000 / 49,152 tokens (100%)", "never more than full");
});

test("after each answer: the meter, the speed, and a hint when the model slows to half its speed", () => {
  assert.equal(status(4915, 49152, 0.65, 41.2, 0), `${meter(4915, 49152)}  · 41 tok/s`);
  assert.match(status(40000, 49152, 0.65, 40, 41), /\(81%\)  · compacts before the next answer  · 40 tok\/s$/);
  assert.match(status(4915, 49152, 0.65, 17, 41), /17 tok\/s  · slower than earlier \(41 tok\/s\): is another program using the GPU\?$/);
});

// A real conversation with the 9B, typed in through a pipe, one line per turn.
const ready = existsSync(ENGINE) && existsSync("D:\\LMStudio\\models");
test("the chat remembers earlier turns, forgets on /clear, and stops at /exit", { skip: !ready && "engine or model drive missing", timeout: 300_000 }, async (t) => {
  if (!(await serverStatus())) t.after(() => stopServer()); // the chat starts the server; stop it again if we did
  t.after(() => unload("qwen/qwen3.5-9b"));
  const lines = [
    "My favourite colour is purple. Reply with exactly: noted",
    "What is my favourite colour? Reply with one word.",
    "/clear",
    "What is my favourite colour? If you don't know, reply with exactly: unknown",
    "/exit",
    "this line comes after /exit and must not be sent",
  ];
  const cli = fileURLToPath(new URL("../src/cli.ts", import.meta.url));
  // Run in an empty folder: since step 08 the model can search the folder it's in, and in the Squid repo it
  // found "my favourite colour is purple" in this very test file, even after /clear.
  const run = spawnSync(process.execPath, [cli, "chat", "--model", "qwen/qwen3.5-9b"], { cwd: tempDir(), input: lines.join("\n") + "\n", encoding: "utf8", timeout: 280_000 });
  // One meter line per answer (on stderr) counts the answers; an answer can span lines (the 9B sometimes
  // repeats itself, see StrayThinkTag in server.ts), so stdout is read as a whole.
  const meters = run.stderr.match(/\d[\d,]* \/ 8,192 tokens \(\d+%\)/g) ?? [];
  assert.equal(meters.length, 3, "three questions answered against the 9B's 8,192-token context, nothing after /exit");
  // After /clear the conversation still starts with Squid's instructions (about 1,000 tokens): a bug in
  // steps 08-11 dropped them, which showed as a tiny third count.
  const tokens = meters.map((m) => Number(m.split(" ")[0].replace(/,/g, "")));
  assert.ok(tokens[2] > 500, `instructions kept after /clear (${tokens[2]} tokens)`);
  const answers = run.stdout.replace(/^› /, "").trim().toLowerCase();
  assert.doesNotMatch(answers, /<\/think>/);
  const [beforeClear, afterClear] = [answers.slice(0, answers.lastIndexOf("\n")), answers.slice(answers.lastIndexOf("\n") + 1)];
  assert.match(beforeClear, /purple/, "remembered from the first turn");
  assert.doesNotMatch(afterClear, /purple/, "forgotten after /clear (the last line is the last answer)");
  assert.match(run.stderr, /Started a new conversation/);
});
