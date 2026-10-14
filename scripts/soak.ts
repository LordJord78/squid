// `npm run soak` (step 21): the 13 Oct incident, replayed as an acceptance test. The 27B at its 48K context
// builds a project far bigger than its memory, through the same turn runner as `squid chat` and the page
// (takeTurn), in auto mode, with every command approved. Then it's asked, from memory, what it built.
//
// It passes if the build finishes without an error, the memory was compacted at least once on the way,
// and the answer names at least 80% of the Python files it created. It takes 30-60 minutes; the report
// goes to stdout (tee it to a file) and ends with PASS or FAIL.
import { execFileSync } from "node:child_process";
import { existsSync, mkdtempSync, readdirSync, statSync } from "node:fs";
import { tmpdir } from "node:os";
import { join, relative } from "node:path";
import type { AddressInfo } from "node:net";
import { startConversation, takeTurn, type Outcome } from "../src/agent.ts";
import { ENGINE, unload } from "../src/engine.ts";
import type { Output } from "../src/reply.ts";
import { startServer } from "../src/server.ts";
import { newSession } from "../src/tools.ts";
import { readConfig } from "../src/config.ts";

const MODEL = process.argv[2] ?? readConfig().defaultModel ?? "qwen3.8-27b-text";
const BUILD =
  "Build a zero-dependency Python showcase project in this folder. A package lab/ with: expr.py (a tokenizer, recursive-descent parser and evaluator for arithmetic with functions like sqrt and max; no eval), markov.py (a seeded Markov-chain text generator with its own sample text), charts.py (ASCII bar charts, labeled bars and sparklines, with an ASCII fallback), snake.py (a Snake engine with no I/O, a greedy autopilot and a playable loop), showcase.py (a headless demo report) and cli.py (argparse subcommands for each). Also tests/ with unittest tests for every module, run_tests.py, demo.py and a README.md. Write complete, working code. When everything is written, run the tests with `python run_tests.py` and fix any failures.";
const RECALL = "Without using any tools: list every Python file you created in this conversation, each with one line on what it does.";

if (!existsSync(ENGINE)) throw new Error("the engine isn't downloaded: npm run engine:get");
const root = mkdtempSync(join(tmpdir(), "squid-soak-"));
const server = await startServer(0); // a private Squid server: loads the model on demand
const base = `http://127.0.0.1:${(server.address() as AddressInfo).port}`;
const session = newSession(root, async (request) => (log(`  approved: ${request.title}`), { ok: true }), undefined, { mode: "auto" });
const history = startConversation(root);
const counts = { trimmed: 0, summarised: 0, healed: 0, calls: 0 };
const started = Date.now();
const log = (text: string) => console.log(`[${((Date.now() - started) / 60_000).toFixed(1)} min] ${text}`);
const out: Output = {
  answer: () => {},
  thinking: () => {},
  note: (text) => {
    if (text.startsWith("●")) counts.calls++, log(text.trim());
    else if (/trimmed old file contents/.test(text)) counts.trimmed++, log(text.trim());
    else if (/compacted into a summary/.test(text)) counts.summarised++, log(text.trim());
    else if (/ran out of memory/.test(text)) counts.healed++, log(text.trim());
    else if (/^\(/.test(text.trim())) log(text.trim());
  },
};

async function turn(text: string, note?: string): Promise<Outcome> {
  log(`you: ${text.slice(0, 100)}${text.length > 100 ? "…" : ""}`);
  const outcome = await takeTurn(base, MODEL, history, text, note, session, undefined, out);
  if (outcome.error) log(`turn broke off: ${outcome.error.message} (kept ${outcome.kept} rounds)`);
  else log(`${outcome.paused ? "paused" : "answered"} · ${outcome.reply?.usage?.total_tokens ?? "?"} tokens in memory`);
  return outcome;
}

const failures: string[] = [];
try {
  // The build: "continue" after a pause (25 rounds) or a broken-off turn, as you would, up to 6 times.
  let outcome = await turn(BUILD);
  for (let more = 0; more < 6 && (outcome.paused || outcome.error); more++) {
    if (outcome.error) failures.push(`the build broke off: ${outcome.error.message}`);
    outcome = await turn("continue", outcome.note);
  }
  if (outcome.paused || outcome.error) failures.push("the build didn't finish after 6 continues");

  const files = pythonFiles(root);
  log(`files: ${files.join(", ")}`);
  let tests = "";
  try {
    tests = execFileSync("python", ["-B", "run_tests.py"], { cwd: root, encoding: "utf8", stdio: ["ignore", "pipe", "pipe"], timeout: 120_000 });
  } catch (err: any) {
    tests = String(err.stderr ?? err.stdout ?? err.message);
  }
  log(`its tests: ${tests.trim().split("\n").filter((l) => /^(Ran|OK|FAILED)/.test(l)).join(" · ") || "no summary"}`);

  const recall = await turn(RECALL);
  const said = (recall.reply?.content ?? "").toLowerCase();
  const named = files.filter((f) => said.includes(f.split("/").pop()!.toLowerCase()));
  log(`recalled ${named.length} of ${files.length}: ${files.filter((f) => !named.includes(f)).join(", ") || "all"}`);

  if (counts.trimmed + counts.summarised === 0) failures.push("the memory was never compacted (the task should outgrow 48K)");
  if (!files.length) failures.push("no Python files were written");
  if (files.length && named.length / files.length < 0.8) failures.push(`it recalled only ${named.length} of ${files.length} files`);
  log(`compactions: ${counts.trimmed} trims, ${counts.summarised} summaries, ${counts.healed} healed cut-offs · ${counts.calls} tool calls · folder ${root}`);
} catch (err) {
  failures.push(`crashed: ${(err as Error).stack}`);
} finally {
  await unload(MODEL).catch(() => {});
  server.close();
}
log(failures.length ? `FAIL\n  ${failures.join("\n  ")}` : "PASS");
process.exitCode = failures.length ? 1 : 0;

function pythonFiles(dir: string): string[] {
  const found: string[] = [];
  for (const name of readdirSync(dir)) {
    const path = join(dir, name);
    if (name === "__pycache__") continue;
    if (statSync(path).isDirectory()) found.push(...pythonFiles(path));
    else if (name.endsWith(".py")) found.push(relative(root, path).replace(/\\/g, "/"));
  }
  return found;
}
