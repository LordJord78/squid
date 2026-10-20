// `squid web`: the same agent as `squid chat`, in a browser tab. Like Jupyter, it's a small server for one
// folder: on this PC only, on a random port, opened with a link that carries a secret key. The page
// (src/page/) only shows what happens and sends your clicks; the agent, the rules and the files stay here.
import { spawn } from "node:child_process";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomBytes, timingSafeEqual } from "node:crypto";
import { readFileSync } from "node:fs";
import type { AddressInfo } from "node:net";
import { relative } from "node:path";
import { brokeOff, MAX_ROUNDS, startConversation, takeTurn } from "./agent.ts";
import { compact, isSummary } from "./compact.ts";
import { answerFrom, planChoiceFrom } from "./approve.ts";
import { readConfig } from "./config.ts";
import { reachServer } from "./daemon.ts";
import { contextLimit, pickModel, type Message, type Output } from "./reply.ts";
import { MODES, type Mode } from "./rules.ts";
import { sseEvent } from "./sse.ts";
import { newSession, writeAtomically, type Approval } from "./tools.ts";
import { applyUndo, describeUndo, planUndo, undoNote, withNote, withoutNote } from "./undo.ts";

// The two files the browser loads. The script is its own file, not written into the page, so the
// security policy can say "only scripts from this server run": anything that sneaks into the page can't.
const PAGE = new URL("./page/", import.meta.url);
const FILES: Record<string, [file: string, type: string]> = {
  "/": ["index.html", "text/html; charset=utf-8"],
  "/page.js": ["page.js", "text/javascript; charset=utf-8"],
};
const POLICY = [
  "default-src 'none'", // nothing is allowed unless listed below
  "script-src 'self'", // scripts: page.js from this server, nothing inline
  "style-src 'unsafe-inline'", // the page's own <style>
  "img-src data:", // the tab icon
  "connect-src 'self'", // fetch() may only talk to this server
  "frame-ancestors 'none'", // no other site may show this page inside its own (and trick you into clicking Yes)
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");
const MAX_BODY_BYTES = 1024 * 1024;

// What the page is told during a turn: one Server-Sent Event each, its data written as JSON.
export type Event =
  | { type: "thinking" | "answer" | "note"; text: string }
  | ({ type: "ask"; id: string } & Approval) // an edit or a command waiting for your Yes / No
  | { type: "plan"; id: string; plan: string } // a plan waiting for your verdict (plan mode)
  | { type: "done"; model: string; used: number; limit?: number; mode: Mode }
  | { type: "error"; message: string };

export type Web = { url: string; port: number; token: string; close: () => void };

// `reconnect`: finds the server again before each message (squid web passes reachServer, which restarts
// a server that has stopped; tests talk to a scripted one and leave it out).
export async function startWeb(options: { base: string; root: string; mode: Mode; model?: string; system?: string; port?: number; reconnect?: () => Promise<string> }): Promise<Web> {
  const { root } = options;
  let base = options.base;
  const token = randomBytes(24).toString("hex"); // 192 random bits: the key in the link
  let model = options.model; // undefined: the server picks the loaded or default model
  let history: Message[] = startConversation(root, options.system);
  let used = 0;
  let pendingNote: string | undefined; // told to the model with your next message (after an undo)
  // The turn being answered: how to stop it, and how to tell the page something. One at a time.
  let turn: { stop: AbortController; send: (event: Event) => void } | undefined;
  // Questions waiting for a click on the page, by id. POST /api/answer settles them.
  const waiting = new Map<string, (answer: string | undefined) => void>();

  // Puts a question on the page and waits for the answer. With no page listening, the answer is
  // undefined, which every caller treats as "no".
  function askPage(make: (id: string) => Event): Promise<string | undefined> {
    if (!turn) return Promise.resolve(undefined);
    const id = randomBytes(8).toString("hex");
    const { send } = turn;
    return new Promise((resolve) => {
      waiting.set(id, resolve);
      send(make(id));
    });
  }
  const answerAllNo = () => {
    for (const [id, resolve] of waiting) {
      waiting.delete(id);
      resolve(undefined);
    }
  };

  // The same session as the chat's, with the page as the one who approves. answerFrom and planChoiceFrom
  // are the terminal's own rules, so "y", "a", "n" or instructions mean exactly the same here.
  const session = newSession(
    root,
    async (request) => answerFrom(await askPage((id) => ({ type: "ask", id, ...request, preview: plain(request.preview) })), Boolean(request.always)),
    (text) => turn?.send({ type: "note", text: plain(text) }),
    { mode: options.mode, compactAt: readConfig().compactAt, choosePlan: async (plan) => planChoiceFrom(await askPage((id) => ({ type: "plan", id, plan }))) },
  );
  const name = (file: string) => relative(root, file) || file;

  // One message from the page, answered as a stream of events that lasts the whole turn.
  async function runTurn(text: string, res: ServerResponse) {
    res.writeHead(200, { "Content-Type": "text/event-stream", "Cache-Control": "no-cache" });
    const stop = new AbortController();
    const send = (event: Event) => res.writableEnded || res.destroyed || res.write(sseEvent(JSON.stringify(event)));
    turn = { stop, send };
    // The tab was closed or reloaded mid-turn: stop the answer (the GPU stops too), and any open
    // question counts as "no". Nobody can click Yes on a page that isn't there.
    res.on("close", () => {
      if (!res.writableEnded) {
        stop.abort();
        answerAllNo();
      }
    });
    const out: Output = {
      answer: (t) => send({ type: "answer", text: t }),
      thinking: (t) => send({ type: "thinking", text: t }),
      note: (t) => send({ type: "note", text: plain(t) }),
    };
    const done = async () => send({ type: "done", model: model ?? "", used, limit: model ? await contextLimit(base, model).catch(() => undefined) : undefined, mode: session.mode });
    try {
      if (options.reconnect) base = await options.reconnect().catch(() => base);
      // /compact, as in the chat (step 19). Typed into the page like a message, but not sent to the model.
      const asked = /^\/compact(\s|$)/i.exec(text);
      if (asked) {
        if (history.length < 3) send({ type: "note", text: "Nothing to compact yet.\n" });
        else used = (await compact(base, model, history, session, out, { force: true, focus: text.slice(asked[0].length).trim() || undefined, signal: stop.signal })).after;
        return await done();
      }
      // The same turn as the chat's (agent.ts): if it breaks off, the rounds that finished are kept.
      const result = await takeTurn(base, model, history, text, pendingNote, session, stop.signal, out);
      pendingNote = result.note;
      if (result.error) send({ type: "error", message: brokeOff(result, stop.signal.aborted, name) });
      else {
        model = result.model!;
        used = result.reply?.usage?.total_tokens ?? used;
        if (result.paused) send({ type: "note", text: `(paused after ${MAX_ROUNDS} rounds of tool calls without an answer; say "continue" to carry on)\n` });
        await done();
      }
    } catch (err) {
      send({ type: "error", message: stop.signal.aborted ? "Stopped: trimmed, but not summarised." : (err as Error).message }); // only /compact gets here
    } finally {
      turn = undefined;
      answerAllNo();
      res.end();
    }
  }

  async function route(req: IncomingMessage, res: ServerResponse) {
    // Step 04's checks, stricter: the exact address and port of this page, nothing else.
    // DNS rebinding: another site's name pointed at 127.0.0.1 still arrives as "Host: that-name".
    const host = req.headers.host;
    if (host !== `127.0.0.1:${port}` && host !== `localhost:${port}`) return reply(res, 403, { error: "Squid's page only answers requests addressed to this PC" });
    // A page from any other website (or any other port) has a different Origin.
    if (req.headers.origin !== undefined && req.headers.origin !== `http://${host}`) return reply(res, 403, { error: "requests from other websites are refused" });

    const path = new URL(req.url ?? "/", "http://localhost").pathname;
    const file = FILES[path];
    if (req.method === "GET" && file) {
      // The page holds no secrets, so it's served without the key; the key arrives in the link's #part.
      // no-referrer: if the page ever links anywhere, the link it came from (with the key) isn't sent.
      res.writeHead(200, { "Content-Type": file[1], "Content-Security-Policy": POLICY, "Referrer-Policy": "no-referrer", "Cache-Control": "no-store" });
      return res.end(readFileSync(new URL(file[0], PAGE)));
    }
    if (!path.startsWith("/api/")) return reply(res, 404, { error: "not found" });

    // Everything that can see the conversation or change files needs the key.
    if (!keyMatches(req.headers.authorization, token)) return reply(res, 401, { error: "Open the link that `squid web` printed: it carries the key." });
    let body: any = {};
    if (req.method === "POST") {
      // JSON only. A plain web form can't send it, and a page elsewhere would first have to ask
      // permission (a "preflight"), which this server never grants.
      if (!String(req.headers["content-type"] ?? "").startsWith("application/json")) return reply(res, 415, { error: "Content-Type must be application/json" });
      try {
        body = JSON.parse((await readBody(req)) || "{}");
      } catch {
        return reply(res, 400, { error: "the body isn't valid JSON, or is over 1 MB" });
      }
    }
    const busy = () => reply(res, 409, { error: "Wait for the current answer to finish, or press Stop." });

    switch (`${req.method} ${path}`) {
      case "GET /api/state": // everything the page needs when it opens (or reloads)
        return reply(res, 200, {
          root, model: model ?? null, mode: session.mode, busy: Boolean(turn), used,
          limit: model ? await contextLimit(base, model).catch(() => undefined) : undefined,
          messages: history
            .filter((m) => (m.role === "user" || m.role === "assistant") && m.content)
            .map((m) =>
              isSummary(m) ? { role: "note", text: "The conversation before this point was compacted into a summary." }
              : { role: m.role, text: m.role === "user" ? withoutNote(m.content) : m.content }),
        });
      case "POST /api/send":
        if (turn) return busy();
        if (typeof body.text !== "string" || !body.text.trim()) return reply(res, 400, { error: "nothing to send" });
        return runTurn(body.text.trim(), res);
      case "POST /api/answer": {
        const settle = waiting.get(String(body.id));
        if (!settle) return reply(res, 404, { error: "that question isn't open any more" });
        waiting.delete(String(body.id));
        settle(typeof body.answer === "string" ? body.answer : "");
        return reply(res, 200, { ok: true });
      }
      case "POST /api/stop": // the Stop button: the chat's Ctrl+C
        turn?.stop.abort();
        answerAllNo();
        return reply(res, 200, { ok: true });
      case "POST /api/mode":
        if (turn) return busy();
        if (!MODES.includes(body.mode)) return reply(res, 400, { error: "the modes are manual, auto and plan" });
        session.mode = body.mode;
        return reply(res, 200, { mode: session.mode });
      case "POST /api/clear": // the New button: /clear
        if (turn) return busy();
        history = startConversation(root, options.system);
        used = 0;
        return reply(res, 200, { ok: true });
      case "GET /api/undo": { // what Undo would do (step 13), for you to confirm
        const plan = planUndo(session.checkpoints);
        if (!plan) return reply(res, 200, { files: 0, text: "Nothing to undo: the agent hasn't changed any files in this conversation. (Changes made by commands can't be undone here; use git.)" });
        return reply(res, 200, { files: plan.restore.length, label: plan.checkpoint.label, text: describeUndo(plan, name, false) });
      }
      case "POST /api/undo": {
        if (turn) return busy();
        const plan = planUndo(session.checkpoints);
        // The page confirms the undo it was shown. If the newest change is a different one now, it looks again.
        if (!plan || plan.checkpoint.label !== body.label) return reply(res, 409, { error: "The last change is different now; press Undo again to see it." });
        applyUndo(session.checkpoints, plan, writeAtomically);
        if (plan.restore.length) pendingNote = withNote([pendingNote, undoNote(plan.checkpoint.label, plan.restore.map((s) => name(s.file)))]);
        return reply(res, 200, { undone: plan.restore.length });
      }
      default:
        return reply(res, 404, { error: `no such request: ${req.method} ${path}` });
    }
  }

  const server = createServer((req, res) => {
    route(req, res).catch((err: Error) => (res.headersSent ? res.destroy() : reply(res, 500, { error: err.message })));
  });
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject);
    server.listen(options.port ?? 0, "127.0.0.1", resolve); // port 0: the system picks a free one
  });
  const port = (server.address() as AddressInfo).port;
  return {
    // The key goes after "#". That part of a link stays in the browser: it's never sent to any server,
    // so it can't end up in a log, or in the Referer header of a request the page makes.
    url: `http://127.0.0.1:${port}/#token=${token}`,
    port,
    token,
    close: () => {
      answerAllNo();
      server.close();
      server.closeAllConnections();
    },
  };
}

// `squid web`: the page for the folder you're in. Prints its link and opens it in your browser.
export async function web(options: { model?: string; mode?: Mode; port?: number; open: boolean }): Promise<void> {
  const base = `http://127.0.0.1:${(await reachServer()).port}`;
  const root = process.cwd();
  const model = options.model && (await pickModel(base, options.model));
  const reconnect = async () => `http://127.0.0.1:${(await reachServer()).port}`;
  const page = await startWeb({ base, root, model, mode: options.mode ?? readConfig().defaultMode ?? "manual", port: options.port, reconnect });
  console.log(`Squid's page for ${root}:\n\n    ${page.url}\n\nOnly this PC can open it, and only with that link. Ctrl+C here stops it.`);
  // rundll32 hands the link to Windows, which opens your default browser. No shell reads the link (step 11).
  if (options.open) spawn("rundll32", ["url.dll,FileProtocolHandler", page.url], { detached: true, stdio: "ignore" }).unref();
}

// Compared in constant time: how long a wrong guess takes to refuse says nothing about how close it was.
export function keyMatches(header: string | undefined, token: string): boolean {
  const given = Buffer.from(header ?? "");
  const wanted = Buffer.from(`Bearer ${token}`);
  return given.length === wanted.length && timingSafeEqual(given, wanted);
}

// Text for the terminal can carry colour codes ("\x1b[2m" dims, "\x1b[31m" is red). The page colours
// things itself, so they're removed.
const plain = (text: string) => text.replace(/\x1b\[[0-9;]*m/g, "");

function reply(res: ServerResponse, status: number, body: unknown) {
  res.writeHead(status, { "Content-Type": "application/json; charset=utf-8", "Cache-Control": "no-store" });
  res.end(JSON.stringify(body));
}

async function readBody(req: IncomingMessage): Promise<string> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    size += chunk.length;
    if (size > MAX_BODY_BYTES) throw new Error("too large");
    chunks.push(chunk);
  }
  return Buffer.concat(chunks).toString("utf8");
}
