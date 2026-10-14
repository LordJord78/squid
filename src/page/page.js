// The browser half of `squid web`. It shows what the agent does and sends what you click or type; the
// agent, the rules and your files all stay in Node (src/web.ts).
// One rule above all: whatever the model or a file says is put on the page as *text* (textContent),
// never as HTML. A file the model reads could contain "<img src=x onerror=…>"; as HTML, that would run
// here, on a page that holds the key, and could click Yes on its own changes.

const token = new URLSearchParams(location.hash.slice(1)).get("token"); // from the link's #part
const $ = (id) => document.getElementById(id);
const log = $("log");
const MODES = ["manual", "auto", "plan"];
const MODE_LABEL = { manual: "manual", auto: "⏵⏵ auto", plan: "⏸ plan" };
const MODE_HELP = {
  manual: "manual: every change, and every command that isn't read-only, is shown to you first",
  auto: "auto: edits inside this folder go ahead without asking; commands still ask",
  plan: "plan: it only looks, then proposes a plan for you to approve",
};
let mode = "manual";
let answering = false;

// A new element with a class and some text. The only way text gets onto this page.
function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

// Keeps the newest text in view, unless you've scrolled up to read something.
let following = true;
log.addEventListener("scroll", () => (following = log.scrollHeight - log.scrollTop - log.clientHeight < 60));
const follow = () => following && (log.scrollTop = log.scrollHeight);
const add = (node) => (log.append(node), follow(), node);

// Every request carries the key. Without it, Squid answers 401 and does nothing.
function api(path, body) {
  const headers = { Authorization: `Bearer ${token}` };
  if (body === undefined) return fetch(path, { headers });
  return fetch(path, { method: "POST", headers: { ...headers, "Content-Type": "application/json" }, body: JSON.stringify(body) });
}

// The turn arrives as Server-Sent Events (step 05), read by hand. The browser's own EventSource can
// only make GET requests and can't send the key, so this is src/sse.ts again, in the browser.
async function* events(response) {
  const reader = response.body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) return;
    buffer += value;
    let end;
    while ((end = buffer.indexOf("\n\n")) !== -1) {
      const data = buffer.slice(0, end).split("\n").filter((l) => l.startsWith("data: ")).map((l) => l.slice(6)).join("\n");
      buffer = buffer.slice(end + 2);
      if (data) yield JSON.parse(data);
    }
  }
}

// Answers are Markdown. A small part of it is shown: ```code blocks```, `code` and **bold**, all built
// from text, so nothing in an answer can become HTML. Re-drawn as each piece of the answer arrives.
function render(target, text) {
  target.replaceChildren();
  const parts = text.split("```");
  parts.forEach((part, i) => {
    if (i % 2) return target.append(el("pre", "", part.replace(/^[^\n]*\n/, ""))); // inside a fence; line 1 is the language
    // A code block has its own space around it, so the line breaks next to it aren't needed.
    if (i > 0) part = part.replace(/^\n+/, "");
    if (i < parts.length - 1) part = part.replace(/\n+$/, "");
    for (const bit of part.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/)) {
      if (/^`[^`]+`$/.test(bit)) target.append(el("code", "", bit.slice(1, -1)));
      else if (/^\*\*[^*]+\*\*$/.test(bit)) target.append(el("strong", "", bit.slice(2, -2)));
      else if (bit) target.append(document.createTextNode(bit));
    }
  });
}

// Tool lines and diffs. Diff rows look like "   2 - old" and "   2 + new" (step 10): coloured by sign.
function block(text, className = "block") {
  const pre = el("pre", className);
  for (const line of text.replace(/\n$/, "").split("\n")) {
    pre.append(el("span", /^\s*\d+ - /.test(line) ? "del" : /^\s*\d+ \+ /.test(line) ? "ins" : "", `${line}\n`));
  }
  return pre;
}

// A card with a question and buttons. `choose(answer, label)` sends the answer once, then the buttons
// are replaced by what you chose, so a second click can't send a second answer.
function card(title, reason, body, makeButtons) {
  const box = el("div", "card");
  box.append(el("div", "title", title));
  if (reason) box.append(el("div", "reason", reason));
  if (body) box.append(body);
  const buttons = el("div", "buttons");
  const choose = (answer, label) => {
    buttons.replaceChildren(el("span", "chosen", label));
    box.classList.add("answered");
    return answer;
  };
  makeButtons(buttons, choose);
  box.append(buttons);
  return box;
}
const button = (text, className, onClick) => {
  const b = el("button", className, text);
  b.type = "button";
  b.onclick = onClick;
  return b;
};
// A text box that sends what you type when you press Enter.
const instead = (placeholder, onText) => {
  const input = el("input");
  input.placeholder = placeholder;
  input.onkeydown = (e) => e.key === "Enter" && input.value.trim() && onText(input.value.trim());
  return input;
};
const answer = (id, text) => api("/api/answer", { id, answer: text });

// An edit or a command waiting for you (step 10): Claude Code's choices, the same words as the terminal.
function askCard(q) {
  return card(q.title, q.reason, q.preview ? block(q.preview) : undefined, (buttons, choose) => {
    buttons.append(button("Yes", "yes", () => answer(q.id, choose("y", "✓ yes"))));
    if (q.always) {
      const always = button("Always", "", () => answer(q.id, choose("a", `✓ always: ${q.always}`)));
      always.title = q.always;
      buttons.append(always);
    }
    buttons.append(button("No", "no", () => answer(q.id, choose("n", "✗ no"))));
    buttons.append(instead("…or tell it what to do instead, then Enter", (text) => answer(q.id, choose(text, `✗ no, instead: ${text}`))));
  });
}

// A plan waiting for your verdict (step 12). Nothing has been changed yet.
function planCard(q) {
  return card("Plan", "nothing has been changed yet", block(q.plan), (buttons, choose) => {
    buttons.append(button("1 · Go ahead in auto mode", "yes", () => answer(q.id, choose("1", "✓ go ahead, auto mode"))));
    buttons.append(button("2 · Go ahead, asking before each change", "", () => answer(q.id, choose("2", "✓ go ahead, asking first"))));
    buttons.append(instead("…or say what to change, then Enter", (text) => answer(q.id, choose(text, `↻ keep planning: ${text}`))));
  });
}

// One event from the turn, added to the page in the order it happened: thinking, words, tool lines,
// questions. A new block of thinking or words starts after each tool line.
function show(turn, event) {
  if (event.type === "thinking") {
    if (!turn.thinking) {
      const details = turn.box.appendChild(el("details", "thinking"));
      details.open = true;
      details.append(el("summary", "", "thinking"), (turn.thinking = el("pre")));
      turn.answer = undefined;
    }
    turn.thinking.append(event.text);
  } else if (event.type === "answer") {
    if (!turn.answer) {
      turn.answer = turn.box.appendChild(el("div", "answer"));
      turn.text = "";
      closeThinking(turn);
    }
    turn.text += event.text;
    render(turn.answer, turn.text.replace(/\n$/, ""));
  } else {
    closeThinking(turn);
    turn.answer = undefined;
    if (event.type === "note") turn.box.append(block(event.text, "note"));
    if (event.type === "ask") turn.box.append(askCard(event));
    if (event.type === "plan") turn.box.append(planCard(event));
    if (event.type === "error") turn.box.append(el("div", "error", event.message));
    if (event.type === "done") {
      setMode(event.mode); // an approved plan or an "always" can change it during the turn
      $("model").textContent = event.model;
      meter(event.used, event.limit);
    }
  }
  follow();
}
function closeThinking(turn) {
  if (turn.thinking) turn.thinking.parentElement.open = false; // folded away, a click opens it again
  turn.thinking = undefined;
}

async function send(text) {
  setAnswering(true);
  add(el("div", "you", text));
  const turn = { box: add(el("div", "turn")) };
  try {
    const response = await api("/api/send", { text });
    if (!response.ok) return turn.box.append(el("div", "error", (await response.json()).error));
    for await (const event of events(response)) show(turn, event);
  } catch (err) {
    turn.box.append(el("div", "error", `Lost touch with squid web (${err.message}). Is it still running?`));
  } finally {
    setAnswering(false);
  }
}

// While an answer is being written, Send becomes Stop, and the buttons that change things wait.
function setAnswering(on) {
  answering = on;
  for (const id of ["mode", "undo", "clear"]) $(id).disabled = on;
  $("send").textContent = on ? "Stop" : "Send";
  $("send").classList.toggle("stop", on);
}
function setMode(next) {
  mode = next;
  $("mode").textContent = MODE_LABEL[mode];
  $("mode").dataset.mode = mode;
  $("mode").title = `${MODE_HELP[mode]}. Click (or Shift+Tab in the message box) to change.`;
}
async function nextMode() {
  if (answering) return;
  const response = await api("/api/mode", { mode: MODES[(MODES.indexOf(mode) + 1) % MODES.length] });
  if (response.ok) setMode((await response.json()).mode);
}
function meter(used, limit) {
  if (!limit) return;
  const share = Math.min(1, used / limit);
  $("meter-fill").style.width = `${share * 100}%`;
  $("meter-text").textContent = `${used.toLocaleString("en-US")} / ${limit.toLocaleString("en-US")} tokens (${Math.round(share * 100)}%)${share > 0.8 ? " · nearly full: New starts fresh" : ""}`;
  $("meter").classList.toggle("full", share > 0.8);
}

$("composer").onsubmit = (e) => {
  e.preventDefault();
  if (answering) return api("/api/stop", {}); // the button says Stop: the chat's Ctrl+C
  const text = $("input").value.trim();
  if (!text) return;
  $("input").value = "";
  send(text);
};
$("input").onkeydown = (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    if (!answering) $("composer").requestSubmit();
  } else if (e.key === "Tab" && e.shiftKey) {
    e.preventDefault(); // Shift+Tab changes the mode here, as in the terminal, instead of leaving the box
    nextMode();
  }
};
$("mode").onclick = nextMode;
$("clear").onclick = async () => {
  if ((await api("/api/clear", {})).ok) {
    log.replaceChildren(el("div", "info", "Started a new conversation."));
    $("meter-fill").style.width = "0";
    $("meter-text").textContent = "";
  }
};
// Undo (step 13): first see what it would put back, then confirm.
$("undo").onclick = async () => {
  const plan = await (await api("/api/undo")).json();
  if (!plan.files) return add(el("div", "info", plan.text));
  add(card("Undo", "", block(plan.text), (buttons, choose) => {
    buttons.append(button("Undo these", "yes", async () => {
      choose(null, "…");
      const response = await api("/api/undo", { label: plan.label });
      const result = await response.json();
      choose(null, response.ok ? `✓ undid ${result.undone} file${result.undone === 1 ? "" : "s"}` : result.error);
    }));
    buttons.append(button("Cancel", "", () => choose(null, "nothing undone")));
  }));
};

// Opening (or reloading) the page: what folder, which model, and the conversation so far.
async function start() {
  if (!token) {
    for (const id of ["mode", "undo", "clear", "send", "input"]) $(id).disabled = true;
    return add(el("div", "error", "This page needs the key from its link. Open the link that `squid web` printed in your terminal."));
  }
  const response = await api("/api/state");
  const state = await response.json();
  if (!response.ok) return add(el("div", "error", state.error));
  $("folder").textContent = state.root;
  $("model").textContent = state.model ?? "default model";
  document.title = `Squid · ${state.root.split(/[\\/]/).pop()}`;
  setMode(state.mode);
  meter(state.used, state.limit);
  for (const m of state.messages) {
    if (m.role === "user") add(el("div", "you", m.text));
    else if (m.role === "note") add(el("div", "info", m.text)); // a compacted summary (step 19)
    else render(add(el("div", "answer")), m.text);
  }
  if (state.busy) add(el("div", "info", "An answer is being written in another tab."));
  add(el("div", "info", `Mode ${MODE_HELP[state.mode]}.`));
  $("input").focus();
}
start().catch((err) => add(el("div", "error", `Couldn't reach squid web (${err.message}).`)));
