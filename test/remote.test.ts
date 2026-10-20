import { test } from "node:test";
import assert from "node:assert/strict";
import { createServer, type Server } from "node:http";
import type { AddressInfo } from "node:net";
import { join } from "node:path";
import { SQUID_HOME } from "../src/config.ts";
import { ensureRemoteServer } from "../src/daemon.ts";
import { ON_THE_PC, quotePwsh, remoteSettings, squidArgs, tunnelArgs, type Remote, type Ssh } from "../src/remote.ts";
import { tempDir } from "./helpers.ts";

// The shared connection's socket in a temp folder: these tests never touch a real tunnel's.
const BLD: Remote = { ssh: "bld", port: 1235, localPort: 1236, socket: join(tempDir(), "ssh.sock") };

test("remote mode is off unless the config names an SSH host; the ports have defaults", () => {
  assert.equal(remoteSettings({}), undefined);
  assert.deepEqual(remoteSettings({ remote: { ssh: "bld" } }), { ssh: "bld", port: 1235, localPort: 1236, socket: join(SQUID_HOME, "ssh-bld.sock") });
  assert.deepEqual(remoteSettings({ remote: { ssh: "bld", port: 2000, localPort: 3000 } })?.localPort, 3000);
});

test("the tunnel: one background SSH connection that brings the PC's port here", () => {
  const args = tunnelArgs(BLD);
  assert.equal(args.at(-1), "bld");
  assert.equal(args[args.indexOf("-S") + 1], BLD.socket);
  assert.ok(args.includes("-M") && args.includes("-f") && args.includes("-N"), "the shared connection's master, in the background, running nothing");
  assert.equal(args[args.indexOf("-L") + 1], "127.0.0.1:1236:127.0.0.1:1235", "both ends on loopback: nothing listens on the network");
  assert.ok(args.includes("ExitOnForwardFailure=yes") && args.includes("BatchMode=yes"));
  assert.ok(!args.includes("ControlMaster=no"));
});

test("words sent to the PC are quoted for PowerShell, so nothing in them runs", () => {
  assert.equal(quotePwsh("ls"), "'ls'");
  assert.equal(quotePwsh("it's"), "'it''s'");
  assert.equal(quotePwsh("it’s"), "'it’’s'", "PowerShell counts curly quotes as quotes too");
  assert.equal(quotePwsh("$(Remove-Item ~); echo"), "'$(Remove-Item ~); echo'", "inside '…', $ and ; are plain text");
  const args = squidArgs(BLD, ["tokenize", "27b", "hello world"], false);
  assert.deepEqual(args.slice(-2), ["bld", "squid 'tokenize' '27b' 'hello world'"]);
  assert.ok(args.includes("-T"), "no terminal when the output is piped");
  assert.ok(squidArgs(BLD, ["ps"], true).includes("-t"));
  assert.ok(args.includes("ControlMaster=no"), "a command rides on the tunnel's connection; it never becomes the master itself (~/.ssh/config may say ControlMaster auto)");
});

test("commands about the PC's models run there; the agent's commands run here", () => {
  for (const command of ["ls", "ps", "info", "tokenize", "next", "unload"]) assert.ok(ON_THE_PC.has(command), command);
  for (const command of ["chat", "ask", "web", "serve", "load", "server", undefined]) assert.ok(!ON_THE_PC.has(command), String(command));
});

// --- reaching the PC's server, with stand-ins for ssh and for the far end of the tunnel ---

// Answers like Squid's server would, on `port`, once started: the PC's server as seen through the tunnel.
function farEnd(port: number) {
  let server: Server | undefined;
  return {
    start: () => void (server = createServer((req, res) => res.end(`{"error":"Unexpected endpoint or method. (${req.method} ${req.url})"}`)).listen(port, "127.0.0.1")),
    stop: () => server?.close(),
  };
}
async function freePort(): Promise<number> {
  const probe = createServer();
  await new Promise<void>((done) => probe.listen(0, "127.0.0.1", done));
  const { port } = probe.address() as AddressInfo;
  await new Promise((done) => probe.close(done));
  return port;
}
// Records every ssh call; `react` plays what the real one would do (open the tunnel, run the task).
function fakeSsh(react: (args: string[]) => { status: number; stderr?: string } | void) {
  const calls: string[][] = [];
  const ssh: Ssh = (args) => (calls.push(args), { status: 0, stderr: "", ...react(args) });
  const kind = (a: string[]) => (a.includes("check") ? "check" : a.includes("forward") ? "forward" : a.includes("-M") ? "tunnel" : a.at(-1)!.startsWith("schtasks") ? "task" : "other");
  return { ssh, calls, kinds: () => calls.map(kind) };
}
const isCheck = (args: string[]) => args.includes("check");

test("when the tunnel already answers, no SSH is needed at all", async () => {
  const remote = { ...BLD, localPort: await freePort() };
  const pc = farEnd(remote.localPort);
  pc.start();
  try {
    const fake = fakeSsh(() => assert.fail("ssh shouldn't run"));
    assert.deepEqual(await ensureRemoteServer(remote, fake.ssh), { port: remote.localPort, remote: "bld" });
    assert.equal(fake.calls.length, 0);
  } finally {
    pc.stop();
  }
});

test("no tunnel yet: it's opened, and the PC's running server answers through it", async () => {
  const remote = { ...BLD, localPort: await freePort() };
  const pc = farEnd(remote.localPort);
  try {
    const fake = fakeSsh((args) => (isCheck(args) ? { status: 255 } : args.includes("-M") ? pc.start() : undefined));
    assert.equal((await ensureRemoteServer(remote, fake.ssh)).port, remote.localPort);
    assert.deepEqual(fake.kinds(), ["check", "tunnel"]);
  } finally {
    pc.stop();
  }
});

test("a connection without the tunnel (another ssh became the master) gets the tunnel added", async () => {
  const remote = { ...BLD, localPort: await freePort() };
  const pc = farEnd(remote.localPort);
  try {
    const fake = fakeSsh((args) => (args.includes("forward") ? pc.start() : undefined)); // check succeeds: a master is there
    await ensureRemoteServer(remote, fake.ssh);
    assert.deepEqual(fake.kinds(), ["check", "forward"]);
    assert.equal(fake.calls[1][fake.calls[1].indexOf("-L") + 1], `127.0.0.1:${remote.localPort}:127.0.0.1:1235`);
  } finally {
    pc.stop();
  }
});

test("the tunnel leads nowhere: the PC's server is started through its scheduled task, not over SSH", async () => {
  const remote = { ...BLD, localPort: await freePort() };
  const pc = farEnd(remote.localPort);
  try {
    const fake = fakeSsh((args) => (args.at(-1)!.startsWith("schtasks") ? pc.start() : undefined)); // check succeeds: the tunnel is up
    await ensureRemoteServer(remote, fake.ssh);
    assert.deepEqual(fake.kinds(), ["check", "forward", "task"]);
    assert.equal(fake.calls[2].at(-1), "schtasks /run /tn 'Squid server'");
  } finally {
    pc.stop();
  }
});

test("failures say what to do: SSH can't connect, the task is missing, the server never answers", async () => {
  const remote = { ...BLD, localPort: await freePort() };
  const unreachable = fakeSsh((args) => (isCheck(args) ? { status: 255 } : { status: 255, stderr: "ssh: connect to host bld port 22: Operation timed out" }));
  await assert.rejects(ensureRemoteServer(remote, unreachable.ssh), /couldn't reach bld over SSH: ssh: connect to host bld port 22: Operation timed out/);
  const noTask = fakeSsh((args) => (args.at(-1)!.startsWith("schtasks") ? { status: 1, stderr: "ERROR: The system cannot find the file specified." } : undefined));
  await assert.rejects(ensureRemoteServer(remote, noTask.ssh), /couldn't start Squid's server on bld.*remote-setup\.ps1/s);
  const silent = fakeSsh(() => undefined);
  await assert.rejects(ensureRemoteServer(remote, silent.ssh, 300), /server on bld didn't answer/);
});
