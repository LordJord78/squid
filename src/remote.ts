// Remote mode (step 22): Squid on a Mac, with the models running on the Windows PC's GPU. The PC's
// server stays on its own 127.0.0.1; an SSH tunnel brings that port to this machine, so chat, ask and
// web work on the files here while every token comes from the PC. Commands about the PC's models
// (ls, ps, unload …) run on the PC itself, over SSH.
import { spawnSync } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import { join } from "node:path";
import { readConfig, SQUID_HOME, type Config } from "./config.ts";

// ssh: the PC as `ssh` knows it (a Host in ~/.ssh/config); port: Squid's port there; localPort: the
// tunnel's end here, 1236 so this machine's own Squid can still have 1235; socket: see COMMON below.
export type Remote = { ssh: string; port: number; localPort: number; socket: string };

export function remoteSettings(config: Config = readConfig()): Remote | undefined {
  if (!config.remote?.ssh) return undefined;
  const { ssh, port = 1235, localPort = 1236 } = config.remote;
  return { ssh, port, localPort, socket: join(SQUID_HOME, `ssh-${ssh}.sock`) };
}

// What the commands for the PC's models are: these run there; everything else runs here.
export const ON_THE_PC = new Set<string | undefined>(["ls", "ps", "info", "tokenize", "next", "unload"]);

// The scheduled task that starts the PC's server (scripts/remote-setup.ps1 creates it).
export const SERVER_TASK = "Squid server";

// One SSH connection, shared: the tunnel's process is its master, and each command for the PC rides
// along on it instead of logging in again (OpenSSH's ControlMaster, through the socket file). BatchMode:
// keys only, never a password prompt that nobody may be there to answer.
const common = (remote: Remote) => ["-S", remote.socket, "-o", "BatchMode=yes"];
// A command rides along, and never becomes the master itself: ~/.ssh/config may say ControlMaster auto,
// and a master made by a command has no tunnel. (On ssh's command line, -o beats the config file.)
const rider = (remote: Remote) => [...common(remote), "-o", "ControlMaster=no"];
const tunnelSpec = (remote: Remote) => `127.0.0.1:${remote.localPort}:127.0.0.1:${remote.port}`;

// -M: be the shared connection's master; -f: go to the background once logged in; -N: run nothing,
// only forward. ExitOnForwardFailure: if the port here is taken, fail rather than run without a tunnel.
export const SSH_LOG = join(SQUID_HOME, "logs", "ssh.log");
export function tunnelArgs(remote: Remote): string[] {
  return [
    ...common(remote), "-M", "-f", "-N", "-E", SSH_LOG, "-o", "ExitOnForwardFailure=yes", "-o", "ServerAliveInterval=30",
    "-L", tunnelSpec(remote), remote.ssh,
  ];
}
// Asking the running master: is it there; add the tunnel (no harm if it has it); close it.
export const checkArgs = (remote: Remote) => [...common(remote), "-O", "check", remote.ssh];
export const forwardArgs = (remote: Remote) => [...common(remote), "-O", "forward", "-L", tunnelSpec(remote), remote.ssh];
export const closeArgs = (remote: Remote) => [...common(remote), "-O", "exit", remote.ssh];
export const taskArgs = (remote: Remote) => [...rider(remote), "-T", remote.ssh, `schtasks /run /tn ${quotePwsh(SERVER_TASK)}`];

// The PC's SSH shell is PowerShell, which reads the command line again, so every word goes in '…':
// inside, nothing is special except the quote itself, written twice. PowerShell counts curly quotes too.
export function quotePwsh(word: string): string {
  return `'${word.replace(/['‘’‚‛]/g, (quote) => quote + quote)}'`;
}

// `squid <words>` on the PC. -t gives it a terminal there too (progress counters, colours, keys);
// -T when the output here is piped.
export function squidArgs(remote: Remote, words: string[], terminal: boolean): string[] {
  return [...rider(remote), terminal ? "-t" : "-T", remote.ssh, ["squid", ...words.map(quotePwsh)].join(" ")];
}

// Runs ssh and waits. The tunnel's -f leaves a process behind that would hold any pipe open, so it gets
// none: its errors go to SSH_LOG (-E) and are read back from there.
export type Ssh = (args: string[]) => { status: number | null; stderr: string };
export const runSsh: Ssh = (args) => {
  if (!args.includes("-f")) {
    const ran = spawnSync("ssh", args, { stdio: ["ignore", "ignore", "pipe"], encoding: "utf8", timeout: 60_000 });
    return { status: ran.status, stderr: ran.stderr ?? ran.error?.message ?? "" };
  }
  mkdirSync(join(SQUID_HOME, "logs"), { recursive: true });
  const ran = spawnSync("ssh", args, { stdio: "ignore", timeout: 60_000 });
  const log = (() => {
    try {
      return readFileSync(SSH_LOG, "utf8").trimEnd().split("\n").at(-1) ?? "";
    } catch {
      return "";
    }
  })();
  return { status: ran.status, stderr: ran.status === 0 ? "" : log || (ran.error?.message ?? "") };
};

// A command on the PC with its output shown here, as if it ran here. Returns its exit code.
export function runOnPc(remote: Remote, words: string[]): number {
  const terminal = Boolean(process.stdin.isTTY && process.stdout.isTTY);
  return spawnSync("ssh", squidArgs(remote, words, terminal), { stdio: "inherit" }).status ?? 1;
}
