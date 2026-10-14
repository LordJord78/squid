// Undo: before the agent changes a file, Squid keeps a copy of how it was. Each of your messages starts a
// new checkpoint (as in Claude Code), and /undo puts back every file the agent changed since the last
// one. Run it again to go further back. Only changes made by edit_file and write_file are tracked; a
// command's changes (npm run format, git checkout) can't be undone here (Claude Code has the same limit).
import { existsSync, readFileSync, rmSync } from "node:fs";
import { renderDiff } from "./diff.ts";

// One file in one checkpoint: how it was before the agent's first change in that turn (null = it didn't
// exist), and how the agent last left it. The second lets undo notice if someone changed it since.
export type Snapshot = { file: string; before: string | null; after: string | null };
export type Checkpoint = { label: string; snapshots: Map<string, Snapshot> };

const KEEP = 100; // like Claude Code, only the most recent 100 checkpoints are kept

export function startCheckpoint(checkpoints: Checkpoint[], label: string): void {
  checkpoints.push({ label: label.length > 60 ? `${label.slice(0, 57)}…` : label, snapshots: new Map() });
  if (checkpoints.length > KEEP) checkpoints.shift();
}

// Called just before a write. Only the first write to a file in a checkpoint records "before": later
// writes in the same turn must not replace the original with an in-between version.
export function remember(checkpoints: Checkpoint[], file: string): void {
  const current = checkpoints.at(-1);
  if (!current || current.snapshots.has(file)) return;
  current.snapshots.set(file, { file, before: existsSync(file) ? readFileSync(file, "utf8") : null, after: null });
}

// Called just after a write, with what the agent wrote.
export function wrote(checkpoints: Checkpoint[], file: string, text: string): void {
  const snapshot = checkpoints.at(-1)?.snapshots.get(file);
  if (snapshot) snapshot.after = text;
}

export type UndoPlan = {
  checkpoint: Checkpoint;
  restore: Snapshot[]; // put back as they were
  skipped: Snapshot[]; // changed since the agent wrote them: left alone
};

// What /undo would do, without doing it: the latest checkpoint that changed anything, split into files
// that can be put back and files someone has changed since the agent did.
export function planUndo(checkpoints: Checkpoint[]): UndoPlan | undefined {
  while (checkpoints.length && !checkpoints.at(-1)!.snapshots.size) checkpoints.pop(); // turns that changed nothing
  const checkpoint = checkpoints.at(-1);
  if (!checkpoint) return undefined;
  const restore: Snapshot[] = [];
  const skipped: Snapshot[] = [];
  for (const snapshot of checkpoint.snapshots.values()) {
    const now = existsSync(snapshot.file) ? readFileSync(snapshot.file, "utf8") : null;
    (now === snapshot.after ? restore : skipped).push(snapshot);
  }
  return { checkpoint, restore, skipped };
}

// The plan in words, for you to approve: each file, what will happen to it, and a short diff.
// `name` shortens paths (relative to the folder); `color` is for a terminal, not the browser page.
export function describeUndo(plan: UndoPlan, name: (file: string) => string, color: boolean): string {
  const lines = [`Undo "${plan.checkpoint.label}":`];
  for (const s of plan.restore) {
    lines.push(s.before === null ? `  ${name(s.file)}: delete it (the agent created it)` : `  ${name(s.file)}: put back as it was`);
    if (s.before !== null) lines.push(renderDiff(s.after ?? "", s.before, { context: 1, color }).split("\n").slice(0, 12).map((l) => `    ${l}`).join("\n"));
  }
  for (const s of plan.skipped) lines.push(`  ${name(s.file)}: changed since the agent wrote it, so it's left alone`);
  return lines.join("\n");
}

// After an undo the model's picture of those files is out of date, so it's told, in front of your next
// message. The checkpoint that message starts is still named after your own words, not the note.
export const undoNote = (label: string, files: string[]) =>
  `(Squid: the user undid your changes from "${label}". These files are back as they were before: ${files.join(", ")}.)`;
// Squid's notes (this one, and the agent's "your last answer was interrupted") all start "(Squid: ";
// there can be more than one in front of a message.
export const withoutNote = (text: string) => text.replace(/^(\(Squid: [\s\S]*?\.\)\n\n)+/, "");
export const withNote = (notes: (string | undefined)[]) => notes.filter(Boolean).join("\n\n") || undefined;

// Puts the files back and removes the checkpoint. A file the agent created is deleted again.
// `write` is the atomic writer from tools.ts, so a crash mid-undo can't leave half a file either.
export function applyUndo(checkpoints: Checkpoint[], plan: UndoPlan, write: (file: string, text: string) => void): void {
  for (const snapshot of plan.restore) {
    if (snapshot.before === null) rmSync(snapshot.file, { force: true });
    else write(snapshot.file, snapshot.before);
  }
  checkpoints.splice(checkpoints.indexOf(plan.checkpoint), 1);
}
