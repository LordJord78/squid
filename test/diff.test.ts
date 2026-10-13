import { test } from "node:test";
import assert from "node:assert/strict";
import { diffLines, renderDiff } from "../src/diff.ts";

const kinds = (a: string[], b: string[]) => diffLines(a, b).map((l) => `${l.kind}${l.text}`);

test("one changed line reads as - old, + new", () => {
  assert.deepEqual(kinds(["a", "b", "c"], ["a", "B", "c"]), [" a", "-b", "+B", " c"]);
});

test("inserted and deleted lines, with the shortest set of changes", () => {
  assert.deepEqual(kinds(["a", "c"], ["a", "b", "c"]), [" a", "+b", " c"]);
  assert.deepEqual(kinds(["a", "b", "c"], ["a", "c"]), [" a", "-b", " c"]);
  // Moving "x" from the top to the bottom: keep the three shared lines, move one.
  assert.deepEqual(kinds(["x", "1", "2", "3"], ["1", "2", "3", "x"]), ["-x", " 1", " 2", " 3", "+x"]);
});

test("line numbers: old for removed lines, new for the rest", () => {
  assert.deepEqual(diffLines(["a", "b", "c"], ["a", "x", "y", "c"]).map((l) => `${l.kind}${l.line}`), [" 1", "-2", "+2", "+3", " 4"]);
});

test("the screen shows 3 lines of context, and … for long unchanged stretches", () => {
  const before = Array.from({ length: 20 }, (_, i) => `line ${i + 1}`);
  const after = before.map((line, i) => (i === 9 ? "line ten" : line));
  assert.equal(renderDiff(before.join("\n"), after.join("\n")), [
    "   …",
    "   7   line 7",
    "   8   line 8",
    "   9   line 9",
    "  10 - line 10",
    "  10 + line ten",
    "  11   line 11",
    "  12   line 12",
    "  13   line 13",
    "   …",
  ].join("\n"));
});

test("a new file is all additions, starting from nothing", () => {
  assert.equal(renderDiff("", "hello\nworld"), "   1 + hello\n   2 + world");
});
