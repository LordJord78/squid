// A line diff: what an edit changes, shown as removed (-) and added (+) lines with a little context,
// like `git diff`. You see this before approving an edit.
export type DiffLine = { kind: " " | "-" | "+"; text: string; line: number }; // line: old number for "-", new number otherwise

// Finds the smallest set of removed and added lines that turns `before` into `after`.
// 1. Lines shared at the start and end are set aside first: an edit usually touches a few lines in the
//    middle, so this leaves almost nothing for step 2.
// 2. What's left is compared with the "longest common subsequence" method: a table where cell [i][j]
//    holds how many lines a[i..] and b[j..] have in common, in order. Walking the table from the top
//    left, each step keeps a shared line, or removes from a, or adds from b, whichever keeps the most.
export function diffLines(before: string[], after: string[]): DiffLine[] {
  let start = 0;
  while (start < before.length && start < after.length && before[start] === after[start]) start++;
  let endA = before.length;
  let endB = after.length;
  while (endA > start && endB > start && before[endA - 1] === after[endB - 1]) endA--, endB--;
  const a = before.slice(start, endA);
  const b = after.slice(start, endB);

  const middle: DiffLine[] = [];
  if (a.length * b.length > 4_000_000) {
    // A table this big would take too much memory (a rewrite of a huge file): show it as "all old lines out, all new lines in".
    a.forEach((text, i) => middle.push({ kind: "-", text, line: start + i + 1 }));
    b.forEach((text, j) => middle.push({ kind: "+", text, line: start + j + 1 }));
  } else {
    const width = b.length + 1;
    const common = new Uint32Array((a.length + 1) * width); // common[i * width + j]
    for (let i = a.length - 1; i >= 0; i--) {
      for (let j = b.length - 1; j >= 0; j--) {
        common[i * width + j] = a[i] === b[j] ? common[(i + 1) * width + j + 1] + 1 : Math.max(common[(i + 1) * width + j], common[i * width + j + 1]);
      }
    }
    let i = 0;
    let j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) middle.push({ kind: " ", text: a[i++], line: start + ++j });
      // On a tie, remove before adding, so a changed line reads "- old" then "+ new", as in git.
      else if (i < a.length && (j === b.length || common[(i + 1) * width + j] >= common[i * width + j + 1])) middle.push({ kind: "-", text: a[i++], line: start + i });
      else middle.push({ kind: "+", text: b[j++], line: start + j });
    }
  }
  const head = before.slice(0, start).map((text, i): DiffLine => ({ kind: " ", text, line: i + 1 }));
  const tail = before.slice(endA).map((text, k): DiffLine => ({ kind: " ", text, line: start + b.length + k + 1 }));
  return [...head, ...middle, ...tail];
}

// The diff as text for the screen: changed lines plus `context` unchanged lines around them; longer
// unchanged stretches become "…". Removed lines red, added lines green (when `color` is on).
export function renderDiff(before: string, after: string, { context = 3, color = false } = {}): string {
  const toLines = (text: string) => (text === "" ? [] : text.split(/\r?\n/)); // a new file starts from no lines, not one empty line
  const lines = diffLines(toLines(before), toLines(after));
  const near = (index: number) => lines.slice(Math.max(0, index - context), index + context + 1).some((l) => l.kind !== " ");
  const paint = (code: number, text: string) => (color ? `\x1b[${code}m${text}\x1b[0m` : text);
  const out: string[] = [];
  let skipped = false;
  lines.forEach((l, index) => {
    if (l.kind === " " && !near(index)) {
      if (!skipped) out.push("   …");
      skipped = true;
      return;
    }
    skipped = false;
    const row = `${String(l.line).padStart(4)} ${l.kind} ${l.text}`;
    out.push(l.kind === "-" ? paint(31, row) : l.kind === "+" ? paint(32, row) : row);
  });
  return out.join("\n");
}
