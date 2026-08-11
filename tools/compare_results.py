"""Numeric comparison of regenerated results against the committed record.

THE CI RESULTS GATE. After the suite reruns, this compares every file under
results/ and report/ against its committed (HEAD) version and fails if any
NUMBER moved — which is the gate's purpose: a moved number means the
committed record is stale.

WHY NOT `git diff`. The gate went through three versions, each failing for a
reason worth keeping:

  1. A verbatim diff fails on every run, because every record carries
     timestamp and git_sha fields that change BY DESIGN.
  2. `git diff -I'"timestamp"' -I'"git_sha"'` fixed the JSON records and then
     failed on report/order_of_accuracy.md, whose provenance lines are
     markdown (`- timestamp: ...`), not JSON — the pattern encoded the file
     format, not the intent.
  3. With both patterns fixed, the gate still fails across MACHINES: the
     moments runner's scipy `expm` produces trailing-digit differences
     between library versions, e.g. a metric of 4.6588e-7 locally against
     4.6610e-7 on the CI runner — a 2e-10 absolute difference in a quantity
     that sits at its own noise floor. Text comparison cannot distinguish
     that from a real regression; only a numeric comparison with a STATED
     tolerance can.

So: floats agree when |a - b| <= atol + rtol * max(|a|, |b|), with
rtol = 1e-3 and atol = 2e-7. Both numbers are MEASURED claims, stated here
so they can be argued with:

  atol = 2e-7: the largest cross-machine jitter observed in the first CI
     run was 1.07e-7 absolute, in a cumulant-slope metric whose value
     (~1e-4) sits at its own fitting noise floor. The OpenFOAM-produced
     records reproduced BIT-EXACTLY across machines; the jitter is in the
     pure-Python cases, through scipy's expm changing in trailing digits
     between versions.
  rtol = 1e-3: recorded convergence orders are ROUNDED to three decimals
     before being written, so a vanishing jitter underneath quantises to a
     visible 1e-3 step (2.001 vs 2.0 in the same CI run). One quantisation
     step at order ~2 must pass.

The gate's job is STALE-RECORD detection, not micro-regression detection —
the tests, with their own tolerances, do that. Any change worth
recommitting the record for exceeds these bounds by orders of magnitude.

Provenance keys (timestamp, git_sha) are skipped by NAME wherever they
appear; markdown provenance lines (`- timestamp:`, `- git:`) likewise. A
failure prints every offending field with both values and the difference, so
a red CI run says WHAT moved, not just that something did.
"""

import json
import re
import subprocess
import sys
from pathlib import Path

RTOL = 1e-3
ATOL = 2e-7
SKIP_KEYS = {"timestamp", "git_sha"}
MD_SKIP = re.compile(r"^- (timestamp|git): ")


def committed(path, repo):
    out = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        cwd=repo, capture_output=True, text=True,
    )
    return out.stdout if out.returncode == 0 else None


def compare_values(a, b, path, failures):
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            if key in SKIP_KEYS:
                continue
            if key not in a or key not in b:
                failures.append(f"{path}.{key}: present in only one version")
                continue
            compare_values(a[key], b[key], f"{path}.{key}", failures)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            failures.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b)):
            compare_values(x, y, f"{path}[{i}]", failures)
    elif isinstance(a, bool) or isinstance(b, bool):
        if a is not b:
            failures.append(f"{path}: {a} vs {b}")
    elif isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if abs(a - b) > ATOL + RTOL * max(abs(a), abs(b)):
            failures.append(f"{path}: {a!r} vs {b!r} (|diff| {abs(a - b):.3e})")
    else:
        if a != b:
            failures.append(f"{path}: {a!r} vs {b!r}")


def compare_file(rel, repo):
    old = committed(rel, repo)
    new_path = Path(repo) / rel
    if old is None:
        return [f"{rel}: not in HEAD (new file — commit it)"]
    if not new_path.exists():
        return [f"{rel}: committed but missing from the working tree"]
    new = new_path.read_text()
    failures = []
    if rel.endswith(".json"):
        compare_values(json.loads(old), json.loads(new), rel, failures)
    else:
        old_lines = [l for l in old.splitlines() if not MD_SKIP.match(l)]
        new_lines = [l for l in new.splitlines() if not MD_SKIP.match(l)]
        if old_lines != new_lines:
            for i, (x, y) in enumerate(zip(old_lines, new_lines)):
                if x != y:
                    failures.append(f"{rel}:{i}: {x!r} vs {y!r}")
            if len(old_lines) != len(new_lines):
                failures.append(f"{rel}: line count changed")
    return failures


def main():
    repo = Path(__file__).resolve().parents[1]
    tracked = subprocess.run(
        ["git", "ls-files", "results/", "report/"],
        cwd=repo, capture_output=True, text=True,
    ).stdout.split()
    all_failures = []
    for rel in tracked:
        all_failures.extend(compare_file(rel, repo))
    if all_failures:
        print(f"RESULTS GATE: {len(all_failures)} field(s) moved beyond "
              f"rtol={RTOL}, atol={ATOL}:")
        for f in all_failures:
            print("  ", f)
        print("A moved number means the committed record is stale: rerun the "
              "suite locally, inspect the change, and commit the new record "
              "WITH an explanation.")
        return 1
    print(f"results gate: {len(tracked)} files match the committed record "
          f"(rtol={RTOL}, atol={ATOL}; provenance keys skipped)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
