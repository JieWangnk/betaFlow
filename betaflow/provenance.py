"""Provenance stamps for results records."""

import subprocess

# Paths the harness itself rewrites on every run; their churn is not code dirt
# and must not mark a revalidation run "-dirty".
_OUTPUT_PREFIXES = ("results/", "report/")


def git_sha(repo):
    """12-char HEAD SHA, suffixed '-dirty' if code or case inputs differ from HEAD.

    Untracked files count as dirty (unlike `git diff`); the harness's own
    regenerated outputs (results/, report/) do not.
    """
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=repo, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        )
        # Porcelain lines are "XY path" — path starts at column 3.
        dirty = [
            line
            for line in status.splitlines()
            if line.strip() and not line[3:].startswith(_OUTPUT_PREFIXES)
        ]
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
