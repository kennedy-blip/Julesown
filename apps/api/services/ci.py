"""Failed checks: turning what CI reports about a commit into a task for the agent.

GitHub knows two kinds of checks. *Check runs* come from GitHub Actions and
other check apps; for an Actions job the log can be read, and any check run
can carry annotations (file, line, message). *Commit statuses* are what
deployment platforms such as Vercel post; they say that something failed and
link to a page, but there is no log to read - the agent has to reproduce the
failure in its sandbox.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .github import GitHubClient

# Hidden in the comment that announces an automatic fix. Counting these on a
# pull request says how often a fix was already attempted - without the server
# having to remember anything between requests.
_MARKER_RE = re.compile(r"<!-- jules-ci-fix:([0-9a-f]{7,40}) -->")
UNFINISHED_PREFIX = "[Unfinished] "
DETAILED_CHECKS = 4

CONTINUE_NOTE = (
    "A previous run started this work and stopped before it was finished. Look at what is there "
    "first (list the files and read the ones that matter), then complete the task and check that "
    "it works."
)


def fix_marker(sha: str) -> str:
    return f"<!-- jules-ci-fix:{sha} -->"


def attempted_fixes(comments: List[Dict[str, Any]]) -> List[str]:
    """The commits an automatic fix was already announced for, oldest first."""
    seen: List[str] = []
    for comment in comments:
        for sha in _MARKER_RE.findall(str(comment.get("body") or "")):
            if sha not in seen:
                seen.append(sha)
    return seen


def check_source(check: Dict[str, Any]) -> str:
    if check.get("kind") == "status":
        return "commit status"
    if check.get("app") == "github-actions":
        return "GitHub Actions"
    return f"check by {check['app']}" if check.get("app") else "check"


def names_of(checks: Dict[str, Any], limit: int = 6) -> str:
    """"build, Vercel and 2 more" - the failed checks in a few words."""
    names = [str(check["name"]) for check in checks.get("failing") or []]
    shown = names[:limit]
    more = int(checks.get("failed") or len(names)) - len(shown)
    return ", ".join(shown) + (f" and {more} more" if more > 0 else "")


def describe_failed_checks(github: GitHubClient, slug: str, checks: Dict[str, Any]) -> str:
    """The failed checks of a commit, with what GitHub can tell about each."""
    lines: List[str] = []
    for position, check in enumerate(checks.get("failing") or [], 1):
        line = f"{position}. {check['name']} ({check_source(check)}): {check['conclusion']}"
        if check.get("summary"):
            line += f" - {check['summary']}"
        lines.append(line)
        details = github.check_failure_details(slug, check) if position <= DETAILED_CHECKS else {"annotations": [], "log": ""}
        for annotation in details["annotations"]:
            lines.append(f"   - {annotation}")
        if details["log"]:
            lines.append("   Log, the part around the failure:")
            lines += ["   | " + text for text in details["log"].splitlines()]
        elif not details["annotations"]:
            page = f" (its page: {check['url']})" if check.get("url") else ""
            lines.append(
                f"   GitHub has no log for this check{page}. Reproduce the failure in the sandbox: install "
                "the dependencies and run the project's build and tests."
            )
    more = int(checks.get("failed") or 0) - len(checks.get("failing") or [])
    if more > 0:
        lines.append(f"... and {more} more failed checks.")
    return "\n".join(lines)


def build_fix_prompt(*, target: str, sha: str, checks_text: str, instructions: str = "") -> str:
    """What the agent is asked to do about failed checks."""
    parts = [
        f"The automated checks fail on {target} (commit {sha[:10]}). Find the cause and fix it.",
        f"Failed checks:\n{checks_text}",
        "How to go about it:\n"
        "- Reproduce the failure in the sandbox first: run the command that failed (the build, the "
        "tests, the linter).\n"
        "- Fix the cause in the code. Do not delete, skip or weaken tests and checks to make them pass, "
        "and change CI or deployment configuration only if that configuration is what is wrong.\n"
        "- Run the command again to confirm that it passes, and say in the summary what was wrong.\n"
        "- If the cause is not in the repository (a secret or environment variable missing on the CI or "
        "hosting service, an outage there), do not change code to hide it: finish, and say in the "
        "summary what has to be set up.",
    ]
    if instructions.strip():
        parts.append(f"Additional instructions:\n{instructions.strip()}")
    return "\n\n".join(parts)


def build_follow_up_prompt(
    *,
    instructions: str,
    number: int,
    title: str,
    head: str,
    base: str,
    checks_text: Optional[str] = None,
    continuing: bool = False,
) -> str:
    """Instructions for more work on a pull request that already exists."""
    parts = [instructions.strip()] if instructions.strip() else []
    if continuing:
        parts.append(CONTINUE_NOTE)
    shown = title[len(UNFINISHED_PREFIX) :] if title.startswith(UNFINISHED_PREFIX) else title
    parts.append(
        f"You are working on pull request #{number} \"{shown}\" (branch `{head}`, to be merged into "
        f"`{base}`). What you change is added to it as a new commit."
    )
    if checks_text:
        parts.append(f"These checks currently fail on it:\n{checks_text}")
    return "\n\n".join(parts)
