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

from .common import clip
from .github import GitHubClient

# Hidden in the comment that announces an automatic fix. Counting these on a
# pull request says how often a fix was already attempted - without the server
# having to remember anything between requests.
_MARKER_RE = re.compile(r"<!-- jules-ci-fix:([0-9a-f]{7,40}) -->")
UNFINISHED_PREFIX = "[Unfinished] "
DETAILED_CHECKS = 4

CONTINUE_NOTE = (
    "A previous run started this work and stopped before it was finished. What it did is in the "
    "repository now: do not start over. Look at what is there first (list the files and read the "
    "ones that matter), run the install and the build or tests to see where it stands and fix what "
    "fails. Then complete what is missing and check that it works."
)
# How a run that stopped by itself describes why (the agent called finish with success=false).
GAVE_UP = "The agent reported that it could not complete the task"


def build_handoff(*, summary: str, reason: str, last_failure: Optional[Dict[str, Any]] = None) -> str:
    """What a run that stopped early leaves for the run that continues it: the
    agent's own account, the command that still fails, and why it stopped.

    It travels with the task record. The dashboard sends it back with the
    follow-up task, so the next run starts from where this one was instead of
    finding everything out again - on a serverless platform that run is served
    by an instance that knows nothing about this one.
    """
    parts: List[str] = []
    if summary.strip():
        parts.append("Its own account when it stopped:\n" + clip(summary.strip(), 1500, tail_share=0.3))
    if last_failure and last_failure.get("command"):
        status = "timed out" if last_failure.get("timed_out") else f"exit code {last_failure.get('exit_code')}"
        output = str(last_failure.get("output") or "").strip() or "(no output)"
        parts.append(
            f"The last command that failed and has not passed since: `{last_failure['command']}` ({status}). "
            f"Its output ended with:\n{clip(output, 1500)}"
        )
    parts.append(f"Why it stopped: {reason.strip()}")
    return "\n\n".join(parts)


STANDING_INSTRUCTIONS = (
    "Standing instructions (they apply to every task here; where they contradict the task above, the task wins):"
)


def standing_instructions(*texts: str) -> str:
    """The block that carries the owner's own rules into a task: what the
    server is configured with (JULES_TASK_GUIDELINES), then what the caller
    sent along. Empty when there are none."""
    parts: List[str] = []
    for text in texts:
        cleaned = (text or "").strip()
        if cleaned and cleaned not in parts:
            parts.append(cleaned)
    return f"{STANDING_INSTRUCTIONS}\n" + "\n\n".join(parts) if parts else ""


# How the commit of unfinished work names the command that still failed.
STILL_FAILING = "Still failing when the run stopped: "


def failure_note(last_failure: Optional[Dict[str, Any]], limit: int = 900) -> str:
    """The command that still failed when a run stopped, for the message of the
    commit that keeps its work: the next run reads it there (see
    ``handoff_from_commit``), and so does whoever looks at the history."""
    if not last_failure or not last_failure.get("command"):
        return ""
    status = "timed out" if last_failure.get("timed_out") else f"exit code {last_failure.get('exit_code')}"
    command = " ".join(str(last_failure["command"]).split())
    lines = [line.rstrip() for line in str(last_failure.get("output") or "").splitlines() if line.strip()]
    # Indented: a line of output is never taken for a paragraph or a trailer of the message.
    output = "\n".join("    " + line for line in clip("\n".join(lines), limit).splitlines() if line.strip())
    return f"{STILL_FAILING}`{clip(command, 200, tail_share=0.0)}` ({status})" + (f"\n{output}" if output else "")


def handoff_from_commit(message: str) -> str:
    """What the commit of unfinished work says about where the work stood: its
    message without the title and the trailers. Empty for any other commit.

    This is what a continuation starts from when nobody passed the handoff of
    the earlier run along - a task started with ``/jules continue`` on GitHub,
    or from a browser that did not run the first one.
    """
    text = message.strip()
    if not text.startswith(UNFINISHED_PREFIX):
        return ""
    paragraphs = [part.strip("\n") for part in re.split(r"\n[ \t]*\n", text) if part.strip()]
    if len(paragraphs) < 3 or not re.search(r"^Task-Id: \S+", paragraphs[-1], re.MULTILINE):
        return ""  # not a commit a task made
    body = paragraphs[1:-1]
    if not any(part.startswith("Unfinished: ") for part in body):
        return ""
    return clip("\n\n".join(body), 3500, tail_share=0.5)


def continue_note(handoff: str = "") -> str:
    """What a task that continues unfinished work is told on top of its instructions."""
    if not handoff.strip():
        return CONTINUE_NOTE
    return f"{CONTINUE_NOTE}\n\nWhat the previous run left behind:\n{clip(handoff.strip(), 4000, tail_share=0.5)}"


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
    handoff: str = "",
) -> str:
    """Instructions for more work on a pull request that already exists."""
    parts = [instructions.strip()] if instructions.strip() else []
    if continuing:
        parts.append(continue_note(handoff))
    shown = title[len(UNFINISHED_PREFIX) :] if title.startswith(UNFINISHED_PREFIX) else title
    parts.append(
        f"You are working on pull request #{number} \"{shown}\" (branch `{head}`, to be merged into "
        f"`{base}`). What you change is added to it as a new commit."
    )
    if checks_text:
        parts.append(f"These checks currently fail on it:\n{checks_text}")
    return "\n\n".join(parts)
