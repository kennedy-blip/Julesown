"""The coding agent: an LLM tool-calling loop that works inside the sandbox.

The agent can read, search and edit files and run commands in a real clone of
the repository. It cannot commit, push or open pull requests - once it calls
``finish`` the task runner inspects the work tree, commits it through the
GitHub API and opens the pull request. Every tool result the model sees is the
real output of the sandbox; nothing is simulated.

A repository without code (empty, or holding only a README, licence or
``.gitignore``) is a different job: the agent builds a project from scratch.
It then works from ``NEW_PROJECT_PROMPT``, which asks for a conventional,
deployable layout instead of the smallest change to existing code.
"""
from __future__ import annotations

import json
import posixpath
import re
import shlex
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

from .common import Deadline, Emit, clip, first_line, plain_text, strip_ansi
from .llm import ChatClient, LLMError, LLMResponse, ToolCall
from .sandbox import MAX_CHANGED_FILES, MAX_FILE_BYTES, KeyFile, RepoCheckout, SandboxError
from .verification import VerificationRun, no_tests_reason, tests_status_of

MAX_READ_LINES = 400
MAX_NUDGES = 2
# How often one model call is repeated when the provider refuses the model's
# reply as an invalid tool call (a known habit of some hosted open models).
MAX_FORMAT_RETRIES = 3
# How often one model call may move on to another provider (several can be
# configured as fallbacks) before the failure is final.
MAX_PROVIDER_SWITCHES = 6
MAX_REPORTED_FILES = 60
# When this little of the time budget is left, the model is asked once to wrap
# up: to say what is done and what is missing. What it has changed is kept
# either way (the runner delivers it marked as unfinished), and its report is
# what a follow-up run continues from. The larger value applies once the
# provider's rate limit has made the run wait: the next reply may be a minute away.
WRAP_UP_SECONDS = 40.0
WRAP_UP_SECONDS_RATE_LIMITED = 80.0
# Once this share of the agent's time is used it is told how much is left, so
# that it checks what it has (build, tests) while there is time to fix it.
# Runs shorter than TIME_CHECK_MIN_SECONDS get no such reminder.
TIME_CHECK_SHARE = 0.5
TIME_CHECK_MIN_SECONDS = 120.0
TIME_UP = "The time budget ran out before the agent finished."
INITIAL_CONTEXT_CHARS = 24_000
MAX_CONTEXT_CHARS = 160_000
MIN_CONTEXT_CHARS = 8_000
# Files the agent wrote stay in the conversation word for word only while it is
# shorter than this; beyond it the oldest large ones are replaced by an outline
# (see ``outline_of``). Every request repeats the whole conversation, so a
# project of thirty files would otherwise be sent thirty times over.
SOFT_CONTEXT_CHARS = 48_000
OUTLINE_MIN_CHARS = 1_500
# How long one model call may take. With other providers to fall back on the
# pool decides (see FailoverClient.patience): a provider that hangs is given up
# on sooner than one that has been answering.
MODEL_CALL_SECONDS = 90.0
_TPM_LIMIT_RE = re.compile(r"Limit\s+(\d+)", re.IGNORECASE)

COMMAND_ENV = {
    "CI": "true",
    "TERM": "dumb",
    "NO_COLOR": "1",
    "GIT_TERMINAL_PROMPT": "0",
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUNBUFFERED": "1",
    # Python validates cached bytecode by source mtime (1 s resolution) and size.
    # An edit made right after a test run that keeps the file size ("a - b" ->
    # "a + b") would otherwise be invisible to the next run, which would then
    # report the result of the *old* code.
    "PYTHONDONTWRITEBYTECODE": "1",
}

SYSTEM_PROMPT = """You are Jules, an autonomous software engineer. You are working in a Linux sandbox \
on a fresh clone of a GitHub repository, and you act only through the provided tools.

How to work:
- Look before you write. Read the files your change depends on or describes (the manifest, the \
source code, the README) and base everything you write on what they actually contain. Never invent \
dependencies, scripts, commands, versions, licences or features: what the repository does not show, \
leave out or describe as not specified.
- Make the smallest change that fully solves the task and match the existing style. Paths are \
relative to the repository root. Use replace_in_file for small edits to existing files and \
write_file for new files or full rewrites.
- Verify code changes with run_tests, using the project's own test command (install dependencies \
first if needed). Check first that the project has tests. If it has none - for example the "test" \
script in package.json is npm's placeholder that only prints "Error: no test specified" - do not \
run a test command: for a code change do a quick sanity check with run_command instead (such as a \
syntax check) and say in your summary that the project has no automated tests. A change that only \
touches documentation needs no install, build or test run.
- When a command you rely on fails (an install, the build, the tests), fix what it reports before \
you do anything else: read the error, change only what it points at and run the command again.
- Reply only by calling the provided tools; no other tools exist. Explanations belong in the \
summary you pass to finish.
- Do not run git commit, git push or branch commands and do not try to open a pull request: after \
you call finish, the platform commits your working tree and opens the pull request for you.
- Stay inside the repository and never print credentials or environment secrets.
- When the work is done call finish with a short summary of what changed and why, and a pull \
request title. If the task cannot be done, call finish with success=false and explain why."""

# A repository without code: the task is to build a project, not to change one.
NEW_PROJECT_PROMPT = """You are Jules, an autonomous software engineer. You are working in a Linux sandbox \
in a GitHub repository that has no code yet: you are creating the project from scratch, and you act \
only through the provided tools.

How to work:
- Build what the task asks for: complete and working rather than broad and half-finished. If the \
task leaves the stack open, choose the simplest one that fits (plain HTML, CSS and JavaScript for \
a small site or browser app).
- Lay the project out the way its stack expects, so that a hosting platform (Vercel, Netlify, \
Render, GitHub Pages ...) can deploy it without extra configuration: a static site has index.html \
at the repository root; a Node project has a package.json with a name, a version, the scripts \
platforms run (build, start, dev) and only the dependencies it really uses; a Python app has \
requirements.txt and a documented start command. Read configuration from environment variables \
(the port from PORT). Never hard-code secrets or localhost addresses. Add platform-specific config \
files or CI workflows only if the task asks for them.
- Always add a README.md (what the project is, how to run it locally, how to deploy it) and a \
.gitignore that fits the stack. If the app needs environment variables, list them in .env.example \
- never commit a real .env.
- Paths are relative to the repository root. You can write several files in one reply; a file \
you wrote is on disk as you sent it, so do not read it back. Keep files lean: sample or mock data \
a handful of entries, no filler code, no long comments - what you write costs time, and the time \
of a run is limited.
- With a build step (a bundler, TypeScript, a compiler) work in this order: the manifest, the \
configuration, the .gitignore and a minimal entry point first; then install and run the build once, \
while a mistake in the configuration is cheap to find; then the features; then the build again. When an \
install, a build or a test fails, fix what it reports before you write anything else: change only \
what the error points at and run the command again.
- Check your work before you finish with run_command, using what the stack offers: install and \
build, or a syntax check (for example node --check). If you wrote tests, run them with run_tests. \
Say in the summary what you ran and what you could not check.
- Reply only by calling the provided tools; no other tools exist. Explanations belong in the \
summary you pass to finish.
- Do not run git commit, git push or branch commands: {delivery}
- Stay inside the repository and never print credentials or environment secrets.
- When the work is done call finish with a short summary of what you built and how to run it, and \
a title (pr_title). If the task cannot be done, call finish with success=false and explain why."""

_DELIVERY_FIRST_COMMIT = (
    "after you call finish, the platform commits your working tree as the repository's first commit."
)
_DELIVERY_PULL_REQUEST = (
    "after you call finish, the platform commits your working tree and opens the pull request for you."
)


_DELIVERY_UPDATE = (
    "after you call finish, the platform adds your working tree as a new commit to the existing pull request."
)


def system_prompt_for(*, new_project: bool, first_commit: bool, update: bool = False) -> str:
    """The instructions for this kind of repository and this way of delivering.

    ``update``: the work goes onto the branch of a pull request that already
    exists instead of into a new one.
    """
    if not new_project:
        if not update:
            return SYSTEM_PROMPT
        return SYSTEM_PROMPT.replace(
            "after you call finish, the platform commits your working tree and opens the pull request for you.",
            _DELIVERY_UPDATE,
        ).replace("and a pull request title.", "and a title for the commit (pr_title).")
    if first_commit:
        return NEW_PROJECT_PROMPT.format(delivery=_DELIVERY_FIRST_COMMIT)
    return NEW_PROJECT_PROMPT.format(delivery=_DELIVERY_UPDATE if update else _DELIVERY_PULL_REQUEST)


AGENT_TOOLS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files in the repository (tracked and new files, ignoring build output).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Directory to list. Defaults to the repository root."}
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a text file, optionally only a range of lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the repository root."},
                    "start_line": {"type": "integer", "description": "First line to return (1-based)."},
                    "end_line": {"type": "integer", "description": "Last line to return."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search the repository for an exact string. Returns matching lines as path:line:text.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Exact text to search for."},
                    "path": {"type": "string", "description": "Limit the search to this file or directory."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Create a file or replace its entire content.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the repository root."},
                    "content": {"type": "string", "description": "The complete new content of the file."},
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "replace_in_file",
            "description": (
                "Replace one exact occurrence of old_text with new_text in a file. old_text must match "
                "the file exactly (including whitespace) and must be unique within it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path relative to the repository root."},
                    "old_text": {"type": "string", "description": "Exact text to replace."},
                    "new_text": {"type": "string", "description": "Replacement text."},
                },
                "required": ["path", "old_text", "new_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": (
                "Run a shell command in the repository root (install dependencies, build, inspect). "
                "Returns the exit code and output."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "The bash command to run."},
                    "timeout_seconds": {"type": "integer", "description": "Optional time limit."},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": (
                "Run the project's own automated tests to verify a code change. Use it only when the "
                "project really has tests. The result is recorded and shown in the pull request."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Test command, e.g. 'pytest -q' or 'npm test'."}
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Signal that the task is complete (or cannot be completed).",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {"type": "string", "description": "What changed and why, in a few sentences."},
                    "pr_title": {"type": "string", "description": "Concise pull request title."},
                    "success": {
                        "type": "boolean",
                        "description": "false if the task could not be completed. Defaults to true.",
                    },
                },
                "required": ["summary"],
            },
        },
    },
]


TOOL_NAMES = tuple(tool["function"]["name"] for tool in AGENT_TOOLS)


class ToolError(Exception):
    """A tool call could not be carried out; the message goes back to the model.

    ``log`` is a shorter wording for the task log when the message to the model
    is long, and ``level`` its log level: ``warn`` for a mistake of the model,
    ``info`` when the platform is merely steering it (see ``_check_blind_write``).
    """

    def __init__(self, message: str, *, log: Optional[str] = None, level: str = "warn") -> None:
        super().__init__(message)
        self.log = log
        self.level = level


@dataclass
class AgentOutcome:
    """How the loop ended. ``stop_reason`` is one of ``finished``, ``gave_up``,
    ``max_turns``, ``timeout``, ``llm_error`` or ``sandbox_error``."""

    stop_reason: str
    summary: str = ""
    pr_title: str = ""
    turns: int = 0
    error: Optional[str] = None
    model: str = ""
    verification: List[VerificationRun] = field(default_factory=list)
    # Repository files whose content the model was shown, in the order it saw them.
    files_read: List[str] = field(default_factory=list)
    # Files the agent wrote or edited with its own tools (as opposed to what a
    # command it ran left behind: installed dependencies, build output).
    files_written: List[str] = field(default_factory=list)
    # The last command that failed and was not run again with success since:
    # ``command``, ``exit_code``, ``timed_out`` and the end of its ``output``.
    # It is where a follow-up run picks the work up.
    last_failure: Optional[Dict[str, Any]] = None
    # The models that answered when the configured name is a router or an alias
    # that chooses for each request (openrouter/free, gemini-flash-latest): name -> answers.
    served: Dict[str, int] = field(default_factory=dict)

    @property
    def finished(self) -> bool:
        return self.stop_reason == "finished"

    @property
    def tests_status(self) -> str:
        """``passed``, ``failed``, ``no_tests`` or ``not_run`` (see ``tests_status_of``)."""
        return tests_status_of(self.verification)


def history_size(messages: List[Dict[str, Any]]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            total += len(content)
        for call in message.get("tool_calls") or []:
            total += len(call.get("function", {}).get("arguments") or "")
    return total


_OUTLINE_LINE_RE = re.compile(
    r"^\s{0,2}(?:export\b|module\.exports\b|def\s|async\s+def\s|class\s|function\s|async\s+function\s|"
    r"interface\s|type\s+[A-Z]|enum\s|struct\s|fn\s|func\s|pub\s|#{1,3}\s)"
)
_OUTLINED_MARK = "["
# What compaction puts in the place of old file content: the outline note at
# the start (see ``_outlined_write``) or the cut mark at the end (see
# ``compact_history``). A model that copies its own earlier call would write
# that note into the file, so content that carries one is refused.
_STAND_IN_RE = re.compile(
    r"\A\[\d+ lines, written to [^\n]{1,300} as sent\. They are not repeated here"
    r"|\[\.\.\. \d+ characters omitted \.\.\.\]\s*\Z"
)


def is_stand_in(content: str) -> bool:
    """Whether ``content`` is what the conversation keeps of an older file
    instead of the file's content."""
    return bool(_STAND_IN_RE.search(content))


def outline_of(content: str, *, limit: int = 1200) -> str:
    """What a file declares, in a few lines: its exports, functions, classes
    and headings - enough to remember what is in it without repeating it."""
    lines = content.splitlines()
    picked = [line.rstrip()[:140] for line in lines if _OUTLINE_LINE_RE.match(line)]
    if not picked:
        picked = [line.rstrip()[:140] for line in lines[:8] if line.strip()]
    text = "\n".join(picked[:40])
    if len(text) > limit:
        text = text[:limit].rsplit("\n", 1)[0] + "\n..."
    return text


def _outlined_write(path: str, content: str) -> str:
    """What stands in for the content of a file the agent wrote some turns ago."""
    count = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
    # The model can no longer see what is in the file: before it changes the
    # file it has to look, or it would rewrite it from this outline.
    return (
        f"{_OUTLINED_MARK}{count} lines, written to {path or 'the file'} as sent. They are not repeated "
        "here to keep the conversation short: read the file with read_file before you change it. "
        "It declares:]\n" + outline_of(content)
    )


def _outline_old_writes(messages: List[Dict[str, Any]], target: int, *, keep_recent: int) -> bool:
    """Replace the content of files written a while ago by an outline, oldest
    first, until the history is no longer than ``target`` characters.

    The model's latest reply is left as it is, however many files it wrote:
    what it does next (run the build, fix what that reports) is about them.
    """
    changed = False
    latest_reply = max(
        (index for index, message in enumerate(messages) if message.get("role") == "assistant"), default=len(messages)
    )
    for index in range(2, max(2, min(len(messages) - keep_recent, latest_reply))):
        message = messages[index]
        if message.get("role") != "assistant":
            continue
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            raw = function.get("arguments") or ""
            if function.get("name") != "write_file" or len(raw) <= OUTLINE_MIN_CHARS:
                continue
            try:
                arguments = json.loads(raw)
            except ValueError:
                continue
            content = arguments.get("content") if isinstance(arguments, dict) else None
            if not isinstance(content, str) or len(content) <= OUTLINE_MIN_CHARS:
                continue
            arguments["content"] = _outlined_write(str(arguments.get("path") or ""), content)
            function["arguments"] = json.dumps(arguments)
            # Provider-native blocks (Anthropic's content, Gemini's parts)
            # would no longer match the shortened call.
            for key in [key for key in message if key.startswith("_")]:
                message.pop(key, None)
            changed = True
        if history_size(messages) <= target:
            break
    return changed


def compact_history(
    messages: List[Dict[str, Any]], budget: int, *, keep_recent: int = 4, soft_budget: Optional[int] = None
) -> bool:
    """Shrink old tool traffic until the history fits ``budget`` characters.

    Above ``soft_budget`` (if given) the files the agent wrote a while ago are
    reduced to an outline: the model wrote them, the sandbox has them, and
    every further request would otherwise carry them again. Above ``budget``
    old tool output and what is left of large arguments is cut as well.
    The system prompt, the task and the most recent messages are never touched.
    Returns True if anything was shortened.
    """
    changed = False
    target = min(budget, soft_budget) if soft_budget else budget
    if history_size(messages) > target:
        changed = _outline_old_writes(messages, target, keep_recent=keep_recent)
    if history_size(messages) <= budget:
        return changed
    for index in range(2, max(2, len(messages) - keep_recent)):
        message = messages[index]
        if message.get("role") == "tool":
            content = message.get("content") or ""
            if len(content) > 260:
                message["content"] = content[:200] + "\n[... older tool output dropped to save context ...]"
                changed = True
        elif message.get("role") == "assistant":
            native = [key for key in message if key.startswith("_")]
            for call in message.get("tool_calls") or []:
                function = call.get("function", {})
                raw = function.get("arguments") or ""
                if len(raw) <= 400:
                    continue
                try:
                    arguments = json.loads(raw)
                except ValueError:
                    continue
                if not isinstance(arguments, dict):
                    continue
                for key, value in list(arguments.items()):
                    if isinstance(value, str) and len(value) > 200:
                        arguments[key] = value[:120] + f" [... {len(value) - 120} characters omitted ...]"
                function["arguments"] = json.dumps(arguments)
                # Provider-native blocks (Anthropic's content, Gemini's parts)
                # would no longer match the shortened calls.
                for key in native:
                    message.pop(key, None)
                native = []
                changed = True
        if history_size(messages) <= budget:
            break
    return changed


_RATE_LIMIT_SCOPE_RE = re.compile(
    r"\b(tokens|requests) per (minute|day)\b[^:]{0,20}:\s*Limit\s+(\d+)", re.IGNORECASE
)


def _rate_limit_scope(message: str, tokens_per_minute: Optional[int]) -> str:
    """Which limit was hit, in words ("8,000 tokens per minute"), if known."""
    match = _RATE_LIMIT_SCOPE_RE.search(message or "")
    if match:
        return f"{int(match.group(3)):,} {match.group(1).lower()} per {match.group(2).lower()}"
    if tokens_per_minute:
        return f"{tokens_per_minute:,} tokens per minute"
    return ""


def _refused_reply_detail(exc: LLMError) -> str:
    if exc.rejected_tool:
        return f"it called '{exc.rejected_tool}', which is not one of its tools"
    return exc.message.strip().rstrip(".")


def _refused_reply_correction(exc: LLMError) -> str:
    """What the model is told before a refused reply is requested again."""
    if exc.rejected_tool:
        problem = f"it called a tool named '{exc.rejected_tool}', which does not exist"
    else:
        problem = f"it was not a valid tool call. The provider reported: {exc.message.strip()}"
    parts = [
        f"Your previous reply was rejected and nothing was executed: {problem.rstrip('.')}.",
        f"Reply again with a call to one of these tools: {', '.join(TOOL_NAMES)}.",
        "There is no tool for comments or for free-form JSON: put explanations in the summary "
        "argument of finish.",
    ]
    if exc.failed_generation:
        parts.append("This is what you sent:\n" + clip(exc.failed_generation.strip(), 600, tail_share=0.3))
    return " ".join(parts)


def _storable_assistant_message(response: LLMResponse) -> Optional[Dict[str, Any]]:
    """The assistant message in a form every provider accepts back.

    Providers validate the history they are sent: a past call to a tool that
    was never offered, or one whose arguments are not valid JSON, makes them
    refuse every later request of the run. Such calls are therefore not kept
    as calls - the model is told about them in a plain message instead.
    Returns None when nothing of the message is left.
    """
    unknown = {call.id for call in response.tool_calls if call.name not in TOOL_NAMES}
    unreadable = {call.id for call in response.tool_calls if call.error}
    message = response.assistant_message
    if not response.tool_calls and not (message.get("content") or "").strip():
        return None  # an empty reply: providers refuse an assistant turn without content
    if not unknown and not unreadable:
        return message
    # Provider-native blocks would still contain the calls that are removed here.
    message = {key: value for key, value in message.items() if not key.startswith("_")}
    kept: List[Dict[str, Any]] = []
    for wire in message.get("tool_calls") or []:
        if wire.get("id") in unknown:
            continue
        if wire.get("id") in unreadable:
            wire = {**wire, "function": {**(wire.get("function") or {}), "arguments": "{}"}}
        kept.append(wire)
    if kept:
        message["tool_calls"] = kept
        return message
    message.pop("tool_calls", None)
    return message if message.get("content") else None


def format_key_files(key_files: Sequence[KeyFile]) -> str:
    """The manifests (and README excerpt) of the repository, for the first message."""
    if not key_files:
        return ""
    lines = [
        "Key files of the repository, read just now (use read_file for any other file, and for the "
        "rest of a file shown only in part):"
    ]
    for item in key_files:
        size = f"{item.size} bytes" if not item.truncated else f"beginning only; the file has {item.size} bytes"
        lines += ["", f"--- {item.path} ({size}) ---", item.content.rstrip("\n")]
    return "\n".join(lines)


class CodingAgent:
    def __init__(
        self,
        *,
        llm: ChatClient,
        checkout: RepoCheckout,
        emit: Emit,
        deadline: Deadline,
        max_turns: int = 30,
        command_timeout: float = 120.0,
        max_output_tokens: int = 8192,
        reserve_seconds: float = 25.0,
        sleep: Callable[[float], None] = time.sleep,
        new_project: bool = False,
        update: bool = False,
    ) -> None:
        self.llm = llm
        # The work is added to a pull request that already exists.
        self.update = update
        self.checkout = checkout
        self.workspace = checkout.workspace
        # The repository has no code: the agent builds a project from scratch.
        # Always the case for an empty repository.
        self.new_project = bool(new_project or getattr(checkout, "empty", False))
        self.emit = emit
        self.deadline = deadline
        self.max_turns = max_turns
        self.command_timeout = command_timeout
        self.reserve_seconds = reserve_seconds
        self._sleep = sleep
        self._configured_output_tokens = max_output_tokens
        # Limits differ from provider to provider, and a run can use several
        # (fallbacks): what was learnt about one must not size requests to another.
        self._tpm_by_provider: Dict[str, int] = {}
        self._scale_by_provider: Dict[str, float] = {}
        self.context_budget = INITIAL_CONTEXT_CHARS
        self.max_output_tokens = max_output_tokens
        self._recompute_budgets()
        self.verification: List[VerificationRun] = []
        # What the model has looked at. It decides whether a write is "blind".
        self.files_read: List[str] = []
        self.files_written: set = set()
        self._known_content: set[str] = set()  # files whose current content the model has seen
        self._inspections = 0  # read_file / search_code / run_command calls so far
        self._blind_write_refused = False
        self._overwrite_refused: set[str] = set()
        self._unwritten: set[str] = set()  # writes that were refused and not repeated since
        self._finish_questioned = False
        self._gaps_questioned = False  # a new project was sent back once for what it lacks
        self._wrap_up_sent = False  # the model was told that time is nearly up
        self._time_check_sent = False  # ... and, earlier, how much time is left
        self._agent_seconds = 0.0  # the time the agent had when it started
        self._rate_limited = False  # the provider's rate limit has made this run wait
        # Commands that failed, by command line, with their position among the
        # commands run: a failure counts until the same command passes.
        self._commands_run = 0
        self._failed_commands: Dict[str, tuple[int, Dict[str, Any]]] = {}
        self._last_command_failed = False
        self._answering_model = ""  # the model that answered last, when a router chose it
        self._served: Dict[str, int] = {}  # every model a router chose, with how often it answered

    # ------------------------------------------------------------------ #
    # Budgets
    # ------------------------------------------------------------------ #
    def _time_left(self) -> float:
        return self.deadline.remaining() - self.reserve_seconds

    @property
    def _provider(self) -> str:
        """The provider the next request goes to."""
        return str(getattr(self.llm, "provider", "") or "")

    @property
    def _tokens_per_minute(self) -> Optional[int]:
        """The tokens-per-minute limit of the provider in use: what it reported,
        else what its client says may be assumed, else unknown."""
        known = self._tpm_by_provider.get(self._provider)
        if known:
            return known
        assumed = getattr(self.llm, "assumed_tokens_per_minute", None)
        return int(assumed) if isinstance(assumed, (int, float)) and assumed > 0 else None

    @property
    def _budget_scale(self) -> float:
        return self._scale_by_provider.get(self._provider, 1.0)

    @property
    def _has_fallback(self) -> bool:
        """Whether a second provider is configured (a provider's own second
        model is no help when the key, the account or the service is the problem)."""
        chain = getattr(self.llm, "chain", None) or ()
        return len({str(label).split(" / ", 1)[0] for label in chain}) > 1

    def _recompute_budgets(self) -> None:
        """Size the context window (characters) and reply length (tokens).

        Providers enforce a tokens-per-minute limit - 8k on Groq's free tier,
        hundreds of thousands on paid plans - and a single request may not
        exceed it. Until the provider reports its limit the budgets stay
        conservative; ``_budget_scale`` shrinks them after a "request too
        large" rejection so the same mistake is not repeated.
        """
        if self._tokens_per_minute:
            context_chars = int(self._tokens_per_minute * 0.45) * 4
            output_tokens = int(self._tokens_per_minute * 0.3)
        else:
            context_chars = INITIAL_CONTEXT_CHARS
            output_tokens = 4096
        self.context_budget = max(
            MIN_CONTEXT_CHARS // 2, min(MAX_CONTEXT_CHARS, int(context_chars * self._budget_scale))
        )
        self.max_output_tokens = max(
            512,
            min(self._configured_output_tokens, int(output_tokens * max(self._budget_scale, 0.5))),
        )

    def _apply_token_limit(self, tokens_per_minute: Optional[int], provider: str = "") -> None:
        if tokens_per_minute and tokens_per_minute > 0:
            self._tpm_by_provider[provider or self._provider] = tokens_per_minute
        self._recompute_budgets()

    @property
    def _result_limit(self) -> int:
        return max(2500, min(12_000, self.context_budget // 4))

    # ------------------------------------------------------------------ #
    # Tools
    # ------------------------------------------------------------------ #
    def _resolve(self, raw_path: Any) -> tuple[str, str]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ToolError("path is required")
        root = self.checkout.directory.rstrip("/")
        candidate = raw_path.strip()
        absolute = posixpath.normpath(
            candidate if candidate.startswith("/") else posixpath.join(root, candidate)
        )
        if absolute != root and not absolute.startswith(root + "/"):
            raise ToolError(f"{raw_path} is outside the repository")
        relative = posixpath.relpath(absolute, root)
        if relative == ".git" or relative.startswith(".git/"):
            raise ToolError("the .git directory is managed by the platform")
        return absolute, relative

    def _tool_list_files(self, args: Dict[str, Any]) -> str:
        _absolute, relative = self._resolve(args.get("path") or ".")
        result = self.workspace.run(
            f"git ls-files --cached --others --exclude-standard -- {shlex.quote(relative)} | head -n 401",
            cwd=self.checkout.directory,
            timeout=30.0,
        )
        files = [line for line in result.stdout.splitlines() if line.strip()]
        self.emit("tool", f"list_files {relative} ({len(files)} entries)")
        if not files:
            return f"No files found under {relative}."
        listing = "\n".join(files[:400])
        if len(files) > 400:
            listing += "\n[... more files not shown; list a subdirectory ...]"
        return clip(listing, self._result_limit, tail_share=0.0)

    def _tool_read_file(self, args: Dict[str, Any]) -> str:
        absolute, relative = self._resolve(args.get("path"))
        try:
            start = max(1, int(args.get("start_line") or 1))
            end = int(args.get("end_line") or 0) or start + MAX_READ_LINES - 1
        except (TypeError, ValueError):
            raise ToolError("start_line and end_line must be integers")
        end = max(start, min(end, start + MAX_READ_LINES - 1))
        quoted = shlex.quote(absolute)
        script = (
            f"f={quoted}; "
            'if [ -d "$f" ]; then echo __JULES_DIR__; exit 0; fi; '
            'if [ ! -f "$f" ]; then echo __JULES_MISSING__; exit 0; fi; '
            'if [ -s "$f" ] && ! grep -Iq \'\' "$f"; then echo "__JULES_BINARY__ $(wc -c < "$f")"; exit 0; fi; '
            'echo "__JULES_LINES__ $(wc -l < "$f")"; '
            f"sed -n '{start},{end}p' \"$f\""
        )
        result = self.workspace.run(script, cwd=self.checkout.directory, timeout=30.0)
        marker, _, body = result.stdout.partition("\n")
        if marker.startswith("__JULES_MISSING__"):
            raise ToolError(f"{relative} does not exist")
        if marker.startswith("__JULES_DIR__"):
            raise ToolError(f"{relative} is a directory; use list_files")
        if marker.startswith("__JULES_BINARY__"):
            self.emit("tool", f"read_file {relative} (binary)")
            return f"{relative} is a binary file ({marker.split()[-1]} bytes); it cannot be shown as text."
        if not marker.startswith("__JULES_LINES__"):
            raise ToolError(f"could not read {relative}: {result.output()[-200:] or 'no output'}")
        try:
            total_lines = int(marker.split()[-1])
        except ValueError:
            total_lines = 0

        lines = body.split("\n")
        if lines and lines[-1] == "":
            lines.pop()
        shown: List[str] = []
        used = 0
        for line in lines:
            if shown and used + len(line) + 1 > self._result_limit:
                break
            shown.append(line)
            used += len(line) + 1
        last = start + len(shown) - 1
        self.emit("tool", f"read_file {relative} (lines {start}-{max(last, start)})")
        self._note_read(relative)
        if not shown:
            return f"{relative} has no content in lines {start}-{end} (the file has {total_lines} lines)."
        header = f"{relative} - lines {start}-{last} of {max(total_lines, last)}"
        footer = ""
        if last < total_lines:
            footer = f"\n[more lines follow; call read_file with start_line={last + 1}]"
        return f"{header}\n" + "\n".join(shown) + footer

    def _tool_search_code(self, args: Dict[str, Any]) -> str:
        query = args.get("query")
        if not isinstance(query, str) or not query:
            raise ToolError("query is required")
        _absolute, relative = self._resolve(args.get("path") or ".")
        result = self.workspace.run(
            "git grep -n -I --no-color --untracked -F -e "
            f"{shlex.quote(query)} -- {shlex.quote(relative)} | cut -c1-240 | head -n 80",
            cwd=self.checkout.directory,
            timeout=30.0,
        )
        matches = [line for line in result.stdout.splitlines() if line.strip()]
        self.emit("tool", f"search_code {query!r} ({len(matches)} matches)")
        self._inspections += 1
        if not matches:
            return f"No matches for {query!r}."
        return clip("\n".join(matches), self._result_limit, tail_share=0.0)

    def _tool_write_file(self, args: Dict[str, Any]) -> str:
        absolute, relative = self._resolve(args.get("path"))
        content = args.get("content")
        if not isinstance(content, str):
            raise ToolError("content must be a string")
        if is_stand_in(content):
            if relative not in self._known_content:
                self._unwritten.add(relative)  # it is still to be written
            raise ToolError(
                f"{relative} was not written: what you sent is not the content of a file but the note "
                "that stands in for one in this conversation. Older file contents are shortened here to "
                "save space; the files themselves are complete on disk. Send the whole content of the "
                "file - or, if it is on disk already as it should be, leave it alone.",
                log=f"write_file {relative} held back: the content was the conversation's shortened copy, not a file",
            )
        self._check_blind_write(absolute, relative)
        self.workspace.write_file(absolute, content)
        self._known_content.add(relative)
        self.files_written.add(relative)
        self._unwritten.discard(relative)
        line_count = content.count("\n") + (0 if content.endswith("\n") or not content else 1)
        self.emit("edit", f"write_file {relative} ({line_count} lines)")
        return f"Wrote {relative} ({line_count} lines)."

    def _note_read(self, relative: str) -> None:
        self._inspections += 1
        self._known_content.add(relative)
        if relative not in self.files_read and len(self.files_read) < MAX_REPORTED_FILES:
            self.files_read.append(relative)

    def _existing_line_count(self, absolute: str, relative: str = "") -> Optional[int]:
        """Lines of the file at ``absolute`` if it exists and has content, else None
        (replacing an empty file discards nothing).

        While a new project is being built, only files of the repository itself
        count: one that a command generated during this run (``npm init``, a
        project template) holds nothing of the user's that could be lost.
        """
        script = f'f={shlex.quote(absolute)}; if [ -s "$f" ] && [ -f "$f" ]; then wc -l < "$f"; else echo none; fi'
        if self.new_project:
            if not self.checkout.base_sha:
                return None  # the repository was empty: nothing in it predates this run
            in_base = f"{self.checkout.base_sha}:{relative}"
            script = f"if git cat-file -e {shlex.quote(in_base)} 2>/dev/null; then {script}; else echo none; fi"
        result = self.workspace.run(script, cwd=self.checkout.directory, timeout=30.0)
        try:
            return max(1, int(result.stdout.strip()))
        except ValueError:
            return None

    def _check_blind_write(self, absolute: str, relative: str) -> None:
        """Stop the two ways of writing without looking, once each.

        A model that writes before it reads produces text that sounds right and
        is not (a description of a project's dependencies, say, written without
        opening its manifest). The model may insist - the same call is accepted
        the second time - but it is told first what it skipped.
        """
        if relative in self._known_content or relative in self._overwrite_refused:
            return
        existing_lines = self._existing_line_count(absolute, relative)
        if existing_lines is not None:
            self._overwrite_refused.add(relative)
            self._unwritten.add(relative)
            raise ToolError(
                f"{relative} already exists ({existing_lines} lines) and you have not read it in this "
                "run, so writing it now would discard content you have not seen. Read it first; to "
                "change part of it use replace_in_file. If you do mean to replace the whole file, "
                "call write_file again.",
                log=f"write_file {relative} held back: the existing file has not been read - the agent was asked to read it first",
                level="info",
            )
        if self.new_project:
            return  # there is no code to look at first: the project is being created
        if self._inspections == 0 and not self._blind_write_refused:
            self._blind_write_refused = True
            self._unwritten.add(relative)
            raise ToolError(
                f"{relative} was not written: you have not opened any file of this repository yet. "
                "Read the files this change describes or depends on first (for example the manifest "
                "and the relevant source files) so that what you write matches the project, then "
                "call write_file again. If the file needs no knowledge of the repository, simply "
                "call write_file again.",
                log=f"write_file {relative} held back: nothing has been read yet - the agent was asked to look at the repository first",
                level="info",
            )

    def _new_project_gaps(self) -> List[str]:
        """What a freshly built project still lacks before it is committed.

        Only things that can be checked for certain and that bite later: no
        README (nobody knows how to run or deploy it), dependencies without a
        ``.gitignore``, a ``package.json`` no tool can parse, a ``.env`` that
        would be committed - and what would make the commit itself fail (a file
        above the size limit, such as the lock file of a large dependency tree,
        or more files than a task may commit), while the agent can still do
        something about it. Best effort: an unreadable answer means no gaps.
        """
        script = (
            # Files that would be committed: everything git does not ignore.
            "git ls-files -co --exclude-standard 2>/dev/null | { n=0; while IFS= read -r f; do "
            'n=$((n+1)); if [ -f "$f" ] && [ "$(wc -c < "$f")" -gt '
            + str(MAX_FILE_BYTES)
            + ' ]; then echo "TOO_BIG $f"; fi; done; '
            + f'if [ "$n" -gt {MAX_CHANGED_FILES} ]; then echo "TOO_MANY $n"; fi; }}; '
            "ls -A | grep -iq '^readme' || echo NO_README; "
            "if [ -f package.json ] || [ -f requirements.txt ] || [ -f pyproject.toml ]; then "
            "[ -f .gitignore ] || echo NO_GITIGNORE; fi; "
            "if [ -f package.json ]; then "
            "if command -v python3 >/dev/null 2>&1; then "
            "python3 -c 'import json; json.load(open(\"package.json\", encoding=\"utf-8\"))' >/dev/null 2>&1 "
            "|| echo BAD_PACKAGE_JSON; "
            "elif command -v node >/dev/null 2>&1; then "
            "node -e 'JSON.parse(require(\"fs\").readFileSync(\"package.json\", \"utf8\"))' >/dev/null 2>&1 "
            "|| echo BAD_PACKAGE_JSON; fi; fi; "
            "for f in .env .env.local .env.production .env.development; do "
            'if [ -f "$f" ] && ! git check-ignore -q "$f"; then echo "ENV_FILE $f"; fi; done; '
            "ls -A | grep -v '^\\.git$' | grep -q . || echo NOTHING"
        )
        try:
            result = self.workspace.run(script, cwd=self.checkout.directory, timeout=30.0)
        except SandboxError:
            return []
        found = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if "NOTHING" in found:
            return []  # nothing was created at all: that is reported as "no changes"
        gaps: List[str] = []
        if "NO_README" in found:
            gaps.append(
                "README.md is missing - add one that says what the project is, how to run it "
                "locally and how to deploy it"
            )
        if "NO_GITIGNORE" in found:
            gaps.append(
                ".gitignore is missing although the project installs dependencies - add one for "
                "the stack (dependencies, build output, .env)"
            )
        if "BAD_PACKAGE_JSON" in found:
            gaps.append("package.json is not valid JSON - nothing can install or deploy the project until it is")
        for line in found:
            if line.startswith("ENV_FILE "):
                name = line.split(" ", 1)[1]
                gaps.append(
                    f"{name} would be committed - list the variable names without values in "
                    f".env.example and add {name} to .gitignore"
                )
            elif line.startswith("TOO_BIG "):
                name = line.split(" ", 1)[1]
                gaps.append(
                    f"{name} is larger than {MAX_FILE_BYTES // 1000} kB, more than a task can commit "
                    "- delete it or add it to .gitignore (a lock file is recreated by installing), "
                    "otherwise nothing at all can be committed"
                )
            elif line.startswith("TOO_MANY "):
                count = line.split(" ", 1)[1]
                gaps.append(
                    f"{count} files would be committed, more than the limit of {MAX_CHANGED_FILES} "
                    "- add build output and installed dependencies to .gitignore, otherwise nothing "
                    "at all can be committed"
                )
        return gaps

    def _tool_replace_in_file(self, args: Dict[str, Any]) -> str:
        absolute, relative = self._resolve(args.get("path"))
        old_text, new_text = args.get("old_text"), args.get("new_text")
        if not isinstance(old_text, str) or not old_text:
            raise ToolError("old_text is required")
        if not isinstance(new_text, str):
            raise ToolError("new_text must be a string")
        if is_stand_in(new_text):
            raise ToolError(
                f"{relative} was not changed: new_text ends with the mark this conversation puts where "
                "older text was cut to save space - it is not the text itself. Send the complete new text.",
                log=f"replace_in_file {relative} held back: new_text was the conversation's shortened copy",
            )
        try:
            current = self.workspace.read_file(absolute)
        except FileNotFoundError:
            raise ToolError(f"{relative} does not exist")
        except UnicodeDecodeError:
            current = "\ufffd"
        if "\ufffd" in current and "\ufffd" not in old_text:
            # Decoded with replacement characters: writing it back would corrupt
            # every byte that is not valid UTF-8.
            raise ToolError(
                f"{relative} is not valid UTF-8 text, so it cannot be edited with replace_in_file. "
                "Use run_command (for example with sed or a short script) instead."
            )
        occurrences = current.count(old_text)
        if occurrences == 0:
            raise ToolError(
                f"old_text was not found in {relative}. It must match the file exactly, including "
                "whitespace and indentation - read the file again and copy the text verbatim."
            )
        if occurrences > 1:
            raise ToolError(
                f"old_text occurs {occurrences} times in {relative}; include more surrounding lines "
                "so that it is unique."
            )
        self.workspace.write_file(absolute, current.replace(old_text, new_text, 1))
        self._known_content.add(relative)
        self.files_written.add(relative)
        self._unwritten.discard(relative)
        self.emit("edit", f"replace_in_file {relative}")
        return f"Updated {relative}."

    def _run_shell(self, command: Any, requested_timeout: Any, *, is_test: bool) -> str:
        if not isinstance(command, str) or not command.strip():
            raise ToolError("command is required")
        try:
            requested = float(requested_timeout) if requested_timeout else self.command_timeout
        except (TypeError, ValueError):
            requested = self.command_timeout
        timeout = min(max(5.0, requested), self.command_timeout, self._time_left())
        if timeout < 5.0:
            raise ToolError("there is not enough time left in this run to start another command")

        self.emit("command", f"$ {command.strip()}")
        started = time.monotonic()
        result = self.workspace.run(
            command, cwd=self.checkout.directory, env=dict(COMMAND_ENV), timeout=timeout
        )
        duration = time.monotonic() - started
        output = strip_ansi(result.output())
        for line in output.splitlines()[-12:]:
            if line.strip():
                self.emit("output", line[:300])
        if not is_test:
            self._inspections += 1
        # A test command that had nothing to test did not fail (and did not pass).
        nothing_to_test = "" if result.timed_out or not is_test else no_tests_reason(result.exit_code, output)
        note = ""
        if result.timed_out:
            self.emit("error", f"timed out after {int(timeout)}s")
            status = f"timed out after {int(timeout)} seconds"
        elif nothing_to_test:
            self.emit("info", f"exit {result.exit_code} · {duration:.1f}s · no tests to run: {nothing_to_test}")
            status = f"exit code: {result.exit_code}"
            note = (
                f"\n\n[There are no tests to run here: {nothing_to_test}. This says nothing about your "
                "change and there is nothing to fix. Do not set up a test framework unless the task "
                "asks for one; state in your summary that the project has no automated tests.]"
            )
        else:
            level = "success" if result.exit_code == 0 else "error"
            self.emit(level, f"exit {result.exit_code} · {duration:.1f}s")
            status = f"exit code: {result.exit_code}"
        if is_test:
            self.verification.append(
                VerificationRun(
                    command=command.strip(),
                    exit_code=result.exit_code,
                    duration=duration,
                    output_tail=clip(output, 3000),
                    timed_out=result.timed_out,
                    no_tests=nothing_to_test,
                )
            )
        self._commands_run += 1
        self._last_command_failed = bool(result.timed_out or (result.exit_code != 0 and not nothing_to_test))
        if self._last_command_failed:
            self._failed_commands[command.strip()] = (
                self._commands_run,
                {
                    "command": command.strip(),
                    "exit_code": result.exit_code,
                    "timed_out": bool(result.timed_out),
                    "output": clip(output, 1500),
                },
            )
        else:
            self._failed_commands.pop(command.strip(), None)
        return f"{status}\n{clip(output, self._result_limit) or '(no output)'}{note}"

    def _unresolved_failure(self) -> Optional[Dict[str, Any]]:
        """The most recent command that failed and has not passed since, if it
        is one of the last few commands (an old failure the agent has moved on
        from says little about where the work stands)."""
        if not self._failed_commands:
            return None
        position, record = max(self._failed_commands.values(), key=lambda item: item[0])
        return dict(record) if self._commands_run - position <= 3 else None

    def _execute(self, call: ToolCall) -> str:
        if call.error:
            self.emit("warn", f"The model's {call.name or 'tool'} call was unreadable ({call.error}); asked it to repeat the call")
            return (
                f"Error: the {call.name or 'tool'} call could not be read because its {call.error}. "
                "Call the tool again with a valid JSON object."
            )
        handlers: Dict[str, Callable[[Dict[str, Any]], str]] = {
            "list_files": self._tool_list_files,
            "read_file": self._tool_read_file,
            "search_code": self._tool_search_code,
            "write_file": self._tool_write_file,
            "replace_in_file": self._tool_replace_in_file,
            "run_command": lambda a: self._run_shell(a.get("command"), a.get("timeout_seconds"), is_test=False),
            "run_tests": lambda a: self._run_shell(a.get("command"), None, is_test=True),
        }
        handler = handlers.get(call.name)
        if handler is None:
            return self._unknown_tool_message(call.name)
        try:
            return handler(call.arguments)
        except ToolError as exc:
            self.emit(exc.level, exc.log or f"{call.name}: {exc}")
            return f"Error: {exc}"

    @staticmethod
    def _unknown_tool_message(names: Any) -> str:
        listed = [names] if isinstance(names, str) else list(names)
        quoted = ", ".join(f"'{name}'" for name in listed) or "that"
        return (
            f"Error: unknown tool {quoted}. Nothing was executed. "
            f"Available tools: {', '.join(TOOL_NAMES)}."
        )

    # ------------------------------------------------------------------ #
    # Model calls
    # ------------------------------------------------------------------ #
    def _announce_switch(self, exc: LLMError) -> None:
        """Log that another provider has taken over, and why."""
        who = exc.provider or "the provider"
        if exc.kind == "rate_limit":
            scope = _rate_limit_scope(exc.message, self._tpm_by_provider.get(who))
            if exc.daily:
                reason = f"{who} has used up its daily limit"
            else:
                reason = f"{who}'s rate limit is reached" + (f" ({scope})" if scope else "")
            level = "info"
        elif exc.kind == "too_large":
            reason, level = f"The request is larger than {who}'s limits allow", "info"
        elif exc.kind == "tool_format":
            reason = f"{who}'s model keeps answering with invalid tool calls ({_refused_reply_detail(exc)})"
            level = "warn"
        else:
            reason, level = exc.message.strip().rstrip("."), "warn"
        self.emit(level, f"{reason}: switching to {exc.switched_to}")

    def _complete(self, messages: List[Dict[str, Any]]) -> LLMResponse:
        transient_retries = 0
        format_retries = 0
        rate_limit_waits = 0
        shrinks = 0
        switches = 0
        # Sent with a repeated request only; it never becomes part of the history.
        correction: Optional[Dict[str, Any]] = None
        while True:
            time_left = self._time_left()
            if time_left <= 5.0:
                raise LLMError("The time budget ran out while waiting for the model.", kind="timeout")
            self._recompute_budgets()  # sized for the provider that answers this request
            # With fallbacks the pool says how long the provider in use is
            # worth waiting for; alone, it gets the full time.
            patience = getattr(self.llm, "patience", None)
            allowed = patience() if callable(patience) else MODEL_CALL_SECONDS
            try:
                return self.llm.complete(
                    messages + [correction] if correction else messages,
                    AGENT_TOOLS,
                    max_output_tokens=self.max_output_tokens,
                    timeout=min(MODEL_CALL_SECONDS, max(10.0, float(allowed)), time_left),
                )
            except LLMError as exc:
                if exc.switched_to and switches < MAX_PROVIDER_SWITCHES:
                    # Several providers are configured and another one has taken
                    # over: no waiting, the request goes to it right away.
                    switches += 1
                    self._announce_switch(exc)
                    correction = None  # it was about a reply the new model never wrote
                    format_retries = transient_retries = 0
                    self._recompute_budgets()
                    compact_history(messages, self.context_budget, soft_budget=SOFT_CONTEXT_CHARS)
                    continue
                # With one provider there is nothing to fall back on: say how to change that.
                alone = not self._has_fallback
                advice = (
                    " Add a key for a second provider (Gemini or OpenRouter, in Settings): the agent "
                    "then carries on with it instead of waiting."
                    if alone
                    else ""
                )
                if exc.kind == "rate_limit":
                    if exc.daily:
                        raise LLMError(
                            f"{exc.message} That is a daily limit: it does not lift during a run.{advice}",
                            kind="rate_limit",
                            status_code=exc.status_code,
                        ) from exc
                    # What the provider asks for, and what is waited before trying
                    # again: limits are counted per minute, so a longer pause than
                    # that is never needed to find out whether the limit has lifted.
                    needed = max(exc.retry_after or 5.0, 1.0) + 0.5
                    wait = min(needed, 65.0)
                    rate_limit_waits += 1
                    if rate_limit_waits > 8 or wait >= self._time_left() - 5.0 or needed > self._time_left():
                        raise LLMError(
                            f"{exc.message} The provider's rate limit left no room to finish within "
                            f"this run's time budget.{advice}",
                            kind="rate_limit",
                            status_code=exc.status_code,
                        ) from exc
                    # Expected on small plans (Groq's free tier allows 8,000 tokens
                    # per minute): the run simply continues after the pause.
                    if exc.next_provider:
                        # Every provider is waiting; this one is free first.
                        who, scope = exc.next_provider.split(" / ", 1)[0], "every provider is rate limited"
                    else:
                        who = exc.provider or self.llm.provider
                        scope = _rate_limit_scope(exc.message, self._tokens_per_minute)
                        if exc.all_limited:
                            scope = f"{scope}; every provider is rate limited" if scope else "every provider is rate limited"
                    self.emit(
                        "info",
                        f"Waiting {wait:.0f}s for {who}'s rate limit" + (f" ({scope})" if scope else ""),
                    )
                    self._rate_limited = True
                    self._sleep(wait)
                    continue
                if exc.kind == "too_large" and shrinks < 3:
                    shrinks += 1
                    match = _TPM_LIMIT_RE.search(exc.message)
                    if match:
                        self._tpm_by_provider[self._provider] = int(match.group(1))
                    self._scale_by_provider[self._provider] = self._budget_scale * 0.6
                    self._recompute_budgets()
                    compact_history(messages, self.context_budget, keep_recent=2)
                    self.emit("warn", "The request was too large for the model; retrying with less context")
                    continue
                if exc.kind == "tool_format":
                    # The provider refused the model's reply (for example a call to
                    # a tool that does not exist). Nothing ran. Sending the very
                    # same request again tends to fail the very same way, so the
                    # model is told what was wrong with its reply.
                    detail = _refused_reply_detail(exc)
                    if format_retries >= MAX_FORMAT_RETRIES:
                        variable = getattr(self.llm, "model_env_var", "")
                        raise LLMError(
                            f"The model ({self.llm.model}) answered {format_retries + 1} times in a row "
                            f"with an invalid tool call ({detail}), so the task could not continue. "
                            "Nothing was wrong with the task itself: run it again"
                            + (f", or choose another model with {variable}." if variable else ".")
                            + (
                                " With a second provider configured (Gemini or OpenRouter, in Settings) "
                                "the agent switches to it when a model keeps doing this."
                                if alone
                                else ""
                            ),
                            kind="tool_format",
                            status_code=exc.status_code,
                            rejected_tool=exc.rejected_tool,
                            failed_generation=exc.failed_generation,
                        ) from exc
                    format_retries += 1
                    correction = {"role": "user", "content": _refused_reply_correction(exc)}
                    self.emit(
                        "warn",
                        f"The model's reply was not a valid tool call ({detail}); "
                        f"asking it again ({format_retries}/{MAX_FORMAT_RETRIES})",
                    )
                    self._sleep(0.25 * format_retries)
                    continue
                if exc.kind in ("server", "network") and transient_retries < 2:
                    transient_retries += 1
                    self.emit("warn", f"{exc.message} - retrying ({transient_retries}/2)")
                    self._sleep(1.5 * transient_retries)
                    continue
                raise

    # ------------------------------------------------------------------ #
    # Main loop
    # ------------------------------------------------------------------ #
    def run(
        self,
        *,
        prompt: str,
        repository: str,
        branch: str,
        file_listing: str = "",
        key_files: Sequence[KeyFile] = (),
    ) -> AgentOutcome:
        first_commit = bool(getattr(self.checkout, "empty", False))
        if first_commit:
            where = (
                f"Repository: {repository} - empty, it has no commits yet. Everything you create "
                f"becomes its first commit on branch `{branch}`."
            )
        else:
            where = f"Repository: {repository} (branch `{branch}`, commit {self.checkout.base_sha[:10]})"
            if self.new_project:
                where += " - it has no code yet."
        task = f"{where}\n\nTask:\n{prompt.strip()}\n"
        self._agent_seconds = max(self._time_left(), 0.0)
        if self.new_project:
            # A project is easy to make too big for one run, so the model is
            # told how much time there is - and that running out of it does not
            # lose the work: getting something small to build comes first.
            minutes = max(1, int(round(self._agent_seconds / 60.0)))
            task += (
                f"\nTime budget: about {minutes} minute{'s' if minutes != 1 else ''} for this run, "
                "including installs and builds. Get a small version working first, then extend it. "
                "If the time runs out, what you have is kept and a follow-up run continues from it.\n"
            )
        if file_listing:
            task += f"\n{file_listing}\n"
        if key_files:
            # The facts every task depends on (what the project is, what it is
            # built with, how it is tested) are in front of the model from the
            # first turn, so it does not have to guess them - or spend a turn.
            task += f"\n{format_key_files(key_files)}\n"
            for item in key_files:
                if item.truncated:
                    continue  # only its beginning was shown: it counts once the model reads it
                self._known_content.add(item.path)
                if item.path not in self.files_read:
                    self.files_read.append(item.path)
            self.emit(
                "tool",
                "Shown to the agent up front: "
                + ", ".join(item.path + (" (beginning)" if item.truncated else "") for item in key_files),
            )
        messages: List[Dict[str, Any]] = [
            {
                "role": "system",
                "content": system_prompt_for(
                    new_project=self.new_project, first_commit=first_commit, update=self.update
                ),
            },
            {"role": "user", "content": task},
        ]
        nudges = 0
        turn = 0
        last_text = ""

        def outcome(reason: str, **extra: Any) -> AgentOutcome:
            return AgentOutcome(
                stop_reason=reason,
                turns=turn,
                model=self.llm.model,
                verification=list(self.verification),
                files_read=list(self.files_read),
                files_written=sorted(self.files_written),
                last_failure=self._unresolved_failure(),
                served=dict(self._served),
                **extra,
            )

        while turn < self.max_turns:
            if self._time_left() <= 5.0:
                return outcome("timeout", error=TIME_UP)
            wrap_up_at = WRAP_UP_SECONDS_RATE_LIMITED if self._rate_limited else WRAP_UP_SECONDS
            if turn >= 1 and not self._wrap_up_sent and self._time_left() <= wrap_up_at:
                # One more reply still fits. What the agent has changed is kept
                # whatever it answers; its account of what is done and what is
                # missing is what a follow-up run starts from.
                self._wrap_up_sent = True
                self.emit("info", "The time budget is nearly used up: the agent was asked to wrap up")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Time is almost up for this run: there is room for one more reply. Start "
                            "nothing new and call finish now, with a title for the work (pr_title). In "
                            "the summary say what is done, what is still missing or unchecked and - if "
                            "a command failed last - what it reported. Your work is kept as it is and a "
                            "follow-up run continues from your summary. Pass success=false if the work "
                            "is not complete or the last build or test run failed."
                        ),
                    }
                )
            elif (
                turn >= 1
                and not self._time_check_sent
                and not self._wrap_up_sent
                and self._agent_seconds >= TIME_CHECK_MIN_SECONDS
                and self._time_left() <= self._agent_seconds * (1.0 - TIME_CHECK_SHARE)
            ):
                # Models do not feel time pass. Told once how much is left, they
                # can check what they have while there is still time to fix it.
                self._time_check_sent = True
                left = int(max(self._time_left() - wrap_up_at, 10.0) // 10 * 10)
                self.emit("info", f"About {left}s of working time are left: the agent was told to check its work")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Time check: about {left} seconds of working time are left in this run. Get "
                            "what you have into a state that builds or runs, and check it now with the "
                            "project's build or test command (a change to documentation alone needs no "
                            "such check). Fix what that reports before you add anything else. A smaller "
                            "result that works beats a larger one that does not; whatever is still "
                            "missing at the end is continued in a follow-up run."
                        ),
                    }
                )
            turn += 1
            compact_history(messages, self.context_budget, soft_budget=SOFT_CONTEXT_CHARS)
            try:
                response = self._complete(messages)
            except LLMError as exc:
                if exc.kind == "timeout":
                    return outcome("timeout", error=exc.message)
                return outcome("llm_error", error=exc.message)
            self._apply_token_limit(response.tokens_per_minute_limit, response.provider)
            answering = str(response.model or "")
            if answering and answering != self.llm.model:
                # A router (openrouter/free) or an alias (gemini-flash-latest) chose the model.
                self._served[answering] = self._served.get(answering, 0) + 1
                if answering != self._answering_model:
                    self._answering_model = answering
                    self.emit("info", f"{response.provider or self.llm.provider} answered with {answering}")
            if response.recovered:
                # The provider had refused the reply over its form (a prefixed
                # tool name, a line break inside JSON); what it meant was clear.
                self.emit(
                    "info",
                    f"{response.provider or self.llm.provider} refused the model's reply over its form; "
                    "the call in it was used as meant ("
                    + ", ".join(call.name for call in response.tool_calls)
                    + ")",
                )
            stored = _storable_assistant_message(response)
            if stored is not None:
                messages.append(stored)

            text = response.content.strip()
            if text:
                last_text = text
                self.emit("agent", clip(plain_text(text), 400, tail_share=0.0))

            if not response.tool_calls:
                if response.finish_reason == "length":
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                "Your reply was cut off because it was too long. Work in smaller steps: "
                                + (
                                    "write fewer files per reply and keep each file small (split a "
                                    "large one into modules)."
                                    if self.new_project
                                    else "prefer replace_in_file over rewriting whole files."
                                )
                            ),
                        }
                    )
                    continue
                nudges += 1
                if nudges > MAX_NUDGES:
                    # A plain-text answer is the model's way of saying it is done.
                    return outcome("finished", summary=last_text, pr_title="")
                messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Continue by calling a tool. If the task is complete, call finish with a "
                            "summary; if it cannot be completed, call finish with success=false. "
                            "A reply without a tool call is not acted on."
                        ),
                    }
                )
                continue

            unknown: List[str] = []
            for call in response.tool_calls:
                if call.name not in TOOL_NAMES:
                    unknown.append(call.name or "(no name)")
                    continue
                if call.name == "finish" and not call.error:
                    summary = str(call.arguments.get("summary") or last_text or "").strip()
                    title = first_line(str(call.arguments.get("pr_title") or ""), 100)
                    succeeded = call.arguments.get("success", True)
                    if isinstance(succeeded, str):
                        succeeded = succeeded.strip().lower() not in ("false", "0", "no")
                    if self._wrap_up_sent:
                        # Asked to stop because time is nearly up. If it says the
                        # work is not complete - or the last command it ran failed,
                        # whatever it says - the run was cut short: its work is
                        # kept as unfinished. Otherwise it made it; sending it
                        # back now would turn a result into a timeout.
                        if succeeded is False or self._last_command_failed:
                            return outcome("timeout", summary=summary, pr_title=title, error=TIME_UP)
                        return outcome("finished", summary=summary, pr_title=title)
                    if succeeded is False:
                        return outcome("gave_up", summary=summary, error=summary or "The agent gave up.")
                    gaps = self._new_project_gaps() if self.new_project and not self._gaps_questioned else []
                    if gaps:
                        self._gaps_questioned = True
                        self.emit(
                            "info",
                            "finish held back: the new project is not ready to be committed ("
                            + "; ".join(gap.split(" - ", 1)[0] for gap in gaps)
                            + ") - the agent was asked to complete it",
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    "Not finished yet. Before the project is committed:\n- "
                                    + "\n- ".join(gaps)
                                    + "\nFix this, then call finish again. If something is the way "
                                    "it is on purpose, call finish again and say so in the summary."
                                ),
                            }
                        )
                        continue
                    if self._unwritten and not self._finish_questioned:
                        # The model reports success, but a write it attempted was
                        # refused and never repeated: what it describes is not there.
                        self._finish_questioned = True
                        missing = ", ".join(sorted(self._unwritten))
                        self.emit("warn", f"finish held back: {missing} was never written - the agent was asked to complete it")
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call.id,
                                "content": (
                                    f"Not finished: {missing} has not been written - your write_file call "
                                    "was refused (see the earlier message). Read what the change depends "
                                    "on, call write_file again and then call finish. If the file is not "
                                    "needed after all, call finish again."
                                ),
                            }
                        )
                        continue
                    return outcome("finished", summary=summary, pr_title=title)
                try:
                    result = self._execute(call)
                except SandboxError as exc:
                    return outcome("sandbox_error", error=f"The sandbox failed during {call.name}: {exc}")
                messages.append({"role": "tool", "tool_call_id": call.id, "content": result})
            if unknown:
                # Not answered as a tool result: the call was removed from the
                # history (see _storable_assistant_message), so there is nothing
                # a tool result could belong to.
                self.emit(
                    "warn",
                    "The model called a tool that does not exist ("
                    + ", ".join(unknown)
                    + "); it was told which tools there are",
                )
                messages.append({"role": "user", "content": self._unknown_tool_message(unknown)})

        return outcome(
            "max_turns",
            error=f"The agent used all {self.max_turns} turns without finishing the task.",
        )
