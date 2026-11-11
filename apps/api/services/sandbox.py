"""Sandboxed workspaces: where a repository is cloned, edited and tested.

``Workspace`` is the small interface the rest of the backend depends on.
``E2BWorkspace`` is the real implementation (an E2B cloud microVM). There is
deliberately no simulated fallback: if a sandbox cannot be created the task
fails and says why, instead of pretending that commands ran.
"""
from __future__ import annotations

import base64
import binascii
import hashlib
import posixpath
import shlex
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

from .github import FileChange

DEFAULT_WORKDIR = "/home/user"
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "LC_ALL": "C.UTF-8"}

# Untracked build/dependency output that must never end up in a pull request,
# even when the target repository's own .gitignore forgets about it.
JUNK_EXCLUDES = (
    "node_modules/",
    ".venv/",
    "venv/",
    "__pycache__/",
    "*.pyc",
    ".pytest_cache/",
    ".mypy_cache/",
    ".ruff_cache/",
    ".tox/",
    ".next/",
    "*.egg-info/",
    ".coverage",
    ".DS_Store",
)

MAX_CHANGED_FILES = 300
MAX_FILE_BYTES = 1_000_000
MAX_TOTAL_BYTES = 4_000_000
_BLOB_BATCH_BYTES = 400_000

# Files that say what a project is and what it is built with. Their content is
# handed to the agent with the task (see RepoCheckout.key_files), within a
# small budget: on providers with tight per-minute token limits every
# character of the first message is paid for again on every turn.
MANIFEST_FILES = (
    "package.json",
    "pyproject.toml",
    "requirements.txt",
    "setup.py",
    "setup.cfg",
    "Pipfile",
    "go.mod",
    "Cargo.toml",
    "Gemfile",
    "composer.json",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "Makefile",
)
README_FILES = ("README.md", "README.rst", "README.txt", "README", "Readme.md", "readme.md")
KEY_FILE_BYTES = 1600
README_BYTES = 500
KEY_FILES_TOTAL_BYTES = 2400


class SandboxError(RuntimeError):
    """The sandbox could not do what was asked. The message is user-facing."""


@dataclass
class CommandResult:
    exit_code: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    def output(self) -> str:
        """stdout followed by stderr, the way a terminal would show it."""
        parts = [part.rstrip("\n") for part in (self.stdout, self.stderr) if part and part.strip()]
        return "\n".join(parts)


class Workspace(Protocol):
    """An isolated Linux environment with a shell and a filesystem."""

    workdir: str
    description: str

    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> CommandResult:
        """Run ``command`` with bash. Non-zero exits are returned, not raised."""
        ...

    def read_file(self, path: str) -> str: ...

    def write_file(self, path: str, content: str) -> None: ...

    def close(self) -> None: ...


# --------------------------------------------------------------------------- #
# E2B implementation
# --------------------------------------------------------------------------- #
def _describe_e2b_error(exc: BaseException) -> str:
    name = exc.__class__.__name__
    text = str(exc).strip() or name
    if "Authentication" in name or "401" in text:
        return f"E2B rejected the API key - check E2B_API_KEY ({text})."
    if "RateLimit" in name or "429" in text:
        return f"E2B refused to start a sandbox: rate or concurrency limit reached ({text})."
    return f"E2B sandbox error ({name}): {text}"


class E2BWorkspace:
    """A real E2B sandbox (https://e2b.dev)."""

    def __init__(self, sandbox: Any) -> None:
        self._sandbox = sandbox
        self.sandbox_id: str = str(getattr(sandbox, "sandbox_id", "") or "")
        self.workdir = DEFAULT_WORKDIR
        self.description = f"E2B sandbox {self.sandbox_id}".strip()

    @classmethod
    def create(
        cls,
        *,
        api_key: str,
        template: Optional[str] = None,
        lifetime_seconds: int = 600,
        metadata: Optional[Dict[str, str]] = None,
    ) -> "E2BWorkspace":
        if not api_key:
            raise SandboxError("E2B_API_KEY is not configured, so no sandbox can be started.")
        try:
            from e2b import Sandbox
        except Exception as exc:  # ImportError, or a platform-specific load failure
            raise SandboxError(
                f"The E2B SDK could not be loaded ({exc.__class__.__name__}: {exc})."
            ) from exc

        options: Dict[str, Any] = {
            "timeout": int(lifetime_seconds),  # the sandbox destroys itself after this
            "api_key": api_key,
            "request_timeout": 30.0,
        }
        if template:
            options["template"] = template
        if metadata:
            options["metadata"] = metadata
        try:
            sandbox = Sandbox.create(**options)
        except Exception as exc:
            raise SandboxError(_describe_e2b_error(exc)) from exc

        workspace = cls(sandbox)
        try:
            located = workspace.run("pwd", timeout=20.0)
            if located.ok and located.stdout.strip().startswith("/"):
                workspace.workdir = located.stdout.strip()
        except SandboxError:
            workspace.close()
            raise
        return workspace

    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> CommandResult:
        try:
            from e2b import CommandExitException, TimeoutException
        except Exception as exc:  # pragma: no cover - import already succeeded in create()
            raise SandboxError(f"The E2B SDK could not be loaded ({exc}).") from exc
        try:
            from e2b.exceptions import SandboxNotRunningException, SandboxUnreachableException

            fatal_timeouts: tuple = (SandboxNotRunningException, SandboxUnreachableException)
        except Exception:  # pragma: no cover - older SDKs
            fatal_timeouts = ()

        try:
            result = self._sandbox.commands.run(
                command,
                cwd=cwd or self.workdir,
                envs=dict(env) if env else None,
                timeout=float(timeout),
            )
        except CommandExitException as exc:
            return CommandResult(int(exc.exit_code), exc.stdout or "", exc.stderr or "")
        except TimeoutException as exc:
            if fatal_timeouts and isinstance(exc, fatal_timeouts):
                raise SandboxError(f"The sandbox is no longer reachable ({exc}).") from exc
            return CommandResult(
                124, "", f"Command timed out after {int(timeout)} seconds.", timed_out=True
            )
        except Exception as exc:
            raise SandboxError(_describe_e2b_error(exc)) from exc
        return CommandResult(int(result.exit_code or 0), result.stdout or "", result.stderr or "")

    def read_file(self, path: str) -> str:
        try:
            return self._sandbox.files.read(path)
        except Exception as exc:
            name = exc.__class__.__name__
            # FileNotFoundException (or its deprecated base NotFoundException) means the
            # file is missing; SandboxNotFoundException means the sandbox itself is gone.
            if "NotFound" in name and not name.startswith("Sandbox"):
                raise FileNotFoundError(path) from exc
            raise SandboxError(_describe_e2b_error(exc)) from exc

    def write_file(self, path: str, content: str) -> None:
        try:
            self._sandbox.files.write(path, content)
        except Exception as exc:
            raise SandboxError(_describe_e2b_error(exc)) from exc

    def close(self) -> None:
        try:
            self._sandbox.kill()
        except Exception:
            # Best effort: the sandbox also destroys itself when its lifetime ends.
            pass


# --------------------------------------------------------------------------- #
# Git checkout inside a workspace
# --------------------------------------------------------------------------- #
@dataclass
class KeyFile:
    """A manifest or README (excerpt) of the repository, as read at the start of a run."""

    path: str
    content: str
    size: int  # bytes of the whole file
    truncated: bool = False


@dataclass
class ChangeSet:
    changes: List[FileChange] = field(default_factory=list)
    tree_sha: str = ""
    total_bytes: int = 0

    @property
    def empty(self) -> bool:
        return not self.changes


def _decode_b64_output(text: str, what: str) -> bytes:
    try:
        return base64.b64decode("".join(text.split()), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SandboxError(f"Could not decode {what} returned by the sandbox.") from exc


class RepoCheckout:
    """A git clone living inside a :class:`Workspace`."""

    def __init__(self, workspace: Workspace, directory: str) -> None:
        self.workspace = workspace
        self.directory = directory
        self.base_sha = ""
        self.base_tree_sha = ""

    # ------------------------------------------------------------------ #
    @classmethod
    def clone(
        cls,
        workspace: Workspace,
        *,
        clone_url: str,
        branch: str,
        directory: Optional[str] = None,
        token: Optional[str] = None,
        timeout: float = 120.0,
    ) -> "RepoCheckout":
        directory = directory or posixpath.join(workspace.workdir, "repo")
        version = workspace.run("git --version", timeout=20.0, env=dict(GIT_ENV))
        if not version.ok:
            raise SandboxError(
                "git is not available in the sandbox image. Use an E2B template that includes git."
            )

        env = dict(GIT_ENV)
        if token:
            # Hand the credential to this one command through the environment
            # (the same mechanism actions/checkout uses). It is neither written
            # to disk nor visible to later commands in the sandbox.
            basic = base64.b64encode(f"x-access-token:{token}".encode("utf-8")).decode("ascii")
            env.update(
                {
                    "GIT_CONFIG_COUNT": "1",
                    "GIT_CONFIG_KEY_0": "http.https://github.com/.extraheader",
                    "GIT_CONFIG_VALUE_0": f"AUTHORIZATION: basic {basic}",
                }
            )
        command = (
            f"rm -rf {shlex.quote(directory)} && "
            "git clone --quiet --depth 1 --single-branch --no-tags "
            f"--branch {shlex.quote(branch)} {shlex.quote(clone_url)} {shlex.quote(directory)}"
        )
        result = workspace.run(command, env=env, timeout=timeout)
        if not result.ok:
            detail = (result.stderr or result.stdout).strip()
            if token:
                detail = detail.replace(token, "***")
            reason = "timed out" if result.timed_out else f"exit code {result.exit_code}"
            raise SandboxError(f"git clone failed ({reason}): {detail[-600:] or 'no output'}")

        checkout = cls(workspace, directory)
        heads = checkout.git("rev-parse HEAD 'HEAD^{tree}'")
        shas = heads.stdout.split()
        if not heads.ok or len(shas) != 2:
            raise SandboxError("The cloned repository has no commits to work from.")
        checkout.base_sha, checkout.base_tree_sha = shas[0], shas[1]

        patterns = " ".join(shlex.quote(pattern) for pattern in JUNK_EXCLUDES)
        checkout.git_shell(f"mkdir -p .git/info && printf '%s\\n' {patterns} >> .git/info/exclude")
        return checkout

    # ------------------------------------------------------------------ #
    def git(self, args: str, *, timeout: float = 60.0) -> CommandResult:
        return self.workspace.run(f"git {args}", cwd=self.directory, env=dict(GIT_ENV), timeout=timeout)

    def git_shell(self, script: str, *, timeout: float = 60.0) -> CommandResult:
        return self.workspace.run(
            f"set -o pipefail; {script}", cwd=self.directory, env=dict(GIT_ENV), timeout=timeout
        )

    def list_files(self, limit: int = 300) -> tuple[List[str], int]:
        """Tracked files (at most ``limit``) and the total number of tracked files."""
        result = self.git_shell(f"git ls-files | wc -l; git ls-files | head -n {int(limit)}")
        if not result.ok:
            return [], 0
        lines = result.stdout.splitlines()
        try:
            total = int(lines[0].strip())
        except (IndexError, ValueError):
            total = 0
        return [line for line in lines[1:] if line.strip()], total

    def key_files(self) -> List[KeyFile]:
        """The repository's manifests and the start of its README.

        Best effort: an empty list if nothing could be read. Symbolic links,
        binary and empty files are skipped; the total size is bounded.
        """
        names = " ".join(shlex.quote(name) for name in MANIFEST_FILES + README_FILES)
        readmes = "|".join(README_FILES)
        script = (
            f"for f in {names}; do "
            'if [ -f "$f" ] && [ ! -L "$f" ] && [ -s "$f" ] && grep -Iq \'\' "$f"; then '
            f'case "$f" in {readmes}) n={README_BYTES};; *) n={KEY_FILE_BYTES};; esac; '
            'printf \'%s %s \' "$f" "$(wc -c < "$f" | tr -d \' \')"; '
            'head -c "$n" "$f" | base64 | tr -d \'\\n\'; printf \'\\n\'; '
            "fi; done"
        )
        try:
            result = self.workspace.run(script, cwd=self.directory, env=dict(GIT_ENV), timeout=30.0)
        except SandboxError:
            return []
        if not result.ok:
            return []
        found: List[KeyFile] = []
        seen_readme = False
        used = 0
        for line in result.stdout.splitlines():
            pieces = line.split(" ")
            if len(pieces) != 3 or not pieces[1].isdigit():
                continue
            path, size = pieces[0], int(pieces[1])
            is_readme = path in README_FILES
            if is_readme and seen_readme:
                continue  # README.md and readme.md are the same file on some filesystems
            try:
                content = base64.b64decode(pieces[2], validate=True).decode("utf-8", "replace")
            except (binascii.Error, ValueError):
                continue
            truncated = size > (README_BYTES if is_readme else KEY_FILE_BYTES)
            if truncated:
                # End on a whole line rather than in the middle of one.
                last_newline = content.rfind("\n")
                if last_newline >= len(content) // 2:
                    content = content[: last_newline + 1]
            if not content.strip() or used + len(content) > KEY_FILES_TOTAL_BYTES:
                continue
            seen_readme = seen_readme or is_readme
            used += len(content)
            found.append(KeyFile(path=path, content=content, size=size, truncated=truncated))
        return found

    def diff_stat(self) -> str:
        result = self.git_shell(
            f"git add -A && git diff --cached --stat {shlex.quote(self.base_sha)} | tail -n 25"
        )
        return result.stdout.strip() if result.ok else ""

    # ------------------------------------------------------------------ #
    def collect_changes(self) -> ChangeSet:
        """Stage everything and describe the difference to the base commit.

        The comparison is against the commit that was cloned (not ``HEAD``), so
        the result is correct even if a command inside the sandbox committed.
        """
        base = shlex.quote(self.base_sha)
        staged = self.git("add -A")
        if not staged.ok:
            raise SandboxError(f"git add failed: {staged.output()[-400:]}")

        raw = self.git_shell(
            f"git diff --cached --raw -z --no-renames --no-abbrev {base} | base64 | tr -d '\\n'"
        )
        if not raw.ok:
            raise SandboxError(f"git diff failed: {raw.output()[-400:]}")
        tokens = _decode_b64_output(raw.stdout, "the list of changed files").split(b"\0")

        changes: List[FileChange] = []
        index = 0
        while index + 1 < len(tokens):
            meta_bytes, path_bytes = tokens[index], tokens[index + 1]
            index += 2
            if not meta_bytes:
                continue
            fields = meta_bytes.decode("ascii", "replace").lstrip(":").split(" ")
            if len(fields) != 5:
                raise SandboxError("Unexpected output from git diff; cannot determine the changes.")
            old_mode, new_mode, _old_sha, new_sha, status = fields
            try:
                path = path_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SandboxError("A changed file has a name that is not valid UTF-8.") from exc
            letter = status[:1]
            if letter == "U":
                raise SandboxError(f"{path} has unresolved merge conflicts.")
            if letter == "D":
                changes.append(FileChange(path=path, status="D", mode=old_mode))
            else:
                changes.append(FileChange(path=path, status=letter, mode=new_mode, sha=new_sha))

        if not changes:
            return ChangeSet()
        if len(changes) > MAX_CHANGED_FILES:
            raise SandboxError(
                f"The run changed {len(changes)} files, more than the limit of {MAX_CHANGED_FILES}. "
                "This usually means build output or dependencies were not ignored."
            )

        self._attach_line_counts(changes)
        total = self._attach_contents(changes)

        tree = self.git("write-tree")
        tree_sha = tree.stdout.strip()
        if not tree.ok or len(tree_sha) != 40:
            raise SandboxError(f"git write-tree failed: {tree.output()[-400:]}")
        return ChangeSet(changes=changes, tree_sha=tree_sha, total_bytes=total)

    def _attach_line_counts(self, changes: List[FileChange]) -> None:
        numstat = self.git_shell(
            "git diff --cached --numstat -z --no-renames "
            f"{shlex.quote(self.base_sha)} | base64 | tr -d '\\n'"
        )
        if not numstat.ok:
            return
        try:
            records = _decode_b64_output(numstat.stdout, "the diff statistics").split(b"\0")
        except SandboxError:
            return
        counts: Dict[str, tuple[Optional[int], Optional[int]]] = {}
        for record in records:
            pieces = record.split(b"\t", 2)
            if len(pieces) != 3:
                continue
            added, deleted, path_bytes = pieces
            try:
                path = path_bytes.decode("utf-8")
            except UnicodeDecodeError:
                continue
            counts[path] = (
                int(added) if added.isdigit() else None,
                int(deleted) if deleted.isdigit() else None,
            )
        for change in changes:
            if change.path in counts:
                change.additions, change.deletions = counts[change.path]

    def _attach_contents(self, changes: List[FileChange]) -> int:
        """Read the staged blob of every added/modified file out of the sandbox."""
        blobs = [c for c in changes if c.status != "D" and c.mode != "160000" and c.sha]
        if not blobs:
            return 0
        unique = sorted({c.sha for c in blobs if c.sha})
        listing = self.git_shell(
            "printf '%s\\n' " + " ".join(shlex.quote(sha) for sha in unique) + " | git cat-file --batch-check"
        )
        if not listing.ok:
            raise SandboxError(f"Could not inspect the changed files: {listing.output()[-400:]}")
        sizes: Dict[str, int] = {}
        for line in listing.stdout.splitlines():
            pieces = line.split()
            if len(pieces) == 3 and pieces[1] == "blob" and pieces[2].isdigit():
                sizes[pieces[0]] = int(pieces[2])
        missing = [sha for sha in unique if sha not in sizes]
        if missing:
            raise SandboxError("Could not determine the size of some changed files.")

        too_big = [c.path for c in blobs if sizes[c.sha or ""] > MAX_FILE_BYTES]
        if too_big:
            raise SandboxError(
                f"These changed files are larger than {MAX_FILE_BYTES // 1000} kB and cannot be "
                f"committed by the agent: {', '.join(sorted(set(too_big))[:5])}"
            )
        total = sum(sizes[sha] for sha in unique)
        if total > MAX_TOTAL_BYTES:
            raise SandboxError(
                f"The change is {total // 1000} kB in total, above the limit of "
                f"{MAX_TOTAL_BYTES // 1000} kB."
            )

        contents: Dict[str, bytes] = {}
        batch: List[str] = []
        batch_bytes = 0

        def flush() -> None:
            nonlocal batch, batch_bytes
            if not batch:
                return
            script = (
                "for s in "
                + " ".join(shlex.quote(sha) for sha in batch)
                + "; do printf '%s ' \"$s\"; git cat-file blob \"$s\" | base64 | tr -d '\\n' || exit 1; "
                "printf '\\n'; done"
            )
            fetched = self.git_shell(script, timeout=120.0)
            if not fetched.ok:
                raise SandboxError(f"Could not read the changed files: {fetched.output()[-400:]}")
            for line in fetched.stdout.splitlines():
                sha, _, encoded = line.partition(" ")
                if sha in sizes:
                    contents[sha] = _decode_b64_output(encoded, "a changed file")
            batch, batch_bytes = [], 0

        for sha in unique:
            if batch and batch_bytes + sizes[sha] > _BLOB_BATCH_BYTES:
                flush()
            batch.append(sha)
            batch_bytes += sizes[sha]
        flush()

        for sha in unique:
            data = contents.get(sha)
            if data is None:
                raise SandboxError("A changed file could not be read back from the sandbox.")
            digest = hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()
            if digest != sha:
                raise SandboxError("A changed file was corrupted while being read from the sandbox.")
        for change in blobs:
            change.content_b64 = base64.b64encode(contents[change.sha or ""]).decode("ascii")
        return total
