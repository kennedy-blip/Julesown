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
from typing import Any, Dict, List, Optional, Protocol, Sequence

from .github import FileChange

DEFAULT_WORKDIR = "/home/user"
GIT_ENV = {"GIT_TERMINAL_PROMPT": "0", "GIT_PAGER": "cat", "LC_ALL": "C.UTF-8"}
# The tree without any entry. Git knows it without it being stored anywhere; it
# is what a repository that has no commits yet is compared against.
EMPTY_TREE_SHA = "4b825dc642cb6eb9a060e54bf8d69288fbee4904"

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

# Where builds and test runs put their output: in a directory of one of these
# names next to the project's manifest (dist/ beside package.json), or in a
# tool's own cache directory anywhere. When a run stops early, its .gitignore
# may not be written yet: what a command left there is then not part of the
# work that is kept (see ``RepoCheckout.collect_changes``). This never applies
# to a file the agent wrote itself, nor to a directory the repository keeps
# under version control.
BUILD_OUTPUT_DIRS = ("dist", "build", "out", "target", "coverage", "htmlcov")
TOOL_CACHE_DIRS = (".vite", ".turbo", ".cache", ".parcel-cache", ".output", ".nuxt", ".svelte-kit", ".gradle")
BUILD_OUTPUT_SUFFIXES = (".tsbuildinfo",)


def build_output_root(path: str) -> Optional[str]:
    """The directory named like build output that ``path`` lies in
    (``packages/app/dist`` for ``packages/app/dist/assets/a.js``), if any.
    Whether it *is* build output depends on what lies next to it."""
    parts = path.split("/")
    for index, part in enumerate(parts[:-1]):
        if part in BUILD_OUTPUT_DIRS or part in TOOL_CACHE_DIRS:
            return "/".join(parts[: index + 1])
    return None


def _in_batches(arguments: Sequence[str], limit: int = 40_000) -> List[List[str]]:
    """Split shell arguments into groups that each fit on one command line: a
    script is handed to the shell as one argument, and the kernel limits the
    length of a single argument (128 kB on Linux)."""
    batches: List[List[str]] = []
    length = 0
    for argument in arguments:
        cost = len(argument.encode("utf-8", "replace")) + 3
        if not batches or length + cost > limit:
            batches.append([])
            length = 0
        batches[-1].append(argument)
        length += cost
    return batches


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
    # Files of the work tree that are not part of the changes although they
    # differ from the base, each with the reason: "dist/app.js (build output)".
    # Only when unfinished work is salvaged (see ``collect_changes``).
    left_out: List[str] = field(default_factory=list)

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
        # True for a repository that had no commits: there is no base commit and
        # everything in the work tree is new.
        self.empty = False

    @property
    def diff_base(self) -> str:
        """What the work tree is compared with: the cloned commit, or the empty
        tree when the repository had none."""
        return self.base_sha or EMPTY_TREE_SHA

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
        checkout._exclude_junk()
        return checkout

    @classmethod
    def init_empty(
        cls,
        workspace: Workspace,
        *,
        branch: str,
        clone_url: Optional[str] = None,
        directory: Optional[str] = None,
    ) -> "RepoCheckout":
        """The work tree for a repository that has no commits yet.

        There is nothing to clone, so the repository is created in place, on
        the branch the first commit will be made on. Whatever is in the work
        tree when the agent finishes becomes that first commit.
        """
        directory = directory or posixpath.join(workspace.workdir, "repo")
        version = workspace.run("git --version", timeout=20.0, env=dict(GIT_ENV))
        if not version.ok:
            raise SandboxError(
                "git is not available in the sandbox image. Use an E2B template that includes git."
            )
        quoted = shlex.quote(directory)
        command = (
            f"rm -rf {quoted} && mkdir -p {quoted} && cd {quoted} && git init --quiet . && "
            f"git symbolic-ref HEAD {shlex.quote('refs/heads/' + branch)}"
        )
        if clone_url:
            command += f" && git remote add origin {shlex.quote(clone_url)}"
        result = workspace.run(command, env=dict(GIT_ENV), timeout=60.0)
        if not result.ok:
            detail = (result.stderr or result.stdout).strip()
            raise SandboxError(f"Could not set up the empty repository: {detail[-600:] or 'no output'}")
        checkout = cls(workspace, directory)
        checkout.empty = True
        checkout.base_tree_sha = EMPTY_TREE_SHA
        checkout._exclude_junk()
        return checkout

    def head_message(self) -> str:
        """The message of the commit the work starts from ("" when there is none)."""
        if self.empty:
            return ""
        result = self.git("log -1 --format=%B")
        return result.stdout.strip() if result.ok else ""

    def _exclude_junk(self) -> None:
        patterns = " ".join(shlex.quote(pattern) for pattern in JUNK_EXCLUDES)
        self.git_shell(f"mkdir -p .git/info && printf '%s\\n' {patterns} >> .git/info/exclude")

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
            f"git add -A && git diff --cached --stat {shlex.quote(self.diff_base)} | tail -n 25"
        )
        return result.stdout.strip() if result.ok else ""

    # ------------------------------------------------------------------ #
    def collect_changes(self, *, salvage: bool = False, written: Sequence[str] = ()) -> ChangeSet:
        """Stage everything and describe the difference to the base commit.

        The comparison is against the commit that was cloned (not ``HEAD``), so
        the result is correct even if a command inside the sandbox committed.
        For a repository that had no commits it is against the empty tree:
        every file is an addition.

        A file that is too large, or a change that is too big altogether, is an
        error: the agent is told while it can still act on it. With ``salvage``
        - for a run that was stopped, where nobody can act any more - such
        files and build output are left out instead (``ChangeSet.left_out``),
        so that one stray bundle does not cost all the work. ``written`` names
        the files the agent wrote itself: those are never taken for build output.
        """
        base = shlex.quote(self.diff_base)
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
        previous: Dict[str, tuple[str, str]] = {}  # path -> (mode, blob) in the base
        index = 0
        while index + 1 < len(tokens):
            meta_bytes, path_bytes = tokens[index], tokens[index + 1]
            index += 2
            if not meta_bytes:
                continue
            fields = meta_bytes.decode("ascii", "replace").lstrip(":").split(" ")
            if len(fields) != 5:
                raise SandboxError("Unexpected output from git diff; cannot determine the changes.")
            old_mode, new_mode, old_sha, new_sha, status = fields
            try:
                path = path_bytes.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SandboxError("A changed file has a name that is not valid UTF-8.") from exc
            letter = status[:1]
            if letter == "U":
                raise SandboxError(f"{path} has unresolved merge conflicts.")
            previous[path] = (old_mode, old_sha)
            if letter == "D":
                changes.append(FileChange(path=path, status="D", mode=old_mode))
            else:
                changes.append(FileChange(path=path, status=letter, mode=new_mode, sha=new_sha))

        left_out: List[str] = []
        if salvage and changes:
            changes, left_out = self._leave_out_what_cannot_be_kept(changes, previous, set(written))
        if not changes:
            return ChangeSet(left_out=left_out)
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
        return ChangeSet(changes=changes, tree_sha=tree_sha, total_bytes=total, left_out=left_out)

    def _blob_sizes(self, shas: List[str]) -> Dict[str, int]:
        """Size in bytes of each of the given blobs."""
        unique = sorted(set(shas))
        sizes: Dict[str, int] = {}
        for batch in _in_batches(unique):
            listing = self.git_shell(
                "printf '%s\\n' " + " ".join(shlex.quote(sha) for sha in batch) + " | git cat-file --batch-check"
            )
            if not listing.ok:
                raise SandboxError(f"Could not inspect the changed files: {listing.output()[-400:]}")
            for line in listing.stdout.splitlines():
                pieces = line.split()
                if len(pieces) == 3 and pieces[1] == "blob" and pieces[2].isdigit():
                    sizes[pieces[0]] = int(pieces[2])
        if any(sha not in sizes for sha in unique):
            raise SandboxError("Could not determine the size of some changed files.")
        return sizes

    def _leave_out_what_cannot_be_kept(
        self, changes: List[FileChange], previous: Dict[str, "tuple[str, str]"], written: set
    ) -> "tuple[List[FileChange], List[str]]":
        """Take out of the staged changes what would make them impossible to
        commit, or what is no part of the work: build output a command of this
        run produced (new files in a build directory that the agent did not
        write itself), files above the size limit and - if the change is still
        too big in total - the largest new files.

        They are taken out of the index as well, so that the tree that is
        committed is still the tree the commit is checked against. Returns the
        changes that remain and a description of each thing left out.
        """
        # 1. Build output: decided by where a file lies, before anything is measured.
        generated = [change for change in changes if change.status == "A" and change.path not in written]
        candidates = sorted({root for root in (build_output_root(change.path) for change in generated) if root})
        manifests_beside = {
            root: [posixpath.join(posixpath.dirname(root), name) for name in MANIFEST_FILES]
            for root in candidates
            if posixpath.basename(root) in BUILD_OUTPUT_DIRS
        }
        in_base = self._paths_in_base(candidates + [path for paths in manifests_beside.values() for path in paths])
        in_work_tree = {change.path for change in changes if change.status != "D"} | (in_base or set())

        def is_build_output(root: str) -> bool:
            if in_base is None or root in in_base:
                return False  # the repository keeps this directory: what is new in it is work
            beside = manifests_beside.get(root)
            return beside is None or any(path in in_work_tree for path in beside)

        roots = {root for root in candidates if is_build_output(root)}
        output_of: Dict[str, List[str]] = {}  # build directory -> its files that are left out
        dropped: Dict[str, str] = {}  # path -> why
        for change in generated:
            root = build_output_root(change.path)
            if root in roots:
                output_of.setdefault(root, []).append(change.path)
                dropped[change.path] = "build output"
            elif change.path.endswith(BUILD_OUTPUT_SUFFIXES):
                dropped[change.path] = "build output"
        described = [f"{path} (build output)" for path, why in dropped.items() if build_output_root(path) not in output_of]
        described += [
            f"{root}/ (build output, {len(paths)} file{'' if len(paths) == 1 else 's'})"
            for root, paths in output_of.items()
        ]

        # 2. Size: of what is left.
        remaining = [change for change in changes if change.path not in dropped]
        sizes = self._blob_sizes([c.sha for c in remaining if c.status != "D" and c.mode != "160000" and c.sha])

        def size(change: FileChange) -> int:
            return 0 if change.status == "D" else sizes.get(change.sha or "", 0)

        for change in remaining:
            if size(change) > MAX_FILE_BYTES:
                dropped[change.path] = f"{size(change) // 1000:,} kB, too large to commit"
                described.append(f"{change.path} ({dropped[change.path]})")
        remaining = [change for change in remaining if change.path not in dropped]
        total = sum(size(change) for change in remaining)
        for change in sorted((c for c in remaining if c.status == "A"), key=size, reverse=True):
            if total <= MAX_TOTAL_BYTES:
                break
            dropped[change.path] = f"{size(change) // 1000:,} kB, the change was too large in total"
            described.append(f"{change.path} ({dropped[change.path]})")
            total -= size(change)
        if not dropped:
            return changes, []

        # 3. The index follows. A build directory none of whose files is kept goes in one piece.
        kept_paths = {change.path for change in changes if change.path not in dropped}
        whole = [root for root in output_of if not any(path.startswith(root + "/") for path in kept_paths)]
        single = [
            change.path
            for change in changes
            if change.path in dropped and change.status == "A" and build_output_root(change.path) not in whole
        ]
        for batch in _in_batches(whole + single):
            # Literal: "app/[id]/page.js" is a path here, not a pattern.
            paths = " ".join(shlex.quote(path) for path in batch)
            result = self.git(f"--literal-pathspecs rm -r --cached -q --ignore-unmatch -- {paths}")
            if not result.ok:
                raise SandboxError(f"git rm --cached failed: {result.output()[-400:]}")
        for change in changes:
            if change.path in dropped and change.status != "A":
                # A file of the repository that became too large: it stays as it was.
                old_mode, old_sha = previous[change.path]
                result = self.git(f"update-index --cacheinfo {old_mode},{old_sha},{shlex.quote(change.path)}")
                if not result.ok:
                    raise SandboxError(f"git update-index failed: {result.output()[-400:]}")
        return [change for change in changes if change.path not in dropped], sorted(described)

    def _paths_in_base(self, paths: List[str]) -> Optional[set]:
        """Which of the given files and directories exist in the commit that
        was cloned. None when that cannot be told."""
        found: set = set()
        for batch in _in_batches(paths):
            names = " ".join(shlex.quote(path) for path in batch)
            result = self.git(
                "-c core.quotePath=false --literal-pathspecs ls-tree --name-only "
                f"{shlex.quote(self.diff_base)} -- {names}"
            )
            if not result.ok:
                return None
            found.update(line.strip() for line in result.stdout.splitlines() if line.strip())
        return found

    def _attach_line_counts(self, changes: List[FileChange]) -> None:
        numstat = self.git_shell(
            "git diff --cached --numstat -z --no-renames "
            f"{shlex.quote(self.diff_base)} | base64 | tr -d '\\n'"
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
        sizes = self._blob_sizes(unique)

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
