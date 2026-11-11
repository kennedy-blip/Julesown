"""Local stand-ins for the external systems, used only by the test-suite.

* ``LocalWorkspace`` - a sandbox backed by a temporary directory and real bash.
* ``GitRemote``      - real bare git repositories acting as "GitHub remotes".
* ``FakeGitHub``     - the subset of the GitHub REST API the backend uses,
  implemented on top of those bare repositories with real git plumbing, so
  blob/tree/commit shas are genuine and integrity checks are exercised.
* ``ScriptedLLM``    - a chat client that replays a fixed list of tool calls.

Together they let the whole pipeline (clone -> edit -> test -> commit -> pull
request) run for real, offline, without touching github.com, E2B or an LLM.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import unquote

import httpx

from apps.api.config import Settings
from apps.api.services.github import GitHubClient
from apps.api.services.llm import LLMResponse, ToolCall
from apps.api.services.runner import RunnerDeps
from apps.api.services.sandbox import CommandResult

GIT_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_AUTHOR_NAME": "Test Author",
    "GIT_AUTHOR_EMAIL": "author@example.com",
    "GIT_COMMITTER_NAME": "Test Committer",
    "GIT_COMMITTER_EMAIL": "committer@example.com",
    "GIT_TERMINAL_PROMPT": "0",
}


def git(cwd: Union[str, Path], *args: str, stdin: Optional[bytes] = None, env: Optional[Dict[str, str]] = None) -> str:
    """Run git and return stdout (stripped). Raises on failure."""
    completed = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        input=stdin,
        capture_output=True,
        env={**os.environ, **GIT_ENV, **(env or {})},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)} failed: {completed.stderr.decode('utf-8', 'replace')}")
    return completed.stdout.decode("utf-8", "replace").strip()


# --------------------------------------------------------------------------- #
# Sandbox
# --------------------------------------------------------------------------- #
class LocalWorkspace:
    """A Workspace that runs commands with bash inside a temporary directory."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.workdir = str(self.root)
        self.description = "local test workspace"
        self.closed = False
        self.commands: List[str] = []
        self.environments: List[Dict[str, str]] = []

    def run(
        self,
        command: str,
        *,
        cwd: Optional[str] = None,
        env: Optional[Dict[str, str]] = None,
        timeout: float = 60.0,
    ) -> CommandResult:
        self.commands.append(command)
        self.environments.append(dict(env or {}))
        full_env = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": str(self.root),
            "LANG": "C.UTF-8",
            **GIT_ENV,
            **(env or {}),
        }
        try:
            completed = subprocess.run(
                ["bash", "-c", command],
                cwd=cwd or self.workdir,
                env=full_env,
                capture_output=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return CommandResult(124, "", f"Command timed out after {int(timeout)} seconds.", timed_out=True)
        return CommandResult(
            completed.returncode,
            completed.stdout.decode("utf-8", "replace"),
            completed.stderr.decode("utf-8", "replace"),
        )

    def read_file(self, path: str) -> str:
        target = Path(path)
        if not target.is_file():
            raise FileNotFoundError(path)
        return target.read_text(encoding="utf-8")

    def write_file(self, path: str, content: str) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def close(self) -> None:
        self.closed = True


# --------------------------------------------------------------------------- #
# Git remotes
# --------------------------------------------------------------------------- #
class GitRemote:
    """Bare repositories on disk standing in for repositories on GitHub."""

    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for(self, slug: str) -> Path:
        return self.root / f"{slug.replace('/', '__')}.git"

    def clone_url(self, slug: str) -> str:
        return f"file://{self.path_for(slug)}"

    def create(self, slug: str, files: Dict[str, str], *, branch: str = "main") -> Path:
        bare = self.path_for(slug)
        git(self.root, "init", "--quiet", "--bare", f"--initial-branch={branch}", str(bare))
        with tempfile.TemporaryDirectory(dir=str(self.root)) as scratch:
            git(scratch, "init", "--quiet", f"--initial-branch={branch}")
            for name, content in files.items():
                target = Path(scratch) / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content, encoding="utf-8")
            git(scratch, "add", "-A")
            git(scratch, "commit", "--quiet", "-m", "Initial commit")
            git(scratch, "push", "--quiet", str(bare), f"HEAD:refs/heads/{branch}")
        return bare

    def rev(self, slug: str, ref: str) -> Optional[str]:
        try:
            return git(self.path_for(slug), "rev-parse", "--verify", "--quiet", ref)
        except RuntimeError:
            return None

    def show(self, slug: str, ref: str, path: str) -> str:
        return git(self.path_for(slug), "show", f"{ref}:{path}")

    def files(self, slug: str, ref: str) -> List[str]:
        listing = git(self.path_for(slug), "ls-tree", "-r", "--name-only", ref)
        return [line for line in listing.splitlines() if line]

    def changed(self, slug: str, base: str, head: str) -> List[str]:
        listing = git(self.path_for(slug), "diff", "--name-status", "--no-renames", base, head)
        return [line for line in listing.splitlines() if line]


# --------------------------------------------------------------------------- #
# GitHub REST API
# --------------------------------------------------------------------------- #
class FakeGitHub:
    APP_SLUG = "jules-test-app"
    PAT = "ghp_" + "t" * 36

    def __init__(self, remote: GitRemote) -> None:
        self.remote = remote
        self.repos: Dict[str, Dict[str, Any]] = {}
        self.installed: Dict[str, int] = {}  # slug -> installation id
        self.app_permissions: Dict[str, str] = {
            "contents": "write",
            "pull_requests": "write",
            "issues": "write",
            "metadata": "read",
        }
        self.pulls: Dict[str, List[Dict[str, Any]]] = {}
        self.comments: List[Dict[str, Any]] = []
        self.requests: List[Tuple[str, str]] = []
        self.token_requests: List[Dict[str, Any]] = []
        self.issued_tokens: Dict[str, Dict[str, Any]] = {}
        self.failures: List[Tuple[str, "re.Pattern[str]", int, Dict[str, Any]]] = []
        self.transport = httpx.MockTransport(self._handle)

    # -- configuration ---------------------------------------------------
    def add_repo(
        self,
        slug: str,
        files: Dict[str, str],
        *,
        default_branch: str = "main",
        private: bool = False,
        archived: bool = False,
        push: bool = True,
        installed: bool = True,
    ) -> None:
        self.remote.create(slug, files, branch=default_branch)
        self.repos[slug.lower()] = {
            "full_name": slug,
            "default_branch": default_branch,
            "private": private,
            "archived": archived,
            "push": push,
        }
        self.pulls[slug.lower()] = []
        if installed:
            self.installed[slug.lower()] = 1000 + len(self.installed)

    def fail_next(self, method: str, path_pattern: str, status: int, body: Dict[str, Any]) -> None:
        self.failures.append((method.upper(), re.compile(path_pattern), status, body))

    def client(self, settings: Optional[Settings] = None, **overrides: Any) -> GitHubClient:
        options: Dict[str, Any] = {"transport": self.transport}
        if settings is not None:
            options.update(
                token=settings.github_token,
                app_id=settings.github_app_id,
                app_private_key=settings.github_app_private_key,
            )
        options.update(overrides)
        return GitHubClient(**options)

    def calls(self, method: str, pattern: str) -> List[str]:
        compiled = re.compile(pattern)
        return [path for verb, path in self.requests if verb == method.upper() and compiled.search(path)]

    # -- helpers ---------------------------------------------------------
    @staticmethod
    def _json(status: int, body: Any) -> httpx.Response:
        return httpx.Response(status, json=body)

    def _not_found(self) -> httpx.Response:
        return self._json(404, {"message": "Not Found", "documentation_url": "https://docs.github.com/rest"})

    def _repo_json(self, repo: Dict[str, Any]) -> Dict[str, Any]:
        slug = repo["full_name"]
        return {
            "full_name": slug,
            "private": repo["private"],
            "archived": repo["archived"],
            "default_branch": repo["default_branch"],
            "html_url": f"https://github.com/{slug}",
            "permissions": {"pull": True, "push": repo["push"], "admin": False},
        }

    def _authorize(self, request: httpx.Request) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Returns (kind, token_info); kind is 'jwt', 'pat', 'installation' or None."""
        header = request.headers.get("authorization", "")
        token = header.split(" ", 1)[1] if " " in header else ""
        if token.count(".") == 2 and token.startswith("eyJ"):
            return "jwt", None
        if token == self.PAT:
            return "pat", None
        if token in self.issued_tokens:
            return "installation", self.issued_tokens[token]
        return None, None

    # -- request handling --------------------------------------------------
    def _handle(self, request: httpx.Request) -> httpx.Response:
        method = request.method.upper()
        path = unquote(request.url.path)
        self.requests.append((method, path))

        for index, (verb, pattern, status, body) in enumerate(self.failures):
            if verb == method and pattern.search(path):
                del self.failures[index]
                return self._json(status, body)

        kind, token_info = self._authorize(request)
        if kind is None:
            return self._json(401, {"message": "Bad credentials"})
        payload: Dict[str, Any] = {}
        if request.content:
            try:
                payload = json.loads(request.content.decode("utf-8"))
            except ValueError:
                payload = {}

        # ----- GitHub App endpoints (JWT) ------------------------------
        if path == "/app" and method == "GET":
            if kind != "jwt":
                return self._json(401, {"message": "A JSON web token could not be decoded"})
            return self._json(
                200,
                {
                    "slug": self.APP_SLUG,
                    "name": "Jules Test App",
                    "html_url": f"https://github.com/apps/{self.APP_SLUG}",
                },
            )
        if path == "/app/installations" and method == "GET":
            return self._json(200, [{"id": inst} for inst in sorted(set(self.installed.values()))])
        match = re.fullmatch(r"/app/installations/(\d+)/access_tokens", path)
        if match and method == "POST":
            if kind != "jwt":
                return self._json(401, {"message": "A JSON web token could not be decoded"})
            installation_id = int(match.group(1))
            permissions = dict(payload.get("permissions") or self.app_permissions)
            token = f"ghs_{installation_id}_{len(self.issued_tokens):04d}_" + "x" * 20
            info = {
                "installation_id": installation_id,
                "repositories": payload.get("repositories"),
                "permissions": permissions,
            }
            self.issued_tokens[token] = info
            self.token_requests.append({"installation_id": installation_id, **payload})
            return self._json(
                201, {"token": token, "expires_at": "2099-01-01T00:00:00Z", "permissions": permissions}
            )
        if path == "/installation/repositories" and method == "GET":
            visible = [self._repo_json(repo) for key, repo in self.repos.items() if key in self.installed]
            return self._json(200, {"total_count": len(visible), "repositories": visible})
        if path == "/user/repos" and method == "GET":
            return self._json(200, [self._repo_json(repo) for repo in self.repos.values()])

        # ----- repository endpoints ------------------------------------
        match = re.match(r"^/repos/([^/]+)/([^/]+)(/.*)?$", path)
        if not match:
            return self._not_found()
        slug = f"{match.group(1)}/{match.group(2)}"
        rest = match.group(3) or ""
        repo = self.repos.get(slug.lower())

        if rest == "/installation" and method == "GET":
            if kind != "jwt":
                return self._json(401, {"message": "A JSON web token could not be decoded"})
            if repo is None or slug.lower() not in self.installed:
                return self._not_found()
            return self._json(200, {"id": self.installed[slug.lower()], "app_slug": self.APP_SLUG})

        if kind == "jwt":
            return self._json(401, {"message": "Bad credentials"})
        if repo is None:
            return self._not_found()
        if kind == "installation":
            assert token_info is not None
            allowed = token_info.get("repositories")
            if slug.lower() not in self.installed or (
                allowed is not None and repo["full_name"].split("/", 1)[1] not in allowed
            ):
                return self._not_found()
        can_write = repo["push"] and (
            kind == "pat" or (token_info or {}).get("permissions", {}).get("contents") == "write"
        )
        bare = self.remote.path_for(repo["full_name"])
        slug = repo["full_name"]

        if rest == "" and method == "GET":
            return self._json(200, self._repo_json(repo))

        if rest.startswith("/git/ref/heads/") and method == "GET":
            branch = rest[len("/git/ref/heads/") :]
            sha = self.remote.rev(slug, f"refs/heads/{branch}")
            if not sha:
                return self._not_found()
            return self._json(200, {"ref": f"refs/heads/{branch}", "object": {"sha": sha, "type": "commit"}})

        if rest == "/pulls" and method == "GET":
            head = request.url.params.get("head", "")
            wanted = head.split(":", 1)[1] if ":" in head else head
            found = [pr for pr in self.pulls[slug.lower()] if not wanted or pr["head"]["ref"] == wanted]
            return self._json(200, found)

        if method != "POST":
            return self._not_found()
        if not can_write:
            return self._json(403, {"message": "Resource not accessible by integration"})

        if rest == "/git/blobs":
            data = base64.b64decode(payload["content"])
            sha = git(bare, "hash-object", "-w", "--stdin", stdin=data)
            return self._json(201, {"sha": sha})

        if rest == "/git/trees":
            with tempfile.TemporaryDirectory() as scratch:
                # A throwaway index (and empty work tree) so plumbing commands
                # can build the tree without touching the bare repository's refs.
                work_tree = Path(scratch) / "tree"
                work_tree.mkdir()
                index_env = {
                    "GIT_INDEX_FILE": str(Path(scratch) / "index"),
                    "GIT_DIR": str(bare),
                    "GIT_WORK_TREE": str(work_tree),
                }
                git(work_tree, "read-tree", payload["base_tree"], env=index_env)
                for entry in payload["tree"]:
                    if entry.get("sha") is None:
                        git(work_tree, "update-index", "--force-remove", "--", entry["path"], env=index_env)
                    else:
                        git(
                            work_tree,
                            "update-index",
                            "--add",
                            "--cacheinfo",
                            f"{entry['mode']},{entry['sha']},{entry['path']}",
                            env=index_env,
                        )
                sha = git(work_tree, "write-tree", env=index_env)
            return self._json(201, {"sha": sha})

        if rest == "/git/commits":
            sha = git(
                bare,
                "commit-tree",
                payload["tree"],
                *[item for parent in payload["parents"] for item in ("-p", parent)],
                "-m",
                payload["message"],
            )
            return self._json(201, {"sha": sha, "html_url": f"https://github.com/{slug}/commit/{sha}"})

        if rest == "/git/refs":
            ref = payload["ref"]
            if self.remote.rev(slug, ref):
                return self._json(422, {"message": "Reference already exists"})
            git(bare, "update-ref", ref, payload["sha"])
            return self._json(201, {"ref": ref, "object": {"sha": payload["sha"]}})

        if rest == "/pulls":
            if (token_info or {}).get("permissions", {"pull_requests": "write"}).get("pull_requests") != "write" and kind != "pat":
                return self._json(403, {"message": "Resource not accessible by integration"})
            head, base = payload["head"], payload["base"]
            head_sha = self.remote.rev(slug, f"refs/heads/{head}")
            base_sha = self.remote.rev(slug, f"refs/heads/{base}")
            if not head_sha or not base_sha:
                return self._json(
                    422,
                    {"message": "Validation Failed", "errors": [{"resource": "PullRequest", "field": "head", "code": "invalid"}]},
                )
            if head_sha == base_sha:
                return self._json(
                    422,
                    {"message": "Validation Failed", "errors": [{"message": f"No commits between {base} and {head}"}]},
                )
            number = len(self.pulls[slug.lower()]) + 1
            pull = {
                "number": number,
                "html_url": f"https://github.com/{slug}/pull/{number}",
                "state": "open",
                "title": payload["title"],
                "body": payload.get("body") or "",
                "head": {"ref": head, "sha": head_sha},
                "base": {"ref": base, "sha": base_sha},
                "merged_at": None,
            }
            self.pulls[slug.lower()].append(pull)
            return self._json(201, pull)

        match = re.fullmatch(r"/issues/(\d+)/comments", rest)
        if match:
            comment = {"repository": slug, "issue": int(match.group(1)), "body": payload.get("body") or ""}
            self.comments.append(comment)
            return self._json(
                201, {"html_url": f"https://github.com/{slug}/issues/{match.group(1)}#issuecomment-{len(self.comments)}"}
            )
        return self._not_found()


# --------------------------------------------------------------------------- #
# LLM
# --------------------------------------------------------------------------- #
Step = Union[str, Sequence[Tuple[str, Dict[str, Any]]], Callable[[List[Dict[str, Any]]], Any]]


class ScriptedLLM:
    """Replays a script. Each step is either a text reply (``str``), a list of
    ``(tool_name, arguments)`` calls, or a callable returning one of those (it
    receives the conversation so far, to decide based on real tool output)."""

    provider = "scripted"

    def __init__(self, steps: Sequence[Step], *, model: str = "scripted-model", tokens_per_minute: Optional[int] = None) -> None:
        self.steps = list(steps)
        self.model = model
        self.tokens_per_minute = tokens_per_minute
        self.requests: List[List[Dict[str, Any]]] = []
        self.output_token_limits: List[int] = []
        self.closed = False

    def complete(self, messages, tools, *, max_output_tokens, timeout) -> LLMResponse:
        self.requests.append(json.loads(json.dumps(messages)))
        self.output_token_limits.append(max_output_tokens)
        if not self.steps:
            raise AssertionError("ScriptedLLM ran out of steps")
        step = self.steps.pop(0)
        if callable(step):
            step = step(messages)
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, str):
            return LLMResponse(
                content=step,
                tool_calls=[],
                finish_reason="stop",
                model=self.model,
                assistant_message={"role": "assistant", "content": step},
                tokens_per_minute_limit=self.tokens_per_minute,
            )
        calls: List[ToolCall] = []
        wire: List[Dict[str, Any]] = []
        for position, (name, arguments) in enumerate(step):
            call_id = f"call_{len(self.requests)}_{position}"
            calls.append(ToolCall(id=call_id, name=name, arguments=arguments))
            wire.append({"id": call_id, "type": "function", "function": {"name": name, "arguments": json.dumps(arguments)}})
        return LLMResponse(
            content="",
            tool_calls=calls,
            finish_reason="tool_calls",
            model=self.model,
            assistant_message={"role": "assistant", "content": None, "tool_calls": wire},
            tokens_per_minute_limit=self.tokens_per_minute,
        )

    def close(self) -> None:
        self.closed = True

    # Convenience for assertions -------------------------------------------
    def tool_results(self) -> List[str]:
        """Content of every tool message the model has been shown, in order."""
        if not self.requests:
            return []
        return [m["content"] for m in self.requests[-1] if m.get("role") == "tool"]


def make_deps(
    *,
    github: FakeGitHub,
    llm: Any,
    workspace_root: Union[str, Path],
    workspaces: Optional[List[LocalWorkspace]] = None,
) -> RunnerDeps:
    """Runner dependencies wired to the local stand-ins."""

    def workspace_factory(_settings: Settings, task_id: str, _lifetime: int) -> LocalWorkspace:
        workspace = LocalWorkspace(Path(workspace_root) / task_id)
        if workspaces is not None:
            workspaces.append(workspace)
        return workspace

    def llm_factory(_settings: Settings, _notice: Callable[[str], None]) -> Any:
        return llm(_notice) if callable(llm) and not hasattr(llm, "complete") else llm

    return RunnerDeps(
        github_factory=lambda settings: github.client(settings),
        workspace_factory=workspace_factory,
        llm_factory=llm_factory,
        clone_url_for=github.remote.clone_url,
    )
