"""GitHub integration: authentication, repository lookups, commits and pull requests.

Nothing in this module fabricates a result. Every URL it returns was handed
back by the GitHub API for an object that GitHub confirmed it created.

Commits are created through the Git Data API (blobs -> tree -> commit -> ref)
rather than by ``git push`` from the sandbox, so write credentials never have
to be exposed to the environment where model-chosen commands run. A repository
without any commit is the one case that API refuses; its first commit is made
through the Contents API (see ``create_initial_commit``).
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from urllib.parse import quote, urlparse

import httpx

API_VERSION = "2022-11-28"
USER_AGENT = "jules-clone"

_OWNER_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})$")
_REPO_RE = re.compile(r"^[A-Za-z0-9._-]{1,100}$")
_BRANCH_FORBIDDEN_RE = re.compile(r"[\x00-\x20\x7f~^:?*\[\\]")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")

REQUIRED_APP_PERMISSIONS = {"contents": "write", "pull_requests": "write"}

# Listing: GitHub returns at most 100 items per page. Every page is fetched, up
# to this many repositories (an account can have thousands).
PAGE_SIZE = 100
MAX_LISTED_REPOSITORIES = 2000
MAX_LISTED_INSTALLATIONS = 300
_DESCRIPTION_CHARS = 140
# Pages of a list that are requested at the same time once the first page has
# said how many there are. A page of 100 repositories takes most of a second.
LIST_CONCURRENCY = 4
_LAST_PAGE_RE = re.compile(r'<[^<>]*[?&]page=(\d+)[^<>]*>\s*;\s*rel="last"')

# Committing: text files travel inside the tree request (GitHub writes the blobs
# itself), so a change of fifty files is one request instead of fifty. Larger and
# binary files are uploaded as blobs one by one.
INLINE_FILE_BYTES = 200_000
INLINE_TREE_BYTES = 1_000_000


def _inline_text(change: "FileChange") -> Optional[str]:
    """The file's content as text if it can be sent inside a tree request.

    Only regular files that are non-empty, valid UTF-8 and free of NUL bytes:
    for those a JSON string round-trips to exactly the same bytes.
    """
    if change.mode not in ("100644", "100755") or not change.content_b64:
        return None
    if len(change.content_b64) > INLINE_FILE_BYTES * 4 // 3 + 4:
        return None
    try:
        text = base64.b64decode(change.content_b64, validate=True).decode("utf-8")
    except (binascii.Error, UnicodeDecodeError, ValueError):
        return None
    return None if "\x00" in text else text


class GitHubError(Exception):
    """A GitHub operation failed. ``message`` is safe to show to the user."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: str = "github_error",
        hint: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.hint = hint
        # Machine-readable context (e.g. what was already written when a
        # multi-step operation failed half way).
        self.details: Dict[str, Any] = dict(details or {})

    def describe(self) -> str:
        return f"{self.message} {self.hint}" if self.hint else self.message


# --------------------------------------------------------------------------- #
# Input validation
# --------------------------------------------------------------------------- #
def parse_repository(raw: Optional[str]) -> str:
    """Normalise user input into an ``owner/name`` slug.

    Accepts ``owner/name``, https/http/ssh GitHub URLs, a trailing ``.git`` and
    URLs with extra path segments (``/tree/main``, ``/pull/3`` ...). Raises
    ``ValueError`` with a user-facing message for anything else - it never
    guesses or substitutes a default repository.
    """
    text = (raw or "").strip()
    if not text:
        raise ValueError("Repository is required. Use owner/name or a GitHub URL.")

    ssh_match = re.match(r"^(?:ssh://)?git@github\.com[:/](.+)$", text, re.IGNORECASE)
    if ssh_match:
        path = ssh_match.group(1)
    else:
        candidate = text
        if re.match(r"^(?:www\.)?github\.com/", candidate, re.IGNORECASE):
            candidate = "https://" + candidate
        if "://" in candidate:
            parsed = urlparse(candidate)
            host = (parsed.hostname or "").lower()
            if host not in ("github.com", "www.github.com"):
                raise ValueError(
                    f"Only github.com repositories are supported (got '{host or text}')."
                )
            path = parsed.path
        else:
            path = candidate.split("?", 1)[0].split("#", 1)[0]

    parts = [segment for segment in path.strip("/").split("/") if segment]
    if len(parts) < 2:
        raise ValueError(f"'{text}' is not a repository. Use owner/name or a GitHub URL.")

    owner, name = parts[0], parts[1]
    if name.lower().endswith(".git"):
        name = name[: -len(".git")]
    if not _OWNER_RE.match(owner):
        raise ValueError(f"'{owner}' is not a valid GitHub owner name.")
    if not _REPO_RE.match(name) or name in (".", ".."):
        raise ValueError(f"'{name}' is not a valid GitHub repository name.")
    return f"{owner}/{name}"


def validate_branch_name(raw: Optional[str]) -> str:
    """Validate a branch name (a conservative subset of git-check-ref-format)."""
    name = (raw or "").strip()
    if name.startswith("refs/heads/"):
        name = name[len("refs/heads/") :]
    invalid = (
        not name
        or len(name) > 200
        or name == "@"
        or name.startswith(("-", "/", "."))
        or name.endswith(("/", ".", ".lock"))
        or ".." in name
        or "//" in name
        or "@{" in name
        or _BRANCH_FORBIDDEN_RE.search(name) is not None
        or any(part.startswith(".") or part.endswith(".lock") for part in name.split("/"))
    )
    if invalid:
        raise ValueError(f"'{raw}' is not a valid branch name.")
    return name


def normalize_private_key(raw: str) -> str:
    """Repair the usual ways a PEM key gets mangled in a hosting dashboard:
    surrounding quotes, literal ``\\n`` sequences, base64-wrapping, or all
    newlines collapsed into spaces."""
    key = (raw or "").strip()
    if len(key) >= 2 and key[0] == key[-1] and key[0] in "\"'":
        key = key[1:-1].strip()
    if "\\n" in key:
        key = key.replace("\\r", "").replace("\\n", "\n")
    if "-----BEGIN" not in key:
        try:
            decoded = base64.b64decode("".join(key.split()), validate=True).decode("utf-8")
        except (binascii.Error, UnicodeDecodeError, ValueError):
            decoded = ""
        if "-----BEGIN" in decoded:
            key = decoded.strip()
    match = re.match(r"^(-----BEGIN [A-Z0-9 ]+-----)(.*?)(-----END [A-Z0-9 ]+-----)\s*$", key, re.DOTALL)
    if match and "\n" not in match.group(2).strip():
        body = "".join(match.group(2).split())
        lines = [body[i : i + 64] for i in range(0, len(body), 64)]
        key = "\n".join([match.group(1), *lines, match.group(3)])
    return key + "\n" if key and not key.endswith("\n") else key


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def build_app_jwt(app_id: str, private_key_pem: str, *, now: Optional[float] = None) -> str:
    """Create the short-lived RS256 JWT a GitHub App authenticates with."""
    try:
        from cryptography.hazmat.primitives import hashes, serialization
        from cryptography.hazmat.primitives.asymmetric import padding, rsa
    except ImportError as exc:  # pragma: no cover - dependency is in requirements.txt
        raise GitHubError(
            "The 'cryptography' package is required for GitHub App authentication.",
            code="app_key_invalid",
        ) from exc

    pem = normalize_private_key(private_key_pem)
    try:
        key = serialization.load_pem_private_key(pem.encode("utf-8"), password=None)
    except Exception as exc:
        raise GitHubError(
            "GITHUB_APP_PRIVATE_KEY could not be read as a PEM private key.",
            code="app_key_invalid",
            hint=(
                "Paste the full contents of the .pem file GitHub generated for the App, "
                "including the BEGIN/END lines."
            ),
        ) from exc
    if not isinstance(key, rsa.RSAPrivateKey):
        raise GitHubError(
            "GITHUB_APP_PRIVATE_KEY is not an RSA private key.", code="app_key_invalid"
        )

    issued_at = int(now if now is not None else time.time())
    header = {"alg": "RS256", "typ": "JWT"}
    # iat is backdated to tolerate clock drift; GitHub caps exp at 10 minutes.
    payload = {"iat": issued_at - 60, "exp": issued_at + 540, "iss": str(app_id).strip()}
    signing_input = (
        _b64url(json.dumps(header, separators=(",", ":")).encode("utf-8"))
        + "."
        + _b64url(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
    )
    signature = key.sign(signing_input.encode("ascii"), padding.PKCS1v15(), hashes.SHA256())
    return signing_input + "." + _b64url(signature)


# --------------------------------------------------------------------------- #
# Data containers
# --------------------------------------------------------------------------- #
@dataclass
class RepoAccess:
    slug: str
    token: str
    mode: str  # "app" | "token"
    expires_at: float = float("inf")
    installation_id: Optional[int] = None
    permissions: Dict[str, str] = field(default_factory=dict)


@dataclass
class RepositoryInfo:
    slug: str
    default_branch: str
    private: bool
    archived: bool
    html_url: str
    can_push: Optional[bool]


@dataclass
class FileChange:
    """One entry of the staged diff between the base commit and the work tree."""

    path: str
    status: str  # A (added), M (modified), D (deleted), T (type changed)
    mode: str  # git file mode of the new entry, e.g. "100644"
    sha: Optional[str] = None  # blob sha computed by git; None for deletions
    content_b64: Optional[str] = None  # blob content; None for deletions/submodules
    additions: Optional[int] = None
    deletions: Optional[int] = None

    def summary(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "status": self.status,
            "additions": self.additions,
            "deletions": self.deletions,
        }


@dataclass
class PullRequest:
    number: int
    html_url: str
    state: str
    title: str
    head_branch: str
    base_branch: str
    merged: bool = False
    head_sha: str = ""
    head_repository: str = ""  # owner/name of the repository the head branch lives in
    author: str = ""


# A check that ended in one of these says the commit is broken. "cancelled"
# (superseded by a newer run) and "action_required" (waiting for a person) do not.
FAILED_CHECK_CONCLUSIONS = ("failure", "timed_out", "startup_failure")
FAILED_STATUS_STATES = ("failure", "error")
MAX_REPORTED_CHECKS = 12
CHECK_LOG_CHARS = 5000
_LOG_TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T[\d:.]+Z ?")
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def log_excerpt(log: str, limit: int = CHECK_LOG_CHARS) -> str:
    """The part of a CI job log that says what went wrong.

    GitHub Actions marks failures with ``##[error]``; the lines leading up to
    the last such marker are what a person would scroll to. Without a marker
    the end of the log is used.
    """
    lines = [_ANSI_RE.sub("", _LOG_TIMESTAMP_RE.sub("", line)).rstrip() for line in log.splitlines()]
    lines = [line for line in lines if line and not line.startswith("##[debug]")]
    if not lines:
        return ""
    marks = [index for index, line in enumerate(lines) if "##[error]" in line]
    end = min(len(lines), marks[-1] + 4) if marks else len(lines)
    start = max(0, (marks[0] if marks else end) - 60, end - 140)
    excerpt = "\n".join(lines[start:end])
    if len(excerpt) > limit:
        excerpt = "[...]\n" + excerpt[-limit:]
    return excerpt


# --------------------------------------------------------------------------- #
# Client
# --------------------------------------------------------------------------- #
class GitHubClient:
    """Small synchronous GitHub REST client.

    Authentication is resolved per repository: a GitHub App installation token
    when the App is configured and installed on the repository, otherwise
    ``GITHUB_TOKEN``.
    """

    def __init__(
        self,
        *,
        token: Optional[str] = None,
        app_id: Optional[str] = None,
        app_private_key: Optional[str] = None,
        api_url: str = "https://api.github.com",
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = 20.0,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._token = token or None
        self._app_id = app_id or None
        self._app_private_key = app_private_key or None
        self._clock = clock
        self._lock = threading.Lock()
        self._access_cache: Dict[str, RepoAccess] = {}
        self._extra_secrets: List[str] = []
        self._jwt_cache: Optional[Tuple[str, float]] = None
        self._app_info: Optional[Dict[str, Any]] = None
        # Who the commits are made by ({} once looked up without success).
        self._committer: Optional[Dict[str, str]] = None
        self._pause: Callable[[float], None] = time.sleep  # between retries
        self._http = httpx.Client(
            base_url=api_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
            headers={
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": API_VERSION,
                "User-Agent": USER_AGENT,
            },
        )

    # ------------------------------------------------------------------ #
    @property
    def app_configured(self) -> bool:
        return bool(self._app_id and self._app_private_key)

    @property
    def auth_mode(self) -> str:
        if self.app_configured:
            return "app"
        if self._token:
            return "token"
        return "none"

    def secrets(self) -> List[str]:
        """Token values currently held, so callers can redact them from logs."""
        with self._lock:
            values = [access.token for access in self._access_cache.values()]
            values.extend(self._extra_secrets)
        if self._token:
            values.append(self._token)
        return [value for value in values if value]

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "GitHubClient":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Low-level request helpers
    # ------------------------------------------------------------------ #
    def _send(
        self,
        method: str,
        path: str,
        *,
        bearer: str,
        action: str,
        json_body: Any = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        try:
            return self._http.request(
                method,
                path,
                headers={"Authorization": f"Bearer {bearer}"},
                json=json_body,
                params=params,
            )
        except httpx.HTTPError as exc:
            raise GitHubError(
                f"Could not reach the GitHub API while {action} ({exc.__class__.__name__}: {exc}).",
                code="github_unreachable",
            ) from exc

    @staticmethod
    def _error(response: httpx.Response, action: str) -> GitHubError:
        status = response.status_code
        try:
            data = response.json()
        except ValueError:
            data = {}
        message = data.get("message") if isinstance(data, dict) else None
        details: List[str] = []
        raw_errors = data.get("errors") if isinstance(data, dict) else None
        if isinstance(raw_errors, list):
            for item in raw_errors:
                if isinstance(item, dict):
                    text = item.get("message") or " ".join(
                        str(item[key]) for key in ("resource", "field", "code") if item.get(key)
                    )
                    if text:
                        details.append(str(text))
                elif item:
                    details.append(str(item))
        text = message or (response.text or "").strip()[:200] or "no details"
        if details:
            text = f"{text} ({'; '.join(details)})"

        hint: Optional[str] = None
        code = "github_error"
        lowered = text.lower()
        if status == 401:
            code = "bad_credentials"
            hint = "GitHub rejected the credentials - check the configured token or App key."
        elif status == 403 and "rate limit" in lowered:
            code = "rate_limited"
            hint = "The GitHub API rate limit was hit; try again later."
        elif "workflow" in lowered and "permission" in lowered:
            # "refusing to allow a GitHub App to create or update workflow ... without
            # `workflows` permission"
            code = "workflow_permission"
            hint = (
                "GitHub only accepts files under .github/workflows from credentials with the "
                "'Workflows' permission. Grant it to the GitHub App (or token), or ask for the "
                "change without workflow files."
            )
        elif status == 403:
            code = "forbidden"
            hint = "The GitHub App or token lacks permission for this operation."
        elif status == 404:
            code = "not_found"
        return GitHubError(
            f"GitHub refused {action}: HTTP {status} - {text}",
            status_code=status,
            code=code,
            hint=hint,
        )

    def _json(
        self,
        method: str,
        path: str,
        *,
        bearer: str,
        action: str,
        json_body: Any = None,
        params: Optional[Dict[str, Any]] = None,
        ok: Iterable[int] = (200, 201),
    ) -> Any:
        response = self._send(
            method, path, bearer=bearer, action=action, json_body=json_body, params=params
        )
        if response.status_code not in tuple(ok):
            raise self._error(response, action)
        if not response.content:
            return {}
        try:
            return response.json()
        except ValueError as exc:
            raise GitHubError(
                f"GitHub returned an unreadable response while {action}.", code="github_error"
            ) from exc

    # ------------------------------------------------------------------ #
    # GitHub App authentication
    # ------------------------------------------------------------------ #
    def _app_jwt(self) -> str:
        if not self.app_configured:
            raise GitHubError("No GitHub App is configured.", code="no_credentials")
        now = self._clock()
        with self._lock:
            if self._jwt_cache and self._jwt_cache[1] - 60 > now:
                return self._jwt_cache[0]
        token = build_app_jwt(self._app_id or "", self._app_private_key or "", now=now)
        with self._lock:
            self._jwt_cache = (token, now + 540)
        return token

    def app_info(self) -> Dict[str, Any]:
        """Metadata of the configured GitHub App (``slug``, ``name``, ``html_url``)."""
        if self._app_info is None:
            response = self._send("GET", "/app", bearer=self._app_jwt(), action="identifying the GitHub App")
            if response.status_code == 401:
                raise GitHubError(
                    "GitHub rejected the App credentials.",
                    status_code=401,
                    code="bad_credentials",
                    hint="Check that GITHUB_APP_ID matches the App the private key was generated for.",
                )
            if response.status_code != 200:
                raise self._error(response, "identifying the GitHub App")
            self._app_info = response.json()
        return self._app_info

    def app_install_url(self) -> Optional[str]:
        """Where the owner can install the App on more repositories."""
        try:
            html_url = self.app_info().get("html_url")
        except GitHubError:
            return None
        return f"{html_url}/installations/new" if html_url else None

    @staticmethod
    def _parse_expiry(value: Optional[str], fallback: float) -> float:
        if not value:
            return fallback
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return fallback

    def _installation_token(
        self, installation_id: int, repository_name: Optional[str] = None
    ) -> Tuple[str, float, Dict[str, str]]:
        body = {"repositories": [repository_name]} if repository_name else None
        data = self._json(
            "POST",
            f"/app/installations/{installation_id}/access_tokens",
            bearer=self._app_jwt(),
            action="requesting an installation token",
            json_body=body,
            ok=(201,),
        )
        token = data.get("token")
        if not token:
            raise GitHubError("GitHub did not return an installation token.", code="github_error")
        expires = self._parse_expiry(data.get("expires_at"), self._clock() + 3000)
        permissions = data.get("permissions") or {}
        return token, expires, {str(k): str(v) for k, v in permissions.items()}

    def _app_access(self, slug: str) -> RepoAccess:
        response = self._send(
            "GET",
            f"/repos/{slug}/installation",
            bearer=self._app_jwt(),
            action=f"looking up the App installation for {slug}",
        )
        if response.status_code == 404:
            install_url = self.app_install_url()
            raise GitHubError(
                f"The GitHub App is not installed on {slug} (or the repository does not exist).",
                status_code=404,
                code="app_not_installed",
                hint=(
                    f"Install it on that repository: {install_url}"
                    if install_url
                    else "Install the GitHub App on that repository, then try again."
                ),
            )
        if response.status_code == 401:
            raise GitHubError(
                "GitHub rejected the App credentials.",
                status_code=401,
                code="bad_credentials",
                hint="Check that GITHUB_APP_ID matches the App the private key was generated for.",
            )
        if response.status_code != 200:
            raise self._error(response, f"looking up the App installation for {slug}")

        installation_id = int(response.json()["id"])
        token, expires, permissions = self._installation_token(installation_id, slug.split("/", 1)[1])
        missing = [
            name
            for name, level in REQUIRED_APP_PERMISSIONS.items()
            if permissions.get(name) != level
        ]
        if missing:
            raise GitHubError(
                "The GitHub App is installed but lacks write access to: "
                + ", ".join(name.replace("_", " ") for name in missing)
                + ".",
                code="app_missing_permissions",
                hint=(
                    "In the App settings open 'Permissions & events' and set Repository permissions "
                    "'Contents' and 'Pull requests' to 'Read and write', then approve the new "
                    "permissions on the installation."
                ),
            )
        return RepoAccess(
            slug=slug,
            token=token,
            mode="app",
            expires_at=expires,
            installation_id=installation_id,
            permissions=permissions,
        )

    def access_for(self, slug: str) -> RepoAccess:
        """Resolve (and cache) credentials able to act on ``slug``."""
        key = slug.lower()
        now = self._clock()
        with self._lock:
            cached = self._access_cache.get(key)
        if cached and cached.expires_at - 120 > now:
            return cached

        access: Optional[RepoAccess] = None
        if self.app_configured:
            try:
                access = self._app_access(slug)
            except GitHubError as exc:
                # A personal token can still cover repositories the App cannot.
                recoverable = exc.code in ("app_not_installed", "app_missing_permissions")
                if not recoverable or not self._token:
                    raise
        if access is None:
            if not self._token:
                raise GitHubError(
                    "No GitHub credentials are configured.",
                    code="no_credentials",
                    hint="Set GITHUB_APP_ID + GITHUB_APP_PRIVATE_KEY, or GITHUB_TOKEN.",
                )
            access = RepoAccess(slug=slug, token=self._token, mode="token")
        with self._lock:
            self._access_cache[key] = access
        return access

    def clone_token(self, slug: str) -> str:
        """A credential for cloning a private repository inside the sandbox.

        With a GitHub App this is a separate token limited to *reading* that one
        repository, so the write-capable token never enters the sandbox. A
        personal access token cannot be narrowed, so it is used as is.
        """
        access = self.access_for(slug)
        if access.mode != "app" or access.installation_id is None:
            return access.token
        data = self._json(
            "POST",
            f"/app/installations/{access.installation_id}/access_tokens",
            bearer=self._app_jwt(),
            action="requesting a read-only clone token",
            json_body={
                "repositories": [slug.split("/", 1)[1]],
                "permissions": {"contents": "read"},
            },
            ok=(201,),
        )
        token = data.get("token")
        if not token:
            raise GitHubError("GitHub did not return a clone token.", code="github_error")
        with self._lock:
            self._extra_secrets.append(token)
        return token

    # ------------------------------------------------------------------ #
    # Repository metadata
    # ------------------------------------------------------------------ #
    def get_repository(self, slug: str) -> RepositoryInfo:
        access = self.access_for(slug)
        response = self._send("GET", f"/repos/{slug}", bearer=access.token, action=f"reading {slug}")
        if response.status_code == 404:
            raise GitHubError(
                f"Repository {slug} was not found, or the configured credentials cannot see it.",
                status_code=404,
                code="repo_not_found",
            )
        if response.status_code != 200:
            raise self._error(response, f"reading {slug}")
        data = response.json()
        permissions = data.get("permissions")
        can_push: Optional[bool]
        if access.mode == "app":
            can_push = True  # verified from the installation token's permissions
        elif isinstance(permissions, dict) and "push" in permissions:
            can_push = bool(permissions.get("push"))
        else:
            can_push = None
        return RepositoryInfo(
            slug=data.get("full_name") or slug,
            default_branch=data.get("default_branch") or "main",
            private=bool(data.get("private")),
            archived=bool(data.get("archived")),
            html_url=data.get("html_url") or f"https://github.com/{slug}",
            can_push=can_push,
        )

    def get_branch_sha(self, slug: str, branch: str) -> Optional[str]:
        """Commit sha of ``branch``, or ``None`` when the branch does not exist.

        A repository without any commit has no branches at all; GitHub answers
        409 for it and this raises ``GitHubError`` with code ``repo_empty``.
        """
        access = self.access_for(slug)
        response = self._send(
            "GET",
            f"/repos/{slug}/git/ref/heads/{quote(branch, safe='/')}",
            bearer=access.token,
            action=f"reading branch {branch}",
        )
        if response.status_code == 404:
            return None
        if response.status_code == 409:
            raise self._conflict(response, slug, f"reading branch {branch}")
        if response.status_code != 200:
            raise self._error(response, f"reading branch {branch}")
        data = response.json()
        if isinstance(data, dict):
            return (data.get("object") or {}).get("sha")
        return None

    def _conflict(self, response: httpx.Response, slug: str, action: str) -> GitHubError:
        """GitHub's 409 for the Git database: the repository is empty, or (rarely)
        still being created."""
        error = self._error(response, action)
        if "empty" in error.message.lower():
            return GitHubError(
                f"Repository {slug} is empty - it has no commits yet.", status_code=409, code="repo_empty"
            )
        error.code = "repo_unavailable"
        error.hint = "GitHub may still be setting the repository up; try again in a moment."
        return error

    def is_empty(self, slug: str, default_branch: str) -> bool:
        """True when the repository has no commits (and therefore no branch)."""
        try:
            self.get_branch_sha(slug, default_branch)
        except GitHubError as exc:
            if exc.code == "repo_empty":
                return True
            raise
        return False

    # ------------------------------------------------------------------ #
    # Listing repositories
    # ------------------------------------------------------------------ #
    def _collect_pages(
        self,
        path: str,
        *,
        bearer: str,
        action: str,
        limit: int,
        items_key: Optional[str] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[Any], Optional[int], bool, Optional[GitHubError]]:
        """Every page of a list endpoint, up to ``limit`` items.

        The first page says how many pages there are (GitHub's own count, or
        the ``Link`` header). The others are then requested a few at a time
        instead of one after the other, so that a long list takes the time of
        two or three requests rather than of all of them. Without that
        information the pages are read one by one until a short one arrives.

        Returns ``(items, total, more, error)``: ``total`` is GitHub's own count
        when the endpoint reports one, ``more`` says that items were left out,
        and ``error`` is what kept a page after the first from being read (the
        pages that could be read are returned; a failing first page raises).
        """

        def fetch(page: int) -> Tuple[List[Any], Optional[int], Optional[int]]:
            """One page: its items, GitHub's total count and the number of the last page."""
            response = self._send(
                "GET",
                path,
                bearer=bearer,
                action=action,
                params={**(params or {}), "per_page": PAGE_SIZE, "page": page},
            )
            if response.status_code != 200:
                raise self._error(response, action)
            try:
                data = response.json()
            except ValueError as exc:
                raise GitHubError(
                    f"GitHub returned an unreadable response while {action}.", code="github_error"
                ) from exc
            count: Optional[int] = None
            if items_key is not None:
                raw = data.get(items_key) if isinstance(data, dict) else None
                if isinstance(data, dict) and isinstance(data.get("total_count"), int):
                    count = int(data["total_count"])
            else:
                raw = data
            last = _LAST_PAGE_RE.search(response.headers.get("link") or "")
            return (raw if isinstance(raw, list) else []), count, (int(last.group(1)) if last else None)

        max_pages = max(1, -(-limit // PAGE_SIZE))
        items, total, last_page = fetch(1)
        error: Optional[GitHubError] = None
        pages: Optional[int] = None  # how many pages the list has, when known
        if total is not None:
            pages = max(1, -(-total // PAGE_SIZE))
        elif last_page is not None:
            pages = last_page
        # Read page by page and stopped by the limit after a full page: there may be more.
        open_ended = len(items) == PAGE_SIZE and max_pages == 1 and pages is None

        if len(items) == PAGE_SIZE and max_pages > 1:
            if pages is not None:
                wanted = list(range(2, min(pages, max_pages) + 1))
                batches: Dict[int, List[Any]] = {}
                failed: Dict[int, GitHubError] = {}

                def attempt(page: int) -> None:
                    try:
                        batches[page] = fetch(page)[0]
                        failed.pop(page, None)
                    except GitHubError as exc:
                        failed[page] = exc

                if len(wanted) > 1:
                    with ThreadPoolExecutor(max_workers=min(LIST_CONCURRENCY, len(wanted))) as pool:
                        list(pool.map(attempt, wanted))
                else:
                    for page in wanted:
                        attempt(page)
                # A page refused while others were under way gets a second try on its own.
                for page in sorted(failed):
                    attempt(page)
                for page in wanted:
                    items.extend(batches.get(page, []))
                if failed:
                    error = failed[min(failed)]
            else:
                page = 2
                while page <= max_pages:
                    try:
                        batch, count, _last = fetch(page)
                    except GitHubError as exc:
                        error = exc
                        break
                    if count is not None:
                        total = count
                    items.extend(batch)
                    if len(batch) < PAGE_SIZE:
                        break
                    page += 1
                else:
                    open_ended = True

        if error is not None:
            more = True
        elif total is not None:
            more = total > min(len(items), limit)
        elif pages is not None:
            more = pages > max_pages or len(items) > limit
        else:
            more = len(items) > limit or open_ended
        return items[:limit], total, more, error

    @staticmethod
    def _repository_summary(repo: Dict[str, Any]) -> Dict[str, Any]:
        name = str(repo.get("full_name") or "")
        description = repo.get("description")
        if isinstance(description, str):
            description = " ".join(description.split())
            if len(description) > _DESCRIPTION_CHARS:
                description = description[: _DESCRIPTION_CHARS - 1].rstrip() + "…"
        return {
            "full_name": name,
            "private": bool(repo.get("private")),
            "default_branch": repo.get("default_branch") or "main",
            "html_url": repo.get("html_url") or f"https://github.com/{name}",
            "archived": bool(repo.get("archived")),
            "fork": bool(repo.get("fork")),
            "description": description or None,
            "language": repo.get("language") or None,
            "pushed_at": repo.get("pushed_at") or None,
        }

    def repository_catalog(self, limit: int = MAX_LISTED_REPOSITORIES) -> Dict[str, Any]:
        """Every repository the configured credentials can act on.

        With a GitHub App that is every repository of every installation of the
        App; with a personal token, every repository the token's user can reach.
        All pages are fetched (GitHub hands out at most 100 repositories per
        request). The result is ordered by last push, newest first, and says
        how each App installation is set up, so that the dashboard can explain
        why a repository is missing.

        ``repositories`` holds at most ``limit`` entries; ``total`` is the number
        GitHub reports, and ``truncated`` tells whether some were left out.
        """
        repositories: List[Dict[str, Any]] = []
        seen: set[str] = set()
        installations: List[Dict[str, Any]] = []
        total = 0
        truncated = False
        error: Optional[str] = None

        def add(items: Iterable[Any]) -> None:
            for repo in items:
                if not isinstance(repo, dict):
                    continue
                name = repo.get("full_name")
                if not name or str(name).lower() in seen:
                    continue
                seen.add(str(name).lower())
                repositories.append(self._repository_summary(repo))

        try:
            if self.app_configured:
                found, _count, more, failure = self._collect_pages(
                    "/app/installations",
                    bearer=self._app_jwt(),
                    action="listing App installations",
                    limit=MAX_LISTED_INSTALLATIONS,
                )
                truncated = truncated or more
                error = failure.describe() if failure else error
                for installation in found:
                    if not isinstance(installation, dict) or "id" not in installation:
                        continue
                    account = installation.get("account") or {}
                    entry: Dict[str, Any] = {
                        "id": int(installation["id"]),
                        "account": account.get("login"),
                        "account_type": account.get("type"),
                        # "all": every repository of the account, including future ones.
                        # "selected": only the repositories picked when the App was installed.
                        "repository_selection": installation.get("repository_selection"),
                        "manage_url": installation.get("html_url"),
                        "suspended": bool(installation.get("suspended_at")),
                        "repository_count": None,
                    }
                    installations.append(entry)
                    if entry["suspended"]:
                        continue
                    remaining = limit - len(repositories)
                    if remaining <= 0:
                        truncated = True
                        continue
                    token, _expires, _permissions = self._installation_token(entry["id"])
                    with self._lock:
                        self._extra_secrets.append(token)
                    listed, count, more, failure = self._collect_pages(
                        "/installation/repositories",
                        bearer=token,
                        action="listing installation repositories",
                        items_key="repositories",
                        limit=remaining,
                    )
                    entry["repository_count"] = count if count is not None else len(listed)
                    total += entry["repository_count"]
                    truncated = truncated or more
                    error = failure.describe() if failure else error
                    add(listed)
            elif self._token:
                if self._token.startswith("ghs_"):
                    listed, count, more, failure = self._collect_pages(
                        "/installation/repositories",
                        bearer=self._token,
                        action="listing repositories",
                        items_key="repositories",
                        limit=limit,
                    )
                else:
                    listed, count, more, failure = self._collect_pages(
                        "/user/repos",
                        bearer=self._token,
                        action="listing repositories",
                        limit=limit,
                        params={"sort": "pushed"},
                    )
                truncated = truncated or more
                error = failure.describe() if failure else error
                add(listed)
                total = count if count is not None else len(repositories)
        except GitHubError as exc:
            if not repositories and not installations:
                raise
            # Part of the list is better than none, as long as it says it is partial.
            error = exc.describe()
            truncated = True

        # Most recently pushed first: the repositories being worked on are on top.
        repositories.sort(key=lambda repo: repo["pushed_at"] or "", reverse=True)
        return {
            "repositories": repositories,
            "total": max(total, len(repositories)),
            "truncated": truncated,
            "installations": installations,
            "error": error,
        }

    def list_repositories(self, limit: int = MAX_LISTED_REPOSITORIES) -> List[Dict[str, Any]]:
        """Repositories the configured credentials can act on (see ``repository_catalog``)."""
        return self.repository_catalog(limit)["repositories"]

    # ------------------------------------------------------------------ #
    # Writing: commit, branch, pull request, comment
    # ------------------------------------------------------------------ #
    def create_commit(
        self,
        slug: str,
        *,
        parent_sha: str,
        base_tree_sha: Optional[str],
        changes: List[FileChange],
        message: str,
        expected_tree_sha: Optional[str] = None,
        known_blobs: Iterable[str] = (),
        inline_text: bool = True,
        author: Optional[Dict[str, str]] = None,
        committer: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Create a commit on top of ``parent_sha`` containing ``changes``.

        ``author`` (``name`` and ``email``) attributes the commit to a person.
        Without it GitHub uses the App (or the owner of the token) as author
        and committer - and signs the commit, which it only does for commits
        made in the App's own name.

        ``committer`` names who made the commit when it has another author
        (see ``committer_identity``). This request needs it spelled out: left
        out, GitHub records the author as the committer too.

        The resulting tree is compared with the sandbox index tree, so the
        commit is guaranteed to be exactly what was tested. Files uploaded as
        blobs are additionally compared one by one with the shas git computed
        in the sandbox.

        Text files are sent inside the tree request when the tree can be
        verified afterwards (``expected_tree_sha``): one request for all of
        them. Should GitHub build a different tree from that, everything is
        uploaded blob by blob instead, which also names the file that differs.

        ``base_tree_sha=None`` builds the tree from ``changes`` alone (the whole
        content of a new repository). ``known_blobs`` are blob shas the
        repository already has, which are not uploaded again.
        """
        if not changes:
            raise GitHubError("There are no changes to commit.", code="no_changes")
        token = self.access_for(slug).token
        already_stored = set(known_blobs)
        entries: List[Dict[str, Any]] = []
        inline_budget = INLINE_TREE_BYTES if inline_text and expected_tree_sha else 0
        inlined = 0
        for change in changes:
            if change.status == "D":
                entries.append({"path": change.path, "mode": "100644", "type": "blob", "sha": None})
                continue
            if change.mode == "160000":
                entries.append(
                    {"path": change.path, "mode": change.mode, "type": "commit", "sha": change.sha}
                )
                continue
            if change.sha and change.sha in already_stored:
                entries.append({"path": change.path, "mode": change.mode, "type": "blob", "sha": change.sha})
                continue
            text = _inline_text(change) if inline_budget > 0 else None
            if text is not None and len(text.encode("utf-8")) <= inline_budget:
                inline_budget -= len(text.encode("utf-8"))
                inlined += 1
                entries.append({"path": change.path, "mode": change.mode, "type": "blob", "content": text})
                continue
            if change.content_b64 is None:
                raise GitHubError(
                    f"Missing content for changed file {change.path}.", code="integrity_error"
                )
            blob = self._json(
                "POST",
                f"/repos/{slug}/git/blobs",
                bearer=token,
                action=f"uploading {change.path}",
                json_body={"content": change.content_b64, "encoding": "base64"},
                ok=(201,),
            )
            blob_sha = blob.get("sha")
            if change.sha and blob_sha != change.sha:
                raise GitHubError(
                    f"Integrity check failed for {change.path}: GitHub stored blob {blob_sha} "
                    f"but the sandbox produced {change.sha}.",
                    code="integrity_error",
                )
            entries.append({"path": change.path, "mode": change.mode, "type": "blob", "sha": blob_sha})
            if blob_sha:
                already_stored.add(str(blob_sha))

        def blob_by_blob() -> Dict[str, Any]:
            return self.create_commit(
                slug,
                parent_sha=parent_sha,
                base_tree_sha=base_tree_sha,
                changes=changes,
                message=message,
                expected_tree_sha=expected_tree_sha,
                known_blobs=already_stored,
                inline_text=False,
                author=author,
                committer=committer,
            )

        tree_request: Dict[str, Any] = {"tree": entries}
        if base_tree_sha:
            tree_request["base_tree"] = base_tree_sha
        try:
            tree = self._json(
                "POST",
                f"/repos/{slug}/git/trees",
                bearer=token,
                action="creating the git tree",
                json_body=tree_request,
                ok=(201,),
            )
        except GitHubError as exc:
            # Sending text along is a shortcut, never a requirement: if GitHub does
            # not take the request (other than for lack of access), go the long way.
            if inlined and exc.status_code not in (401, 403, 404):
                return blob_by_blob()
            raise
        tree_sha = tree.get("sha")
        if expected_tree_sha and tree_sha != expected_tree_sha and inlined:
            # Not the tree the sandbox has. Upload every file as a blob instead:
            # that either works or says which file differs.
            return blob_by_blob()
        if expected_tree_sha and tree_sha != expected_tree_sha:
            raise GitHubError(
                f"Integrity check failed: GitHub built tree {tree_sha} but the sandbox work tree is "
                f"{expected_tree_sha}. Nothing was committed.",
                code="integrity_error",
            )
        commit = self._json(
            "POST",
            f"/repos/{slug}/git/commits",
            bearer=token,
            action="creating the commit",
            json_body={
                "message": message,
                "tree": tree_sha,
                "parents": [parent_sha],
                **({"author": commit_author(author)} if author else {}),
                **({"committer": commit_author(committer)} if author and committer else {}),
            },
            ok=(201,),
        )
        commit_sha = commit.get("sha")
        if not commit_sha or not _SHA_RE.match(str(commit_sha)):
            raise GitHubError("GitHub did not return a commit sha.", code="github_error")
        return {"sha": commit_sha, "tree_sha": tree_sha, "html_url": commit.get("html_url")}

    def create_branch(self, slug: str, branch: str, sha: str) -> None:
        token = self.access_for(slug).token
        response = self._send(
            "POST",
            f"/repos/{slug}/git/refs",
            bearer=token,
            action=f"creating branch {branch}",
            json_body={"ref": f"refs/heads/{branch}", "sha": sha},
        )
        if response.status_code == 201:
            return
        error = self._error(response, f"creating branch {branch}")
        if response.status_code == 422 and "already exists" in error.message.lower():
            error.code = "branch_exists"
        raise error

    def update_branch(self, slug: str, branch: str, sha: str) -> None:
        """Move ``branch`` forward to ``sha``. Never forced: GitHub refuses the
        update unless ``sha`` descends from the commit the branch points to."""
        token = self.access_for(slug).token
        self._json(
            "PATCH",
            f"/repos/{slug}/git/refs/heads/{quote(branch, safe='/')}",
            bearer=token,
            action=f"updating branch {branch}",
            json_body={"sha": sha, "force": False},
            ok=(200,),
        )

    def _landed_commit(
        self, slug: str, branch: str, *, parent_sha: str, tree_sha: Optional[str]
    ) -> Optional[Dict[str, Any]]:
        """The commit ``branch`` points to, if it is the one that was being made:
        a child of ``parent_sha`` with exactly the expected tree.

        Asked after a request failed, to tell "it did not happen" from "it
        happened and the answer got lost on the way". ``None`` whenever that
        cannot be said for certain.
        """
        if not tree_sha:
            return None
        try:
            response = self._send(
                "GET",
                f"/repos/{slug}/commits",
                bearer=self.access_for(slug).token,
                action=f"checking what {branch} points to",
                params={"sha": branch, "per_page": 1},
            )
            listed = response.json() if response.status_code == 200 else None
        except (GitHubError, ValueError):
            return None
        tip = listed[0] if isinstance(listed, list) and listed and isinstance(listed[0], dict) else None
        if tip is None:
            return None
        commit = tip.get("commit") if isinstance(tip.get("commit"), dict) else {}
        parents = [str(parent.get("sha")) for parent in tip.get("parents") or [] if isinstance(parent, dict)]
        sha = tip.get("sha")
        if parents != [parent_sha] or (commit.get("tree") or {}).get("sha") != tree_sha:
            return None
        if not sha or not _SHA_RE.match(str(sha)):
            return None
        return {
            "sha": sha,
            "tree_sha": tree_sha,
            "html_url": tip.get("html_url") or f"https://github.com/{slug}/commit/{sha}",
        }

    # ------------------------------------------------------------------ #
    # The first commit of an empty repository
    # ------------------------------------------------------------------ #
    @staticmethod
    def _first_file(changes: List[FileChange]) -> FileChange:
        """The file that becomes the repository's first commit.

        The Contents API writes one regular, non-executable file. A README is
        preferred (it is what GitHub itself starts a repository with), then
        ``.gitignore``, then the smallest file.
        """
        files = [c for c in changes if c.status != "D" and c.mode in ("100644", "100755")]
        candidates = [c for c in files if c.mode == "100644" and c.content_b64] or [
            c for c in files if c.content_b64
        ] or [c for c in files if c.content_b64 is not None]
        if not candidates:
            raise GitHubError(
                "None of the new files can be committed as the repository's first file.",
                code="no_changes",
            )

        def rank(change: FileChange) -> Tuple[int, int, str]:
            name = change.path.lower()
            if name == "readme.md":
                order = 0
            elif "/" not in name and name.startswith("readme"):
                order = 1
            elif name == ".gitignore":
                order = 2
            else:
                order = 3
            return (order, len(change.content_b64 or ""), change.path)

        return min(candidates, key=rank)

    def create_initial_commit(
        self,
        slug: str,
        *,
        branch: str,
        changes: List[FileChange],
        message: str,
        first_message: str = "Initial commit",
        expected_tree_sha: Optional[str] = None,
        author: Optional[Dict[str, str]] = None,
        committer: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Put ``changes`` into a repository that has no commits yet.

        GitHub's Git Data API (blobs, trees, commits, refs) answers 409 for an
        empty repository. The documented way in is the Contents API: writing one
        file with it creates the repository's first commit and its default
        branch. If there are more files, they follow in a second commit made
        the usual way on top of it. Blob shas and the final tree are compared
        with what the sandbox produced, exactly as for a pull request.

        Returns ``sha`` / ``html_url`` of the last commit, the ``branch`` it is
        on, and ``commits`` (1 or 2). If the first commit was made but the rest
        failed, the raised error has code ``initial_commit_incomplete`` and
        says in ``details`` what is already in the repository.
        """
        if not changes:
            raise GitHubError("There are no changes to commit.", code="no_changes")
        if any(change.status == "D" for change in changes):
            raise GitHubError(
                "An empty repository has nothing that could be deleted.", code="integrity_error"
            )
        token = self.access_for(slug).token
        first = self._first_file(changes)
        only_file = len(changes) == 1 and first.mode == "100644"

        response = self._send(
            "PUT",
            f"/repos/{slug}/contents/{quote(first.path, safe='/')}",
            bearer=token,
            action=f"creating the first commit ({first.path})",
            json_body={
                "message": message if only_file else first_message,
                "content": first.content_b64 or "",
                **({"author": commit_author(author)} if author else {}),
                **({"committer": commit_author(committer)} if author and committer else {}),
            },
        )
        refusal = (response.text or "").lower()
        if response.status_code == 422 and "sha" in refusal and "supplied" in refusal:
            # "sha" wasn't supplied: the file is already there, so somebody
            # committed to the repository while the task was running.
            raise GitHubError(
                f"{slug} is no longer empty: {first.path} was added to it while the task was running. "
                "Nothing was written.",
                status_code=422,
                code="repo_not_empty",
            )
        if response.status_code not in (200, 201):
            raise self._error(response, f"creating the first commit ({first.path})")
        try:
            data = response.json()
        except ValueError as exc:
            raise GitHubError(
                "GitHub returned an unreadable response while creating the first commit.",
                code="github_error",
            ) from exc
        commit = data.get("commit") if isinstance(data, dict) else None
        commit = commit if isinstance(commit, dict) else {}
        first_sha = commit.get("sha")
        if not first_sha or not _SHA_RE.match(str(first_sha)):
            raise GitHubError("GitHub did not return the sha of the first commit.", code="github_error")
        first_url = commit.get("html_url")
        written = {"first_commit_sha": first_sha, "first_commit_url": first_url, "first_path": first.path}

        def incomplete(reason: str, cause: Optional[BaseException] = None) -> GitHubError:
            error = GitHubError(reason, code="initial_commit_incomplete", details=written)
            if cause is not None:
                error.__cause__ = cause
            return error

        blob_sha = ((data.get("content") or {}) if isinstance(data, dict) else {}).get("sha")
        if first.sha and blob_sha != first.sha:
            raise incomplete(
                f"Integrity check failed for {first.path}: GitHub stored blob {blob_sha} but the "
                f"sandbox produced {first.sha}."
            )

        def points_to_first_commit(name: str) -> bool:
            # The branch was created a moment ago; allow GitHub a moment to show it.
            for attempt in range(3):
                try:
                    if self.get_branch_sha(slug, name) == first_sha:
                        return True
                except GitHubError as exc:
                    if exc.code != "repo_empty":
                        raise
                if attempt < 2:
                    self._pause(0.6)
            return False

        try:
            # Which branch did GitHub create? Normally the repository's default branch.
            if not points_to_first_commit(branch):
                branch = self.get_repository(slug).default_branch
                if not points_to_first_commit(branch):
                    raise incomplete(
                        "GitHub created the first commit, but no branch points to it "
                        f"(expected '{branch}')."
                    )
            written["branch"] = branch

            first_tree = (commit.get("tree") or {}).get("sha")
            if expected_tree_sha and first_tree == expected_tree_sha:
                return {
                    "sha": first_sha,
                    "tree_sha": first_tree,
                    "html_url": first_url,
                    "branch": branch,
                    "commits": 1,
                }
            if only_file and not expected_tree_sha:
                return {
                    "sha": first_sha,
                    "tree_sha": first_tree,
                    "html_url": first_url,
                    "branch": branch,
                    "commits": 1,
                }

            # Everything, as one tree, on top of the first commit. Half a first
            # commit is the worst outcome, so a hiccup on GitHub's side is given a
            # second attempt: blobs and trees are addressed by their content
            # (sending them again changes nothing) and the branch only ever moves
            # forward.
            for attempt in (1, 2):
                try:
                    result = self.create_commit(
                        slug,
                        parent_sha=str(first_sha),
                        base_tree_sha=None,
                        changes=changes,
                        message=message,
                        expected_tree_sha=expected_tree_sha,
                        known_blobs=[first.sha] if first.sha else (),
                        author=author,
                        committer=committer,
                    )
                    self.update_branch(slug, branch, result["sha"])
                    break
                except GitHubError as exc:
                    # The request may have been carried out with only its answer
                    # lost: then the branch already holds the project.
                    landed = self._landed_commit(
                        slug, branch, parent_sha=str(first_sha), tree_sha=expected_tree_sha
                    )
                    if landed is not None:
                        result = landed
                        break
                    passing = exc.code == "github_unreachable" or (exc.status_code or 0) >= 500
                    if attempt == 2 or not passing:
                        raise
                    self._pause(1.0)
        except GitHubError as exc:
            if exc.code == "initial_commit_incomplete":
                raise
            raise incomplete(exc.describe(), exc) from exc
        return {**result, "branch": branch, "commits": 2}

    def find_task_commit(self, slug: str, task_id: str, start: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """A commit a task made, found by its ``Task-Id`` trailer.

        Without ``start`` the default branch is searched: that is where the
        first commit of a repository that was empty went. ``start`` (a commit
        sha or a branch) searches from there instead, which finds the commit a
        task added to a pull request.
        """
        token = self.access_for(slug).token
        params: Dict[str, Any] = {"per_page": 30}
        if start:
            params["sha"] = start
        response = self._send(
            "GET",
            f"/repos/{slug}/commits",
            bearer=token,
            action="looking up the task's commit",
            params=params,
        )
        if response.status_code in (404, 409):
            return None  # no such repository for these credentials, or still empty
        if response.status_code != 200:
            raise self._error(response, "looking up the task's commit")
        try:
            listed = response.json()
        except ValueError:
            return None
        marker = f"Task-Id: {task_id}"
        for item in listed if isinstance(listed, list) else []:
            if not isinstance(item, dict):
                continue
            text = str((item.get("commit") or {}).get("message") or "")
            sha = item.get("sha")
            if marker in text.splitlines() and sha and _SHA_RE.match(str(sha)):
                return {
                    "sha": sha,
                    "html_url": item.get("html_url") or f"https://github.com/{slug}/commit/{sha}",
                    "title": text.splitlines()[0] if text else "",
                }
        return None

    @staticmethod
    def _pull_request_from(data: Dict[str, Any]) -> PullRequest:
        number = data.get("number")
        html_url = data.get("html_url")
        if not isinstance(number, int) or not isinstance(html_url, str) or not html_url.startswith("http"):
            raise GitHubError(
                "GitHub's response did not include a pull request number and URL.", code="github_error"
            )
        head = data.get("head") or {}
        return PullRequest(
            number=number,
            html_url=html_url,
            state=str(data.get("state") or "open"),
            title=str(data.get("title") or ""),
            head_branch=str(head.get("ref") or ""),
            base_branch=str((data.get("base") or {}).get("ref") or ""),
            merged=bool(data.get("merged_at")),
            head_sha=str(head.get("sha") or ""),
            head_repository=str((head.get("repo") or {}).get("full_name") or ""),
            author=str((data.get("user") or {}).get("login") or ""),
        )

    def create_pull_request(
        self, slug: str, *, title: str, body: str, head: str, base: str
    ) -> PullRequest:
        token = self.access_for(slug).token
        data = self._json(
            "POST",
            f"/repos/{slug}/pulls",
            bearer=token,
            action="opening the pull request",
            json_body={"title": title, "body": body, "head": head, "base": base},
            ok=(201,),
        )
        return self._pull_request_from(data)

    def find_pull_request(self, slug: str, head_branch: str) -> Optional[PullRequest]:
        """The pull request (any state) whose head is ``head_branch``, if one exists."""
        token = self.access_for(slug).token
        owner = slug.split("/", 1)[0]
        data = self._json(
            "GET",
            f"/repos/{slug}/pulls",
            bearer=token,
            action="looking up the pull request",
            params={"head": f"{owner}:{head_branch}", "state": "all", "per_page": 5},
        )
        for item in data if isinstance(data, list) else []:
            if isinstance(item, dict) and (item.get("head") or {}).get("ref") == head_branch:
                return self._pull_request_from(item)
        return None

    def create_issue_comment(self, slug: str, issue_number: int, body: str) -> Optional[str]:
        token = self.access_for(slug).token
        data = self._json(
            "POST",
            f"/repos/{slug}/issues/{int(issue_number)}/comments",
            bearer=token,
            action="commenting on the issue",
            json_body={"body": body},
            ok=(201,),
        )
        return data.get("html_url")

    def list_issue_comments(self, slug: str, issue_number: int, limit: int = 300) -> List[Dict[str, Any]]:
        """Comments on an issue or pull request, oldest first: ``body`` and ``author``."""
        listed, _total, _more, _error = self._collect_pages(
            f"/repos/{slug}/issues/{int(issue_number)}/comments",
            bearer=self.access_for(slug).token,
            action="reading the comments",
            limit=limit,
        )
        return [
            {"body": str(item.get("body") or ""), "author": str((item.get("user") or {}).get("login") or "")}
            for item in listed
            if isinstance(item, dict)
        ]

    # ------------------------------------------------------------------ #
    # Pull requests that already exist
    # ------------------------------------------------------------------ #
    def get_pull_request(self, slug: str, number: int) -> PullRequest:
        token = self.access_for(slug).token
        response = self._send(
            "GET", f"/repos/{slug}/pulls/{int(number)}", bearer=token, action=f"reading pull request #{number}"
        )
        if response.status_code == 404:
            raise GitHubError(f"{slug} has no pull request #{number}.", status_code=404, code="pull_request_not_found")
        if response.status_code != 200:
            raise self._error(response, f"reading pull request #{number}")
        return self._pull_request_from(response.json())

    def update_pull_request(self, slug: str, number: int, *, title: Optional[str] = None, body: Optional[str] = None) -> None:
        changes = {key: value for key, value in (("title", title), ("body", body)) if value is not None}
        if not changes:
            return
        self._json(
            "PATCH",
            f"/repos/{slug}/pulls/{int(number)}",
            bearer=self.access_for(slug).token,
            action=f"updating pull request #{number}",
            json_body=changes,
            ok=(200,),
        )

    # ------------------------------------------------------------------ #
    # Checks: what CI says about a commit
    # ------------------------------------------------------------------ #
    def checks_for(self, slug: str, sha: str) -> Dict[str, Any]:
        """What GitHub knows about the checks of commit ``sha``.

        Two sources are read: check runs (GitHub Actions and other check apps)
        and commit statuses (how Vercel, among others, reports a deployment).
        ``state`` is ``failure`` if anything failed, else ``pending`` if
        anything is still running, else ``success``, or ``none`` when the
        commit has no checks. ``failing`` lists what failed. A source the
        credentials may not read is named in ``notes`` instead of being
        counted as "no checks".
        """
        token = self.access_for(slug).token
        failing: List[Dict[str, Any]] = []
        notes: List[str] = []
        passed = pending = 0
        readable = False

        response = self._send(
            "GET",
            f"/repos/{slug}/commits/{sha}/check-runs",
            bearer=token,
            action="reading the checks",
            params={"per_page": 100, "filter": "latest"},
        )
        if response.status_code == 200:
            readable = True
            try:
                runs = response.json().get("check_runs") or []
            except (ValueError, AttributeError):
                runs = []
            for run in runs:
                if not isinstance(run, dict):
                    continue
                conclusion = str(run.get("conclusion") or "")
                if run.get("status") != "completed":
                    pending += 1
                elif conclusion in FAILED_CHECK_CONCLUSIONS:
                    output = run.get("output") if isinstance(run.get("output"), dict) else {}
                    failing.append(
                        {
                            "kind": "check_run",
                            "id": run.get("id"),
                            "name": str(run.get("name") or "check"),
                            "conclusion": conclusion,
                            "app": str((run.get("app") or {}).get("slug") or ""),
                            "url": run.get("html_url") or run.get("details_url"),
                            "summary": " - ".join(
                                str(output.get(key)).strip() for key in ("title", "summary") if output.get(key)
                            )[:600],
                        }
                    )
                else:
                    passed += 1
        elif response.status_code in (403, 404):
            notes.append("check runs could not be read (GitHub App permission 'Checks: Read-only' is missing)")
        elif response.status_code != 422:  # 422: no commit with this sha
            raise self._error(response, "reading the checks")

        response = self._send(
            "GET", f"/repos/{slug}/commits/{sha}/status", bearer=token, action="reading the commit statuses"
        )
        if response.status_code == 200:
            readable = True
            try:
                statuses = response.json().get("statuses") or []
            except (ValueError, AttributeError):
                statuses = []
            for status in statuses:
                if not isinstance(status, dict):
                    continue
                state = str(status.get("state") or "")
                if state == "pending":
                    pending += 1
                elif state in FAILED_STATUS_STATES:
                    failing.append(
                        {
                            "kind": "status",
                            "id": status.get("id"),
                            "name": str(status.get("context") or "status"),
                            "conclusion": state,
                            "app": "",
                            "url": status.get("target_url"),
                            "summary": str(status.get("description") or "")[:600],
                        }
                    )
                else:
                    passed += 1
        elif response.status_code in (403, 404):
            notes.append(
                "commit statuses could not be read (GitHub App permission 'Commit statuses: Read-only' is missing)"
            )
        elif response.status_code != 422:
            raise self._error(response, "reading the commit statuses")

        if failing:
            state = "failure"
        elif pending:
            state = "pending"
        elif passed:
            state = "success"
        else:
            state = "none" if readable else "unknown"
        return {
            "sha": sha,
            "state": state,
            "failing": failing[:MAX_REPORTED_CHECKS],
            "failed": len(failing),
            "passed": passed,
            "pending": pending,
            "notes": notes,
        }

    def check_failure_details(self, slug: str, check: Dict[str, Any]) -> Dict[str, Any]:
        """What can be learnt about one failed check: its annotations and, for
        a GitHub Actions job, the part of the log that shows the failure.
        Best effort - whatever cannot be read is simply left out."""
        details: Dict[str, Any] = {"annotations": [], "log": ""}
        if check.get("kind") != "check_run" or not check.get("id"):
            return details
        token = self.access_for(slug).token
        try:
            response = self._send(
                "GET",
                f"/repos/{slug}/check-runs/{check['id']}/annotations",
                bearer=token,
                action="reading the check's annotations",
                params={"per_page": 30},
            )
            if response.status_code == 200:
                for item in response.json() or []:
                    if not isinstance(item, dict) or item.get("annotation_level") == "notice":
                        continue
                    # GitHub Actions files a failure of the job itself under ".github".
                    where = str(item.get("path") or "")
                    where = "" if where == ".github" else where
                    if where and item.get("start_line"):
                        where += f":{item['start_line']}"
                    text = " ".join(str(item.get("message") or "").split())
                    if text:
                        details["annotations"].append(f"{where}: {text}" if where else text)
            if check.get("app") == "github-actions":
                # For GitHub Actions the check run is the job; its log is behind a redirect.
                response = self._send(
                    "GET",
                    f"/repos/{slug}/actions/jobs/{check['id']}/logs",
                    bearer=token,
                    action="reading the job log",
                )
                if response.status_code == 200:
                    details["log"] = log_excerpt(response.text or "")
        except (GitHubError, ValueError):
            pass
        details["annotations"] = details["annotations"][:12]
        return details

    # ------------------------------------------------------------------ #
    # People
    # ------------------------------------------------------------------ #
    def _reading_token(self) -> str:
        """Any credential that can read public data: the token, else the token
        of one of the App's installations."""
        if self._token:
            return self._token
        if self.app_configured:
            found, _count, _more, _error = self._collect_pages(
                "/app/installations", bearer=self._app_jwt(), action="listing App installations", limit=PAGE_SIZE
            )
            for installation in found:
                if isinstance(installation, dict) and "id" in installation and not installation.get("suspended_at"):
                    token, _expires, _permissions = self._installation_token(int(installation["id"]))
                    with self._lock:
                        self._extra_secrets.append(token)
                    return token
        raise GitHubError("No GitHub credentials are configured.", code="no_credentials")

    def committer_identity(self, slug: str) -> Optional[Dict[str, str]]:
        """Name and address of the credential that makes the commits: the App's
        bot account, or the owner of the token.

        It is what GitHub itself puts on a commit made without further ado.
        A commit authored as somebody else has to name it, or that person is
        recorded as the committer as well. None when it cannot be found out.
        """
        with self._lock:
            if self._committer is not None:
                return dict(self._committer) or None
        found: Dict[str, str] = {}
        try:
            token = self.access_for(slug).token
            if self.auth_mode == "app":
                login = f"{self.app_info().get('slug')}[bot]"
                response = self._send(
                    "GET", f"/users/{quote(login, safe='')}", bearer=token, action="looking up the App's bot account"
                )
            else:
                login = ""
                response = self._send("GET", "/user", bearer=token, action="looking up the owner of the token")
            if response.status_code == 200:
                data = response.json()
                account = str(data.get("login") or login)
                address = noreply_address(account, data.get("id"))
                if account and address:
                    # A bot is known by its login; a person by the name on the account.
                    name = account if account.endswith("[bot]") else str(data.get("name") or account)
                    found = {"name": name, "email": address}
        except (GitHubError, ValueError, TypeError):
            found = {}
        with self._lock:
            self._committer = found
        return dict(found) or None

    def user_identity(self, login: str) -> Dict[str, Any]:
        """Name and commit e-mail address of a GitHub account.

        The address is the account's ``noreply`` one: GitHub attributes commits
        that carry it to the account without publishing a private address.
        """
        cleaned = (login or "").strip().lstrip("@")
        if not re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})(?:\[bot\])?", cleaned):
            raise GitHubError(f"'{login}' is not a GitHub user name.", code="invalid_user")
        response = self._send(
            "GET", f"/users/{cleaned}", bearer=self._reading_token(), action=f"looking up the GitHub user {cleaned}"
        )
        if response.status_code == 404:
            raise GitHubError(f"There is no GitHub user '{cleaned}'.", status_code=404, code="user_not_found")
        if response.status_code != 200:
            raise self._error(response, f"looking up the GitHub user {cleaned}")
        data = response.json()
        account, user_id = str(data.get("login") or cleaned), data.get("id")
        return {
            "login": account,
            "name": str(data.get("name") or account),
            "email": f"{user_id}+{account}@users.noreply.github.com" if user_id else None,
            "avatar_url": data.get("avatar_url"),
            "html_url": data.get("html_url") or f"https://github.com/{account}",
        }


def commit_author(author: Dict[str, str]) -> Dict[str, str]:
    """The ``author`` object of a commit request."""
    return {"name": str(author.get("name") or ""), "email": str(author.get("email") or "")}


def noreply_address(login: str, user_id: Any) -> Optional[str]:
    """The commit address GitHub ties to an account: ``id+login@users.noreply.github.com``."""
    try:
        return f"{int(user_id)}+{login}@users.noreply.github.com" if login else None
    except (TypeError, ValueError):
        return None


def compare_url(slug: str, base: str, head: str) -> str:
    """GitHub page for opening a pull request by hand from an existing branch."""
    return f"https://github.com/{slug}/compare/{quote(base, safe='/')}...{quote(head, safe='/')}?expand=1"
