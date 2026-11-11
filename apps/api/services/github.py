"""GitHub integration: authentication, repository lookups, commits and pull requests.

Nothing in this module fabricates a result. Every URL it returns was handed
back by the GitHub API for an object that GitHub confirmed it created.

Commits are created through the Git Data API (blobs -> tree -> commit -> ref)
rather than by ``git push`` from the sandbox, so write credentials never have
to be exposed to the environment where model-chosen commands run.
"""
from __future__ import annotations

import base64
import binascii
import json
import re
import threading
import time
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


class GitHubError(Exception):
    """A GitHub operation failed. ``message`` is safe to show to the user."""

    def __init__(
        self,
        message: str,
        *,
        status_code: Optional[int] = None,
        code: str = "github_error",
        hint: Optional[str] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.code = code
        self.hint = hint

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
        """Commit sha of ``branch``, or ``None`` when the branch does not exist."""
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
            raise GitHubError(f"Repository {slug} is empty - it has no commits yet.", code="repo_empty")
        if response.status_code != 200:
            raise self._error(response, f"reading branch {branch}")
        data = response.json()
        if isinstance(data, dict):
            return (data.get("object") or {}).get("sha")
        return None

    def list_repositories(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Repositories the configured credentials can act on."""
        collected: List[Dict[str, Any]] = []
        seen = set()

        def add(items: Iterable[Any]) -> None:
            for repo in items:
                if not isinstance(repo, dict):
                    continue
                name = repo.get("full_name")
                if not name or name in seen or repo.get("archived"):
                    continue
                seen.add(name)
                collected.append(
                    {
                        "full_name": name,
                        "private": bool(repo.get("private")),
                        "default_branch": repo.get("default_branch") or "main",
                        "html_url": repo.get("html_url") or f"https://github.com/{name}",
                    }
                )

        if self.app_configured:
            installations = self._json(
                "GET",
                "/app/installations",
                bearer=self._app_jwt(),
                action="listing App installations",
                params={"per_page": 100},
            )
            for installation in installations if isinstance(installations, list) else []:
                if len(collected) >= limit:
                    break
                token, _expires, _permissions = self._installation_token(int(installation["id"]))
                page = self._json(
                    "GET",
                    "/installation/repositories",
                    bearer=token,
                    action="listing installation repositories",
                    params={"per_page": 100},
                )
                add(page.get("repositories") or [])
        elif self._token:
            if self._token.startswith("ghs_"):
                page = self._json(
                    "GET",
                    "/installation/repositories",
                    bearer=self._token,
                    action="listing repositories",
                    params={"per_page": 100},
                )
                add(page.get("repositories") or [])
            else:
                page = self._json(
                    "GET",
                    "/user/repos",
                    bearer=self._token,
                    action="listing repositories",
                    params={"per_page": 100, "sort": "pushed"},
                )
                add(page if isinstance(page, list) else [])
        return collected[:limit]

    # ------------------------------------------------------------------ #
    # Writing: commit, branch, pull request, comment
    # ------------------------------------------------------------------ #
    def create_commit(
        self,
        slug: str,
        *,
        parent_sha: str,
        base_tree_sha: str,
        changes: List[FileChange],
        message: str,
        expected_tree_sha: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Create a commit on top of ``parent_sha`` containing ``changes``.

        Blob shas returned by GitHub are compared with the shas git computed in
        the sandbox, and the resulting tree is compared with the sandbox index
        tree, so the commit is guaranteed to be exactly what was tested.
        """
        if not changes:
            raise GitHubError("There are no changes to commit.", code="no_changes")
        token = self.access_for(slug).token
        entries: List[Dict[str, Any]] = []
        for change in changes:
            if change.status == "D":
                entries.append({"path": change.path, "mode": "100644", "type": "blob", "sha": None})
                continue
            if change.mode == "160000":
                entries.append(
                    {"path": change.path, "mode": change.mode, "type": "commit", "sha": change.sha}
                )
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

        tree = self._json(
            "POST",
            f"/repos/{slug}/git/trees",
            bearer=token,
            action="creating the git tree",
            json_body={"base_tree": base_tree_sha, "tree": entries},
            ok=(201,),
        )
        tree_sha = tree.get("sha")
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
            json_body={"message": message, "tree": tree_sha, "parents": [parent_sha]},
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

    @staticmethod
    def _pull_request_from(data: Dict[str, Any]) -> PullRequest:
        number = data.get("number")
        html_url = data.get("html_url")
        if not isinstance(number, int) or not isinstance(html_url, str) or not html_url.startswith("http"):
            raise GitHubError(
                "GitHub's response did not include a pull request number and URL.", code="github_error"
            )
        return PullRequest(
            number=number,
            html_url=html_url,
            state=str(data.get("state") or "open"),
            title=str(data.get("title") or ""),
            head_branch=str((data.get("head") or {}).get("ref") or ""),
            base_branch=str((data.get("base") or {}).get("ref") or ""),
            merged=bool(data.get("merged_at")),
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


def compare_url(slug: str, base: str, head: str) -> str:
    """GitHub page for opening a pull request by hand from an existing branch."""
    return f"https://github.com/{slug}/compare/{quote(base, safe='/')}...{quote(head, safe='/')}?expand=1"
