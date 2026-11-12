"""GitHub-facing endpoints used by the dashboard.

``/setup`` and ``/callback`` exist so that the URLs GitHub redirects a browser
to after a GitHub App is installed or authorised lead back to the dashboard
instead of a 404.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Optional
from urllib.parse import urlencode

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import RedirectResponse

from ..config import get_settings
from ..dependencies import get_runner_deps
from ..security import require_access
from ..services.github import GitHubError, parse_repository
from ..services.runner import RunnerDeps

router = APIRouter(prefix="/api/v1/github", tags=["github"])
_SHA_RE = re.compile(r"[0-9a-f]{40}")

_SETUP_ACTIONS = ("install", "update", "request")


def is_github_app_redirect(params: Mapping[str, str], path: str = "") -> bool:
    """True when a request looks like GitHub sending a browser back to us after
    an App installation (``installation_id`` / ``setup_action``) or an OAuth
    authorisation (``code`` on a callback-style path)."""
    if params.get("installation_id") or params.get("setup_action"):
        return True
    lowered = path.lower()
    return bool(params.get("code")) and any(word in lowered for word in ("callback", "oauth", "auth"))


def dashboard_landing_url(params: Mapping[str, str]) -> str:
    """Dashboard URL carrying only the harmless, validated redirect parameters
    (never the OAuth ``code``)."""
    query: Dict[str, str] = {}
    installation_id = params.get("installation_id") or ""
    setup_action = params.get("setup_action") or ""
    if setup_action == "request":
        query["github_app"] = "requested"
    elif installation_id or setup_action:
        query["github_app"] = "installed"
    else:
        query["github_app"] = "authorized"
    if installation_id.isdigit():
        query["installation_id"] = installation_id
    if setup_action in _SETUP_ACTIONS:
        query["setup_action"] = setup_action
    return "/?" + urlencode(query)


@router.get("/setup", include_in_schema=True)
async def github_app_setup(request: Request):
    """Landing URL for a GitHub App's *Setup URL*: returns the browser to the dashboard."""
    return RedirectResponse(dashboard_landing_url(request.query_params), status_code=307)


@router.get("/callback", include_in_schema=True)
async def github_app_callback(request: Request):
    """Landing URL for a GitHub App's *Callback URL*: returns the browser to the dashboard.

    The dashboard authenticates as the App itself, so the OAuth ``code`` is not needed
    and is deliberately dropped.
    """
    return RedirectResponse(dashboard_landing_url(request.query_params), status_code=307)


@router.get("/repos", dependencies=[Depends(require_access)])
def list_repositories(deps: RunnerDeps = Depends(get_runner_deps)) -> Dict[str, Any]:
    """Every repository the configured GitHub credentials can work on.

    All of them, most recently pushed first - not only the first page GitHub
    returns. ``installations`` says, per account the GitHub App is installed
    on, whether it covers all repositories or only selected ones, and where
    that can be changed: a repository that is missing here is one the App has
    not been given.
    """
    settings = get_settings()
    mode = settings.github_auth_mode
    result: Dict[str, Any] = {
        "auth_mode": mode,
        "repositories": [],
        "total": 0,
        "truncated": False,
        "installations": [],
        "install_url": None,
        "error": None,
    }
    if mode == "none":
        result["error"] = "No GitHub credentials are configured."
        return result
    github = deps.github_factory(settings)
    try:
        if github.app_configured:
            result["install_url"] = github.app_install_url()
        catalog = github.repository_catalog()
        result.update(
            repositories=catalog["repositories"],
            total=catalog["total"],
            truncated=catalog["truncated"],
            installations=catalog["installations"],
            error=catalog["error"],
        )
    except GitHubError as exc:
        result["error"] = exc.describe()
    finally:
        github.close()
    return result


@router.get("/check", dependencies=[Depends(require_access)])
def check_repository(
    repository: str = Query(..., description="owner/name or a GitHub URL"),
    branch: Optional[str] = Query(None, description="Base branch to verify (optional)"),
    deps: RunnerDeps = Depends(get_runner_deps),
) -> Dict[str, Any]:
    """Verify, without running anything, that a task could act on a repository."""
    settings = get_settings()
    result: Dict[str, Any] = {
        "ok": False,
        "repository": repository,
        "default_branch": None,
        "private": None,
        # True: the repository has no commits. A task then builds the project
        # from scratch and its result becomes the first commit.
        "empty": None,
        "auth_mode": settings.github_auth_mode,
        "code": None,
        "error": None,
    }
    try:
        slug = parse_repository(repository)
    except ValueError as exc:
        result.update(code="invalid_repository", error=str(exc))
        return result
    result["repository"] = slug
    if settings.github_auth_mode == "none":
        result.update(code="no_credentials", error="No GitHub credentials are configured.")
        return result

    github = deps.github_factory(settings)
    try:
        info = github.get_repository(slug)
        result.update(
            repository=info.slug,
            default_branch=info.default_branch,
            private=info.private,
            auth_mode=github.access_for(slug).mode,
        )
        if info.archived:
            result.update(code="archived", error=f"{info.slug} is archived.")
            return result
        if info.can_push is False:
            result.update(code="no_push_access", error=f"The configured token cannot push to {info.slug}.")
            return result
        wanted = (branch or "").strip() or info.default_branch
        try:
            branch_exists = github.get_branch_sha(info.slug, wanted) is not None
            result["empty"] = False
        except GitHubError as exc:
            if exc.code != "repo_empty":
                raise
            branch_exists = False
            result["empty"] = True
        if result["empty"]:
            if wanted != info.default_branch:
                result.update(
                    code="branch_not_found",
                    error=(
                        f"{info.slug} is empty, so it has no branch '{wanted}'. Leave the base branch "
                        f"empty: the first commit is made on '{info.default_branch}'."
                    ),
                )
            else:
                result["ok"] = True
        elif not branch_exists:
            result.update(
                code="branch_not_found",
                error=f"Branch '{wanted}' does not exist. The default branch is '{info.default_branch}'.",
            )
        else:
            result["ok"] = True
    except GitHubError as exc:
        result.update(code=exc.code, error=exc.describe())
    finally:
        github.close()
    return result


@router.get("/checks", dependencies=[Depends(require_access)])
def commit_checks(
    repository: str = Query(..., description="owner/name or a GitHub URL"),
    ref: str = Query(..., description="A commit sha or a branch name"),
    deps: RunnerDeps = Depends(get_runner_deps),
) -> Dict[str, Any]:
    """What CI says about a commit: check runs (GitHub Actions and other check
    apps) and commit statuses (deployments by Vercel and the like).

    ``state`` is ``failure``, ``pending``, ``success``, ``none`` (the commit has
    no checks) or ``unknown`` (the credentials may not read them; ``notes``
    says which permission is missing).
    """
    settings = get_settings()
    result: Dict[str, Any] = {"ok": False, "repository": repository, "sha": None, "state": "unknown", "error": None}
    try:
        slug = parse_repository(repository)
    except ValueError as exc:
        result["error"] = str(exc)
        return result
    result["repository"] = slug
    if settings.github_auth_mode == "none":
        result["error"] = "No GitHub credentials are configured."
        return result
    github = deps.github_factory(settings)
    try:
        wanted = ref.strip()
        sha = wanted if _SHA_RE.fullmatch(wanted) else github.get_branch_sha(slug, wanted)
        if not sha:
            result["error"] = f"{slug} has no branch or commit '{wanted}'."
            return result
        result.update(ok=True, **github.checks_for(slug, sha))
    except GitHubError as exc:
        result["error"] = exc.describe()
    finally:
        github.close()
    return result


@router.get("/identity", dependencies=[Depends(require_access)])
def github_identity(
    login: str = Query(..., min_length=1, max_length=60, description="A GitHub user name"),
    deps: RunnerDeps = Depends(get_runner_deps),
) -> Dict[str, Any]:
    """Name and commit e-mail address of a GitHub account, for commits that are
    to be attributed to a person. The address is the account's ``noreply``
    address, which GitHub ties to the account without exposing a private one."""
    settings = get_settings()
    result: Dict[str, Any] = {"ok": False, "login": login, "name": None, "email": None, "error": None}
    if settings.github_auth_mode == "none":
        result["error"] = "No GitHub credentials are configured."
        return result
    github = deps.github_factory(settings)
    try:
        result.update(ok=True, **github.user_identity(login))
    except GitHubError as exc:
        result["error"] = exc.describe()
    finally:
        github.close()
    return result


# What the GitHub App needs in order to be told about failed checks, and to read them.
AUTOMATION_EVENTS = {
    "check_suite": "Check suite",
    "status": "Status",
    "workflow_run": "Workflow run",
    "issue_comment": "Issue comment",
}
AUTOMATION_PERMISSIONS = {
    "checks": "Checks",
    "statuses": "Commit statuses",
    "actions": "Actions",
}


@router.get("/automation", dependencies=[Depends(require_access)])
def automation_status(deps: RunnerDeps = Depends(get_runner_deps)) -> Dict[str, Any]:
    """Whether failed checks can be fixed automatically, and what is missing.

    Fixing on request (from the dashboard or with ``/jules`` in a comment)
    only needs to read the checks. Fixing unasked needs GitHub to tell this
    server about failed checks: a webhook secret, and the GitHub App
    subscribed to the events.
    """
    settings = get_settings()
    result: Dict[str, Any] = {
        "auth_mode": settings.github_auth_mode,
        "ci_autofix": settings.ci_autofix,
        "attempts": settings.ci_autofix_attempts,
        "webhook_secret": bool(settings.github_webhook_secret),
        "webhook_path": "/api/v1/webhooks/github",
        "app": None,
        "events": [],
        "permissions": [],
        "ready": False,
        "error": None,
    }
    if not settings.github_app_configured:
        # A personal token reads checks without further ado, but GitHub has
        # nobody to send events to on its behalf except a repository webhook.
        result["events"] = [
            {"id": key, "label": label, "subscribed": None} for key, label in AUTOMATION_EVENTS.items()
        ]
        result["ready"] = bool(settings.ci_autofix and settings.github_webhook_secret and settings.github_token)
        return result
    github = deps.github_factory(settings)
    try:
        info = github.app_info()
        subscribed = set(info.get("events") or [])
        granted = info.get("permissions") or {}
        slug, owner = info.get("slug"), info.get("owner") or {}
        # An App owned by an organization is configured under that organization.
        settings_url = None
        if slug:
            if owner.get("type") == "Organization" and owner.get("login"):
                settings_url = f"https://github.com/organizations/{owner['login']}/settings/apps/{slug}"
            else:
                settings_url = f"https://github.com/settings/apps/{slug}"
        result["app"] = {
            "name": info.get("name"),
            "slug": slug,
            "settings_url": settings_url,
            "permissions_url": f"{settings_url}/permissions" if settings_url else None,
        }
        result["events"] = [
            {"id": key, "label": label, "subscribed": key in subscribed} for key, label in AUTOMATION_EVENTS.items()
        ]
        result["permissions"] = [
            {"id": key, "label": label, "granted": granted.get(key) in ("read", "write")}
            for key, label in AUTOMATION_PERMISSIONS.items()
        ]
        result["ready"] = bool(
            settings.ci_autofix
            and settings.github_webhook_secret
            and subscribed & {"check_suite", "status", "workflow_run"}
        )
    except GitHubError as exc:
        result["error"] = exc.describe()
    finally:
        github.close()
    return result
