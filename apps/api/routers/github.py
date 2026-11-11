"""GitHub-facing endpoints used by the dashboard.

``/setup`` and ``/callback`` exist so that the URLs GitHub redirects a browser
to after a GitHub App is installed or authorised lead back to the dashboard
instead of a 404.
"""
from __future__ import annotations

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
