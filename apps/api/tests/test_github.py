"""GitHub integration: input validation, App authentication and API behaviour."""
from __future__ import annotations

import base64
import json

import pytest

from apps.api.services.github import (
    FileChange,
    GitHubError,
    build_app_jwt,
    compare_url,
    normalize_private_key,
    parse_repository,
    validate_branch_name,
)

from .support import FakeGitHub, git


# --------------------------------------------------------------------------- #
# Repository and branch parsing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("kennedy-blip/sql-hub", "kennedy-blip/sql-hub"),
        ("  kennedy-blip/sql-hub  ", "kennedy-blip/sql-hub"),
        ("https://github.com/kennedy-blip/Julesown", "kennedy-blip/Julesown"),
        ("https://github.com/kennedy-blip/Julesown/", "kennedy-blip/Julesown"),
        ("https://github.com/kennedy-blip/Julesown.git", "kennedy-blip/Julesown"),
        ("http://github.com/kennedy-blip/Julesown", "kennedy-blip/Julesown"),
        ("github.com/kennedy-blip/Julesown", "kennedy-blip/Julesown"),
        ("https://www.github.com/kennedy-blip/Julesown", "kennedy-blip/Julesown"),
        ("git@github.com:kennedy-blip/Julesown.git", "kennedy-blip/Julesown"),
        # Extra path segments used to end up inside the slug and produce dead links.
        ("https://github.com/kennedy-blip/Julesown/tree/some-branch", "kennedy-blip/Julesown"),
        ("https://github.com/kennedy-blip/Julesown/pull/3", "kennedy-blip/Julesown"),
        ("https://github.com/kennedy-blip/Julesown?tab=readme#top", "kennedy-blip/Julesown"),
        # ".git" must only be stripped from the end: this used to become "octo/octohub.io".
        ("octo/octo.github.io", "octo/octo.github.io"),
        ("https://github.com/octo/octo.github.io.git", "octo/octo.github.io"),
        ("octo/.github", "octo/.github"),
    ],
)
def test_parse_repository_accepts(raw, expected):
    assert parse_repository(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        None,
        "just-a-name",
        "https://gitlab.com/owner/repo",
        "https://github.com/owner",
        "owner/repo name",
        "-owner/repo",
        "owner/..",
        "owner/re$po",
    ],
)
def test_parse_repository_rejects(raw):
    with pytest.raises(ValueError):
        parse_repository(raw)


@pytest.mark.parametrize("name", ["main", "feature/x", "release-1.2", "refs/heads/dev", "jules/task-abc123"])
def test_branch_names_accepted(name):
    assert validate_branch_name(name) == name.replace("refs/heads/", "")


@pytest.mark.parametrize(
    "name", ["", "-rf", "a..b", "a b", "a~1", "a:b", "x.lock", "/abs", "a//b", "a/", ".hidden", "a;rm -rf /", "a\tb"]
)
def test_branch_names_rejected(name):
    with pytest.raises(ValueError):
        validate_branch_name(name)


def test_compare_url_keeps_branch_slashes():
    assert (
        compare_url("acme/demo", "main", "jules/task-abc")
        == "https://github.com/acme/demo/compare/main...jules/task-abc?expand=1"
    )


# --------------------------------------------------------------------------- #
# GitHub App credentials
# --------------------------------------------------------------------------- #
def _decode_segment(segment: str) -> dict:
    return json.loads(base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4)))


def test_app_jwt_is_a_valid_rs256_token(rsa_private_key_pem):
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import padding

    token = build_app_jwt("123456", rsa_private_key_pem, now=1_800_000_000)
    header, payload, signature = token.split(".")
    assert _decode_segment(header) == {"alg": "RS256", "typ": "JWT"}
    claims = _decode_segment(payload)
    assert claims["iss"] == "123456"
    assert claims["iat"] == 1_800_000_000 - 60
    assert 0 < claims["exp"] - claims["iat"] <= 600  # GitHub rejects lifetimes over 10 minutes

    public_key = serialization.load_pem_private_key(rsa_private_key_pem.encode(), password=None).public_key()
    public_key.verify(  # raises InvalidSignature if the token is not correctly signed
        base64.urlsafe_b64decode(signature + "=" * (-len(signature) % 4)),
        f"{header}.{payload}".encode(),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )


def test_private_key_survives_dashboard_mangling(rsa_private_key_pem):
    pem = rsa_private_key_pem
    variants = {
        "escaped newlines": pem.replace("\n", "\\n"),
        "quoted": f'"{pem}"',
        "base64": base64.b64encode(pem.encode()).decode(),
        "spaces instead of newlines": pem.strip().replace("\n", " "),
        "extra whitespace": f"\n\n  {pem}  \n",
    }
    for label, mangled in variants.items():
        assert normalize_private_key(mangled).strip() == pem.strip(), label
        assert build_app_jwt("1", mangled).count(".") == 2, label


def test_unusable_private_key_gives_actionable_error():
    with pytest.raises(GitHubError) as caught:
        build_app_jwt("1", "definitely not a key")
    assert caught.value.code == "app_key_invalid"
    assert "GITHUB_APP_PRIVATE_KEY" in caught.value.message


def test_app_access_uses_a_repo_scoped_installation_token(github: FakeGitHub, rsa_private_key_pem):
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    access = client.access_for("acme/demo")
    assert access.mode == "app"
    assert access.token.startswith("ghs_")
    assert access.permissions["contents"] == "write"
    # Least privilege: the token is requested for this one repository only.
    assert github.token_requests[-1]["repositories"] == ["demo"]
    # Cached: a second lookup does not mint another token.
    client.access_for("acme/demo")
    assert len(github.token_requests) == 1
    assert client.get_repository("acme/demo").default_branch == "main"


def test_app_not_installed_explains_how_to_install(github: FakeGitHub, rsa_private_key_pem):
    github.add_repo("acme/other", {"README.md": "x\n"}, installed=False)
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    with pytest.raises(GitHubError) as caught:
        client.access_for("acme/other")
    assert caught.value.code == "app_not_installed"
    assert "https://github.com/apps/jules-test-app/installations/new" in caught.value.describe()


def test_app_not_installed_falls_back_to_token(github: FakeGitHub, rsa_private_key_pem):
    github.add_repo("acme/other", {"README.md": "x\n"}, installed=False)
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem, token=FakeGitHub.PAT)
    assert client.access_for("acme/other").mode == "token"


def test_app_without_write_permission_is_reported(github: FakeGitHub, rsa_private_key_pem):
    github.app_permissions = {"contents": "read", "pull_requests": "write", "metadata": "read"}
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    with pytest.raises(GitHubError) as caught:
        client.access_for("acme/demo")
    assert caught.value.code == "app_missing_permissions"
    assert "contents" in caught.value.message
    assert "Read and write" in caught.value.describe()


def test_clone_token_is_read_only_for_apps(github: FakeGitHub, rsa_private_key_pem):
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    write_token = client.access_for("acme/demo").token
    clone_token = client.clone_token("acme/demo")
    assert clone_token != write_token
    assert github.issued_tokens[clone_token]["permissions"] == {"contents": "read"}
    assert github.issued_tokens[clone_token]["repositories"] == ["demo"]
    assert clone_token in client.secrets() and write_token in client.secrets()


def test_no_credentials_is_an_error_not_a_guess(github: FakeGitHub):
    with pytest.raises(GitHubError) as caught:
        github.client().access_for("acme/demo")
    assert caught.value.code == "no_credentials"


# --------------------------------------------------------------------------- #
# Repository lookups and writes
# --------------------------------------------------------------------------- #
def test_missing_repository_and_branch(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    with pytest.raises(GitHubError) as caught:
        client.get_repository("acme/does-not-exist")
    assert caught.value.code == "repo_not_found"
    assert client.get_branch_sha("acme/demo", "main") == github.remote.rev("acme/demo", "refs/heads/main")
    assert client.get_branch_sha("acme/demo", "nope") is None


def test_commit_branch_and_pull_request_round_trip(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    bare = github.remote.path_for("acme/demo")
    parent = github.remote.rev("acme/demo", "refs/heads/main")
    base_tree = git(bare, "rev-parse", f"{parent}^{{tree}}")
    content = b"def add(a, b):\n    return a + b\n"
    blob_sha = git(bare, "hash-object", "--stdin", stdin=content)

    commit = client.create_commit(
        "acme/demo",
        parent_sha=parent,
        base_tree_sha=base_tree,
        message="Fix add()",
        changes=[
            FileChange(path="calc.py", status="M", mode="100644", sha=blob_sha, content_b64=base64.b64encode(content).decode()),
            FileChange(path="README.md", status="D", mode="100644"),
        ],
    )
    client.create_branch("acme/demo", "jules/task-1", commit["sha"])
    assert github.remote.show("acme/demo", "jules/task-1", "calc.py") == content.decode().strip()
    assert "README.md" not in github.remote.files("acme/demo", "jules/task-1")
    assert "README.md" in github.remote.files("acme/demo", "main")  # base branch untouched

    pull = client.create_pull_request("acme/demo", title="Fix add()", body="body", head="jules/task-1", base="main")
    assert (pull.number, pull.html_url) == (1, "https://github.com/acme/demo/pull/1")
    found = client.find_pull_request("acme/demo", "jules/task-1")
    assert found is not None and found.number == 1
    assert client.find_pull_request("acme/demo", "jules/task-unknown") is None

    with pytest.raises(GitHubError) as caught:
        client.create_branch("acme/demo", "jules/task-1", commit["sha"])
    assert caught.value.code == "branch_exists"


def test_commit_integrity_checks(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    bare = github.remote.path_for("acme/demo")
    parent = github.remote.rev("acme/demo", "refs/heads/main")
    base_tree = git(bare, "rev-parse", f"{parent}^{{tree}}")
    content = base64.b64encode(b"new\n").decode()

    with pytest.raises(GitHubError) as caught:  # blob sha does not match the content
        client.create_commit(
            "acme/demo",
            parent_sha=parent,
            base_tree_sha=base_tree,
            message="x",
            changes=[FileChange(path="a.txt", status="A", mode="100644", sha="0" * 40, content_b64=content)],
        )
    assert caught.value.code == "integrity_error"

    with pytest.raises(GitHubError) as caught:  # resulting tree differs from the sandbox tree
        client.create_commit(
            "acme/demo",
            parent_sha=parent,
            base_tree_sha=base_tree,
            message="x",
            changes=[FileChange(path="a.txt", status="A", mode="100644", content_b64=content)],
            expected_tree_sha="f" * 40,
        )
    assert caught.value.code == "integrity_error"
    assert "Nothing was committed" in caught.value.message


def test_pull_request_errors_carry_githubs_reason(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    with pytest.raises(GitHubError) as caught:
        client.create_pull_request("acme/demo", title="t", body="b", head="main", base="main")
    assert caught.value.status_code == 422
    assert "No commits between main and main" in caught.value.message


def test_write_without_permission_is_forbidden(github: FakeGitHub):
    github.add_repo("acme/readonly", {"README.md": "x\n"}, push=False)
    client = github.client(token=FakeGitHub.PAT)
    assert client.get_repository("acme/readonly").can_push is False
    with pytest.raises(GitHubError) as caught:
        client.create_branch("acme/readonly", "x", "0" * 40)
    assert caught.value.code == "forbidden"


def test_list_repositories_for_app_and_token(github: FakeGitHub, rsa_private_key_pem):
    github.add_repo("acme/second", {"README.md": "x\n"}, private=True)
    github.add_repo("acme/not-installed", {"README.md": "x\n"}, installed=False)
    as_app = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    names = [repo["full_name"] for repo in as_app.list_repositories()]
    assert sorted(names) == ["acme/demo", "acme/second"]
    assert as_app.app_install_url() == "https://github.com/apps/jules-test-app/installations/new"

    as_token = github.client(token=FakeGitHub.PAT)
    assert len(as_token.list_repositories()) == 3


def test_bad_credentials_are_reported(github: FakeGitHub):
    client = github.client(token="ghp_" + "w" * 36)
    with pytest.raises(GitHubError) as caught:
        client.get_repository("acme/demo")
    assert caught.value.code == "bad_credentials"
