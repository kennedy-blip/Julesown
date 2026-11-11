"""GitHub integration: input validation, App authentication and API behaviour."""
from __future__ import annotations

import base64
import json

import pytest

from typing import Dict, List, Tuple

from apps.api.services.github import (
    FileChange,
    GitHubError,
    build_app_jwt,
    compare_url,
    normalize_private_key,
    parse_repository,
    validate_branch_name,
)
from apps.api.services.sandbox import RepoCheckout

from .support import FakeGitHub, LocalWorkspace, git


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


# --------------------------------------------------------------------------- #
# Listing repositories: all of them, not the first page
# --------------------------------------------------------------------------- #
def add_many(github: FakeGitHub, count: int, owner: str = "busy") -> List[str]:
    names = [f"{owner}/repo-{index:03d}" for index in range(count)]
    for name in names:
        github.add_listed_repo(name)
    return names


def test_listing_fetches_every_page(github: FakeGitHub, rsa_private_key_pem):
    """An account with 250 repositories used to show its first 100: GitHub hands
    out at most 100 per request, and only one request was made."""
    names = add_many(github, 250)
    client = github.client(app_id="123456", app_private_key=rsa_private_key_pem)
    catalog = client.repository_catalog()
    listed = [repo["full_name"] for repo in catalog["repositories"]]
    assert len(listed) == 251 and set(names) <= set(listed) and "acme/demo" in listed
    assert catalog["total"] == 251 and catalog["truncated"] is False and catalog["error"] is None
    # busy: pages of 100, 100 and 50; acme: one page.
    assert len(github.calls("GET", r"^/installation/repositories$")) == 4
    assert client.list_repositories() == catalog["repositories"]


def test_listing_with_a_token_fetches_every_page(github: FakeGitHub):
    add_many(github, 230)
    catalog = github.client(token=FakeGitHub.PAT).repository_catalog()
    assert len(catalog["repositories"]) == 231 and catalog["total"] == 231 and catalog["truncated"] is False
    assert catalog["installations"] == []
    assert len(github.calls("GET", r"^/user/repos$")) == 3


def test_listing_is_ordered_by_last_push_and_describes_each_repository(github: FakeGitHub):
    github.add_listed_repo("acme/old", pushed_at="2020-05-01T00:00:00Z", archived=True)
    github.add_listed_repo(
        "acme/newest",
        pushed_at="2030-01-01T00:00:00Z",
        private=True,
        language="TypeScript",
        description="  A dashboard\nfor things " + "x" * 300,
        fork=True,
    )
    github.add_listed_repo("acme/never-pushed", pushed_at=None)
    listed = github.client(token=FakeGitHub.PAT).list_repositories()
    assert [repo["full_name"] for repo in listed] == ["acme/newest", "acme/demo", "acme/old", "acme/never-pushed"]
    newest, old = listed[0], listed[2]
    assert newest["private"] is True and newest["language"] == "TypeScript" and newest["fork"] is True
    assert newest["description"].startswith("A dashboard for things xxx") and len(newest["description"]) == 140
    assert newest["html_url"] == "https://github.com/acme/newest" and newest["default_branch"] == "main"
    # Archived repositories are listed (and marked), not silently left out.
    assert old["archived"] is True and newest["archived"] is False


def test_listing_says_how_the_app_is_installed(github: FakeGitHub, rsa_private_key_pem):
    """A repository that is missing from the list is one the App was not given:
    the listing says per account whether the App covers all or selected
    repositories, and where that is changed."""
    add_many(github, 3, owner="busy")
    github.add_repo("acme/not-installed", {"README.md": "x\n"}, installed=False)
    busy = github.installation_for("busy")
    github.installations[busy].update(repository_selection="all", account_type="Organization")
    catalog = github.client(app_id="123456", app_private_key=rsa_private_key_pem).repository_catalog()
    by_account = {item["account"]: item for item in catalog["installations"]}
    assert by_account["acme"] == {
        "id": github.installation_for("acme"),
        "account": "acme",
        "account_type": "User",
        "repository_selection": "selected",
        "manage_url": f"https://github.com/settings/installations/{github.installation_for('acme')}",
        "suspended": False,
        "repository_count": 1,
    }
    assert by_account["busy"]["repository_selection"] == "all" and by_account["busy"]["repository_count"] == 3
    assert by_account["busy"]["manage_url"] == f"https://github.com/organizations/busy/settings/installations/{busy}"
    assert "acme/not-installed" not in [repo["full_name"] for repo in catalog["repositories"]]
    assert catalog["total"] == 4


def test_suspended_installation_is_reported_but_not_queried(github: FakeGitHub, rsa_private_key_pem):
    add_many(github, 2, owner="paused")
    github.installations[github.installation_for("paused")]["suspended"] = True
    catalog = github.client(app_id="123456", app_private_key=rsa_private_key_pem).repository_catalog()
    assert [repo["full_name"] for repo in catalog["repositories"]] == ["acme/demo"]
    paused = next(item for item in catalog["installations"] if item["account"] == "paused")
    assert paused["suspended"] is True and paused["repository_count"] is None


def test_listing_is_capped_and_says_so(github: FakeGitHub, rsa_private_key_pem):
    add_many(github, 250)
    as_app = github.client(app_id="123456", app_private_key=rsa_private_key_pem).repository_catalog(limit=120)
    assert len(as_app["repositories"]) == 120 and as_app["truncated"] is True and as_app["total"] == 251
    as_token = github.client(token=FakeGitHub.PAT).repository_catalog(limit=200)
    assert len(as_token["repositories"]) == 200 and as_token["truncated"] is True


def test_listing_keeps_what_it_has_when_a_page_fails(github: FakeGitHub):
    add_many(github, 150)
    client = github.client(token=FakeGitHub.PAT)
    calls = {"count": 0}
    original = github._handle

    def flaky(request):
        if request.url.path == "/user/repos":
            calls["count"] += 1
            if calls["count"] == 2:
                return github._json(502, {"message": "Server Error"})
        return original(request)

    github.transport.handler = flaky  # type: ignore[attr-defined]
    catalog = client.repository_catalog()
    assert len(catalog["repositories"]) == 100 and catalog["truncated"] is True
    assert "HTTP 502" in catalog["error"]

    # Nothing at all is an error, not an empty list that looks like "no repositories".
    calls["count"] = 1
    with pytest.raises(GitHubError):
        client.repository_catalog()


# --------------------------------------------------------------------------- #
# Empty repositories
# --------------------------------------------------------------------------- #
def new_project_changes(tmp_path, files: Dict[str, str], executable: Tuple[str, ...] = ()) -> Tuple[List[FileChange], str]:
    """The change set of a project written into an empty repository's work tree."""
    workspace = LocalWorkspace(tmp_path / "work")
    checkout = RepoCheckout.init_empty(workspace, branch="main")
    for name, content in files.items():
        workspace.write_file(f"{checkout.directory}/{name}", content)
    for name in executable:
        workspace.run(f"chmod +x {name}", cwd=checkout.directory)
    changeset = checkout.collect_changes()
    return changeset.changes, changeset.tree_sha


def test_empty_repository_is_recognised(github: FakeGitHub):
    github.add_repo("acme/blank", None, empty=True)
    client = github.client(token=FakeGitHub.PAT)
    assert client.is_empty("acme/blank", "main") is True
    assert client.is_empty("acme/demo", "main") is False
    assert client.get_repository("acme/blank").default_branch == "main"
    with pytest.raises(GitHubError) as caught:
        client.get_branch_sha("acme/blank", "main")
    assert caught.value.code == "repo_empty" and "no commits yet" in caught.value.message

    # GitHub's other reason for a 409 is not mistaken for "empty".
    github.fail_next("GET", r"/git/ref/heads/main$", 409, {"message": "Git Repository is unavailable."})
    with pytest.raises(GitHubError) as caught:
        client.is_empty("acme/demo", "main")
    assert caught.value.code == "repo_unavailable" and "try again" in caught.value.describe()


def test_git_data_api_refuses_an_empty_repository(github: FakeGitHub):
    """Why the first commit cannot be made the usual way."""
    github.add_repo("acme/blank", None, empty=True)
    client = github.client(token=FakeGitHub.PAT)
    with pytest.raises(GitHubError) as caught:
        client.create_commit(
            "acme/blank",
            parent_sha="0" * 40,
            base_tree_sha=None,
            message="x",
            changes=[FileChange(path="a.txt", status="A", mode="100644", content_b64=base64.b64encode(b"a\n").decode())],
        )
    assert caught.value.status_code == 409 and "empty" in caught.value.message.lower()


def test_first_commit_of_a_single_file(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Blank\n"})
    client = github.client(token=FakeGitHub.PAT)
    result = client.create_initial_commit(
        "acme/blank", branch="main", changes=changes, message="Describe the project\n\nTask-Id: abc", expected_tree_sha=tree
    )
    head = github.remote.rev("acme/blank", "refs/heads/main")
    assert result["sha"] == head and result["commits"] == 1 and result["branch"] == "main"
    assert result["html_url"] == f"https://github.com/acme/blank/commit/{head}"
    assert github.remote.log("acme/blank", "main") == ["Describe the project"]
    assert github.remote.show("acme/blank", "main", "README.md") == "# Blank"
    assert github.calls("POST", r"/git/") == []  # the Contents API alone did it


def test_first_commit_of_a_whole_project(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True)
    files = {
        "README.md": "# Tasks\n",
        ".gitignore": "node_modules/\n",
        "package.json": '{"name": "tasks", "version": "1.0.0"}\n',
        "src/index.js": "console.log('hi');\n",
        "scripts/start.sh": "#!/bin/sh\nnode src/index.js\n",
        "public/.gitkeep": "",
    }
    changes, tree = new_project_changes(tmp_path, files, executable=("scripts/start.sh",))
    client = github.client(token=FakeGitHub.PAT)
    result = client.create_initial_commit(
        "acme/blank", branch="main", changes=changes, message="Create the tasks app\n\nTask-Id: abc", expected_tree_sha=tree
    )
    head = github.remote.rev("acme/blank", "refs/heads/main")
    assert result["sha"] == head and result["commits"] == 2 and result["tree_sha"] == tree
    # The README goes in first (the way GitHub itself starts a repository), the project follows.
    assert github.remote.log("acme/blank", "main") == ["Create the tasks app", "Initial commit"]
    assert github.remote.files("acme/blank", "main~1") == ["README.md"]
    assert sorted(github.remote.files("acme/blank", "main")) == sorted(files)
    for name, content in files.items():
        assert github.remote.show("acme/blank", "main", name) == content.strip()
    # Exactly the tree the sandbox had, file modes included.
    assert git(github.remote.path_for("acme/blank"), "rev-parse", "main^{tree}") == tree
    assert github.remote.mode("acme/blank", "main", "scripts/start.sh") == "100755"
    # The README was uploaded once, with the Contents API; nothing was forced.
    assert len(github.calls("PUT", r"/contents/README\.md$")) == 1
    assert len(github.calls("PATCH", r"/git/refs/heads/main$")) == 1
    # The text files went along in the one tree request; only the empty file needed a blob.
    assert len(github.calls("POST", r"/git/trees$")) == 1 and len(github.calls("POST", r"/git/blobs$")) == 1
    assert github.pauses == []


def test_first_file_is_one_the_contents_api_can_write(github: FakeGitHub, tmp_path):
    """No README: the smallest regular file goes first, never an executable or an empty one."""
    github.add_repo("acme/blank", None, empty=True)
    files = {"run.sh": "#!/bin/sh\necho hi\n", "empty.txt": "", "lib/long.py": "x = 1\n" * 40, "lib/short.py": "y = 2\n"}
    changes, tree = new_project_changes(tmp_path, files, executable=("run.sh",))
    client = github.client(token=FakeGitHub.PAT)
    result = client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Add scripts", expected_tree_sha=tree)
    assert github.remote.files("acme/blank", "main~1") == ["lib/short.py"]
    assert result["commits"] == 2 and github.remote.mode("acme/blank", "main", "run.sh") == "100755"
    assert git(github.remote.path_for("acme/blank"), "rev-parse", "main^{tree}") == tree


def test_first_commit_does_not_overwrite_a_repository_that_got_content(github: FakeGitHub, tmp_path):
    """The repository was empty when the task started and somebody pushed since."""
    github.add_repo("acme/raced", {"README.md": "# Mine\n"})
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Generated\n", "app.js": "1\n"})
    client = github.client(token=FakeGitHub.PAT)
    before = github.remote.rev("acme/raced", "refs/heads/main")
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/raced", branch="main", changes=changes, message="x", expected_tree_sha=tree)
    assert caught.value.code == "repo_not_empty" and "Nothing was written" in caught.value.message
    assert github.remote.rev("acme/raced", "refs/heads/main") == before
    assert github.remote.show("acme/raced", "main", "README.md") == "# Mine"


def test_first_commit_says_what_is_there_when_the_rest_fails(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Tasks\n", "app.js": "1\n"})
    client = github.client(token=FakeGitHub.PAT)
    github.fail_next("POST", r"/git/trees$", 502, {"message": "Server Error"}, times=99)
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    error = caught.value
    assert error.code == "initial_commit_incomplete" and "HTTP 502" in error.message
    # Tried twice, each time with the text inside the tree request and then blob by blob.
    assert github.pauses == [1.0] and len(github.calls("POST", r"/git/trees$")) == 4
    head = github.remote.rev("acme/blank", "refs/heads/main")
    assert error.details == {
        "first_commit_sha": head,
        "first_commit_url": f"https://github.com/acme/blank/commit/{head}",
        "first_path": "README.md",
        "branch": "main",
    }
    assert github.remote.files("acme/blank", "main") == ["README.md"]


def test_first_commit_survives_a_hiccup(github: FakeGitHub, tmp_path):
    """Half a first commit is the worst outcome. Git objects are addressed by
    their content, so the second half can simply be tried again."""
    github.add_repo("acme/blank", None, empty=True)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Tasks\n", "app.js": "1\n", "b.js": "2\n"})
    client = github.client(token=FakeGitHub.PAT)
    github.fail_next("PATCH", r"/git/refs/heads/main$", 503, {"message": "Service Unavailable"})
    result = client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    assert result["commits"] == 2 and github.remote.rev("acme/blank", "refs/heads/main") == result["sha"]
    assert github.remote.log("acme/blank", "main") == ["Create", "Initial commit"]
    assert github.pauses == [1.0]

    # A refusal is not a hiccup: it is not tried again, in any way.
    github.add_repo("acme/blank2", None, empty=True)
    github.fail_next("POST", r"/git/trees$", 403, {"message": "Resource not accessible by integration"}, times=99)
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/blank2", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    assert caught.value.code == "initial_commit_incomplete" and github.pauses == [1.0]
    assert len(github.calls("POST", r"/acme/blank2/git/trees$")) == 1


def test_first_commit_whose_answer_got_lost_is_recognised(github: FakeGitHub, tmp_path):
    """The branch update went through, but its answer did not arrive. Trying
    again would not be a fast-forward - and the project is already there."""
    github.add_repo("acme/blank", None, empty=True)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Tasks\n", "app.js": "1\n", "b.js": "2\n"})
    client = github.client(token=FakeGitHub.PAT)
    github.lose_answer("PATCH", r"/git/refs/heads/main$")
    result = client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    head = github.remote.rev("acme/blank", "refs/heads/main")
    assert result["sha"] == head and result["commits"] == 2 and result["tree_sha"] == tree
    assert result["html_url"] == f"https://github.com/acme/blank/commit/{head}"
    assert github.remote.log("acme/blank", "main") == ["Create", "Initial commit"]
    assert github.pauses == [] and len(github.calls("PATCH", r"/git/refs/heads/main$")) == 1  # no second attempt needed


def test_a_branch_that_holds_something_else_is_not_taken_for_the_project(github: FakeGitHub, tmp_path):
    """Only the exact tree on top of the first commit counts as "it landed"."""
    github.add_repo("acme/blank", None, empty=True)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# Tasks\n", "app.js": "1\n"})
    client = github.client(token=FakeGitHub.PAT)
    github.fail_next("PATCH", r"/git/refs/heads/main$", 422, {"message": "Update is not a fast forward"})
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    assert caught.value.code == "initial_commit_incomplete"
    assert github.remote.log("acme/blank", "main") == ["Initial commit"]


def test_first_commit_is_checked_against_the_sandbox(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True)
    changes, _tree = new_project_changes(tmp_path, {"README.md": "# Tasks\n", "app.js": "1\n"})
    client = github.client(token=FakeGitHub.PAT)
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/blank", branch="main", changes=changes, message="Create", expected_tree_sha="f" * 40)
    assert caught.value.code == "initial_commit_incomplete" and "Integrity check failed" in caught.value.message
    # The branch was not moved to a commit that differs from what was tested.
    assert github.remote.files("acme/blank", "main") == ["README.md"]

    github.add_repo("acme/blank2", None, empty=True)
    changes[0].sha = "0" * 40 if changes[0].path == "README.md" else changes[0].sha
    changes[1].sha = "0" * 40 if changes[1].path == "README.md" else changes[1].sha
    with pytest.raises(GitHubError) as caught:
        client.create_initial_commit("acme/blank2", branch="main", changes=changes, message="Create")
    assert caught.value.code == "initial_commit_incomplete" and "README.md" in caught.value.message


def test_first_commit_lands_on_the_branch_github_created(github: FakeGitHub, tmp_path):
    """The default branch is whatever the repository says, not an assumption."""
    github.add_repo("acme/trunk", None, empty=True, default_branch="trunk")
    changes, tree = new_project_changes(tmp_path, {"README.md": "# T\n", "a.txt": "a\n"})
    client = github.client(token=FakeGitHub.PAT)
    result = client.create_initial_commit("acme/trunk", branch="main", changes=changes, message="Create", expected_tree_sha=tree)
    assert result["branch"] == "trunk" and github.remote.rev("acme/trunk", "refs/heads/main") is None
    assert github.pauses == [0.6, 0.6]  # "main" was given a moment to appear first
    assert sorted(github.remote.files("acme/trunk", "trunk")) == ["README.md", "a.txt"]


def test_first_commit_needs_write_access(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True, push=False)
    changes, tree = new_project_changes(tmp_path, {"README.md": "# T\n"})
    with pytest.raises(GitHubError) as caught:
        github.client(token=FakeGitHub.PAT).create_initial_commit(
            "acme/blank", branch="main", changes=changes, message="x", expected_tree_sha=tree
        )
    assert caught.value.code == "forbidden"
    assert github.remote.is_empty("acme/blank")


def test_branch_updates_are_never_forced(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    bare = github.remote.path_for("acme/demo")
    head = github.remote.rev("acme/demo", "refs/heads/main")
    tree = git(bare, "rev-parse", f"{head}^{{tree}}")
    unrelated = git(bare, "commit-tree", tree, "-m", "unrelated history")
    with pytest.raises(GitHubError) as caught:
        client.update_branch("acme/demo", "main", unrelated)
    assert "not a fast forward" in caught.value.message
    assert github.remote.rev("acme/demo", "refs/heads/main") == head
    child = git(bare, "commit-tree", tree, "-p", head, "-m", "child")
    client.update_branch("acme/demo", "main", child)
    assert github.remote.rev("acme/demo", "refs/heads/main") == child


def test_a_tasks_commit_is_found_by_its_trailer(github: FakeGitHub, tmp_path):
    github.add_repo("acme/blank", None, empty=True)
    client = github.client(token=FakeGitHub.PAT)
    assert client.find_task_commit("acme/blank", "a" * 32) is None  # still empty
    changes, tree = new_project_changes(tmp_path, {"README.md": "# T\n", "a.txt": "a\n"})
    result = client.create_initial_commit(
        "acme/blank", branch="main", changes=changes, message=f"Create the app\n\nSummary.\n\nTask-Id: {'a' * 32}", expected_tree_sha=tree
    )
    found = client.find_task_commit("acme/blank", "a" * 32)
    assert found == {"sha": result["sha"], "html_url": result["html_url"], "title": "Create the app"}
    assert client.find_task_commit("acme/blank", "b" * 32) is None
    assert client.find_task_commit("acme/demo", "a" * 32) is None


def test_refused_workflow_files_are_explained(github: FakeGitHub):
    client = github.client(token=FakeGitHub.PAT)
    github.fail_next(
        "POST",
        r"/git/refs$",
        403,
        {"message": "refusing to allow a GitHub App to create or update workflow `.github/workflows/ci.yml` without `workflows` permission"},
    )
    with pytest.raises(GitHubError) as caught:
        client.create_branch("acme/demo", "x", github.remote.rev("acme/demo", "refs/heads/main"))
    assert caught.value.code == "workflow_permission"
    assert "'Workflows' permission" in caught.value.describe()


# --------------------------------------------------------------------------- #
# One request for all text files
# --------------------------------------------------------------------------- #
def demo_commit_inputs(github: FakeGitHub, tmp_path, files: Dict[str, bytes]):
    """A clone of acme/demo with ``files`` written into it, and its change set."""
    workspace = LocalWorkspace(tmp_path / "work")
    checkout = RepoCheckout.clone(workspace, clone_url=github.remote.clone_url("acme/demo"), branch="main")
    for name, content in files.items():
        target = tmp_path / "work" / "repo" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return checkout, checkout.collect_changes()


def tree_requests(github: FakeGitHub) -> List[Dict[str, object]]:
    return [payload for method, path, payload in github.payloads if method == "POST" and path.endswith("/git/trees")]


def test_text_files_travel_inside_the_tree_request(github: FakeGitHub, tmp_path):
    """Fifty changed files used to be fifty uploads. Text now goes along with the
    tree; only what cannot be sent as text is uploaded on its own."""
    files = {f"src/module_{index}.py": f"VALUE = {index}\n".encode() for index in range(12)}
    files.update(
        {
            "notes/ünïcödé.md": "Grüße, 世界 👋\r\nwith CRLF and a BOM \ufeff inside\n".encode("utf-8"),
            "logo.png": b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\xff\xfe",  # binary
            "latin1.txt": "caf\xe9\n".encode("latin-1"),  # text, but not UTF-8
            "empty.txt": b"",
            "big.txt": b"x" * 250_000,
        }
    )
    checkout, changeset = demo_commit_inputs(github, tmp_path, files)
    client = github.client(token=FakeGitHub.PAT)
    commit = client.create_commit(
        "acme/demo",
        parent_sha=checkout.base_sha,
        base_tree_sha=checkout.base_tree_sha,
        changes=changeset.changes,
        message="Add modules",
        expected_tree_sha=changeset.tree_sha,
    )
    assert commit["tree_sha"] == changeset.tree_sha
    requests = tree_requests(github)
    assert len(requests) == 1
    entries = {entry["path"]: entry for entry in requests[0]["tree"]}
    inline = sorted(path for path, entry in entries.items() if "content" in entry)
    assert inline == sorted([f"src/module_{index}.py" for index in range(12)] + ["notes/ünïcödé.md"])
    assert entries["notes/ünïcödé.md"]["content"] == "Grüße, 世界 👋\r\nwith CRLF and a BOM \ufeff inside\n"
    # Binary, non-UTF-8, empty and large files were uploaded one by one.
    assert len(github.calls("POST", r"/git/blobs$")) == 4
    client.create_branch("acme/demo", "jules/task-x", commit["sha"])
    bare = github.remote.path_for("acme/demo")
    assert git(bare, "rev-parse", "jules/task-x^{tree}") == changeset.tree_sha


def test_without_a_tree_to_compare_with_files_are_uploaded_one_by_one(github: FakeGitHub, tmp_path):
    """Sending text along is only done when the result can be verified."""
    checkout, changeset = demo_commit_inputs(github, tmp_path, {"a.py": b"A = 1\n", "b.py": b"B = 2\n"})
    client = github.client(token=FakeGitHub.PAT)
    client.create_commit(
        "acme/demo",
        parent_sha=checkout.base_sha,
        base_tree_sha=checkout.base_tree_sha,
        changes=changeset.changes,
        message="Add",
    )
    assert len(github.calls("POST", r"/git/blobs$")) == 2
    assert all("content" not in entry for entry in tree_requests(github)[0]["tree"])


def test_commit_falls_back_to_blobs_if_github_alters_text(github: FakeGitHub, tmp_path):
    """If text sent along with the tree were not stored byte for byte, the tree
    would not be the sandbox's. That is noticed, and the files are uploaded as
    blobs instead - the commit is still exactly what was tested."""
    github.mangle_inline_content = True
    checkout, changeset = demo_commit_inputs(github, tmp_path, {"win.txt": b"one\r\ntwo\r\n", "unix.txt": b"one\ntwo\n"})
    client = github.client(token=FakeGitHub.PAT)
    commit = client.create_commit(
        "acme/demo",
        parent_sha=checkout.base_sha,
        base_tree_sha=checkout.base_tree_sha,
        changes=changeset.changes,
        message="Add",
        expected_tree_sha=changeset.tree_sha,
    )
    assert commit["tree_sha"] == changeset.tree_sha
    requests = tree_requests(github)
    assert len(requests) == 2
    assert any("content" in entry for entry in requests[0]["tree"])
    assert all("content" not in entry for entry in requests[1]["tree"])
    assert len(github.calls("POST", r"/git/blobs$")) == 2
    client.create_branch("acme/demo", "jules/task-y", commit["sha"])
    assert git(github.remote.path_for("acme/demo"), "cat-file", "blob", "jules/task-y:win.txt") == "one\r\ntwo"


def test_commit_goes_the_long_way_if_github_does_not_take_the_tree_with_text(github: FakeGitHub, tmp_path):
    checkout, changeset = demo_commit_inputs(github, tmp_path, {"a.py": b"A = 1\n", "b.py": b"B = 2\n"})
    client = github.client(token=FakeGitHub.PAT)
    github.fail_next("POST", r"/git/trees$", 422, {"message": "Invalid tree info"})
    commit = client.create_commit(
        "acme/demo",
        parent_sha=checkout.base_sha,
        base_tree_sha=checkout.base_tree_sha,
        changes=changeset.changes,
        message="Add",
        expected_tree_sha=changeset.tree_sha,
    )
    assert commit["tree_sha"] == changeset.tree_sha
    assert len(github.calls("POST", r"/git/trees$")) == 2 and len(github.calls("POST", r"/git/blobs$")) == 2

