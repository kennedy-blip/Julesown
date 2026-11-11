# Jules Clone Monorepo 🤖⚡

An autonomous coding agent you deploy yourself. You describe a change; an LLM agent makes it on a
clone of your GitHub repository inside an [E2B](https://e2b.dev) sandbox, runs the tests there, and
opens a **real pull request** with the result. A Next.js dashboard and a Python FastAPI backend are
deployed together on [Vercel](https://vercel.com/) from this one repository.

The dashboard lists **every repository** the configured GitHub credentials can work on, with a
search box. A repository that is still **empty** works too: the agent builds the project from your
description and pushes it as the repository's first commit, laid out so that it can be deployed on
Vercel or another host afterwards.

---

## 🔁 What happens when you run a task

```text
configuration check → GitHub access → E2B sandbox → git clone → agent loop → commit → pull request
```

1. **Configuration** – the server checks it has an LLM key, the sandbox key and GitHub write credentials.
2. **GitHub access** – it confirms it may open pull requests on the repository and resolves the base
   branch (the repository's default branch unless you name one).
3. **Sandbox + clone** – an E2B sandbox is started and the repository is cloned into it.
4. **Agent** – the model works through tools (`list_files`, `read_file`, `search_code`, `write_file`,
   `replace_in_file`, `run_command`, `run_tests`, `finish`). Every step is streamed to the dashboard.
   It is given the file list, the repository's manifests (`package.json`, `pyproject.toml`, …) and
   the start of the README together with the task, and it has to look before it writes
   (see [What the agent's work is based on](#-what-the-agents-work-is-based-on)).
5. **Commit** – the files that changed are committed to a new branch `jules/task-<id>` through the
   GitHub API. The base branch is never modified.
6. **Pull request** – a pull request is opened from that branch. Its description contains the agent's
   summary, the changed files, the files the agent looked at and the real test results.

An **empty repository** takes a different last step – see
[Empty repositories and new projects](#-empty-repositories-and-new-projects).

Nothing in this pipeline is simulated:

- A pull request link is shown only when it is the URL GitHub returned for the pull request that was
  just created. There are no placeholder or guessed links. The same holds for the link to a first
  commit.
- If any step fails, the task is reported as **failed** with the reason (missing key, App not
  installed, rate limit, time budget…). If the agent changes nothing, the task finishes with
  **no changes** and no pull request.
- Test status is what actually happened in the sandbox: *passed*, *failed*, *no tests* or *not run*.
  A pull request opened with failing or missing tests says so.
  - *No tests* means a test command was run but had nothing to test – for example `npm test` in a
    project whose `test` script is still npm's placeholder (`echo "Error: no test specified" && exit 1`),
    a missing `test` script, or a test runner that found no tests. That exits with an error code, but
    it is not a failing test and is not reported as one.
  - A change that only touches documentation (Markdown, `README`, `LICENSE`, …) needs no tests, so it
    gets no warning about them.

---

## 🔎 What the agent's work is based on

A language model will happily describe a project it has not looked at, and the result reads well and
is wrong. The platform cannot check prose for truth, but it makes sure the model looked first and
shows you what it looked at:

- **Facts up front.** The repository's root manifests and the beginning of its README are part of
  the task message, so the model does not have to guess what the project is built with.
- **Look before you write.** The first `write_file` of a run is held back once if the model has not
  read, searched or run anything yet, and an existing file is not overwritten before it has been read.
  The model is told why and may repeat the call – this is a nudge, not a lock. If a held-back write is
  never repeated, `finish` is questioned once.
- **Shown to the reviewer.** The pull request (and the dashboard) lists the files the agent looked at,
  or warns that it opened none. Check a pull request against that list before merging: a description
  of "the dependencies" is only as good as the manifest behind it.

---

## 📚 Choosing a repository

The **Repositories** panel shows all repositories the server's GitHub credentials can work on – every
page GitHub returns, not just the first hundred – most recently pushed first. Archived repositories
are listed and marked (they are read-only, so no task can run on them).

- **Search** matches the name, the owner, the description and the language; several words must all
  match. Names that start with what you typed come first, and a few letters in the right order
  (`tsktr` for `Task-tracker`) are enough when nothing matches literally. Pasting a GitHub URL finds
  its repository. `↑` `↓` move through the matches, `Enter` chooses, `Esc` clears.
- **Not listed?** With a GitHub App the panel says, per account, whether the App covers *all
  repositories* or only *selected* ones and links to the GitHub page where repositories are added.
  You can also type `owner/name` and use it directly: the form then checks that repository and says
  what is missing.
- The list is read from GitHub when the dashboard opens; the reload button reads it again (for
  example after you created a repository or changed the App's selection).

---

## 🌱 Empty repositories and new projects

A repository without any commit has no branch, so there is nothing a pull request could be opened
against. A task on such a repository builds the project **from your prompt** and its result becomes
the repository's **first commit on its default branch**:

```text
configuration check → GitHub access → E2B sandbox → empty work tree → agent → first commit
```

- The dashboard tells you before you start: *"owner/name is empty. Jules will build the project from
  your instructions and commit it to main as the first commit."* The result shows the commit GitHub
  created (**First commit pushed**, history badge `PUSHED`).
- This is the only case in which a default branch is written to. A repository that has commits always
  gets a pull request, and if somebody pushes to the repository while the task is running, the task
  fails instead of committing on top of their work. Nothing is ever force-pushed.
- GitHub's Git Data API answers `409 Git Repository is empty` for such a repository, so the first
  file is written with the Contents API (GitHub's documented way in) and the rest follows in a second
  commit. The history therefore starts with `Initial commit` (the README) followed by the commit
  that adds the project. The committed tree is compared with the sandbox's, exactly as for a pull
  request.
- From the next task on it is an ordinary repository: changes arrive as pull requests.

**What the agent is asked to build.** In a repository without code – empty, or holding only what
GitHub adds when a repository is created (README, licence, `.gitignore`) – the agent works from
different instructions: build what the prompt asks for, complete rather than broad, in the layout the
chosen stack expects so that a hosting platform (Vercel, Netlify, Render, GitHub Pages …) can deploy
it without extra configuration:

- a static site has `index.html` at the repository root; a Node project a `package.json` with the
  scripts platforms run and only the dependencies it uses; a Python app a `requirements.txt`;
- configuration comes from environment variables (the port from `PORT`), documented in
  `.env.example`; no secrets, no hard-coded localhost addresses;
- a `README.md` (what it is, how to run it, how to deploy it) and a `.gitignore` for the stack.

Before the result is committed the platform checks what it can check for certain, and sends the
agent back **once** if something is missing: no README, dependencies without a `.gitignore`, a
`package.json` that is not valid JSON, a `.env` file that would be committed. Whether the project
*works* is not something the platform can know; the result panel says which checks the agent ran.

A repository that only has a README gets the same instructions, and – because it has commits – the
result as a pull request.

**Size.** A new project has to fit into one run (see [Limits](#️-limits-worth-knowing)). The agent is
told how much time it has, and when the time is nearly used up it is asked once to wrap up, so that a
run ends with a result that says what is missing rather than with a timeout that loses everything.

---

## 🏗 Architecture

```text
.
├── apps/
│   ├── web/                        # Next.js 14 dashboard (App Router + Tailwind CSS)
│   │   ├── app/
│   │   │   ├── page.tsx            # Dashboard: task form, live log, result
│   │   │   ├── layout.tsx          # Root layout with the backend status badge
│   │   │   └── not-found.tsx       # 404 page; forwards stray GitHub App redirects to the dashboard
│   │   ├── components/             # RepositoryPicker, TaskForm, LiveTerminal, PRStatusTracker, StageProgress, …
│   │   └── lib/                    # API client (NDJSON stream), repository search, health store, session history, result wording
│   └── api/                        # Python FastAPI backend (Vercel Python function)
│       ├── main.py                 # App entrypoint
│       ├── config.py               # Environment-driven settings and model defaults
│       ├── security.py             # Access control for task endpoints
│       ├── routers/
│       │   ├── tasks.py            # Create/stream/look up tasks
│       │   ├── github.py           # Repository list, pre-flight check, App redirect landing
│       │   ├── webhooks.py         # GitHub webhook: `/jules` commands on issues
│       │   └── health.py           # Liveness and configuration diagnostics
│       ├── services/
│       │   ├── runner.py           # The pipeline above, end to end (pull request or first commit)
│       │   ├── agent.py            # Tool-calling loop and tools; instructions for existing and for new projects
│       │   ├── llm.py              # Groq / OpenAI / Anthropic clients
│       │   ├── verification.py     # What a test command's result means; documentation-only changes
│       │   ├── sandbox.py          # E2B sandbox adapter, clone and change collection
│       │   ├── github.py           # GitHub App + token auth, repository listing, commits, branches, pull requests
│       │   └── store.py            # In-memory task records
│       ├── tests/                  # Test-suite (runs the whole pipeline offline)
│       └── requirements.txt
├── vercel.json                     # Routing: /api/* → apps/api/main.py, the rest → Next.js, unknown paths → its 404 page
├── .env.example                    # Environment variable template
└── package.json                    # npm workspace root and scripts
```

---

## ⚙️ Configuration

All configuration is environment variables. `GET /api/v1/health/env` (and **Diagnostics Details** in
the dashboard) reports which are present and what is still missing — never their values.

### Required

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY` **or** `OPENAI_API_KEY` **or** `ANTHROPIC_API_KEY` | The model that drives the agent. If several are set, the order of preference is Groq, OpenAI, Anthropic (override with `JULES_LLM_PROVIDER`). |
| `E2B_API_KEY` | Sandbox in which the repository is cloned, edited and tested. |
| `GITHUB_APP_ID` + `GITHUB_APP_PRIVATE_KEY` **or** `GITHUB_TOKEN` | Credentials used to push the branch and open the pull request. See [GitHub setup](#-github-setup). |

### Access control

Running a task spends your LLM and sandbox quota, opens pull requests with your GitHub credentials,
and streams repository content to the caller. Task endpoints are therefore protected:

| Variable | Effect |
| --- | --- |
| `JULES_ACCESS_KEY` | Recommended. Callers must present this value (the dashboard asks for it once and stores it in the browser; API clients send `Authorization: Bearer <key>`). |
| `JULES_ALLOW_PUBLIC_ACCESS=true` | Lets anyone who has the URL run tasks. Only for deployments that are not reachable by others. |

With neither set, tasks are **locked** on Vercel / `ENVIRONMENT=production` and open in local development.

### Optional

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_MODEL` / `OPENAI_MODEL` / `ANTHROPIC_MODEL` | `openai/gpt-oss-120b` / `gpt-6-luna` / `claude-sonnet-5-5` | Model to use. If the configured model has been retired, the backend picks an available one from the provider's model list and says so in the task log. |
| `GROQ_REASONING_EFFORT` / `OPENAI_REASONING_EFFORT` | `low` / `none` | Reasoning effort sent to the model (`default` omits the parameter). |
| `GROQ_TEMPERATURE` | `0.2` | Sampling temperature for Groq, `0`–`2` (`default` leaves it to Groq, which uses `1`). A low value makes malformed tool calls and made-up content less likely. |
| `JULES_TASK_BUDGET_SECONDS` | `270` | Time one task may take. On Vercel it is additionally capped by the function's remaining time. |
| `JULES_MAX_AGENT_TURNS` | `30` | Maximum model turns per task. |
| `JULES_COMMAND_TIMEOUT_SECONDS` | `120` | Time limit for one command in the sandbox. |
| `JULES_MAX_OUTPUT_TOKENS` | `8192` | Upper bound for one model reply. |
| `E2B_TEMPLATE` | E2B default | Sandbox template, e.g. one with your toolchain preinstalled. |
| `GITHUB_WEBHOOK_SECRET` | – | Enables `/jules` commands from GitHub issues (see below). |
| `JULES_CORS_ORIGINS` | – | Comma-separated origins allowed to call the API from a browser. Not needed for the dashboard, which is served from the same origin. |

---

## 🐙 GitHub setup

### Option A — GitHub App (recommended)

1. Create a GitHub App (*Settings → Developer settings → GitHub Apps*) with these **repository permissions**:
   - **Contents: Read and write** (push the task branch)
   - **Pull requests: Read and write** (open the pull request)
   - **Issues: Read and write** (only if you use `/jules` commands; lets the result be commented on the issue)
   - Metadata: Read-only (mandatory)
2. Generate a private key and set `GITHUB_APP_ID` and `GITHUB_APP_PRIVATE_KEY` (paste the whole PEM;
   literal `\n` sequences and base64 are accepted too).
3. **Install the App on the repositories** Jules should work on. A task on a repository without the
   installation fails with a link to install it. When installing, GitHub asks for *All repositories*
   or *Only select repositories*: with the second choice, a repository that was not selected does not
   appear in the dashboard until it is added (*Settings → Applications → Configure*). The dashboard
   says which of the two applies and links to that page.
4. URLs, if you want GitHub to send people back to the dashboard after installing:
   - **Setup URL**: `https://<your-domain>/api/v1/github/setup`
   - **Callback URL**: `https://<your-domain>/api/v1/github/callback`
   - **Webhook URL**: `https://<your-domain>/api/v1/webhooks/github` (with the secret from `GITHUB_WEBHOOK_SECRET`)

Each task uses a short-lived installation token limited to the one repository. That token never
enters the sandbox; private repositories are cloned with a separate read-only token.

### Option B — Personal access token

Set `GITHUB_TOKEN` to a fine-grained token with **Contents: Read and write** and
**Pull requests: Read and write** on the target repositories (or a classic token with `repo`).

### `/jules` commands from issues (optional)

With `GITHUB_WEBHOOK_SECRET` set and the webhook subscribed to **Issues** and **Issue comments**, a
repository owner, member or collaborator can start a task by writing a line that begins with `/jules`
in a new issue or an issue comment:

```text
/jules fix this and add a regression test
```

The pull request link (or the reason for failure) is posted back to the issue as a comment.
Deliveries with an invalid signature are rejected, and nothing is started when no secret is configured.

---

## 🚀 Deploying on Vercel

1. Import the repository in Vercel. `vercel.json` builds the Next.js app and the Python function,
   routes `/api/*` to the backend and sends unknown paths to the app's own 404 page.
2. Add the environment variables above, including `JULES_ACCESS_KEY`.
3. Deploy. Environment variable changes only take effect after a new deployment.
4. Open the site, enter the access key, pick a repository and run a task.

A task runs inside one function invocation and streams its progress back, so it has to fit in the
function's maximum duration (300 s on the Hobby plan; the default task budget of 270 s leaves room
to open the pull request). If the browser loses the connection, the dashboard looks the task up
again and, when necessary, finds its pull request on GitHub by the task's branch name.

When a task fails, the dashboard shows the stage it failed at and the reason. *Diagnostics Details*
(`GET /api/v1/health/env`) shows which settings the deployment actually sees, without their values,
and every task leaves one summary line in the function logs (Vercel → your deployment → Logs).

---

## 📡 API

| Endpoint | Description |
| --- | --- |
| `POST /api/v1/tasks/stream` | Run a task and stream progress as newline-delimited JSON (`task`, `log`, `update`, `ping`, `done`). Used by the dashboard. |
| `POST /api/v1/tasks` | Run a task in the background; poll `GET /api/v1/tasks/{id}`. |
| `GET /api/v1/tasks/{id}?repository=owner/name` | Task status. When the answering server instance does not know the task, its pull request is looked up on GitHub. |
| `GET /api/v1/tasks/{id}/pull-request?repository=owner/name` | The task's pull request as it is on GitHub right now: `open`, `merged` or `closed`. The dashboard uses it to keep the history up to date. |
| `GET /api/v1/github/repos` | Every repository the configured credentials can work on (all pages, up to 2,000), most recently pushed first, with `total`, `truncated` and, for a GitHub App, `installations` (account, `repository_selection`: `all` or `selected`, and where to change it). |
| `GET /api/v1/github/check?repository=…&branch=…` | Check, without running anything, that a task could act on a repository. `empty: true` means it has no commits and a task will create its first commit. |
| `GET /api/v1/health`, `GET /api/v1/health/env` | Liveness and configuration diagnostics. |
| `POST /api/v1/webhooks/github` | GitHub webhook receiver. |

Request body for both task endpoints (JSON, sent with `Content-Type: application/json`):
`{"repository": "owner/name or GitHub URL", "branch": "optional", "prompt": "what to do"}`.
Interactive documentation is served at `/api/docs`.

Fields of a task record that describe how far the result can be trusted:

| Field | Meaning |
| --- | --- |
| `tests_status` | `passed` or `failed` (the last run that really tested something), `no_tests` (test commands were run, but the project has no tests) or `not_run`. `tests_passed` is `true`/`false` only for the first two. |
| `verification[]` | Every `run_tests` call: `command`, `exit_code`, `duration`, `outcome` (`passed`, `failed`, `timed_out`, `no_tests`) and, for `no_tests`, a `note` with the reason. |
| `docs_only` | `true` when the change only touches documentation files. |
| `files_read` | Repository files whose content the agent saw before it finished. |
| `outcome` | `pull_request`, `no_changes`, or `initial_commit` (the repository was empty: `commit_sha` and `commit_url` name its first commit and there is no pull request). |
| `empty_repository`, `new_project` | The repository had no commits / no code when the task started. |

---

## ⚠️ Limits worth knowing

- **Scope of a task.** The time budget suits focused changes ("fix this failing test", "add input
  validation to X"), not large refactorings. A project built from scratch has to be small for the
  same reason: on a rate-limited model a run produces a handful of files, not an application with
  dozens. Ask for a first version and extend it with follow-up tasks.
- **Workflow files.** GitHub refuses files under `.github/workflows` from a GitHub App without the
  *Workflows* permission, which fails the commit. The agent does not add CI workflows unless asked.
- **Review before you merge.** The agent's summary and any documentation it writes are the model's
  words. The platform reports what it can observe (files read, commands run, test results); it does
  not verify that a description is accurate.
- **LLM rate limits.** Free tiers are small (Groq's free tier allows about 8,000 tokens per minute on
  the default model). The agent sizes its context to the provider's limit and waits out rate limits,
  which makes runs slower; a task that cannot finish in time fails and says so. A paid tier or
  another provider removes most of this.
- **Refused model replies.** Open models hosted by Groq sometimes answer with a call to a tool that
  does not exist (`commentary`, `json`, …); Groq refuses such a reply and nothing is executed. The
  agent tells the model what was wrong and asks again, up to three times per step and with a slightly
  higher temperature each time. The log shows these as warnings. If the model keeps doing it, the
  task fails with that reason: run it again or choose another model with `GROQ_MODEL`.
- **Sandbox contents.** The default E2B template has git, Python and Node.js but not your project's
  dependencies; the agent installs what the tests need, which costs time. Use `E2B_TEMPLATE` for a
  prepared image.
- **Task history** is kept in the server's memory and in your browser. It is not a database; the
  durable record of a task is its pull request (or first commit). Whether a pull request has been merged or closed
  since is read from GitHub when the dashboard loads and when you return to its tab.
- **Change size.** A task may change up to 300 files / 4 MB (1 MB per file). Dependency and build
  directories (`node_modules`, `.venv`, `__pycache__`, …) are never committed.

---

## 🛠 Local development

Prerequisites: Node.js 18+, Python 3.11+.

```bash
cp .env.example .env            # fill in your keys (loaded automatically by the backend)

npm install                     # web dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r apps/api/requirements.txt

npm run dev:api                 # FastAPI on http://localhost:8000
npm run dev:web                 # Next.js on http://localhost:3000 (proxies /api/* to the backend)
```

The web dev server forwards `/api/*` to `http://localhost:8000`; set `API_BASE_URL` in its
environment (`API_BASE_URL=http://host:port npm run dev:web`) if the backend runs elsewhere.

Without keys the dashboard still works: it shows what is missing, and a task fails at the
configuration step instead of pretending to run.

---

## 🧪 Tests

```bash
npm run test:api                # or: pytest apps/api
```

The suite runs the whole pipeline offline against local stand-ins: a real `git clone` into a
temporary directory, real file edits and commands, real git objects pushed to a bare repository
through an emulation of the GitHub REST API, and a scripted model. It covers the honest-failure
paths as well (missing configuration, missing App installation, LLM errors, refused pull requests).
The real E2B, LLM and GitHub services are only exercised when you run a task with your own keys.
