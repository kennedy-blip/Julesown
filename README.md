# Jules Clone Monorepo 🤖⚡

An autonomous coding agent you deploy yourself. You describe a change; an LLM agent makes it on a
clone of your GitHub repository inside an [E2B](https://e2b.dev) sandbox, runs the tests there, and
opens a **real pull request** with the result. A Next.js dashboard and a Python FastAPI backend are
deployed together on [Vercel](https://vercel.com/) from this one repository.

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
5. **Commit** – the files that changed are committed to a new branch `jules/task-<id>` through the
   GitHub API. The base branch is never modified.
6. **Pull request** – a pull request is opened from that branch. Its description contains the agent's
   summary, the changed files and the real test results.

Nothing in this pipeline is simulated:

- A pull request link is shown only when it is the URL GitHub returned for the pull request that was
  just created. There are no placeholder or guessed links.
- If any step fails, the task is reported as **failed** with the reason (missing key, App not
  installed, rate limit, time budget…). If the agent changes nothing, the task finishes with
  **no changes** and no pull request.
- Test status is what actually happened in the sandbox: *passed*, *failed* or *not run*. A pull
  request opened with failing or missing tests says so.

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
│   │   ├── components/             # TaskForm, LiveTerminal, PRStatusTracker, StageProgress, …
│   │   └── lib/                    # API client (NDJSON stream), health store, session history
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
│       │   ├── runner.py           # The pipeline above, end to end
│       │   ├── agent.py            # Tool-calling loop and tools
│       │   ├── llm.py              # Groq / OpenAI / Anthropic clients
│       │   ├── sandbox.py          # E2B sandbox adapter, clone and change collection
│       │   ├── github.py           # GitHub App + token auth, commits, branches, pull requests
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
   installation fails with a link to install it. The dashboard lists the repositories it can access.
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
| `GET /api/v1/github/repos` | Repositories the configured credentials can open pull requests on. |
| `GET /api/v1/github/check?repository=…&branch=…` | Check, without running anything, that a task could act on a repository. |
| `GET /api/v1/health`, `GET /api/v1/health/env` | Liveness and configuration diagnostics. |
| `POST /api/v1/webhooks/github` | GitHub webhook receiver. |

Request body for both task endpoints (JSON, sent with `Content-Type: application/json`):
`{"repository": "owner/name or GitHub URL", "branch": "optional", "prompt": "what to do"}`.
Interactive documentation is served at `/api/docs`.

---

## ⚠️ Limits worth knowing

- **Scope of a task.** The time budget suits focused changes ("fix this failing test", "add input
  validation to X"), not large refactorings.
- **LLM rate limits.** Free tiers are small (Groq's free tier allows about 8,000 tokens per minute on
  the default model). The agent sizes its context to the provider's limit and waits out rate limits,
  which makes runs slower; a task that cannot finish in time fails and says so. A paid tier or
  another provider removes most of this.
- **Sandbox contents.** The default E2B template has git, Python and Node.js but not your project's
  dependencies; the agent installs what the tests need, which costs time. Use `E2B_TEMPLATE` for a
  prepared image.
- **Task history** is kept in the server's memory and in your browser. It is not a database; the
  durable record of a task is its pull request.
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
