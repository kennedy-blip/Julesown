# Jules Clone Monorepo 🤖⚡

An autonomous coding agent you deploy yourself. You describe a change; an LLM agent makes it on a
clone of your GitHub repository inside an [E2B](https://e2b.dev) sandbox, runs the tests there, and
opens a **real pull request** with the result. A Next.js dashboard and a Python FastAPI backend are
deployed together on [Vercel](https://vercel.com/) from this one repository.

The dashboard lists **every repository** the configured GitHub credentials can work on, with a
search box. A repository that is still **empty** works too: the agent builds the project from your
description and pushes it as the repository's first commit, laid out so that it can be deployed on
Vercel or another host afterwards.

The agent is not tied to one model. Each provider (Groq, Google Gemini, NVIDIA NIM, Z.AI,
OpenRouter, OpenAI, Anthropic) can be given several models, and with keys for more than one
provider the agent **carries on with the next model** when the one in use reaches its rate limit,
has used up its day or fails. It can
**add more work to a pull request**, **repair the checks that failed** on it (a test run on GitHub
Actions, a deployment on Vercel) and, when a run stops before the work is done, it **delivers what
it had** marked as unfinished - together with a note of where it stopped, so that the next run
**continues from there** instead of starting over. Provider keys, the models, your own **standing
instructions** for every task and who the commits are attributed to can be set in the dashboard's
**Settings**.

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
   (see [What the agent's work is based on](#-what-the-agents-work-is-based-on)). When the model
   in use reaches a limit or fails, the same conversation continues with the next one
   (see [Model providers and fallback](#-model-providers-and-fallback)). Rules of your own that
   apply to every task are added after the task (see [Standing instructions](#-standing-instructions)).
5. **Commit** – the files that changed are committed to a new branch `jules/task-<id>` through the
   GitHub API. The base branch is never modified.
6. **Pull request** – a pull request is opened from that branch. Its description contains the agent's
   summary, the changed files, the files the agent looked at and the real test results.

An **empty repository** takes a different last step – see
[Empty repositories and new projects](#-empty-repositories-and-new-projects). A task can also add
its commit to a pull request that already exists instead of opening a new one – see
[More work on a pull request](#-more-work-on-a-pull-request).

Nothing in this pipeline is simulated:

- A pull request link is shown only when it is the URL GitHub returned for the pull request that was
  just created. There are no placeholder or guessed links. The same holds for the link to a first
  commit.
- If any step fails, the task is reported as **failed** with the reason (missing key, App not
  installed, rate limit, time budget…). If the agent changes nothing, the task finishes with
  **no changes** and no pull request.
- A run that stops after the agent had already changed files - because the time ran out, the
  models were exhausted, or the agent itself reported that it could not complete the task - is not
  reported as a clean result and not thrown away either: the changes are delivered **marked as
  unfinished** (see [Unfinished work](#-unfinished-work)).
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

**Size and time.** One run is short (see [Limits](#️-limits-worth-knowing)): a small project fits
into it, a larger one takes two or three. The agent is told so and works accordingly:

- It is told how much time the run has, to keep files lean (a handful of sample entries, no filler)
  and not to read back what it has just written.
- With a build step it writes the manifest, the configuration, the `.gitignore` and a minimal entry
  point first and runs the install and the build **once, early**, while a mistake in the
  configuration is cheap to find. A command that fails is fixed before anything else is written.
- At half time it is told how many seconds are left and to check what it has.
- When the time is nearly used up it is asked once to wrap up: to say what is done, what is missing
  and what the last failing command reported.

Whatever it has built by then is **kept**: a project whose build does not pass yet is pushed as
`[Unfinished] …` with that report, and the next run continues from it
(see [Unfinished work](#-unfinished-work)). Nothing is discarded because the time ran out.

---

## 🧠 Model providers and fallback

| Provider | Key | Default models | Free of charge |
| --- | --- | --- | --- |
| Groq | `GROQ_API_KEY` | `openai/gpt-oss-120b`, then `openai/gpt-oss-20b` | few tokens per minute and per day |
| Google Gemini | `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) | `gemini-flash-latest`, then `gemini-3.5-flash-lite` | requests per day |
| NVIDIA NIM | `NVIDIA_API_KEY` (or `NVIDIA_NIM_API_KEY`) | `deepseek-ai/deepseek-v4.1-flash`, then `nvidia/nemotron-3-super-120b-a12b` | about 40 requests a minute per account, no daily limit |
| Z.AI (GLM) | `ZAI_API_KEY` (or `GLM_API_KEY`) | `glm-4.7-flash`, then `glm-4.5-flash` | these two models, one request at a time |
| OpenRouter | `OPENROUTER_API_KEY` | `openrouter/free` | 50 requests a day for all free models together |
| OpenAI | `OPENAI_API_KEY` | `gpt-6-luna` | – |
| Anthropic | `ANTHROPIC_API_KEY` | `claude-sonnet-5-5` | – |

One key is enough to run tasks. The more of the free ones are set, the less a single provider's
limit matters: where one has used up its day, the next carries on. (The limits are the providers'
own and change; the column says what they were when this was written.)

**Several models per provider.** A provider can be given up to four models, separated by commas
(`GEMINI_MODEL=gemini-flash-latest,gemini-3.5-flash-lite`, or the **Models** field in the
dashboard's Settings). Providers count their limits per model, so a second model is a second
allowance: when the first is rate limited, overloaded or withdrawn, the next one carries on.

**Defaults that do not go stale.** Where a provider has a name that moves with its catalogue, that
name is the default:

- `gemini-flash-latest` is Google's own alias for its current Flash model. Google moves it when a new
  version is released, so there is no need to follow which version (2.5, 3.x, …) is current or
  being shut down. The task log says which model answered behind the alias.
- `openrouter/free` is OpenRouter's router for its free models: for every request it picks a free
  model that is available and supports what the request needs (tool calls). A single free model is
  throttled whenever it is popular and may be withdrawn; the router has neither problem. The log
  and the result say which models answered. To pin one, or to list several, set `OPENROUTER_MODEL`
  (for example `qwen/qwen3-coder:free,openrouter/free`).
- NVIDIA and Z.AI have no such name: their defaults are models. A model NVIDIA has withdrawn is
  replaced by another one it still lists; a GLM model that is gone is passed over, and the rest of
  the chain carries on.

**NVIDIA NIM** ([build.nvidia.com](https://build.nvidia.com)) hosts open models behind one key
(`nvapi-…`). What to know about it:

- Requests are counted **per account**, not per model, and nothing says how long to stay away. So
  when NVIDIA answers `429`, all of its models are set aside for a minute - asking the second one
  would be refused as well, and users report that asking again while limited prolongs the limit.
- Its catalogue changes every few weeks. When a configured model has been withdrawn, another one
  that NVIDIA still lists and that calls tools takes its place, and the task log says so. To choose
  yourself: `NVIDIA_MODEL=nvidia/nemotron-3-ultra-550b-a55b,deepseek-ai/deepseek-v4.1-flash` (the
  names are those on build.nvidia.com; pick models whose page says *Function Calling: Supported*).
- `404 … Not found for account` for **every** model means the account may list the models but not
  call them: NVIDIA has to enable its hosted endpoints for the account. **Test** in Settings shows it.
- The free endpoints are a trial service for development and evaluation. NVIDIA's notice for them
  says that what is sent and what comes back is recorded and used to improve its products, and asks
  not to send confidential information: leave NVIDIA out for code that must stay private.

**Z.AI** ([z.ai](https://z.ai/model-api)) serves the GLM models. What to know about it:

- Use a **standard API key**. With it `glm-4.7-flash` and `glm-4.5-flash` cost nothing; they
  answer one request at a time, which is all a task needs. The larger models are paid for from the
  account's balance: `ZAI_MODEL=glm-5.3,glm-4.7-flash` uses GLM-5.3 while there is a balance and
  the free model when there is none (only the model without balance is set aside, not Z.AI).
- A **GLM Coding Plan** key is not meant for this. Z.AI restricts that plan to the coding tools it
  lists and may limit accounts that use it elsewhere; this project does not use the plan's endpoint.
- GLM models think before every reply unless told otherwise, at their highest effort - more than a
  run of a few minutes can afford. Thinking is therefore switched off (for GLM-5.3 and later, which
  cannot switch it off, the lowest effort is asked for). `ZAI_REASONING_EFFORT` changes that.

**The chain.** Every provider that has a key takes part, in the order of the table (change it with
`JULES_LLM_ORDER`, or name the first choice with `JULES_LLM_PROVIDER`). A task tries every
provider's first model before anybody's second one - with keys for all five free providers:

```text
groq openai/gpt-oss-120b → gemini gemini-flash-latest → nvidia deepseek-ai/deepseek-v4.1-flash
  → zai glm-4.7-flash → openrouter openrouter/free
  → groq openai/gpt-oss-20b → gemini gemini-3.5-flash-lite
  → nvidia nvidia/nemotron-3-super-120b-a12b → zai glm-4.5-flash
```

OpenRouter comes last among them: its router answers with whichever free model it picks, and its
daily allowance is the smallest.

The agent works with the first one and switches when it has to:

- **Rate limit** – the next model that is ready takes over at once and stays in use; the agent
  does not hop back. Only when *every* model is limited does it wait, for the shortest of the
  waits. If that wait is longer than the time the task has left, the run ends there (and what it
  had changed is kept). Limits are counted per model - except at NVIDIA, where a limit sets all of
  its models aside.
- **A router whose pick is throttled** – when `openrouter/free` lands on a model that is rate
  limited upstream (and no waiting time is named), the next request gets another pick: the router
  is asked again after 3 seconds instead of being set aside like a single model (30 seconds).
- **Daily limit** – that model is out for the rest of the run. OpenRouter counts one daily
  allowance for all its free models, so there it takes out every free model at once; paid models
  on the same key stay usable.
- **Rejected key, no credit** – every model of that provider is out for the rest of the run.
  (At Z.AI, "no balance" is about the paid model that was asked: its free models stay in use.)
- **Unknown model, request the provider refuses** – that model is out for the rest of the run.
- **Outage or timeout** – the model is set aside for a short while and tried again later.
- **A model that does not answer** – a request is not waited for longer than the model has earned:
  45 seconds for one that has not answered yet in this run; one that has gets three times its
  slowest answer, between 45 and 90 seconds. With nothing else to switch to, it gets the full 90.
- **Replies the provider refuses for their form** (Groq does this when an open model calls a tool
  that does not exist) – the reply is rebuilt from what the provider returned when that is possible;
  after two refusals in a row the next model takes over.

The conversation carries over: the next model receives everything the agent has read and done so
far. The task log says when and why it switched, and the result lists what each model did
(`llm_usage` in the task record; for a router or an alias also `served`, the models that actually
answered). `JULES_LLM_FALLBACK=false` restricts tasks to the first model of the first provider.

**Where the keys live.** A key can be an environment variable of the deployment, or be entered in the
dashboard under **Settings → AI models**. A key entered there is kept in that browser
(`localStorage`) and sent to your own backend with each task you start, in a request header
(`X-Groq-Key`, `X-Gemini-Key`, `X-Nvidia-Key`, `X-Zai-Key`, `X-OpenRouter-Key`, `X-OpenAI-Key`,
`X-Anthropic-Key`). The backend uses it for that one task, in place of its own key for that provider; it is not stored and is masked
in logs and error messages. The same page sets the models per provider and the first choice, and
its **Test** button sends one small request with a tool to **each** of the provider's models
(`POST /api/v1/llm/test`) to show whether the key is accepted, each model exists and can call
tools - and, for a router or an alias, which model answered.

Tasks that GitHub starts (a `/jules` comment, an automatic fix of failed checks) do not pass through
a browser. They use the keys of the deployment, so for fallback there, set the keys as environment
variables.

---

## ➕ More work on a pull request

A task does not have to open a new pull request. It can add a commit to one that is open:

- **Dashboard** – choose *Add to a pull request* in the task form and give its number, or press
  *Ask for more changes* on the result of an earlier task.
- **GitHub** – comment `/jules <what to change>` on the pull request.
- **API** – send `"pull_request": <number>` with the task.

The agent works on a clone of the pull request's branch, so it starts from what is there already,
and it is told which checks currently fail on it. Its commit is added on top (the branch is never
force-pushed) and the pull request gets a comment saying what was added. The pull request must be
open and its branch must be in the repository itself (not in a fork). This works for any pull
request, not only for ones Jules opened. The task ends with the outcome `pr_updated`.

### 🧩 Unfinished work

A run can stop before the work is done: the time budget is used up, every model is exhausted, the
turn limit is reached - or the agent itself reports that it cannot complete the task (a build it
could not get to pass, something it needs that is not in the repository). If it had changed files
by then, they are **not discarded**:

- In a repository with code, a pull request titled **`[Unfinished] …`** is opened (or the commit is
  added to the pull request the task was working on). Its description starts with a warning that says
  why the agent stopped, and that it should not be merged as it is.
- In an empty repository the files are pushed as the first commit, marked the same way.
- The commit message records where the work stands: the agent's own account, why it stopped and the
  command that still failed, with the end of its output.
- A run that was stopped cannot tidy up after itself. What a build left behind before the
  `.gitignore` was written, and files too large to commit, are **left out** rather than costing
  the rest; the log and the handoff name them, so the next run knows they are not in the
  repository. Build output means: new files in `dist/`, `build/`, `out/`, `target/`, `coverage/`
  or `htmlcov/` next to the project's manifest (`package.json`, `pyproject.toml`, `Cargo.toml` …),
  in a tool's cache directory (`.vite`, `.turbo`, `.cache` …), and `*.tsbuildinfo`. A file the agent
  wrote itself, or a directory the repository already keeps, is never taken for it.

```text
[Unfinished] Build OmniPulse

I scaffolded the app: package.json, vite.config.ts and src/. The build does not pass yet.

Unfinished: the agent was stopped before it was done. The time budget ran out before the agent finished.

Still failing when the run stopped: `npm run build` (exit code 2)
    vite.config.ts(8,18): error TS2580: Cannot find name 'process'.

Task-Id: 28b06a3f362a4553804385d0471f48a2
```

**Continuing.** The next run does not start over. It is told that a previous run began the work,
what that run reported (the task record's `handoff`: its account, the failing command and its
output, why it stopped), to look at what is there, and to fix the failing command first:

- **Dashboard** – the result offers **Continue the work**. It sends the same task again with the
  handoff of the run it continues: as a follow-up on the pull request, or - after a first commit -
  as a task that delivers the rest as a pull request. *Where it stopped* in the result shows what
  the next run will be told.
- **GitHub** – comment `/jules continue` on the pull request, or on the issue the work was started
  from (for a first commit). Lines after it are further instructions. The run reads where the work
  stands from the message of the unfinished commit.
- **API** – `"continue_work": true`, with the previous task's `handoff` (optional: without it, the
  commit message is read).

When a continuation finishes, the mark is taken off the title; one that finds nothing left to do
takes it off as well. If the agent *gave up* rather than ran out of time, read its summary before
continuing: it says what stood in the way, and the next run may need to be told how to get past it.
`JULES_KEEP_UNFINISHED_WORK=false` restores the old behaviour, where a run that stops early leaves
nothing behind.

---

## 📌 Standing instructions

Rules that should hold for every task - the conventions of your stack, what your deployments need,
what to leave alone - do not have to be repeated in each prompt:

- **Dashboard** – **Settings → Instructions**. The text is kept in that browser and sent with
  every task started from it (request field `guidelines`).
- **Server** – the environment variable `JULES_TASK_GUIDELINES` (up to 4,000 characters; write `\n`
  for a line break). It applies to every task of the deployment, including the ones GitHub starts
  (a `/jules` comment, an automatic fix of failed checks), which do not pass through a browser.

The agent is given them after the task, under the heading *Standing instructions*; the server's come
first, then the browser's. Where they contradict a task, the task wins. They are not part of the
prompt shown in the result or quoted in the pull request.

Some rules are **built in** and need not be written down: read before writing, keep changes small,
fix a failing install, build or test before doing anything else, and - in an empty repository - get
a small version building first and extend it from there. Good standing instructions are the things
only you know, for example:

```text
- In a Vite app read configuration from import.meta.env, never from process.env.
- If vite.config.ts uses Node APIs (path, process, __dirname), add @types/node to devDependencies.
- Deployments are on Vercel: no server of its own, no hard-coded localhost addresses.
```

---

## 🩺 Failed checks

When the checks of a pull request fail – a workflow on GitHub Actions, a deployment reported by
Vercel, any other check run or commit status – the agent can read what failed and repair it. It is
given the names of the failed checks, their annotations (file, line, message), the part of the job
log around the error and, for commit statuses, the description and link; it then reproduces the
failure in the sandbox where it can, fixes it and adds the commit to the pull request.

| How | What it needs |
| --- | --- |
| **From the dashboard.** The result of a task shows the checks of its pull request (or first commit) and, when they fail, a *Fix failed checks* button. The task form has the same option for any pull request or branch. | Permission to read checks (see below). |
| **From GitHub, on request.** Comment `/jules fix the checks` (or just `/jules`) on the pull request. | The webhook with *Issue comment* events. |
| **Unasked.** For pull requests Jules opened itself (branches `jules/…`), as soon as GitHub reports the failure. At most `JULES_CI_AUTOFIX_ATTEMPTS` times per pull request (default 2), once per failing commit; after that it says so in a comment and leaves it to you. | The webhook with *Check suite* and/or *Status* events, and `GITHUB_WEBHOOK_SECRET`. `JULES_CI_AUTOFIX=false` turns it off. |

**Settings → Automation** in the dashboard shows which of these are in place for your GitHub App
(`GET /api/v1/github/automation`) and links to the page where the rest is switched on.

A check that fails for a reason outside the repository (a secret or environment variable missing on
the CI or hosting service, an outage there) cannot be fixed by a commit. The agent is told to report
that in its summary rather than change code to hide it.

---

## 👤 Who the commits are attributed to

By default a commit is made by the GitHub App (or the owner of `GITHUB_TOKEN`), which GitHub marks as
*Verified*. Two other modes attribute the work to a person:

| Mode | Commit author | Notes |
| --- | --- | --- |
| `bot` (default) | the GitHub App | *Verified*. |
| `coauthor` | the GitHub App, with a `Co-authored-by:` line naming you | Still *Verified*; the commit also counts as yours on GitHub. |
| `user` | you; the App appears only as the committer | Not *Verified*: GitHub signs only commits the App authors itself. |

Choose the mode under **Settings → Commits** in the dashboard (it applies to the tasks started from
that browser; *Look up* fills in the name and the `noreply` address of a GitHub account), send
`"commit": {"mode": "user", "name": "…", "email": "…"}` with an API request, or set the server's
default with `JULES_COMMIT_AUTHORSHIP`, `JULES_COMMIT_AUTHOR_NAME` and `JULES_COMMIT_AUTHOR_EMAIL`.
With a mode other than `bot` and no name configured, a task started from GitHub is attributed to the
person who wrote the `/jules` comment, using their `noreply` address. The pull request itself is
always opened by the GitHub App: only commits can carry another author.

---

## 🏗 Architecture

```text
.
├── apps/
│   ├── web/                        # Next.js 14 dashboard (App Router + Tailwind CSS)
│   │   ├── app/
│   │   │   ├── page.tsx            # Workspace (repositories, task form, live log, result, history) and Settings
│   │   │   ├── layout.tsx          # Root layout: navigation and the backend status badge
│   │   │   └── not-found.tsx       # 404 page; forwards stray GitHub App redirects to the dashboard
│   │   ├── components/             # RepositoryPicker, TaskForm, LiveTerminal, PRStatusTracker, TaskHistory, SettingsPanel, ui (shared look), …
│   │   └── lib/                    # API client (NDJSON stream), settings kept in the browser, routes, provider chain, health store, session history, result wording
│   └── api/                        # Python FastAPI backend (Vercel Python function)
│       ├── main.py                 # App entrypoint
│       ├── config.py               # Environment-driven settings and model defaults
│       ├── security.py             # Access control for task endpoints
│       ├── routers/
│       │   ├── tasks.py            # Create/stream/look up tasks; state of a task's pull request and its checks
│       │   ├── github.py           # Repository list, pre-flight check, checks of a commit, account lookup, automation status, App redirect landing
│       │   ├── llm.py              # Trying a provider key out
│       │   ├── webhooks.py         # GitHub webhook: `/jules` commands (also `/jules continue`), failed checks
│       │   └── health.py           # Liveness and configuration diagnostics
│       ├── services/
│       │   ├── runner.py           # The pipeline above, end to end (new pull request, more work on one, first commit, unfinished work)
│       │   ├── agent.py            # Tool-calling loop and tools; instructions for existing and for new projects; time check and wrap-up
│       │   ├── llm.py              # Groq / Gemini / OpenRouter / OpenAI / Anthropic clients and the fallback between their models
│       │   ├── ci.py               # Reading failed checks; what a continued task is told (handoff, standing instructions)
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

The server is configured with environment variables. `GET /api/v1/health/env` (and **Settings →
About** in the dashboard) reports which are present and what is still missing — never their values.
Model provider keys can also be entered in the dashboard instead
(see [Model providers and fallback](#-model-providers-and-fallback)).

### Required

| Variable | Purpose |
| --- | --- |
| `GROQ_API_KEY`, `GEMINI_API_KEY`, `NVIDIA_API_KEY`, `ZAI_API_KEY`, `OPENROUTER_API_KEY`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY` — at least one, here or in the dashboard's Settings | The models that drive the agent. With several, a task starts with the first in that order and carries on with the next when it has to. |
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
| `GROQ_MODEL` / `GEMINI_MODEL` / `NVIDIA_MODEL` / `ZAI_MODEL` / `OPENROUTER_MODEL` / `OPENAI_MODEL` / `ANTHROPIC_MODEL` | see [the table above](#-model-providers-and-fallback) | The model to use, or up to four separated by commas in the order they are tried. If a configured model has been retired, the backend picks an available one from the provider's model list and says so in the task log. |
| `JULES_LLM_PROVIDER` | – | The provider tasks start with (`groq`, `gemini`, `nvidia`, `zai`, `openrouter`, `openai`, `anthropic`). |
| `JULES_LLM_ORDER` | `groq,gemini,nvidia,zai,openrouter,openai,anthropic` | The order in which providers that have a key are used. |
| `JULES_LLM_FALLBACK` | `true` | `false`: a task uses only the first model of the first provider. |
| `JULES_TASK_GUIDELINES` | – | Instructions added to every task of this deployment (see [Standing instructions](#-standing-instructions)). `\n` stands for a line break. |
| `GROQ_REASONING_EFFORT` / `OPENROUTER_REASONING_EFFORT` / `OPENAI_REASONING_EFFORT` | `low` / `low` / `none` | Reasoning effort sent to the model (`default` omits the parameter). |
| `NVIDIA_REASONING_EFFORT` | – | Not sent unless set: the models NVIDIA hosts differ in whether they take it. |
| `NVIDIA_EXTRA_BODY` | – | A JSON object with more fields for every request to NVIDIA, for what the page of your model documents - for example `{"chat_template_kwargs": {"enable_thinking": false}}` to stop a Nemotron model from thinking before each reply. A field a model rejects by name is left out for the rest of the run. |
| `ZAI_REASONING_EFFORT` | `none` | How much a GLM model thinks before it answers: `none` (off; the lowest effort for models that cannot switch it off), `low`, `medium`, `high`, `max`, or `default` to leave it to Z.AI (thinking at the highest effort). |
| `GEMINI_THINKING_LEVEL` | `low` | How much Gemini thinks before it answers (`minimal`, `low`, `medium`, `high`; `default` leaves it to Google). |
| `GROQ_TEMPERATURE` / `OPENROUTER_TEMPERATURE` / `NVIDIA_TEMPERATURE` / `ZAI_TEMPERATURE` | `0.2` | Sampling temperature, `0`–`2` (Z.AI: `0`–`1`; `default` leaves it to the provider). A low value makes malformed tool calls and made-up content less likely. |
| `NVIDIA_BASE_URL` / `ZAI_BASE_URL` | `https://integrate.api.nvidia.com/v1` / `https://api.z.ai/api/paas/v4` | Where the two are reached, for example a NIM you host yourself, or `https://open.bigmodel.cn/api/paas/v4` for a key from Z.AI's platform in mainland China. |
| `JULES_KEEP_UNFINISHED_WORK` | `true` | Deliver what a run that stopped early had changed, marked as unfinished (see [Unfinished work](#-unfinished-work)). |
| `JULES_CI_AUTOFIX` / `JULES_CI_AUTOFIX_ATTEMPTS` | `true` / `2` | Fix failed checks of Jules' own pull requests unasked, and how often per pull request (see [Failed checks](#-failed-checks)). |
| `JULES_COMMIT_AUTHORSHIP` | `bot` | `bot`, `coauthor` or `user`, with `JULES_COMMIT_AUTHOR_NAME` and `JULES_COMMIT_AUTHOR_EMAIL` (see [Who the commits are attributed to](#-who-the-commits-are-attributed-to)). |
| `JULES_TASK_BUDGET_SECONDS` | `270` | Time one task may take. On Vercel it is additionally capped by the function's remaining time. |
| `JULES_MAX_AGENT_TURNS` | `30` | Maximum model turns per task. |
| `JULES_COMMAND_TIMEOUT_SECONDS` | `120` | Time limit for one command in the sandbox. |
| `JULES_MAX_OUTPUT_TOKENS` | `8192` | Upper bound for one model reply. |
| `E2B_TEMPLATE` | E2B default | Sandbox template, e.g. one with your toolchain preinstalled. |
| `GITHUB_WEBHOOK_SECRET` | – | Enables what GitHub starts: `/jules` commands and fixing failed checks unasked (see below). |
| `JULES_CORS_ORIGINS` | – | Comma-separated origins allowed to call the API from a browser. Not needed for the dashboard, which is served from the same origin. |

---

## 🐙 GitHub setup

### Option A — GitHub App (recommended)

1. Create a GitHub App (*Settings → Developer settings → GitHub Apps*) with these **repository permissions**:
   - **Contents: Read and write** (push the task branch)
   - **Pull requests: Read and write** (open the pull request)
   - **Issues: Read and write** (only if you use `/jules` commands; lets the result be commented on the issue)
   - **Checks: Read-only** and **Commit statuses: Read-only** (to see whether the checks of a result
     passed, and to fix them when they failed)
   - **Actions: Read-only** (optional: lets the agent read the log of a failed GitHub Actions job)
   - Metadata: Read-only (mandatory)

   When permissions are added to an App that is already installed, GitHub asks the owner of each
   installation to approve them; until then the installation keeps the old ones.
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

### `/jules` commands and failed checks from GitHub (optional)

With `GITHUB_WEBHOOK_SECRET` set and the webhook subscribed to **Issues** and **Issue comments**, a
repository owner, member or collaborator can start a task by writing a line that begins with `/jules`
in a new issue or an issue comment:

```text
/jules fix this and add a regression test
```

The pull request link (or the reason for failure) is posted back to the issue as a comment.

The same line as a comment **on a pull request** adds the work to that pull request instead of
opening another one, and `/jules fix the checks` (or `/jules` alone) there repairs its failed checks
– see [More work on a pull request](#-more-work-on-a-pull-request) and [Failed checks](#-failed-checks).

`/jules continue` (also `finish it`, `carry on`, `resume`) completes work that a run left
unfinished: on the `[Unfinished]` pull request, or on the issue a new project was started from.
When a task that GitHub started ends unfinished, its comment says so and names this command
– see [Unfinished work](#-unfinished-work).

To have failed checks of Jules' own pull requests fixed **unasked**, also subscribe the webhook to
**Check suite** (GitHub Actions and other check runs) and/or **Status** (commit statuses, which is how
Vercel reports a deployment). *Workflow run* is understood as well.

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

When a task fails, the dashboard shows the stage it failed at and the reason. *Settings → About*
(`GET /api/v1/health/env`) shows which settings the deployment actually sees, without their values,
and every task leaves one summary line in the function logs (Vercel → your deployment → Logs).

---

## 📡 API

| Endpoint | Description |
| --- | --- |
| `POST /api/v1/tasks/stream` | Run a task and stream progress as newline-delimited JSON (`task`, `log`, `update`, `ping`, `done`). Used by the dashboard. |
| `POST /api/v1/tasks` | Run a task in the background; poll `GET /api/v1/tasks/{id}`. |
| `GET /api/v1/tasks/{id}?repository=owner/name` | Task status. When the answering server instance does not know the task, its pull request is looked up on GitHub. For a task that added to a pull request, also pass `pull_request=<number>`: its commit is then looked for on that pull request. |
| `GET /api/v1/tasks/{id}/pull-request?repository=owner/name` | The task's pull request as it is on GitHub right now: `open`, `merged` or `closed`, whether it is still marked `unfinished`, its `head_sha` and, while it is open, its `checks` (`state`: `failure`, `pending`, `success`, `none` or `unknown`; `failing[]`; counts). `number=<n>` names the pull request for a task that added to an existing one. |
| `GET /api/v1/github/checks?repository=…&ref=…` | The checks of a commit or branch, in the same form. |
| `GET /api/v1/github/identity?login=…` | Name and `noreply` e-mail address of a GitHub account, for commit authorship. |
| `GET /api/v1/github/automation` | Whether failed checks can be fixed unasked: webhook secret, the App's event subscriptions and permissions, and where to change them. |
| `POST /api/v1/llm/test` | Try a provider out: `{"provider": "gemini", "model": "optional, one or several"}` with the key in its `X-<Provider>-Key` header (or the server's key without it). One small request per model; `models[]` says how each did and `answered_by` which model answered behind a router or an alias. |
| `GET /api/v1/github/repos` | Every repository the configured credentials can work on (all pages, up to 2,000), most recently pushed first, with `total`, `truncated` and, for a GitHub App, `installations` (account, `repository_selection`: `all` or `selected`, and where to change it). |
| `GET /api/v1/github/check?repository=…&branch=…` | Check, without running anything, that a task could act on a repository. `empty: true` means it has no commits and a task will create its first commit. |
| `GET /api/v1/health`, `GET /api/v1/health/env` | Liveness and configuration diagnostics. |
| `POST /api/v1/webhooks/github` | GitHub webhook receiver. |

Request body for both task endpoints (JSON, sent with `Content-Type: application/json`):

```jsonc
{
  "repository": "owner/name or GitHub URL",
  "prompt": "what to do",              // optional with fix_checks
  "branch": "optional base branch",
  "pull_request": 12,                   // optional: add the result to this pull request
  "fix_checks": true,                   // optional: the task is the failed checks (of the pull request, else of the branch)
  "continue_work": true,                // optional: carry on with work that was left unfinished
  "handoff": "…",                       // optional with continue_work: the `handoff` of the task being continued
  "guidelines": "…",                    // optional: standing instructions, given to the agent after the task
  "llm": {"primary": "gemini", "fallback": true, "models": {"gemini": "gemini-flash-latest, gemini-3.5-flash-lite"}},
  "commit": {"mode": "coauthor", "name": "Ada Example", "email": "1234567+ada-example@users.noreply.github.com"}
}
```

`llm` and `commit` are optional and apply to that task only. Provider keys for the task go in the
headers `X-Groq-Key`, `X-Gemini-Key`, `X-Nvidia-Key`, `X-Zai-Key`, `X-OpenRouter-Key`, `X-OpenAI-Key`,
`X-Anthropic-Key`.
Interactive documentation is served at `/api/docs`.

Fields of a task record that describe how far the result can be trusted:

| Field | Meaning |
| --- | --- |
| `tests_status` | `passed` or `failed` (the last run that really tested something), `no_tests` (test commands were run, but the project has no tests) or `not_run`. `tests_passed` is `true`/`false` only for the first two. |
| `verification[]` | Every `run_tests` call: `command`, `exit_code`, `duration`, `outcome` (`passed`, `failed`, `timed_out`, `no_tests`) and, for `no_tests`, a `note` with the reason. |
| `docs_only` | `true` when the change only touches documentation files. |
| `files_read` | Repository files whose content the agent saw before it finished. |
| `outcome` | `pull_request`, `pr_updated` (a commit was added to an existing pull request), `no_changes`, or `initial_commit` (the repository was empty: `commit_sha` and `commit_url` name its first commit and there is no pull request). |
| `unfinished`, `stop_reason`, `unfinished_reason` | The agent stopped before the work was done - it was interrupted (`timeout`, `llm_error`, `max_turns`) or reported that it could not complete the task (`gave_up`) - and what it had changed was delivered anyway; `unfinished_reason` says why in words. |
| `handoff` | For unfinished work: where it stands (the agent's account, the command that still fails with the end of its output, why it stopped). Send it back as `handoff` with `continue_work` to have the next run start from it. |
| `llm_chain[]`, `llm_usage[]` | The models the task could use, and what each did: `provider`, `model`, `requests`, `failures`, `input_tokens`, `output_tokens`, if it dropped out, why (`out`), and - for a router or an alias - `served`: the models that answered, with how often. |
| `failed_checks[]` | For a task that fixes checks: the checks that had failed when it started (`name`, `kind`, `url`). |
| `commit_authorship` | `bot`, `coauthor` or `user`: who the commit was attributed to. |
| `empty_repository`, `new_project` | The repository had no commits / no code when the task started. |

---

## ⚠️ Limits worth knowing

- **Scope of a task.** The time budget suits focused changes ("fix this failing test", "add input
  validation to X"), not large refactorings. One run is a few minutes (on Vercel's Hobby plan a
  function may run 300 seconds), and a model needs most of that to write a project of twenty or
  thirty files - before the first build has told it what is wrong. Such a project takes two or
  three runs: the first leaves an `[Unfinished]` commit, **Continue the work** (or `/jules continue`)
  finishes it. Asking for a first version and extending it with follow-up tasks works as well.
- **Tokens.** Every request to a model repeats the conversation so far. Files the agent wrote some
  turns ago are therefore replaced in the conversation by an outline of what they declare (they are
  in the sandbox, and it is told to read a file again before it changes it); without that a project
  of thirty files is sent thirty times over. Its latest reply always stays whole. Should a model
  copy such an outline into a file instead of the file's content, the write is refused.
- **Workflow files.** GitHub refuses files under `.github/workflows` from a GitHub App without the
  *Workflows* permission, which fails the commit. The agent does not add CI workflows unless asked.
- **Review before you merge.** The agent's summary and any documentation it writes are the model's
  words. The platform reports what it can observe (files read, commands run, test results); it does
  not verify that a description is accurate.
- **LLM rate limits.** Free tiers are small (Groq's free tier allows about 8,000 tokens per minute on
  the default model). The agent sizes its context to the provider's limit. With a single model it
  has to wait out a rate limit, and a wait that does not fit in the time budget ends the run; with
  a second model or a second provider it carries on with that one instead. Either way, what it had
  changed by then is kept (see [Unfinished work](#-unfinished-work)).
- **Free models differ.** A free model on OpenRouter may not support tool calls or may be
  withdrawn, which is why the default there is the router `openrouter/free` rather than one model.
  OpenRouter's free models share a daily allowance per account (50 requests a day at the time of
  writing, 1,000 once the account has bought credits; failed attempts count), and one larger task
  can use most of the small one. Use the **Test** button in Settings before relying on a model -
  it costs one request per model; `OPENROUTER_MODEL` (or the Models field in Settings) selects others.
- **Refused model replies.** Open models hosted by Groq sometimes answer with a call to a tool that
  does not exist (`commentary`, `json`, …), or with arguments that are not valid JSON; Groq refuses
  such a reply. Where the refused reply contains a usable tool call it is recovered and executed.
  Otherwise the agent tells the model what was wrong and asks again, up to three times; with a
  second provider configured, that one takes over after two refusals in a row. If the model keeps
  doing it and there is nothing to switch to, the task fails with that reason.
- **Automatic fixes are bounded.** Failed checks are fixed unasked only on pull requests Jules
  opened, at most twice per pull request by default. Each attempt is a full task and spends model
  and sandbox quota.
- **Sandbox contents.** The default E2B template has git, Python and Node.js but not your project's
  dependencies; the agent installs what the tests need, which costs time. Use `E2B_TEMPLATE` for a
  prepared image.
- **Task history** is kept in the server's memory and in your browser. It is not a database; the
  durable record of a task is its pull request (or first commit). Whether a pull request has been merged or closed
  since, and what its checks say, is read from GitHub when the dashboard loads, when you return to
  its tab, and for a while after a task finishes.
- **Settings are per browser.** Provider keys, model choices, standing instructions and the commit
  identity entered in the dashboard are kept in that browser's `localStorage`: another browser or
  device starts without them, and anyone with access to the browser profile can read them. Use
  environment variables for what should be shared or what tasks started from GitHub need.
- **Change size.** A task may change up to 300 files / 4 MB (1 MB per file). Dependency and cache
  directories (`node_modules`, `.venv`, `__pycache__`, `.next`, …) are never committed. If a
  finished task left a file that is too large (a bundle, a dump), the pull request is opened
  without it - together with any build output the run produced - and says so under *Changes*; a
  task with nothing else to deliver fails with that reason.

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
