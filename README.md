# Jules Clone Monorepo 🤖⚡

A production-ready, full-stack clone of an autonomous coding agent platform (Project Jules Clone). Architected as a unified monorepo engineered specifically for 1-click deployment on [Vercel](https://vercel.com/) with zero extra server configuration.

---

## 🏗 Architecture Overview

The monorepo uses unified routing where Next.js powers the interactive frontend and Python FastAPI acts as a high-performance serverless backend hosted under Vercel Serverless Functions:

```text
.
├── apps/
│   ├── web/                    # Next.js 14+ Frontend (App Router + Tailwind CSS)
│   │   ├── app/
│   │   │   ├── page.tsx        # Dashboard (Task Launcher, Terminal & PR Tracker)
│   │   │   ├── layout.tsx      # Dark-themed root layout
│   │   │   └── api/            # API Route Proxy & Handlers
│   │   ├── components/
│   │   │   ├── TaskForm.tsx    # Repo/Branch/Prompt submission
│   │   │   ├── LiveTerminal.tsx# Real-time streaming log terminal
│   │   │   └── PRStatusTracker.tsx # Status & Pull Request tracker
│   │   └── package.json
│   └── api/                    # Python 3.12 FastAPI Serverless Backend
│       ├── main.py             # FastAPI entrypoint (Vercel Serverless compatible)
│       ├── routers/
│       │   ├── webhooks.py     # GitHub Webhook HMAC verification endpoint
│       │   └── tasks.py        # Task creation, execution & log streaming
│       ├── services/
│       │   ├── sandbox.py      # E2B Sandbox microVM runtime manager
│       │   └── agent.py        # Autonomous tool-calling loop (Anthropic/OpenAI)
│       └── requirements.txt
├── package.json                # Root monorepo workspace dependencies
├── vercel.json                 # Unified routing: /api/* -> apps/api/main.py
├── .env.example                # Environment variables template
├── .gitignore                  # Clean repository ignore rules
└── README.md                   # Setup & deployment documentation
```

---

## 🚀 1-Click Vercel Deployment

Deploying the complete stack takes less than 3 minutes:

1. **Push or Fork this repository** to your GitHub account.
2. Go to [Vercel Dashboard](https://vercel.com/new) and click **"Add New Project"**.
3. Select this repository. Vercel automatically detects the Next.js frontend and maps the serverless Python backend via `vercel.json`.
4. Add the following **Environment Variables** in Vercel:
   - `E2B_API_KEY`: Get an API key from [e2b.dev](https://e2b.dev) (powers isolated microVM code sandboxes).
   - `ANTHROPIC_API_KEY` (or `OPENAI_API_KEY`): Powers autonomous agent reasoning and tool dispatch.
   - `GITHUB_WEBHOOK_SECRET`: A secret string you choose for validating webhook signatures from GitHub.
   - `GITHUB_TOKEN`: (Optional) GitHub Personal Access Token or App token with repo write permissions.
5. Click **Deploy**. Vercel will build both the Next.js app and the Python serverless runtime under a single unified domain!

---

## ⚙️ GitHub App / Webhook Setup

To trigger Jules automatically on repository issues, pull requests, or push events:

1. Go to your GitHub repository -> **Settings** -> **Webhooks** -> **Add webhook**.
2. **Payload URL**: `https://<your-vercel-domain>.vercel.app/api/v1/webhooks/github`
3. **Content type**: `application/json`
4. **Secret**: Enter the exact string configured for `GITHUB_WEBHOOK_SECRET`.
5. **Events**: Select `Issues`, `Issue comments`, and/or `Push`.
6. Click **Add webhook**. GitHub will send a `ping` event verified by HMAC SHA-256 (`X-Hub-Signature-256`).

---

## 🛠 Local Development

### Prerequisites
- Node.js 18+ (tested with Node 20 / 22)
- Python 3.11 or 3.12
- An [E2B API Key](https://e2b.dev) and Anthropic or OpenAI API Key

### 1. Setup Environment
```bash
cp .env.example .env
# Edit .env with your credentials
```

### 2. Install Dependencies
```bash
# Install root & web workspace dependencies
npm install

# Setup Python environment
python3 -m venv .venv
source .venv/bin/activate
pip install -r apps/api/requirements.txt
```

### 3. Run Locally
Run both frontend and backend concurrently:
```bash
# Terminal 1: Next.js Frontend (http://localhost:3000)
npm run dev:web

# Terminal 2: FastAPI Backend (http://localhost:8000)
npm run dev:api
```

---

## 🧩 Autonomous Agent Capabilities

The core agent (`apps/api/services/agent.py`) executes a goal-directed tool-use loop inside an isolated [E2B Sandbox](https://e2b.dev) microVM:

- 📂 `view_file`: Read source code safely inside the sandbox container.
- ✏️ `edit_file`: Apply modifications, search/replace patches, or rewrite files.
- 🧪 `run_tests`: Execute `pytest` or custom test runners and capture exit codes and failure logs.
- 📦 `git_commit`: Stage files and commit changes with conventional messages.
- 🚀 `open_pr`: Push branches and open Pull Requests back to GitHub.

---

## 🧪 Running Tests

```bash
# Test backend routes and agent tool logic
pytest apps/api
```
