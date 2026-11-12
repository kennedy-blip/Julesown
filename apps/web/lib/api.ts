/**
 * Client for the Jules API (FastAPI backend under /api).
 *
 * All URLs are relative: on Vercel the API is served from the same origin, and
 * in local development Next.js proxies /api/* to the backend (next.config.mjs).
 */
import { getSettings, providerKeyHeaders, taskOptions } from './settings';
import type { ProviderId } from './settings';

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';
/**
 * passed / failed - the last test run that really tested something
 * no_tests        - a test command was run, but the project has no tests
 * not_run         - no test command was run
 */
export type TestsStatus = 'passed' | 'failed' | 'no_tests' | 'not_run';
export type VerificationOutcome = 'passed' | 'failed' | 'timed_out' | 'no_tests';
/**
 * pull_request   - the change is on a branch and GitHub opened a pull request for it
 * pr_updated     - the change was added, as a commit, to a pull request that existed
 * initial_commit - the repository was empty: the result is its first commit
 * no_changes     - the agent finished without changing (or creating) any file
 */
export type TaskOutcome = 'pull_request' | 'pr_updated' | 'no_changes' | 'initial_commit';

/** What one model provider did during a run. */
export interface ProviderUsage {
  provider: string;
  model: string;
  requests?: number;
  failures?: number;
  input_tokens?: number;
  output_tokens?: number;
  /** Why the provider was not used again in the run, if it was taken out. */
  out?: string | null;
  /**
   * The models that answered, when the configured one is a router or an alias
   * that chooses for each request (openrouter/free, gemini-flash-latest): name -> answers.
   */
  served?: Record<string, number>;
}

export interface FailedCheck {
  name: string;
  kind: 'check_run' | 'status' | string;
  url?: string | null;
}

export interface LogEntry {
  ts: number; // unix seconds
  level: string; // info | stage | tool | edit | command | output | agent | success | warn | error
  message: string;
}

export interface ChangedFile {
  path: string;
  status: string; // A | M | D | T
  additions: number | null;
  deletions: number | null;
}

export interface VerificationRun {
  command: string;
  exit_code: number;
  passed: boolean;
  timed_out: boolean;
  duration: number;
  /** Not present in results saved by earlier versions of the app. */
  outcome?: VerificationOutcome;
  /** Why the command had nothing to test (when outcome is 'no_tests'). */
  note?: string;
}

export interface TaskRecord {
  task_id: string;
  repository: string;
  branch: string;
  prompt: string;
  status: TaskStatus;
  stage: string | null;
  outcome: TaskOutcome | null;
  pr_url: string | null;
  pr_number: number | null;
  head_branch: string | null;
  base_branch: string | null;
  commit_sha: string | null;
  /** The commit on GitHub when the result was committed directly (outcome 'initial_commit'). */
  commit_url?: string | null;
  repository_url?: string | null;
  /** The repository had no commits when the task started (null until that is known). */
  empty_repository?: boolean | null;
  /** The repository had no code: the agent built a project from scratch. */
  new_project?: boolean | null;
  changed_files: ChangedFile[];
  /** The change only touches documentation files (null until the changes are known). */
  docs_only?: boolean | null;
  /** Repository files the agent looked at (not present in results saved by earlier versions). */
  files_read?: string[];
  verification: VerificationRun[];
  tests_status: TestsStatus | null;
  summary: string | null;
  error: string | null;
  llm_provider: string | null;
  llm_model: string | null;
  /** Every provider configured for the run ("provider / model"), first choice first. */
  llm_chain?: string[];
  llm_usage?: ProviderUsage[];
  /** More work on a pull request that existed: its number. */
  pull_request?: number | null;
  fix_checks?: boolean;
  continue_work?: boolean;
  failed_checks?: FailedCheck[];
  /** The agent stopped before the task was done; what it had changed was delivered anyway. */
  unfinished?: boolean;
  /** timeout | llm_error | max_turns (it was interrupted) | gave_up (it reported that it cannot complete the task) */
  stop_reason?: string | null;
  /** Why the agent stopped, in words. */
  unfinished_reason?: string | null;
  /**
   * Where unfinished work stands: the agent's own account, the command that
   * still fails, why it stopped. Sent back with the task that continues it.
   */
  handoff?: string | null;
  /** bot | coauthor | user, as it was applied to the commit. */
  commit_authorship?: string | null;
  sandbox: string | null;
  recovered: boolean;
  created_at: number;
  started_at: number | null;
  completed_at: number | null;
  logs?: string[];
  log_entries?: LogEntry[];
}

/** What became of a pull request: still open, merged, or closed without a merge. */
export type PullRequestState = 'open' | 'merged' | 'closed';

/** failure | pending | success | none (no checks) | unknown (may not be read). */
export type ChecksState = 'failure' | 'pending' | 'success' | 'none' | 'unknown';

/** What CI says about a commit. */
export interface ChecksSummary {
  sha: string;
  state: ChecksState;
  failing: Array<FailedCheck & { conclusion?: string; summary?: string; app?: string }>;
  failed: number;
  passed: number;
  pending: number;
  notes: string[];
}

export interface PullRequestStatus {
  task_id: string;
  repository: string;
  number: number;
  url: string;
  state: PullRequestState;
  title: string;
  head_branch: string;
  base_branch: string;
  head_sha?: string;
  /** The pull request is still marked as unfinished work. */
  unfinished?: boolean;
  checks?: ChecksSummary | null;
}

export interface ProviderInfo {
  id: ProviderId;
  label: string;
  /** The server has a key for it in its environment. */
  configured: boolean;
  /** The model the server uses first unless told otherwise. */
  model: string;
  /** Every model the server gives the provider, in the order they are tried. */
  models?: string[];
  key_variable: string;
  model_variable: string;
}

export interface HealthEnv {
  status: string; // HEALTHY | INCOMPLETE
  ready: boolean;
  missing: string[];
  variables: Record<string, boolean>;
  llm: {
    provider: string;
    model: string;
    chain?: Array<{ provider: string; model: string }>;
    fallback?: boolean;
  } | null;
  /** How the server chooses among providers, whether or not it has a key for any. */
  llm_preferences?: { order: string[]; first_choice: string | null; fallback: boolean };
  providers?: ProviderInfo[];
  commits?: { authorship: string; author_configured: boolean; author_name: string | null };
  automation?: {
    ci_autofix: boolean;
    ci_autofix_attempts: number;
    keep_unfinished_work: boolean;
    /** The server adds instructions of its own to every task (JULES_TASK_GUIDELINES). */
    task_guidelines?: boolean;
  };
  /** What the server itself still lacks, by kind: a description, or null when it is there. */
  needs?: { llm: string | null; sandbox: string | null; github: string | null };
  github: { auth_mode: 'app' | 'token' | 'none' };
  webhooks?: { secret_configured: boolean };
  access: { mode: 'open' | 'key' | 'locked' };
  limits?: { task_budget_seconds: number; max_agent_turns: number };
  environment?: string;
  region?: string | null;
}

/** The answer of trying a provider key out. */
export interface ProviderTest {
  provider: string;
  model: string | null;
  ok: boolean;
  /** The model called the test tool - what the agent needs it to do. */
  tool_call: boolean;
  latency_ms: number | null;
  kind: string | null;
  message: string;
  key_source?: 'request' | 'server' | null;
  /** The model that answered, when the one asked for is a router or an alias. */
  answered_by?: string | null;
  /** With several models: what each of them did, in the order they are tried. */
  models?: ProviderModelTest[];
}

/** How one model of a provider did in the test. */
export interface ProviderModelTest {
  model: string;
  ok: boolean;
  tool_call: boolean;
  latency_ms: number | null;
  kind: string | null;
  message: string;
  answered_by?: string | null;
}

export interface GitHubIdentity {
  ok: boolean;
  login: string;
  name: string | null;
  email: string | null;
  avatar_url?: string | null;
  html_url?: string | null;
  error: string | null;
}

export interface AutomationStatus {
  auth_mode: string;
  ci_autofix: boolean;
  attempts: number;
  webhook_secret: boolean;
  webhook_path: string;
  app: { name?: string | null; slug?: string | null; settings_url?: string | null; permissions_url?: string | null } | null;
  events: Array<{ id: string; label: string; subscribed: boolean | null }>;
  permissions: Array<{ id: string; label: string; granted: boolean }>;
  ready: boolean;
  error: string | null;
}

export interface CommitChecks extends Partial<ChecksSummary> {
  ok: boolean;
  repository: string;
  error: string | null;
}

export interface RepoSummary {
  full_name: string;
  private: boolean;
  default_branch: string;
  html_url: string;
  archived?: boolean;
  fork?: boolean;
  description?: string | null;
  language?: string | null;
  /** ISO time of the last push; the list is ordered by it, newest first. */
  pushed_at?: string | null;
}

/** One account the GitHub App is installed on. */
export interface AppInstallation {
  id: number;
  account: string | null;
  account_type: string | null;
  /** 'all': every repository of the account. 'selected': only the ones picked on GitHub. */
  repository_selection: 'all' | 'selected' | string | null;
  /** Where the repositories the App can access are chosen. */
  manage_url: string | null;
  suspended: boolean;
  repository_count: number | null;
}

export interface RepoListing {
  auth_mode: 'app' | 'token' | 'none';
  /** Every repository the backend's GitHub credentials can work on. */
  repositories: RepoSummary[];
  /** How many GitHub reports; more than `repositories.length` only when `truncated`. */
  total?: number;
  truncated?: boolean;
  installations?: AppInstallation[];
  install_url: string | null;
  error: string | null;
}

export interface RepoCheck {
  ok: boolean;
  repository: string;
  default_branch: string | null;
  private: boolean | null;
  /** The repository has no commits: a task creates the project and its first commit. */
  empty?: boolean | null;
  auth_mode: string;
  code: string | null;
  error: string | null;
}

export type StreamEvent =
  | { type: 'task'; task: TaskRecord }
  | { type: 'log'; entry: LogEntry }
  | { type: 'update'; fields: Partial<TaskRecord> }
  | { type: 'ping'; ts: number }
  | { type: 'done'; task: TaskRecord };

export class ApiError extends Error {
  status: number;
  code: string;

  constructor(status: number, code: string, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.code = code;
  }
}

// --------------------------------------------------------------------------
// Access key (only needed when the deployment sets JULES_ACCESS_KEY)
// --------------------------------------------------------------------------
const ACCESS_KEY_STORAGE = 'jules_access_key';

export function getAccessKey(): string {
  try {
    return window.localStorage.getItem(ACCESS_KEY_STORAGE) || '';
  } catch {
    return '';
  }
}

export function setAccessKey(value: string): void {
  try {
    const trimmed = value.trim();
    if (trimmed) window.localStorage.setItem(ACCESS_KEY_STORAGE, trimmed);
    else window.localStorage.removeItem(ACCESS_KEY_STORAGE);
  } catch {
    /* storage unavailable (private mode): the key simply is not remembered */
  }
}

function authHeaders(): Record<string, string> {
  const key = getAccessKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

// --------------------------------------------------------------------------
// Errors
// --------------------------------------------------------------------------
function fallbackMessage(status: number): string {
  if (status === 404) {
    return 'The API endpoint was not found (HTTP 404). The deployed backend may be out of date.';
  }
  if (status === 504 || status === 502 || status === 503) {
    return `The server did not answer (HTTP ${status}). The function may have timed out or failed to start.`;
  }
  if (status === 413) return 'The request was too large (HTTP 413).';
  if (status === 429) return 'Too many requests (HTTP 429). Wait a moment and try again.';
  if (status === 500) {
    // No JSON body: the function crashed (Vercel) or the dev proxy could not reach the API.
    return (
      'The backend failed without an explanation (HTTP 500). On Vercel, see the function logs of ' +
      'the deployment; locally, check that the API is running (npm run dev:api).'
    );
  }
  return `The server returned HTTP ${status}.`;
}

async function toApiError(res: Response): Promise<ApiError> {
  let code = `http_${res.status}`;
  let message = '';
  try {
    const text = await res.text();
    try {
      const data = JSON.parse(text);
      const detail = data && data.detail;
      if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
        if (typeof detail.code === 'string') code = detail.code;
        if (typeof detail.message === 'string') message = detail.message;
      } else if (typeof detail === 'string' && detail !== 'Not Found') {
        message = detail;
      }
    } catch {
      /* not JSON: an HTML error page from the hosting platform */
    }
  } catch {
    /* body unreadable */
  }
  return new ApiError(res.status, code, message || fallbackMessage(res.status));
}

async function getJson<T>(url: string, withAuth: boolean): Promise<T> {
  const res = await fetch(url, {
    headers: { Accept: 'application/json', ...(withAuth ? authHeaders() : {}) },
    cache: 'no-store',
  });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as T;
}

/** A message suitable for showing to the user, whatever was thrown. */
export function describeError(error: unknown): string {
  if (error instanceof ApiError) return error.message;
  if (error instanceof Error) {
    if (error.name === 'TypeError') {
      return 'Could not reach the server. Check your connection and that the backend is running.';
    }
    return error.message;
  }
  return 'Unexpected error.';
}

// --------------------------------------------------------------------------
// Endpoints
// --------------------------------------------------------------------------
export function fetchHealth(): Promise<HealthEnv> {
  return getJson<HealthEnv>('/api/v1/health/env', false);
}

export function fetchRepositories(): Promise<RepoListing> {
  return getJson<RepoListing>('/api/v1/github/repos', true);
}

export function checkRepository(repository: string, branch: string): Promise<RepoCheck> {
  const params = new URLSearchParams({ repository });
  if (branch.trim()) params.set('branch', branch.trim());
  return getJson<RepoCheck>(`/api/v1/github/check?${params.toString()}`, true);
}

/**
 * Look a task up. When the server instance that answers does not know the task
 * (serverless instances do not share memory), the backend searches GitHub for
 * what the task left there, which is why the repository is passed: the pull
 * request it opened, the first commit it made or - for a task that added to an
 * existing pull request, whose number is then needed - the commit it added.
 */
export function fetchTask(taskId: string, repository: string, pullRequest?: number | null): Promise<TaskRecord> {
  const params = new URLSearchParams();
  if (repository) params.set('repository', repository);
  if (repository && pullRequest) params.set('pull_request', String(pullRequest));
  const query = params.toString();
  return getJson<TaskRecord>(
    `/api/v1/tasks/${encodeURIComponent(taskId)}${query ? `?${query}` : ''}`,
    true
  );
}

/**
 * The pull request a task opened or added to, as it is on GitHub right now -
 * its state and what CI says about its head commit. `number` is needed for a
 * task that added to an existing pull request (it has no branch of its own).
 */
export function fetchPullRequestStatus(
  taskId: string,
  repository: string,
  number?: number | null
): Promise<PullRequestStatus> {
  const params = new URLSearchParams({ repository });
  if (number) params.set('number', String(number));
  return getJson<PullRequestStatus>(
    `/api/v1/tasks/${encodeURIComponent(taskId)}/pull-request?${params.toString()}`,
    true
  );
}

/** What CI says about a commit or the head of a branch. */
export function fetchChecks(repository: string, ref: string): Promise<CommitChecks> {
  const params = new URLSearchParams({ repository, ref });
  return getJson<CommitChecks>(`/api/v1/github/checks?${params.toString()}`, true);
}

/** Name and commit address (noreply) of a GitHub account. */
export function fetchIdentity(login: string): Promise<GitHubIdentity> {
  const params = new URLSearchParams({ login });
  return getJson<GitHubIdentity>(`/api/v1/github/identity?${params.toString()}`, true);
}

export function fetchAutomation(): Promise<AutomationStatus> {
  return getJson<AutomationStatus>('/api/v1/github/automation', true);
}

/**
 * Try a provider out with one small request. The key typed into the dashboard
 * is sent along (only that provider's); without one the server's own is tried.
 */
export async function testProvider(provider: ProviderId, key: string, model: string): Promise<ProviderTest> {
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
    Accept: 'application/json',
    ...authHeaders(),
  };
  const trimmed = key.trim();
  if (trimmed && /^[\x21-\x7e]+$/.test(trimmed)) headers[`X-${provider}-Key`] = trimmed;
  const body: Record<string, string> = { provider };
  if (model.trim()) body.model = model.trim();
  const res = await fetch('/api/v1/llm/test', {
    method: 'POST',
    headers,
    body: JSON.stringify(body),
    cache: 'no-store',
  });
  if (!res.ok) throw await toApiError(res);
  return (await res.json()) as ProviderTest;
}

export interface TaskRequest {
  repository: string;
  branch: string; // '' = the repository's default branch
  prompt: string;
  /** Add the work to this open pull request instead of opening a new one. */
  pullRequest?: number | null;
  /** The task is to fix the failed checks (of the pull request, or of the branch). */
  fixChecks?: boolean;
  /** The task finishes work an earlier run left unfinished. */
  continueWork?: boolean;
  /** With continueWork: what the run that is continued reported when it stopped. */
  handoff?: string | null;
}

/**
 * Start a task and follow it. The task runs inside this request; progress
 * arrives as newline-delimited JSON events.
 *
 * Resolves `true` when the final `done` event was received and `false` when
 * the connection ended first (the caller should then look the task up).
 * Rejects with ApiError when the server refuses the request.
 */
export async function streamTask(
  request: TaskRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal
): Promise<boolean> {
  const settings = getSettings();
  const body: Record<string, unknown> = {
    repository: request.repository,
    prompt: request.prompt,
    // Which provider to start with and how the commit is attributed.
    ...taskOptions(settings),
  };
  if (request.pullRequest) body.pull_request = request.pullRequest;
  else if (request.branch.trim()) body.branch = request.branch.trim();
  if (request.fixChecks) body.fix_checks = true;
  if (request.continueWork) {
    body.continue_work = true;
    // The server keeps nothing between two runs: where the first one stopped travels with the second.
    const handoff = (request.handoff || '').trim();
    if (handoff) body.handoff = handoff.slice(0, 8000);
  }

  const res = await fetch('/api/v1/tasks/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/x-ndjson',
      ...authHeaders(),
      // Provider keys kept in this browser: used by the backend for this task only.
      ...providerKeyHeaders(settings),
    },
    body: JSON.stringify(body),
    cache: 'no-store',
    signal,
  });
  if (!res.ok) throw await toApiError(res);

  let finished = false;
  const handleLine = (line: string) => {
    const trimmed = line.trim();
    if (!trimmed) return;
    let event: StreamEvent;
    try {
      event = JSON.parse(trimmed) as StreamEvent;
    } catch {
      return; // a truncated line from an interrupted connection
    }
    if (!event || typeof event !== 'object' || typeof (event as { type?: unknown }).type !== 'string') return;
    if (event.type === 'done') finished = true;
    onEvent(event);
  };

  if (!res.body) {
    // Very old browsers: no streaming, read everything at the end.
    (await res.text()).split('\n').forEach(handleLine);
    return finished;
  }

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      let newline = buffer.indexOf('\n');
      while (newline >= 0) {
        handleLine(buffer.slice(0, newline));
        buffer = buffer.slice(newline + 1);
        newline = buffer.indexOf('\n');
      }
    }
    buffer += decoder.decode();
    handleLine(buffer);
  } catch (error) {
    // The connection dropped mid-run. Events received so far were delivered.
    if (signal && signal.aborted) throw error;
    return finished;
  }
  return finished;
}

// --------------------------------------------------------------------------
// Helpers
// --------------------------------------------------------------------------
const PULL_REQUEST_URL = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/pull\/\d+$/;

/**
 * The pull request link to show for a task, or null.
 *
 * A link is only ever shown when the backend reports a pull request that
 * GitHub created (`.../pull/<number>`). Nothing is guessed or constructed here.
 */
export function pullRequestUrl(task: Pick<TaskRecord, 'pr_url'> | null | undefined): string | null {
  const url = task && task.pr_url;
  return typeof url === 'string' && PULL_REQUEST_URL.test(url) ? url : null;
}

const COMMIT_URL = /^https:\/\/github\.com\/[\w.-]+\/[\w.-]+\/commit\/[0-9a-f]{40}$/;

/** The link to the commit a task added to an existing pull request, or null. */
export function addedCommitUrl(
  task: Pick<TaskRecord, 'outcome' | 'commit_url'> | null | undefined
): string | null {
  if (!task || task.outcome !== 'pr_updated') return null;
  const url = task.commit_url;
  return typeof url === 'string' && COMMIT_URL.test(url) ? url : null;
}

/**
 * The link to the first commit of a repository that was empty, or null.
 * Like pull request links, it is only shown when it is the URL GitHub
 * returned for the commit it created.
 */
export function firstCommitUrl(
  task: Pick<TaskRecord, 'outcome' | 'commit_url'> | null | undefined
): string | null {
  if (!task || task.outcome !== 'initial_commit') return null;
  const url = task.commit_url;
  return typeof url === 'string' && COMMIT_URL.test(url) ? url : null;
}

const REPOSITORY_SLUG = /^[\w.-]+\/[\w.-]+$/;

/** Best-effort owner/name from what the user typed (the server validates for real). */
export function normalizeRepository(raw: string): string {
  let value = raw.trim();
  value = value.replace(/^git@github\.com:/i, '');
  value = value.replace(/^(https?:\/\/)?(www\.)?github\.com\//i, '');
  value = value.split(/[?#]/)[0];
  const parts = value.split('/').filter(Boolean);
  if (parts.length >= 2) {
    value = `${parts[0]}/${parts[1].replace(/\.git$/i, '')}`;
  }
  return value;
}

export function looksLikeRepository(raw: string): boolean {
  return REPOSITORY_SLUG.test(normalizeRepository(raw));
}
