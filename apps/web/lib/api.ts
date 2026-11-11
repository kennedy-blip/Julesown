/**
 * Client for the Jules API (FastAPI backend under /api).
 *
 * All URLs are relative: on Vercel the API is served from the same origin, and
 * in local development Next.js proxies /api/* to the backend (next.config.mjs).
 */

export type TaskStatus = 'pending' | 'running' | 'completed' | 'failed';
/**
 * passed / failed - the last test run that really tested something
 * no_tests        - a test command was run, but the project has no tests
 * not_run         - no test command was run
 */
export type TestsStatus = 'passed' | 'failed' | 'no_tests' | 'not_run';
export type VerificationOutcome = 'passed' | 'failed' | 'timed_out' | 'no_tests';
export type TaskOutcome = 'pull_request' | 'no_changes';

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

export interface PullRequestStatus {
  task_id: string;
  repository: string;
  number: number;
  url: string;
  state: PullRequestState;
  title: string;
  head_branch: string;
  base_branch: string;
}

export interface HealthEnv {
  status: string; // HEALTHY | INCOMPLETE
  ready: boolean;
  missing: string[];
  variables: Record<string, boolean>;
  llm: { provider: string; model: string } | null;
  github: { auth_mode: 'app' | 'token' | 'none' };
  access: { mode: 'open' | 'key' | 'locked' };
  limits?: { task_budget_seconds: number; max_agent_turns: number };
  environment?: string;
  region?: string | null;
}

export interface RepoSummary {
  full_name: string;
  private: boolean;
  default_branch: string;
  html_url: string;
}

export interface RepoListing {
  auth_mode: 'app' | 'token' | 'none';
  repositories: RepoSummary[];
  install_url: string | null;
  error: string | null;
}

export interface RepoCheck {
  ok: boolean;
  repository: string;
  default_branch: string | null;
  private: boolean | null;
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
 * the pull request the task opened, which is why the repository is passed.
 */
export function fetchTask(taskId: string, repository: string): Promise<TaskRecord> {
  const params = new URLSearchParams();
  if (repository) params.set('repository', repository);
  const query = params.toString();
  return getJson<TaskRecord>(
    `/api/v1/tasks/${encodeURIComponent(taskId)}${query ? `?${query}` : ''}`,
    true
  );
}

/** The pull request a task opened, as it is on GitHub right now. */
export function fetchPullRequestStatus(taskId: string, repository: string): Promise<PullRequestStatus> {
  const params = new URLSearchParams({ repository });
  return getJson<PullRequestStatus>(
    `/api/v1/tasks/${encodeURIComponent(taskId)}/pull-request?${params.toString()}`,
    true
  );
}

export interface TaskRequest {
  repository: string;
  branch: string; // '' = the repository's default branch
  prompt: string;
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
  const body: Record<string, string> = { repository: request.repository, prompt: request.prompt };
  if (request.branch.trim()) body.branch = request.branch.trim();

  const res = await fetch('/api/v1/tasks/stream', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      Accept: 'application/x-ndjson',
      ...authHeaders(),
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
