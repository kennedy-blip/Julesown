import type { ChecksSummary, LogEntry, PullRequestState, TaskRecord } from './api';
import { firstCommitUrl, pullRequestUrl } from './api';

/**
 * running      - this tab is receiving the task's live stream
 * completed    - the server reported the task finished (pull request, first commit or no changes)
 * failed       - the server reported a failure, or refused the request
 * interrupted  - the live connection ended before a result arrived; the outcome is unknown
 */
export type SessionState = 'running' | 'completed' | 'failed' | 'interrupted';

export interface Session {
  id: string;
  repository: string;
  branch: string; // as requested; '' = default branch
  prompt: string;
  state: SessionState;
  task: TaskRecord | null;
  logs: LogEntry[];
  /** Client-side explanation (request refused, connection lost, ...). */
  note: string | null;
  /**
   * What became of the pull request after the task opened it, as last read
   * from GitHub: its state, whether it is still marked unfinished and what CI
   * says about its head. Absent until it has been looked up.
   */
  pullRequest?: {
    state: PullRequestState;
    checkedAt: number;
    unfinished?: boolean;
    checks?: ChecksSummary | null;
  } | null;
  /** What CI says about the first commit of a repository that was empty. */
  commitChecks?: { checks: ChecksSummary; checkedAt: number } | null;
  /** More work on an existing pull request: what was asked for. */
  target?: {
    pullRequest: number | null;
    fixChecks: boolean;
    continueWork: boolean;
    /** What the run that this one continues reported when it stopped (sent again by a retry). */
    handoff?: string | null;
  } | null;
  createdAt: number; // ms
  updatedAt: number; // ms
}

export type SessionBadge =
  | 'RUNNING'
  | 'PR OPENED'
  | 'PR UPDATED'
  | 'UNFINISHED'
  | 'MERGED'
  | 'PR CLOSED'
  | 'PUSHED'
  | 'NO CHANGES'
  | 'FAILED'
  | 'UNKNOWN';

/** The pull request's state on GitHub if it has been looked up, else null. */
export function pullRequestState(session: Session): PullRequestState | null {
  if (session.state !== 'completed' || !pullRequestUrl(session.task)) return null;
  return session.pullRequest ? session.pullRequest.state : null;
}

/**
 * Whether the result of this session is unfinished work: the agent was cut
 * short and what it had was delivered. For a pull request GitHub has the last
 * word - a follow-up that finished it takes the mark off.
 */
export function isUnfinished(session: Session): boolean {
  if (session.state !== 'completed' || !session.task) return false;
  if (pullRequestUrl(session.task) && session.pullRequest && typeof session.pullRequest.unfinished === 'boolean') {
    return session.pullRequest.unfinished;
  }
  return session.task.unfinished === true;
}

export function sessionBadge(session: Session): SessionBadge {
  if (session.state === 'running') return 'RUNNING';
  if (session.state === 'failed') return 'FAILED';
  if (session.state === 'interrupted') return 'UNKNOWN';
  const outcome = session.task ? session.task.outcome : null;
  if (outcome === 'no_changes') return 'NO CHANGES';
  if (pullRequestUrl(session.task)) {
    const state = pullRequestState(session);
    if (state === 'merged') return 'MERGED';
    if (state === 'closed') return 'PR CLOSED';
    if (isUnfinished(session)) return 'UNFINISHED';
    return outcome === 'pr_updated' ? 'PR UPDATED' : 'PR OPENED';
  }
  // The repository was empty: the result is its first commit, on the default branch.
  if (firstCommitUrl(session.task)) return isUnfinished(session) ? 'UNFINISHED' : 'PUSHED';
  return 'NO CHANGES';
}

/**
 * Whether the pull request of this session is worth asking GitHub about:
 * there is one, and it was not already seen merged (which is final).
 */
export function pullRequestCanChange(session: Session): boolean {
  if (session.state !== 'completed' || !pullRequestUrl(session.task)) return false;
  if (!session.task || !session.task.task_id) return false;
  // A follow-up that changed nothing names the pull request it was about, not a result.
  if (session.task.outcome === 'no_changes') return false;
  return pullRequestState(session) !== 'merged';
}

/** Whether the result is the first commit of a repository that was empty (its checks can be asked for). */
export function isFirstCommit(session: Session): boolean {
  return session.state === 'completed' && !!firstCommitUrl(session.task) && !!session.task && !!session.task.commit_sha;
}

// --------------------------------------------------------------------------
// Persistence
// --------------------------------------------------------------------------
const STORAGE_KEY = 'jules_sessions_v2';
// Earlier versions stored sessions under this key, including pull request links
// that were made up by the app rather than returned by GitHub. It is discarded.
const LEGACY_STORAGE_KEY = 'jules_sessions';
const MAX_SESSIONS = 30;
const MAX_STORED_LOGS = 400;

function isLogEntry(value: unknown): value is LogEntry {
  if (!value || typeof value !== 'object') return false;
  const entry = value as Record<string, unknown>;
  return typeof entry.ts === 'number' && typeof entry.level === 'string' && typeof entry.message === 'string';
}

function reviveChecks(value: unknown): ChecksSummary | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const states = ['failure', 'pending', 'success', 'none', 'unknown'];
  if (typeof raw.state !== 'string' || states.indexOf(raw.state) < 0) return null;
  const count = (item: unknown) => (typeof item === 'number' ? item : 0);
  return {
    sha: typeof raw.sha === 'string' ? raw.sha : '',
    state: raw.state as ChecksSummary['state'],
    failing: Array.isArray(raw.failing)
      ? raw.failing.filter((item) => item && typeof item === 'object' && typeof (item as { name?: unknown }).name === 'string')
      : [],
    failed: count(raw.failed),
    passed: count(raw.passed),
    pending: count(raw.pending),
    notes: Array.isArray(raw.notes) ? raw.notes.filter((item) => typeof item === 'string') : [],
  };
}

function revivePullRequest(value: unknown): Session['pullRequest'] {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const states: PullRequestState[] = ['open', 'merged', 'closed'];
  if (states.indexOf(raw.state as PullRequestState) < 0) return null;
  return {
    state: raw.state as PullRequestState,
    checkedAt: typeof raw.checkedAt === 'number' ? raw.checkedAt : 0,
    unfinished: typeof raw.unfinished === 'boolean' ? raw.unfinished : undefined,
    checks: reviveChecks(raw.checks),
  };
}

function reviveTarget(value: unknown): Session['target'] {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  return {
    pullRequest: typeof raw.pullRequest === 'number' && raw.pullRequest > 0 ? raw.pullRequest : null,
    fixChecks: raw.fixChecks === true,
    continueWork: raw.continueWork === true,
    handoff: typeof raw.handoff === 'string' && raw.handoff.trim() ? raw.handoff.slice(0, 8000) : null,
  };
}

/**
 * Where unfinished work stands, for the task that continues it: the handoff of
 * the latest run on the same work. For a pull request that is the newest run
 * that added to it and stopped early; an earlier one describes a state the
 * pull request has left behind.
 */
export function handoffFor(session: Session, sessions: Session[]): string | null {
  const own = session.task && typeof session.task.handoff === 'string' && session.task.handoff.trim() ? session.task.handoff : null;
  const task = session.task;
  if (!task || !task.pr_number || task.outcome === 'initial_commit') return own;
  const repository = (task.repository || session.repository).toLowerCase();
  // Newest first, as the history is kept.
  const latest = sessions.find(
    (item) =>
      !!item.task &&
      (item.task.repository || item.repository).toLowerCase() === repository &&
      item.task.pr_number === task.pr_number &&
      item.task.outcome !== 'no_changes' &&
      item.state === 'completed'
  );
  if (latest && latest.task && latest.task.unfinished && typeof latest.task.handoff === 'string' && latest.task.handoff.trim()) {
    return latest.task.handoff;
  }
  return latest && latest.id !== session.id ? null : own;
}

function reviveCommitChecks(value: unknown): Session['commitChecks'] {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  const checks = reviveChecks(raw.checks);
  return checks ? { checks, checkedAt: typeof raw.checkedAt === 'number' ? raw.checkedAt : 0 } : null;
}

function reviveSession(value: unknown): Session | null {
  if (!value || typeof value !== 'object') return null;
  const raw = value as Record<string, unknown>;
  if (typeof raw.id !== 'string' || typeof raw.repository !== 'string') return null;
  const states: SessionState[] = ['running', 'completed', 'failed', 'interrupted'];
  const state = states.indexOf(raw.state as SessionState) >= 0 ? (raw.state as SessionState) : 'interrupted';
  const task = raw.task && typeof raw.task === 'object' ? (raw.task as TaskRecord) : null;
  return {
    id: raw.id,
    repository: raw.repository,
    branch: typeof raw.branch === 'string' ? raw.branch : '',
    prompt: typeof raw.prompt === 'string' ? raw.prompt : '',
    // A stream cannot survive a page reload: what was "running" is now unknown
    // until it has been looked up again.
    state: state === 'running' ? 'interrupted' : state,
    task,
    logs: Array.isArray(raw.logs) ? raw.logs.filter(isLogEntry) : [],
    note: typeof raw.note === 'string' ? raw.note : null,
    pullRequest: revivePullRequest(raw.pullRequest),
    commitChecks: reviveCommitChecks(raw.commitChecks),
    target: reviveTarget(raw.target),
    createdAt: typeof raw.createdAt === 'number' ? raw.createdAt : Date.now(),
    updatedAt: typeof raw.updatedAt === 'number' ? raw.updatedAt : Date.now(),
  };
}

export function loadSessions(): Session[] {
  try {
    window.localStorage.removeItem(LEGACY_STORAGE_KEY);
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (!saved) return [];
    const parsed: unknown = JSON.parse(saved);
    if (!Array.isArray(parsed)) return [];
    const sessions: Session[] = [];
    parsed.forEach((item) => {
      const session = reviveSession(item);
      if (session) sessions.push(session);
    });
    return sessions.slice(0, MAX_SESSIONS);
  } catch {
    return [];
  }
}

export function saveSessions(sessions: Session[]): void {
  try {
    const compact = sessions.slice(0, MAX_SESSIONS).map((session) => {
      const task = session.task ? { ...session.task } : null;
      if (task) {
        // The session keeps its own copy of the log; do not store it twice.
        delete task.logs;
        delete task.log_entries;
      }
      return { ...session, task, logs: session.logs.slice(-MAX_STORED_LOGS) };
    });
    if (compact.length === 0) window.localStorage.removeItem(STORAGE_KEY);
    else window.localStorage.setItem(STORAGE_KEY, JSON.stringify(compact));
  } catch {
    /* storage full or unavailable: history is simply not persisted */
  }
}

export function newSessionId(): string {
  return `sess-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}
