import type { LogEntry, TaskRecord } from './api';
import { pullRequestUrl } from './api';

/**
 * running      - this tab is receiving the task's live stream
 * completed    - the server reported the task finished (pull request or no changes)
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
  createdAt: number; // ms
  updatedAt: number; // ms
}

export type SessionBadge = 'RUNNING' | 'PR OPENED' | 'NO CHANGES' | 'FAILED' | 'UNKNOWN';

export function sessionBadge(session: Session): SessionBadge {
  if (session.state === 'running') return 'RUNNING';
  if (session.state === 'failed') return 'FAILED';
  if (session.state === 'interrupted') return 'UNKNOWN';
  if (pullRequestUrl(session.task)) return 'PR OPENED';
  return 'NO CHANGES';
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
