'use client';

import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Plus, FolderGit2, KeyRound, RefreshCw, Trash2, AlertTriangle, CheckCircle2, X, ExternalLink } from 'lucide-react';
import { TaskForm } from '@/components/TaskForm';
import type { RepoCheckState, TaskFormValues } from '@/components/TaskForm';
import { LiveTerminal } from '@/components/LiveTerminal';
import { PRStatusTracker } from '@/components/PRStatusTracker';
import {
  ApiError,
  checkRepository,
  describeError,
  fetchPullRequestStatus,
  fetchRepositories,
  fetchTask,
  getAccessKey,
  looksLikeRepository,
  normalizeRepository,
  setAccessKey,
  streamTask,
} from '@/lib/api';
import type { LogEntry, RepoListing, StreamEvent, TaskRecord } from '@/lib/api';
import { readinessOf, refreshHealth, useHealth } from '@/lib/health';
import { loadSessions, newSessionId, pullRequestCanChange, saveSessions, sessionBadge } from '@/lib/sessions';
import type { Session, SessionBadge } from '@/lib/sessions';

const MAX_SESSIONS = 30;
/** How long after a lost connection the task is still looked up automatically
 *  (a little longer than the longest run the backend allows). */
const AUTO_RECHECK_WINDOW_MS = 7 * 60 * 1000;
const AUTO_RECHECK_INTERVAL_MS = 15 * 1000;
/** A pull request's state is asked from GitHub at most this often per task... */
const PR_STATE_MIN_INTERVAL_MS = 20 * 1000;
/** ...and when the page loads only for this many of the most recent tasks. */
const PR_STATE_STARTUP_LIMIT = 8;

const GITHUB_APP_NOTICES: Record<string, string> = {
  installed:
    'GitHub App installation saved. Jules can now open pull requests on the repositories you selected.',
  requested:
    'Installation requested. An owner of that organization has to approve it before Jules can access its repositories.',
  authorized: 'GitHub authorization complete.',
};

function clientLog(level: string, message: string): LogEntry {
  return { ts: Date.now() / 1000, level, message };
}

/** Log lines written by this page (not the server), kept when the server's log replaces the rest. */
function clientLines(logs: LogEntry[]): LogEntry[] {
  return logs.filter((entry) => entry.level === 'client');
}

function applyStreamEvent(session: Session, event: StreamEvent): Session {
  const updatedAt = Date.now();
  switch (event.type) {
    case 'task':
      return { ...session, task: event.task, repository: event.task.repository || session.repository, updatedAt };
    case 'log':
      return { ...session, logs: session.logs.concat([event.entry]), updatedAt };
    case 'update':
      return session.task ? { ...session, task: { ...session.task, ...event.fields }, updatedAt } : session;
    case 'done': {
      const task = event.task;
      const serverLog = Array.isArray(task.log_entries) ? task.log_entries : [];
      return {
        ...session,
        task,
        logs: serverLog.length > 0 ? clientLines(session.logs).concat(serverLog) : session.logs,
        state: task.status === 'completed' ? 'completed' : 'failed',
        note: null,
        updatedAt,
      };
    }
    default:
      return session;
  }
}

/** Merge the answer of a task lookup into a session whose live connection was lost. */
function applyLookup(session: Session, found: TaskRecord): Session {
  let task: TaskRecord = found;
  let logs: LogEntry[];
  if (found.recovered) {
    // Looked up on GitHub: only the pull request is known. Keep what this
    // browser already learned from the stream and add the pull request to it.
    const known = session.task || found;
    task = {
      ...known,
      status: found.status,
      stage: found.stage,
      outcome: found.outcome,
      pr_url: found.pr_url,
      pr_number: found.pr_number,
      head_branch: found.head_branch || known.head_branch,
      base_branch: found.base_branch || known.base_branch,
      summary: known.summary || found.summary,
      prompt: known.prompt || session.prompt,
      recovered: true,
    };
    logs = session.logs.concat(
      (found.log_entries || []).map((entry) => ({ ...entry, level: entry.level || 'success' }))
    );
  } else {
    const serverLog = Array.isArray(found.log_entries) ? found.log_entries : [];
    logs = serverLog.length > 0 ? clientLines(session.logs).concat(serverLog) : session.logs;
  }

  const finished = task.status === 'completed' || task.status === 'failed';
  return {
    ...session,
    task,
    logs,
    state: finished ? (task.status === 'completed' ? 'completed' : 'failed') : 'interrupted',
    note: finished ? null : 'The task is still running on the server. Its status is checked again automatically.',
    updatedAt: Date.now(),
  };
}

const BADGE_STYLES: Record<SessionBadge, string> = {
  RUNNING: 'bg-amber-500/10 text-amber-400 border border-amber-500/20',
  'PR OPENED': 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20',
  MERGED: 'bg-purple-500/10 text-purple-300 border border-purple-500/30',
  'PR CLOSED': 'bg-slate-700/40 text-slate-300 border border-slate-600/50',
  'NO CHANGES': 'bg-slate-700/40 text-slate-300 border border-slate-600/50',
  FAILED: 'bg-rose-500/10 text-rose-400 border border-rose-500/20',
  UNKNOWN: 'bg-amber-500/10 text-amber-300 border border-amber-500/20',
};

function formatWhen(ms: number): string {
  const date = new Date(ms);
  const sameDay = new Date().toDateString() === date.toDateString();
  const time = date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
  return sameDay ? time : `${date.toLocaleDateString([], { month: 'short', day: 'numeric' })} ${time}`;
}

export default function Dashboard() {
  const health = useHealth();
  const readiness = readinessOf(health);
  const healthData = health.phase === 'ok' ? health.data : null;
  const accessMode = healthData && healthData.access ? healthData.access.mode : 'open';
  const githubAuthMode = healthData && healthData.github ? healthData.github.auth_mode : 'none';

  const [sessions, setSessions] = useState<Session[]>([]);
  const [storageLoaded, setStorageLoaded] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [form, setForm] = useState<TaskFormValues>({ repository: '', branch: '', prompt: '' });
  const [accessKey, setAccessKeyInput] = useState('');
  const [accessProblem, setAccessProblem] = useState<string | null>(null);
  const [listing, setListing] = useState<RepoListing | null>(null);
  const [listingError, setListingError] = useState<string | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [repoCheck, setRepoCheck] = useState<RepoCheckState>({ phase: 'idle' });
  const [notice, setNotice] = useState<string | null>(null);
  const [showDiagnostics, setShowDiagnostics] = useState(false);
  const [recheckingId, setRecheckingId] = useState<string | null>(null);

  const sessionsRef = useRef<Session[]>([]);
  const launching = useRef(false);
  const autoRecheckBusy = useRef(false);
  // Pull request lookups: when each session was last asked about, and the queue
  // that keeps them one at a time.
  const prStateAsked = useRef<Record<string, number>>({});
  const prStateQueue = useRef<Promise<void>>(Promise.resolve());

  const updateSessions = useCallback((change: (previous: Session[]) => Session[]) => {
    setSessions((previous) => {
      const next = change(previous);
      sessionsRef.current = next;
      return next;
    });
  }, []);

  const patchSession = useCallback(
    (id: string, change: (session: Session) => Session) => {
      updateSessions((previous) => previous.map((session) => (session.id === id ? change(session) : session)));
    },
    [updateSessions]
  );

  // ------------------------------------------------------------------
  // Startup: history, access key, GitHub App landing
  // ------------------------------------------------------------------
  useEffect(() => {
    const stored = loadSessions();
    updateSessions(() => stored);
    if (stored.length > 0) {
      setActiveId(stored[0].id);
      setForm((current) => (current.repository ? current : { ...current, repository: stored[0].repository }));
    }
    setAccessKeyInput(getAccessKey());
    setStorageLoaded(true);

    // GitHub sends the browser back here after the App is installed or authorised.
    const params = new URLSearchParams(window.location.search);
    const landing = params.get('github_app');
    if (landing) {
      setNotice(GITHUB_APP_NOTICES[landing] || null);
      window.history.replaceState(null, '', window.location.pathname);
    }
  }, [updateSessions]);

  // Persist history (debounced; log lines can arrive many times per second).
  useEffect(() => {
    if (!storageLoaded) return;
    const timer = window.setTimeout(() => saveSessions(sessions), 300);
    return () => window.clearTimeout(timer);
  }, [sessions, storageLoaded]);

  useEffect(() => {
    if (!storageLoaded) return;
    const flush = () => saveSessions(sessionsRef.current);
    window.addEventListener('pagehide', flush);
    return () => window.removeEventListener('pagehide', flush);
  }, [storageLoaded]);

  // ------------------------------------------------------------------
  // Repositories the backend's GitHub credentials can act on
  // ------------------------------------------------------------------
  const canCallApi = health.phase === 'ok' && accessMode !== 'locked' && (accessMode !== 'key' || accessKey.trim() !== '');

  const loadRepositories = useCallback(async () => {
    setListingLoading(true);
    try {
      const data = await fetchRepositories();
      setListing(data);
      setListingError(data.error);
      setAccessProblem(null);
    } catch (error) {
      setListing(null);
      if (error instanceof ApiError && error.code === 'access_key_required') {
        setListingError(null);
        setAccessProblem(getAccessKey() ? 'That access key was not accepted.' : null);
      } else if (error instanceof ApiError && error.code === 'access_locked') {
        setListingError(null);
        void refreshHealth();
      } else {
        setListingError(describeError(error));
      }
    } finally {
      setListingLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!canCallApi) {
      setListing(null);
      return;
    }
    // Debounced so that typing an access key does not send a request per keystroke.
    const timer = window.setTimeout(() => void loadRepositories(), accessMode === 'key' ? 500 : 0);
    return () => window.clearTimeout(timer);
  }, [canCallApi, accessMode, accessKey, loadRepositories]);

  const repositories = listing ? listing.repositories : [];
  const firstRepository = repositories.length > 0 ? repositories[0].full_name : '';
  useEffect(() => {
    if (firstRepository) {
      setForm((current) => (current.repository.trim() ? current : { ...current, repository: firstRepository }));
    }
  }, [firstRepository]);

  // ------------------------------------------------------------------
  // Pre-flight check of the repository in the form
  // ------------------------------------------------------------------
  useEffect(() => {
    if (!canCallApi || githubAuthMode === 'none' || !looksLikeRepository(form.repository)) {
      setRepoCheck({ phase: 'idle' });
      return;
    }
    let cancelled = false;
    setRepoCheck({ phase: 'checking' });
    const timer = window.setTimeout(async () => {
      try {
        const result = await checkRepository(normalizeRepository(form.repository), form.branch);
        if (cancelled) return;
        if (result.ok) {
          setRepoCheck({
            phase: 'ok',
            repository: result.repository,
            defaultBranch: result.default_branch,
            isPrivate: result.private,
            authMode: result.auth_mode,
          });
        } else {
          setRepoCheck({ phase: 'error', message: result.error || 'This repository cannot be used.' });
        }
      } catch (error) {
        if (cancelled) return;
        if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          setRepoCheck({ phase: 'idle' }); // the access banner already explains this
        } else {
          setRepoCheck({ phase: 'error', message: describeError(error) });
        }
      }
    }, 600);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [canCallApi, githubAuthMode, form.repository, form.branch, accessKey]);

  // ------------------------------------------------------------------
  // Looking up a task whose live connection was lost
  // ------------------------------------------------------------------
  const recheck = useCallback(
    async (id: string) => {
      const session = sessionsRef.current.find((item) => item.id === id);
      if (!session || session.state !== 'interrupted') return;
      const taskId = session.task ? session.task.task_id : '';
      if (!taskId) {
        patchSession(id, (item) => ({
          ...item,
          state: 'failed',
          note: 'The connection ended before the server confirmed the task, so nothing was started that could be looked up. Run the task again.',
          updatedAt: Date.now(),
        }));
        return;
      }
      setRecheckingId(id);
      try {
        const found = await fetchTask(taskId, (session.task && session.task.repository) || session.repository);
        patchSession(id, (item) => (item.state === 'interrupted' ? applyLookup(item, found) : item));
      } catch (error) {
        const recent = Date.now() - session.createdAt < AUTO_RECHECK_WINDOW_MS;
        let note: string;
        if (error instanceof ApiError && error.status === 404) {
          note = recent
            ? 'No pull request from this task exists on GitHub yet. The run may still be in progress on the server, so this is checked again automatically for a few minutes.'
            : 'No pull request from this task exists on GitHub. The run did not get as far as opening one (it may have been stopped by the hosting time limit).';
        } else if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
          note = error.message;
        } else {
          note = `The task could not be looked up: ${describeError(error)}`;
        }
        patchSession(id, (item) => (item.state === 'interrupted' ? { ...item, note, updatedAt: Date.now() } : item));
      } finally {
        setRecheckingId((current) => (current === id ? null : current));
      }
    },
    [patchSession]
  );

  useEffect(() => {
    if (!storageLoaded || !canCallApi) return;
    const tick = () => {
      if (autoRecheckBusy.current) return;
      const candidate = sessionsRef.current.find(
        (session) =>
          session.state === 'interrupted' &&
          !!session.task &&
          !!session.task.task_id &&
          Date.now() - session.createdAt < AUTO_RECHECK_WINDOW_MS
      );
      if (!candidate) return;
      autoRecheckBusy.current = true;
      recheck(candidate.id).then(
        () => {
          autoRecheckBusy.current = false;
        },
        () => {
          autoRecheckBusy.current = false;
        }
      );
    };
    tick();
    const timer = window.setInterval(tick, AUTO_RECHECK_INTERVAL_MS);
    return () => window.clearInterval(timer);
  }, [storageLoaded, canCallApi, recheck]);

  // ------------------------------------------------------------------
  // What became of the pull requests (merged? closed?)
  // ------------------------------------------------------------------
  // A task ends with "pull request opened". Whether it was merged afterwards is
  // only known to GitHub, so it is asked: once when the page loads (for the most
  // recent tasks), and for the task on screen whenever the user comes back to
  // this tab or selects it. There is no polling.
  const refreshPullRequests = useCallback(
    (ids: string[]) => {
      const work = async () => {
        for (const id of ids) {
          const session = sessionsRef.current.find((item) => item.id === id);
          if (!session || !pullRequestCanChange(session) || !session.task) continue;
          const asked = prStateAsked.current[id] || 0;
          if (Date.now() - asked < PR_STATE_MIN_INTERVAL_MS) continue;
          prStateAsked.current[id] = Date.now();
          try {
            const found = await fetchPullRequestStatus(
              session.task.task_id,
              session.task.repository || session.repository
            );
            patchSession(id, (item) => ({
              ...item,
              pullRequest: { state: found.state, checkedAt: Date.now() },
            }));
          } catch (error) {
            // Not being able to ask is not worth a message: the task's own result
            // stays as it is. Without access, stop (and ask again once there is).
            if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
              delete prStateAsked.current[id];
              return;
            }
          }
        }
      };
      // One lookup at a time, in the order they were asked for.
      prStateQueue.current = prStateQueue.current.then(work, work);
    },
    [patchSession]
  );

  useEffect(() => {
    if (!storageLoaded || !canCallApi || githubAuthMode === 'none') return;
    // Debounced like the repository list: no request per keystroke of an access key.
    const timer = window.setTimeout(
      () => {
        const ids = sessionsRef.current
          .filter(pullRequestCanChange)
          .slice(0, PR_STATE_STARTUP_LIMIT)
          .map((session) => session.id);
        if (ids.length > 0) refreshPullRequests(ids);
      },
      accessMode === 'key' ? 800 : 0
    );
    return () => window.clearTimeout(timer);
  }, [storageLoaded, canCallApi, githubAuthMode, accessMode, accessKey, refreshPullRequests]);

  useEffect(() => {
    if (!storageLoaded || !canCallApi || githubAuthMode === 'none' || !activeId) return;
    const check = () => {
      if (document.visibilityState === 'visible') refreshPullRequests([activeId]);
    };
    check(); // the task was just selected
    window.addEventListener('focus', check);
    document.addEventListener('visibilitychange', check);
    return () => {
      window.removeEventListener('focus', check);
      document.removeEventListener('visibilitychange', check);
    };
  }, [storageLoaded, canCallApi, githubAuthMode, activeId, refreshPullRequests]);

  // ------------------------------------------------------------------
  // Running a task
  // ------------------------------------------------------------------
  const runningSession = sessions.find((session) => session.state === 'running') || null;

  const launch = useCallback(async () => {
    if (launching.current) return;
    const repository = normalizeRepository(form.repository);
    const branch = form.branch.trim();
    const prompt = form.prompt.trim();
    if (!looksLikeRepository(repository) || !prompt) return;

    launching.current = true;
    setNotice(null); // a one-off confirmation (GitHub App installed) has done its job by now
    const id = newSessionId();
    const now = Date.now();
    const session: Session = {
      id,
      repository,
      branch,
      prompt,
      state: 'running',
      task: null,
      logs: [clientLog('client', `Sending the task for ${repository} to the server`)],
      note: null,
      createdAt: now,
      updatedAt: now,
    };
    updateSessions((previous) => [session].concat(previous).slice(0, MAX_SESSIONS));
    setActiveId(id);

    try {
      const finished = await streamTask({ repository, branch, prompt }, (event) =>
        patchSession(id, (item) => applyStreamEvent(item, event))
      );
      if (!finished) {
        patchSession(id, (item) => ({
          ...item,
          state: 'interrupted',
          note: 'The live connection ended before the task reported a result. Looking the task up…',
          logs: item.logs.concat([clientLog('warn', 'Connection lost before the result arrived; looking the task up')]),
          updatedAt: Date.now(),
        }));
        window.setTimeout(() => void recheck(id), 1500);
      }
    } catch (error) {
      const message = describeError(error);
      if (error instanceof ApiError && error.code === 'access_key_required') {
        setAccessProblem(getAccessKey() ? 'That access key was not accepted.' : 'This deployment requires an access key.');
      }
      if (error instanceof ApiError && error.code === 'access_locked') void refreshHealth();
      patchSession(id, (item) => ({
        ...item,
        // Nothing was started unless the server had already confirmed a task.
        state: item.task ? 'interrupted' : 'failed',
        note: message,
        logs: item.logs.concat([clientLog('error', message)]),
        updatedAt: Date.now(),
      }));
    } finally {
      launching.current = false;
    }
  }, [form, patchSession, recheck, updateSessions]);

  // ------------------------------------------------------------------
  // Derived view state
  // ------------------------------------------------------------------
  const activeSession = sessions.find((session) => session.id === activeId) || null;
  const activeTask = activeSession ? activeSession.task : null;
  const finishedAt =
    activeSession && activeTask && activeTask.started_at && activeTask.completed_at
      ? activeSession.createdAt + (activeTask.completed_at - activeTask.started_at) * 1000
      : null;

  let blockedReason: string | null = null;
  if (accessMode === 'locked') blockedReason = 'Task execution is locked on this deployment (see above)';
  else if (accessMode === 'key' && !accessKey.trim()) blockedReason = 'Enter the access key to run tasks';

  const modelLabel = healthData && healthData.llm ? `${healthData.llm.provider} · ${healthData.llm.model}` : null;

  let pickerPlaceholder = 'Choose a repository…';
  if (listingLoading) pickerPlaceholder = 'Loading…';
  else if (!listing) {
    // The list was not loaded; do not claim there are no repositories.
    if (accessMode === 'locked') pickerPlaceholder = 'Locked';
    else if (accessMode === 'key' && !accessKey.trim()) pickerPlaceholder = 'Enter the access key first';
    else pickerPlaceholder = 'Not available';
  } else if (repositories.length === 0) pickerPlaceholder = 'None found';

  const selectSession = (session: Session) => {
    setActiveId(session.id);
    setForm({ repository: session.repository, branch: session.branch, prompt: session.prompt });
  };

  const startNewTask = () => {
    setActiveId(null);
    setForm((current) => ({ ...current, prompt: '' }));
    const promptField = document.getElementById('task-prompt');
    if (promptField) promptField.focus();
  };

  const clearHistory = () => {
    updateSessions((previous) => previous.filter((session) => session.state === 'running'));
    setActiveId(runningSession ? runningSession.id : null);
  };

  const onAccessKeyChange = (value: string) => {
    setAccessKeyInput(value);
    setAccessKey(value);
    setAccessProblem(null);
  };

  const statusLabel: Record<typeof readiness, string> = {
    loading: 'CHECKING',
    ready: 'READY',
    incomplete: 'SETUP INCOMPLETE',
    locked: 'LOCKED',
    unreachable: 'UNREACHABLE',
  };

  return (
    <div className="flex flex-col min-h-[calc(100vh-100px)]">
      {/* ---- Notices ---- */}
      {notice && (
        <div className="bg-emerald-500/10 border border-emerald-500/30 rounded-lg px-4 py-2.5 mb-4 flex items-start justify-between gap-3 text-xs text-emerald-200" data-testid="notice">
          <span className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 flex-shrink-0 text-emerald-400" />
            {notice}
          </span>
          <button onClick={() => setNotice(null)} aria-label="Dismiss" className="text-emerald-300 hover:text-white">
            <X className="w-3.5 h-3.5" />
          </button>
        </div>
      )}

      {health.phase === 'unreachable' && (
        <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg px-4 py-3 mb-4 text-xs text-rose-100" data-testid="banner-unreachable">
          <div className="flex items-center gap-2 font-semibold text-rose-300 mb-1">
            <AlertTriangle className="w-4 h-4" /> The backend API is not reachable
          </div>
          <p className="mb-2">{health.message}</p>
          <button
            onClick={() => void refreshHealth()}
            className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 text-slate-100"
          >
            <RefreshCw className="w-3 h-3" /> Try again
          </button>
        </div>
      )}

      {readiness === 'locked' && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3 mb-4 text-xs text-amber-100" data-testid="banner-locked">
          <div className="flex items-center gap-2 font-semibold text-amber-300 mb-1">
            <KeyRound className="w-4 h-4" /> Task execution is locked on this deployment
          </div>
          <p>
            Running a task spends this deployment&apos;s LLM and sandbox quota and opens pull requests with its
            GitHub credentials, so a public deployment does not run tasks for anonymous visitors. To unlock it,
            add an environment variable <code className="font-mono text-amber-200">JULES_ACCESS_KEY</code> (any
            long random value) to the deployment and redeploy, then enter the same value here. To allow anyone
            with the URL to run tasks instead, set{' '}
            <code className="font-mono text-amber-200">JULES_ALLOW_PUBLIC_ACCESS=true</code>.
          </p>
        </div>
      )}

      {readiness === 'incomplete' && healthData && (
        <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg px-4 py-3 mb-4 text-xs text-amber-100" data-testid="banner-incomplete">
          <div className="flex items-center gap-2 font-semibold text-amber-300 mb-1">
            <AlertTriangle className="w-4 h-4" /> The backend is not fully configured, so tasks will fail
          </div>
          <ul className="list-disc pl-5 space-y-0.5">
            {(healthData.missing || []).map((item) => (
              <li key={item}>Missing: {item}</li>
            ))}
          </ul>
        </div>
      )}

      {/* ---- Status row ---- */}
      <div className="bg-[#1e293b]/60 border border-slate-800 rounded-lg px-4 py-2.5 mb-6 flex items-center justify-between gap-3">
        <div className="flex items-center space-x-2 text-xs min-w-0">
          <span
            className={`w-2 h-2 rounded-full flex-shrink-0 ${
              readiness === 'ready'
                ? 'bg-emerald-400 animate-pulse'
                : readiness === 'unreachable'
                  ? 'bg-rose-400'
                  : readiness === 'loading'
                    ? 'bg-slate-500'
                    : 'bg-amber-400'
            }`}
          />
          <span className="text-slate-300 font-medium">Backend status:</span>
          <span className="font-mono text-xs uppercase text-slate-400" data-testid="backend-status">
            {statusLabel[readiness]}
          </span>
          {healthData && (
            <span className="font-mono text-[11px] text-slate-500 truncate hidden sm:inline">
              · GitHub: {githubAuthMode === 'app' ? 'App' : githubAuthMode === 'token' ? 'token' : 'not configured'}
              {modelLabel ? ` · ${modelLabel}` : ''}
            </span>
          )}
        </div>
        <button
          onClick={() => setShowDiagnostics(true)}
          disabled={!healthData}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors font-mono underline disabled:opacity-40 flex-shrink-0"
        >
          Diagnostics Details
        </button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 flex-1 items-start">
        {/* ---- Sidebar ---- */}
        <div className="lg:col-span-4 bg-[#0f172a] border border-slate-800 rounded-xl p-4 space-y-4">
          <div className="space-y-1.5">
            <label htmlFor="repository-picker" className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider block">
              Repositories Jules can access
            </label>
            <div className="relative">
              <select
                id="repository-picker"
                value={repositories.some((repo) => repo.full_name === form.repository) ? form.repository : ''}
                onChange={(e) => {
                  if (e.target.value) setForm((current) => ({ ...current, repository: e.target.value, branch: '' }));
                }}
                disabled={repositories.length === 0}
                className="w-full bg-[#1e293b] border border-slate-700/60 rounded-lg px-3 py-2 pr-8 text-xs text-slate-100 font-mono focus:border-blue-500 outline-none transition-colors appearance-none disabled:opacity-60"
              >
                <option value="">{pickerPlaceholder}</option>
                {repositories.map((repo) => (
                  <option key={repo.full_name} value={repo.full_name}>
                    {repo.full_name}
                    {repo.private ? ' (private)' : ''}
                  </option>
                ))}
              </select>
              <FolderGit2 className="w-3.5 h-3.5 text-slate-400 absolute right-3 top-2.5 pointer-events-none" />
            </div>

            <div className="text-[11px] text-slate-500 space-y-1" data-testid="repo-listing-status">
              {listingError && <p className="text-amber-400 break-words">{listingError}</p>}
              {!listingError && listing && repositories.length === 0 && githubAuthMode === 'app' && (
                <p className="text-amber-400">The GitHub App is not installed on any repository yet.</p>
              )}
              {listing && listing.install_url && (
                <a
                  href={listing.install_url}
                  target="_blank"
                  rel="noreferrer noopener"
                  className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 underline"
                >
                  Install the GitHub App on {repositories.length === 0 ? 'a repository' : 'more repositories'}
                  <ExternalLink className="w-3 h-3" />
                </a>
              )}
              {listing && githubAuthMode === 'token' && (
                <p>Listed from the configured GITHUB_TOKEN. You can also type any repository it can push to.</p>
              )}
            </div>
          </div>

          {(accessMode === 'key' || accessProblem) && (
            <div className="space-y-1.5">
              <label htmlFor="access-key" className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider flex items-center gap-1.5">
                <KeyRound className="w-3 h-3" /> Access key
              </label>
              <input
                id="access-key"
                type="password"
                value={accessKey}
                onChange={(e) => onAccessKeyChange(e.target.value)}
                placeholder="Value of JULES_ACCESS_KEY"
                autoComplete="off"
                className={`w-full bg-[#1e293b] border rounded-lg px-3 py-2 text-xs text-slate-100 font-mono outline-none transition-colors ${
                  accessProblem ? 'border-amber-500/70 focus:border-amber-400' : 'border-slate-700/60 focus:border-blue-500'
                }`}
              />
              <p className={`text-[11px] ${accessProblem ? 'text-amber-400' : 'text-slate-500'}`} data-testid="access-key-status">
                {accessProblem || 'Required by this deployment. Stored only in this browser.'}
              </p>
            </div>
          )}

          <button
            onClick={startNewTask}
            className="w-full flex items-center justify-center space-x-2 bg-blue-600/90 hover:bg-blue-600 text-white text-xs font-medium py-2 px-3 rounded-lg transition-colors shadow-sm"
          >
            <Plus className="w-3.5 h-3.5" />
            <span>New Task</span>
          </button>

          <div className="pt-2 border-t border-slate-800">
            <div className="flex items-center justify-between mb-2">
              <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">Task History</span>
              {sessions.length > 0 && (
                <button
                  onClick={clearHistory}
                  className="text-[11px] text-slate-500 hover:text-slate-300 inline-flex items-center gap-1"
                  title="Remove finished tasks from this browser's history"
                >
                  <Trash2 className="w-3 h-3" /> Clear
                </button>
              )}
            </div>
            <div className="space-y-2 max-h-[380px] overflow-y-auto pr-1" data-testid="history">
              {sessions.length === 0 ? (
                <div className="text-xs text-slate-500 italic p-3 text-center">
                  No tasks yet. Tasks you run from this browser are listed here.
                </div>
              ) : (
                sessions.map((session) => {
                  const badge = sessionBadge(session);
                  return (
                    <button
                      type="button"
                      key={session.id}
                      onClick={() => selectSession(session)}
                      className={`w-full text-left p-2.5 rounded-lg border text-xs cursor-pointer transition-all ${
                        session.id === activeId
                          ? 'bg-[#1e293b] border-blue-500/50 shadow-sm'
                          : 'bg-[#111827]/40 border-slate-800/80 hover:bg-[#1e293b]/50'
                      }`}
                    >
                      <div className="flex items-center justify-between mb-1 gap-2">
                        <span className="font-mono text-[11px] text-slate-300 truncate">{session.repository}</span>
                        <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono flex-shrink-0 ${BADGE_STYLES[badge]}`}>
                          {badge}
                        </span>
                      </div>
                      <div className="text-slate-400 line-clamp-1 text-[11px]">{session.prompt}</div>
                      <div className="text-slate-600 text-[10px] mt-0.5">{formatWhen(session.createdAt)}</div>
                    </button>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* ---- Workspace ---- */}
        <div className="lg:col-span-8 space-y-6">
          <TaskForm
            values={form}
            onChange={(patch) => setForm((current) => ({ ...current, ...patch }))}
            onSubmit={() => void launch()}
            isRunning={!!runningSession}
            blockedReason={blockedReason}
            repositories={repositories}
            repoCheck={repoCheck}
            modelLabel={modelLabel}
            readiness={readiness}
          />

          <LiveTerminal
            logs={activeSession ? activeSession.logs : []}
            state={activeSession ? activeSession.state : 'idle'}
            sandbox={activeTask ? activeTask.sandbox : null}
            startedAt={activeSession ? activeSession.createdAt : null}
            finishedAt={finishedAt}
          />

          <PRStatusTracker
            session={activeSession}
            onRecheck={() => {
              if (activeSession) void recheck(activeSession.id);
            }}
            rechecking={!!activeSession && recheckingId === activeSession.id}
          />
        </div>
      </div>

      {/* ---- Diagnostics ---- */}
      {showDiagnostics && healthData && (
        <div
          className="fixed inset-0 bg-black/70 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setShowDiagnostics(false)}
        >
          <div
            className="bg-[#0f172a] border border-slate-800 rounded-xl max-w-md w-full p-6 shadow-2xl max-h-[90vh] overflow-y-auto"
            onClick={(e) => e.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-label="Backend diagnostics"
          >
            <h3 className="text-sm font-semibold text-white mb-1">Backend diagnostics</h3>
            <p className="text-xs text-slate-400 mb-4">
              Reported by the backend. It shows whether each setting is present, never its value, and it
              cannot tell whether a key is valid until a task uses it.
            </p>

            <div className="grid grid-cols-2 gap-2 mb-4 text-[11px] font-mono">
              <div className="bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800">
                <div className="text-slate-500">LLM</div>
                <div className="text-slate-200 break-words">{modelLabel || 'not configured'}</div>
              </div>
              <div className="bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800">
                <div className="text-slate-500">GitHub auth</div>
                <div className="text-slate-200">
                  {githubAuthMode === 'app' ? 'GitHub App' : githubAuthMode === 'token' ? 'GITHUB_TOKEN' : 'not configured'}
                </div>
              </div>
              <div className="bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800">
                <div className="text-slate-500">Access</div>
                <div className="text-slate-200">{accessMode}</div>
              </div>
              <div className="bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800">
                <div className="text-slate-500">Time budget per task</div>
                <div className="text-slate-200">
                  {healthData.limits ? `${healthData.limits.task_budget_seconds}s` : 'unknown'}
                </div>
              </div>
            </div>

            <div className="space-y-2 mb-6">
              {Object.keys(healthData.variables || {}).map((key) => {
                const present = healthData.variables[key];
                return (
                  <div
                    key={key}
                    className="flex items-center justify-between bg-[#1e293b]/50 px-3 py-2 rounded-lg border border-slate-800 text-xs font-mono"
                  >
                    <span className="text-slate-300">{key}</span>
                    <span
                      className={`px-2 py-0.5 rounded text-[10px] ${
                        present
                          ? 'bg-emerald-500/10 text-emerald-400 border border-emerald-500/20'
                          : 'bg-slate-700/40 text-slate-400 border border-slate-600/40'
                      }`}
                    >
                      {present ? 'SET' : 'NOT SET'}
                    </span>
                  </div>
                );
              })}
            </div>

            {(healthData.missing || []).length > 0 && (
              <div className="mb-6 text-xs text-amber-200 bg-amber-500/10 border border-amber-500/30 rounded-lg p-3">
                <div className="font-semibold text-amber-300 mb-1">Still needed to run tasks</div>
                <ul className="list-disc pl-5 space-y-0.5">
                  {(healthData.missing || []).map((item) => (
                    <li key={item}>{item}</li>
                  ))}
                </ul>
              </div>
            )}

            <button
              onClick={() => setShowDiagnostics(false)}
              className="w-full py-2 bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-medium rounded-lg transition-colors"
            >
              Close
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
