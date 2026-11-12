'use client';

import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { AlertTriangle, ArrowLeft, CheckCircle2, KeyRound, Loader2, RefreshCw, X } from 'lucide-react';
import { AccessKeyField } from '@/components/AccessKeyField';
import { LiveTerminal } from '@/components/LiveTerminal';
import { PRStatusTracker } from '@/components/PRStatusTracker';
import { RepositoryPicker } from '@/components/RepositoryPicker';
import { SettingsPanel } from '@/components/SettingsPanel';
import { formIsComplete, TaskForm } from '@/components/TaskForm';
import type { RepoCheckState } from '@/components/TaskForm';
import { TaskHistory } from '@/components/TaskHistory';
import { Button, buttonClass, Callout } from '@/components/ui';
import {
  ApiError,
  checkRepository,
  describeError,
  fetchChecks,
  fetchPullRequestStatus,
  fetchRepositories,
  fetchTask,
  getAccessKey,
  looksLikeRepository,
  normalizeRepository,
  setAccessKey,
  streamTask,
} from '@/lib/api';
import type { ChecksSummary, LogEntry, RepoListing, StreamEvent, TaskRecord } from '@/lib/api';
import { readinessOf, refreshHealth, stillMissing, useHealth } from '@/lib/health';
import { effectiveChain } from '@/lib/providers';
import { navigate, routeHref, useRoute } from '@/lib/route';
import {
  isFirstCommit,
  isUnfinished,
  loadSessions,
  newSessionId,
  pullRequestCanChange,
  pullRequestState,
  saveSessions,
} from '@/lib/sessions';
import type { Session } from '@/lib/sessions';
import { getSettings, hasProviderKey, loadSettings, updateSettings, useSettings } from '@/lib/settings';
import type { Draft, SettingsTab } from '@/lib/settings';

const MAX_SESSIONS = 30;
/** How long after a lost connection the task is still looked up automatically
 *  (a little longer than the longest run the backend allows). */
const AUTO_RECHECK_WINDOW_MS = 7 * 60 * 1000;
const AUTO_RECHECK_INTERVAL_MS = 15 * 1000;
/** What became of a result is asked from GitHub at most this often per task... */
const RESULT_MIN_INTERVAL_MS = 20 * 1000;
/** ...and when the page loads only for this many of the most recent tasks. */
const RESULT_STARTUP_LIMIT = 8;
/** While the checks of the result on screen are running they are asked for again this often... */
const CHECKS_POLL_MS = 25 * 1000;
/** ...for at most this long after the task finished... */
const CHECKS_WATCH_MS = 20 * 60 * 1000;
/** ...and checks that have not shown up yet are waited for this long. */
const CHECKS_APPEAR_MS = 3 * 60 * 1000;

const GITHUB_APP_NOTICES: Record<string, string> = {
  installed: 'GitHub App installation saved. Jules can now work on the repositories you selected.',
  requested:
    'Installation requested. An owner of that organization has to approve it before Jules can access its repositories.',
  authorized: 'GitHub authorization complete.',
};

/** Everything a task is started with. */
interface LaunchRequest {
  repository: string;
  branch: string;
  prompt: string;
  /** Add the result to this pull request instead of opening a new one. */
  pullRequest: number | null;
  fixChecks: boolean;
  continueWork: boolean;
}

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
    // Looked up on GitHub: only what the task left there is known. Keep what
    // this browser already learned from the stream and add the result to it.
    const known = session.task || found;
    task = {
      ...known,
      status: found.status,
      stage: found.stage,
      outcome: found.outcome,
      pr_url: found.pr_url,
      pr_number: found.pr_number,
      pull_request: found.pull_request || known.pull_request,
      commit_sha: found.commit_sha || known.commit_sha,
      commit_url: found.commit_url || known.commit_url,
      head_branch: found.head_branch || known.head_branch,
      base_branch: found.base_branch || known.base_branch,
      summary: known.summary || found.summary,
      prompt: known.prompt || session.prompt,
      unfinished: known.unfinished === true || found.unfinished === true,
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

/** Whether GitHub is worth asking about this result: a pull request that can still change, or a first commit. */
function resultCanChange(session: Session): boolean {
  return pullRequestCanChange(session) || isFirstCommit(session);
}

/** What CI said about the result of a session the last time it was asked (null: not asked, or not applicable). */
function checksOf(session: Session): ChecksSummary | null {
  if (pullRequestCanChange(session)) return (session.pullRequest && session.pullRequest.checks) || null;
  if (isFirstCommit(session)) return (session.commitChecks && session.commitChecks.checks) || null;
  return null;
}

export default function Dashboard() {
  const health = useHealth();
  const settings = useSettings();
  const route = useRoute();
  const form = settings.draft;
  const browserHasModelKey = hasProviderKey(settings);
  const readiness = readinessOf(health, browserHasModelKey);
  const healthData = health.phase === 'ok' ? health.data : null;
  const accessMode = healthData && healthData.access ? healthData.access.mode : 'open';
  const githubAuthMode = healthData && healthData.github ? healthData.github.auth_mode : 'none';
  // The model providers a task started here will use, first choice first.
  const chain = useMemo(() => effectiveChain(healthData, settings), [healthData, settings]);

  const [sessions, setSessions] = useState<Session[]>([]);
  const [storageLoaded, setStorageLoaded] = useState(false);
  const [activeId, setActiveId] = useState<string | null>(null);
  const [accessKey, setAccessKeyInput] = useState('');
  const [accessProblem, setAccessProblem] = useState<string | null>(null);
  // The backend has accepted a request made with the key that is entered.
  const [accessAccepted, setAccessAccepted] = useState(false);
  // The key is being asked for on the workspace (stays up while it is typed and checked).
  const [accessPrompt, setAccessPrompt] = useState(false);
  const [listing, setListing] = useState<RepoListing | null>(null);
  const [listingError, setListingError] = useState<string | null>(null);
  const [listingLoading, setListingLoading] = useState(false);
  const [repoCheck, setRepoCheck] = useState<RepoCheckState>({ phase: 'idle' });
  const [notice, setNotice] = useState<string | null>(null);
  const [recheckingId, setRecheckingId] = useState<string | null>(null);
  const [refreshingId, setRefreshingId] = useState<string | null>(null);

  const sessionsRef = useRef<Session[]>([]);
  const launching = useRef(false);
  const autoRecheckBusy = useRef(false);
  // Lookups of results on GitHub: when each session was last asked about, and
  // the queue that keeps them one at a time.
  const resultAsked = useRef<Record<string, number>>({});
  const resultQueue = useRef<Promise<void>>(Promise.resolve());

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

  /** Change the task form. What is in it is kept in this browser, so a reload loses nothing. */
  const setForm = useCallback((patch: Partial<Draft>) => {
    updateSettings((current) => ({ draft: { ...current.draft, ...patch } }));
  }, []);

  // ------------------------------------------------------------------
  // Startup: history, settings, access key, GitHub App landing
  // ------------------------------------------------------------------
  useEffect(() => {
    const stored = loadSessions();
    updateSessions(() => stored);
    const saved = loadSettings();
    if (stored.length > 0) {
      setActiveId(stored[0].id);
      if (!saved.draft.repository.trim()) setForm({ repository: stored[0].repository });
    }
    setAccessKeyInput(getAccessKey());
    setStorageLoaded(true);

    // GitHub sends the browser back here after the App is installed or authorised.
    const params = new URLSearchParams(window.location.search);
    const landing = params.get('github_app');
    if (landing) {
      setNotice(GITHUB_APP_NOTICES[landing] || null);
      window.history.replaceState(null, '', window.location.pathname + window.location.hash);
    }
  }, [updateSessions, setForm]);

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
    // The answer is about the key the request was made with: one that arrives
    // after the key was changed says nothing about the new one.
    const key = getAccessKey();
    const current = () => getAccessKey() === key;
    setListingLoading(true);
    try {
      const data = await fetchRepositories();
      if (!current()) return;
      setListing(data);
      setListingError(data.error);
      setAccessProblem(null);
      setAccessAccepted(true);
    } catch (error) {
      if (!current()) return;
      setListing(null);
      if (error instanceof ApiError && error.code === 'access_key_required') {
        setListingError(null);
        setAccessAccepted(false);
        setAccessProblem(key ? 'That access key was not accepted.' : null);
      } else if (error instanceof ApiError && error.code === 'access_locked') {
        setListingError(null);
        void refreshHealth();
      } else {
        setListingError(describeError(error));
      }
    } finally {
      if (current()) setListingLoading(false);
    }
  }, []);

  useEffect(() => {
    if (!canCallApi) {
      setListing(null);
      setListingLoading(false);
      return;
    }
    // Debounced so that typing an access key does not send a request per keystroke.
    const timer = window.setTimeout(() => void loadRepositories(), accessMode === 'key' ? 500 : 0);
    return () => window.clearTimeout(timer);
  }, [canCallApi, accessMode, accessKey, loadRepositories]);

  // Ask for the access key on the workspace when it is missing or was refused,
  // and keep asking until the backend has accepted one.
  useEffect(() => {
    if (accessMode !== 'key' || accessAccepted) setAccessPrompt(false);
    else if (storageLoaded && (!accessKey.trim() || accessProblem)) setAccessPrompt(true);
  }, [accessMode, accessAccepted, storageLoaded, accessKey, accessProblem]);

  const repositories = listing ? listing.repositories : [];
  // The list is ordered by last push: the first usable entry is the repository worked on most recently.
  const firstUsable = repositories.find((repo) => !repo.archived);
  const firstRepository = firstUsable ? firstUsable.full_name : '';
  useEffect(() => {
    if (firstRepository && storageLoaded && !getSettings().draft.repository.trim()) {
      setForm({ repository: firstRepository });
    }
  }, [firstRepository, storageLoaded, setForm]);

  // ------------------------------------------------------------------
  // Pre-flight check of the repository in the form
  // ------------------------------------------------------------------
  // Asked again when a task ends: a task changes what is true about a
  // repository (an empty one, for instance, is not empty afterwards).
  const taskInProgress = sessions.some((session) => session.state === 'running');
  // A pull request has its own branch: the base branch in the form does not apply to it.
  const branchToCheck = form.target === 'pull_request' ? '' : form.branch;
  useEffect(() => {
    if (!canCallApi || githubAuthMode === 'none' || !looksLikeRepository(form.repository)) {
      setRepoCheck({ phase: 'idle' });
      return;
    }
    let cancelled = false;
    setRepoCheck({ phase: 'checking' });
    const timer = window.setTimeout(async () => {
      try {
        const result = await checkRepository(normalizeRepository(form.repository), branchToCheck);
        if (cancelled) return;
        if (result.ok) {
          setRepoCheck({
            phase: 'ok',
            repository: result.repository,
            defaultBranch: result.default_branch,
            isPrivate: result.private,
            authMode: result.auth_mode,
            empty: result.empty === true,
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
  }, [canCallApi, githubAuthMode, form.repository, branchToCheck, accessKey, taskInProgress]);

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
      // A task that adds to a pull request has no branch of its own to be found
      // by: its commit is looked for on that pull request.
      const pullRequest = (session.target && session.target.pullRequest) || (session.task && session.task.pull_request) || null;
      setRecheckingId(id);
      try {
        const found = await fetchTask(taskId, (session.task && session.task.repository) || session.repository, pullRequest);
        patchSession(id, (item) => (item.state === 'interrupted' ? applyLookup(item, found) : item));
      } catch (error) {
        const recent = Date.now() - session.createdAt < AUTO_RECHECK_WINDOW_MS;
        let note: string;
        if (error instanceof ApiError && error.status === 404) {
          const nothing = pullRequest
            ? `No commit from this task is on pull request #${pullRequest}`
            : 'Nothing from this task exists on GitHub';
          note = recent
            ? `${nothing} yet. The run may still be in progress on the server, so this is checked again automatically for a few minutes.`
            : `${nothing}${pullRequest ? '' : ': no pull request and no commit'}. The run did not get that far (it may have been stopped by the hosting time limit).`;
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
  // What became of the results (merged? closed? did the checks pass?)
  // ------------------------------------------------------------------
  // A task ends with "pull request opened". Whether it was merged afterwards,
  // and what CI makes of it, is only known to GitHub, so it is asked: once when
  // the page loads (for the most recent tasks), for the task on screen whenever
  // the user comes back to this tab or selects it, and a few times while its
  // checks are still running.
  const refreshResults = useCallback(
    (ids: string[]) => {
      const work = async () => {
        for (const id of ids) {
          const session = sessionsRef.current.find((item) => item.id === id);
          if (!session || !session.task || !resultCanChange(session)) continue;
          const asked = resultAsked.current[id] || 0;
          if (Date.now() - asked < RESULT_MIN_INTERVAL_MS) continue;
          resultAsked.current[id] = Date.now();
          const repository = session.task.repository || session.repository;
          try {
            if (pullRequestCanChange(session)) {
              const found = await fetchPullRequestStatus(session.task.task_id, repository, session.task.pr_number);
              patchSession(id, (item) => ({
                ...item,
                pullRequest: {
                  state: found.state,
                  checkedAt: Date.now(),
                  unfinished: typeof found.unfinished === 'boolean' ? found.unfinished : undefined,
                  checks: found.checks || null,
                },
              }));
            } else if (session.task.commit_sha) {
              // The first commit of a repository that was empty: there is no pull request, only its checks.
              const found = await fetchChecks(repository, session.task.commit_sha);
              const state = found.state;
              if (found.ok && state) {
                patchSession(id, (item) => ({
                  ...item,
                  commitChecks: {
                    checks: {
                      sha: found.sha || '',
                      state,
                      failing: found.failing || [],
                      failed: found.failed || 0,
                      passed: found.passed || 0,
                      pending: found.pending || 0,
                      notes: found.notes || [],
                    },
                    checkedAt: Date.now(),
                  },
                }));
              }
            }
          } catch (error) {
            // Not being able to ask is not worth a message: the task's own result
            // stays as it is. Without access, stop (and ask again once there is).
            if (error instanceof ApiError && (error.status === 401 || error.status === 403)) {
              delete resultAsked.current[id];
              return;
            }
          }
        }
      };
      // One lookup at a time, in the order they were asked for.
      resultQueue.current = resultQueue.current.then(work, work);
      return resultQueue.current;
    },
    [patchSession]
  );

  const canAskGitHub = storageLoaded && canCallApi && githubAuthMode !== 'none';

  useEffect(() => {
    if (!canAskGitHub) return;
    // Debounced like the repository list: no request per keystroke of an access key.
    const timer = window.setTimeout(
      () => {
        const ids = sessionsRef.current
          .filter(pullRequestCanChange)
          .slice(0, RESULT_STARTUP_LIMIT)
          .map((session) => session.id);
        if (ids.length > 0) void refreshResults(ids);
      },
      accessMode === 'key' ? 800 : 0
    );
    return () => window.clearTimeout(timer);
  }, [canAskGitHub, accessMode, accessKey, refreshResults]);

  useEffect(() => {
    if (!canAskGitHub || !activeId) return;
    const check = () => {
      if (document.visibilityState === 'visible') void refreshResults([activeId]);
    };
    check(); // the task was just selected
    window.addEventListener('focus', check);
    document.addEventListener('visibilitychange', check);
    return () => {
      window.removeEventListener('focus', check);
      document.removeEventListener('visibilitychange', check);
    };
  }, [canAskGitHub, activeId, refreshResults]);

  // ------------------------------------------------------------------
  // Derived view state
  // ------------------------------------------------------------------
  const runningSession = sessions.find((session) => session.state === 'running') || null;
  const activeSession = sessions.find((session) => session.id === activeId) || null;
  const activeTask = activeSession ? activeSession.task : null;
  const finishedAt =
    activeSession && activeTask && activeTask.started_at && activeTask.completed_at
      ? activeSession.createdAt + (activeTask.completed_at - activeTask.started_at) * 1000
      : null;

  // The checks of the result on screen, while they can still change: CI starts
  // a little after the push and takes a while, so they are watched for a time.
  const watchId = activeSession && resultCanChange(activeSession) && pullRequestState(activeSession) !== 'closed' ? activeSession.id : null;
  const watchedChecks = activeSession ? checksOf(activeSession) : null;
  const watchedState = watchedChecks ? watchedChecks.state : null;
  const watchedSince = activeSession ? activeSession.updatedAt : 0;
  useEffect(() => {
    if (!canAskGitHub || !watchId) return;
    if (watchedState === 'failure' || watchedState === 'unknown') return; // nothing to wait for
    const due = () => {
      const age = Date.now() - watchedSince;
      if (age > CHECKS_WATCH_MS) return false;
      // Checks that are running are waited for; ones that have not appeared (or
      // have all passed already) only for a short while after the push.
      return watchedState === 'pending' || age < CHECKS_APPEAR_MS;
    };
    if (!due()) return;
    void refreshResults([watchId]); // the task has just finished, or was just selected
    const timer = window.setInterval(() => {
      if (document.visibilityState === 'visible' && due()) void refreshResults([watchId]);
    }, CHECKS_POLL_MS);
    return () => window.clearInterval(timer);
  }, [canAskGitHub, watchId, watchedState, watchedSince, refreshResults]);

  // ------------------------------------------------------------------
  // Running a task
  // ------------------------------------------------------------------
  const launch = useCallback(
    async (request: LaunchRequest) => {
      if (launching.current || sessionsRef.current.some((session) => session.state === 'running')) return;
      const repository = normalizeRepository(request.repository);
      const prompt = request.prompt.trim();
      // The failed checks are a task of their own; anything else has to be described.
      if (!looksLikeRepository(repository) || (!prompt && !request.fixChecks)) return;
      const branch = request.pullRequest ? '' : request.branch.trim();

      launching.current = true;
      setNotice(null); // a one-off confirmation (GitHub App installed) has done its job by now
      const id = newSessionId();
      const now = Date.now();
      const where = request.pullRequest ? `pull request #${request.pullRequest} of ${repository}` : repository;
      const session: Session = {
        id,
        repository,
        branch,
        prompt,
        state: 'running',
        task: null,
        logs: [clientLog('client', `Sending the task for ${where} to the server`)],
        note: null,
        target: { pullRequest: request.pullRequest, fixChecks: request.fixChecks, continueWork: request.continueWork },
        createdAt: now,
        updatedAt: now,
      };
      updateSessions((previous) => [session].concat(previous).slice(0, MAX_SESSIONS));
      setActiveId(id);

      try {
        const finished = await streamTask(
          {
            repository,
            branch,
            prompt,
            pullRequest: request.pullRequest,
            fixChecks: request.fixChecks,
            continueWork: request.continueWork,
          },
          (event) => patchSession(id, (item) => applyStreamEvent(item, event))
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
        } else {
          // Earlier results of the same pull request have changed with it (it is
          // no longer unfinished, its checks belong to a new commit): ask again.
          const done = sessionsRef.current.find((item) => item.id === id);
          const number = done && done.task ? done.task.pr_number : null;
          if (number) {
            const others = sessionsRef.current
              .filter((item) => item.id !== id && item.repository.toLowerCase() === repository.toLowerCase() && !!item.task && item.task.pr_number === number)
              .map((item) => item.id);
            others.forEach((other) => delete resultAsked.current[other]);
            if (others.length > 0) void refreshResults(others);
          }
        }
      } catch (error) {
        const message = describeError(error);
        if (error instanceof ApiError && error.code === 'access_key_required') {
          setAccessAccepted(false);
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
    },
    [patchSession, recheck, refreshResults, updateSessions]
  );

  const emptyRepository = repoCheck.phase === 'ok' && repoCheck.empty;

  const submitForm = () => {
    if (!formIsComplete(form, emptyRepository)) return;
    // An empty repository has no pull requests and no checks: the task builds it.
    const toPullRequest = !emptyRepository && form.target === 'pull_request';
    void launch({
      repository: form.repository,
      branch: form.branch,
      prompt: form.prompt,
      pullRequest: toPullRequest ? Number(form.pullRequest) : null,
      fixChecks: form.fixChecks && !emptyRepository,
      continueWork: false,
    });
  };

  // What can be done about the result on screen -------------------------------
  const continueWork = (session: Session) => {
    const task = session.task;
    if (!task) return;
    const repository = task.repository || session.repository;
    const prompt = session.prompt.trim() || 'Finish the work that was left unfinished.';
    if (task.pr_number && task.outcome !== 'initial_commit') {
      // The rest is added to the pull request that holds the unfinished work.
      void launch({ repository, branch: '', prompt, pullRequest: task.pr_number, fixChecks: false, continueWork: true });
    } else {
      // The unfinished project is on the default branch: the rest arrives as a pull request.
      void launch({ repository, branch: '', prompt, pullRequest: null, fixChecks: false, continueWork: true });
    }
  };

  const fixChecks = (session: Session) => {
    const task = session.task;
    if (!task) return;
    const repository = task.repository || session.repository;
    if (task.pr_number && task.outcome !== 'initial_commit') {
      void launch({ repository, branch: '', prompt: '', pullRequest: task.pr_number, fixChecks: true, continueWork: false });
    } else {
      // A first commit has no pull request: the fix for its branch arrives as one.
      void launch({ repository, branch: task.base_branch || '', prompt: '', pullRequest: null, fixChecks: true, continueWork: false });
    }
  };

  const retry = (session: Session) => {
    const target = session.target || null;
    void launch({
      repository: session.repository,
      branch: session.branch,
      prompt: session.prompt,
      pullRequest: target ? target.pullRequest : null,
      fixChecks: !!target && target.fixChecks,
      continueWork: !!target && target.continueWork,
    });
  };

  const focusPrompt = () => {
    window.setTimeout(() => {
      const field = document.getElementById('task-prompt');
      if (field) {
        field.scrollIntoView({ block: 'center', behavior: 'smooth' });
        field.focus({ preventScroll: true });
      }
    }, 0);
  };

  /** Put a pull request into the task form, to describe what else it needs. */
  const requestChanges = (session: Session) => {
    const task = session.task;
    if (!task || !task.pr_number) return;
    setForm({
      repository: task.repository || session.repository,
      branch: '',
      target: 'pull_request',
      pullRequest: String(task.pr_number),
      fixChecks: false,
      prompt: '',
    });
    focusPrompt();
  };

  const refreshNow = (id: string) => {
    delete resultAsked.current[id];
    setRefreshingId(id);
    const done = () => setRefreshingId((current) => (current === id ? null : current));
    refreshResults([id]).then(done, done);
  };

  let blockedReason: string | null = null;
  let blockedHref: string | null = null;
  if (accessMode === 'locked') blockedReason = 'Task execution is locked on this deployment (see above).';
  else if (accessMode === 'key' && !accessKey.trim()) blockedReason = 'Enter the access key to run tasks (see above).';
  else if (accessMode === 'key' && accessProblem) blockedReason = 'The access key was not accepted (see above).';
  else if (healthData && chain.length === 0) {
    blockedReason = 'Add a key for a model provider to run tasks.';
    blockedHref = routeHref('settings', 'models');
  }
  // Why a task cannot be started from the result panel right now.
  const startBlocked = runningSession ? 'A task is running. Wait for it to finish.' : blockedReason;

  // Why there is no repository list, when there is none (never "no repositories").
  let listUnavailable: string | null = null;
  if (!listing && !listingLoading) {
    if (accessMode === 'locked') listUnavailable = 'Locked';
    else if (accessMode === 'key' && !accessKey.trim()) listUnavailable = 'Enter the access key first';
    else if (health.phase !== 'ok') listUnavailable = health.phase === 'unreachable' ? 'Backend not reachable' : 'Checking the backend…';
    else if (githubAuthMode === 'none') listUnavailable = 'GitHub is not configured';
  }

  const selectSession = (session: Session) => {
    setActiveId(session.id);
    // The form follows the task that is selected, unless something new was
    // typed into it: that is never thrown away by looking at an older task.
    const draft = getSettings().draft;
    const typed = draft.prompt.trim();
    const nothingNew = typed === '' || sessionsRef.current.some((item) => item.prompt.trim() === typed);
    if (!nothingNew) return;
    const target = session.target || null;
    const pullRequest = target ? target.pullRequest : null;
    setForm({
      repository: session.repository,
      branch: pullRequest ? '' : session.branch,
      prompt: session.prompt,
      target: pullRequest ? 'pull_request' : 'new',
      pullRequest: pullRequest ? String(pullRequest) : '',
      fixChecks: !!target && target.fixChecks,
    });
  };

  const startNewTask = () => {
    setActiveId(null);
    setForm({ prompt: '', branch: '', target: 'new', pullRequest: '', fixChecks: false });
    focusPrompt();
  };

  const clearHistory = () => {
    updateSessions((previous) => previous.filter((session) => session.state === 'running'));
    setActiveId(runningSession ? runningSession.id : null);
  };

  const onAccessKeyChange = (value: string) => {
    setAccessKeyInput(value);
    setAccessKey(value);
    setAccessProblem(null);
    setAccessAccepted(false);
  };

  const settingsTab: SettingsTab = route.tab || settings.settingsTab;
  const openSettingsTab = (tab: SettingsTab) => {
    updateSettings({ settingsTab: tab });
    navigate('settings', tab);
  };
  // A tab reached by its address (a link, a reload) is the one to come back to as well.
  const visitedTab = route.view === 'settings' ? route.tab : null;
  useEffect(() => {
    if (storageLoaded && visitedTab && getSettings().settingsTab !== visitedTab) updateSettings({ settingsTab: visitedTab });
  }, [storageLoaded, visitedTab]);

  const missing = healthData ? stillMissing(healthData, browserHasModelKey) : [];
  const modelKeyMissing = !!healthData && chain.length === 0;

  // ------------------------------------------------------------------
  // Settings
  // ------------------------------------------------------------------
  if (route.view === 'settings') {
    return (
      <div className="space-y-5">
        <div className="flex flex-wrap items-center justify-between gap-3">
          <div>
            <h1 className="text-lg font-semibold text-white">Settings</h1>
            <p className="text-xs text-slate-400 mt-0.5">
              Kept in this browser. Keys are sent to your own backend with the requests that need them and are not stored there.
            </p>
          </div>
          <a href={routeHref('workspace')} className={buttonClass('secondary')} data-testid="back-to-workspace">
            <ArrowLeft className="w-4 h-4" />
            Back to the workspace
          </a>
        </div>

        {runningSession && (
          <Callout tone="info" icon={<Loader2 className="w-4 h-4 animate-spin text-sky-300" />} data-testid="running-notice">
            A task is running on <span className="font-mono">{runningSession.repository}</span>. It carries on while you are here;
            changes made now apply to the next task.
          </Callout>
        )}

        <SettingsPanel
          health={healthData}
          accessMode={accessMode}
          accessKey={accessKey}
          onAccessKeyChange={onAccessKeyChange}
          accessProblem={accessProblem}
          accessAccepted={accessAccepted}
          canCallApi={canCallApi}
          installations={listing && Array.isArray(listing.installations) ? listing.installations : []}
          tab={settingsTab}
          onTab={openSettingsTab}
        />
      </div>
    );
  }

  // ------------------------------------------------------------------
  // Workspace
  // ------------------------------------------------------------------
  return (
    <div className="space-y-6">
      {/* ---- Notices ---- */}
      {(notice || health.phase === 'unreachable' || readiness === 'locked' || accessPrompt || (readiness === 'incomplete' && healthData)) && (
        <div className="space-y-3">
          {notice && (
            <Callout tone="good" icon={<CheckCircle2 className="w-4 h-4 text-emerald-400" />} data-testid="notice">
              <div className="flex items-start justify-between gap-3">
                <span>{notice}</span>
                <button onClick={() => setNotice(null)} aria-label="Dismiss" className="text-emerald-300 hover:text-white flex-shrink-0">
                  <X className="w-4 h-4" />
                </button>
              </div>
            </Callout>
          )}

          {health.phase === 'unreachable' && (
            <Callout tone="bad" icon={<AlertTriangle className="w-4 h-4 text-rose-300" />} data-testid="banner-unreachable">
              <div className="font-semibold text-rose-200 mb-1">The backend API is not reachable</div>
              <p className="mb-2.5">{health.message}</p>
              <Button onClick={() => void refreshHealth()}>
                <RefreshCw className="w-4 h-4" /> Try again
              </Button>
            </Callout>
          )}

          {readiness === 'locked' && (
            <Callout tone="warn" icon={<KeyRound className="w-4 h-4 text-amber-300" />} data-testid="banner-locked">
              <div className="font-semibold text-amber-200 mb-1">Task execution is locked on this deployment</div>
              <p>
                Running a task spends this deployment&apos;s model and sandbox quota and writes to GitHub with its
                credentials, so a public deployment does not run tasks for anonymous visitors. To unlock it, add an
                environment variable <code className="font-mono text-amber-200">JULES_ACCESS_KEY</code> (any long random
                value) to the deployment and redeploy, then enter the same value here. To allow anyone with the URL to
                run tasks instead, set <code className="font-mono text-amber-200">JULES_ALLOW_PUBLIC_ACCESS=true</code>.
              </p>
            </Callout>
          )}

          {accessPrompt && (
            <Callout tone="warn" icon={<KeyRound className="w-4 h-4 text-amber-300" />} data-testid="banner-access">
              <div className="font-semibold text-amber-200 mb-1">This deployment asks for its access key</div>
              <p className="mb-2.5 text-amber-100/90">
                It is the value of the <code className="font-mono">JULES_ACCESS_KEY</code> variable of the deployment.
                Without it the repositories cannot be listed and no task can be run.
              </p>
              <div className="max-w-xl">
                <AccessKeyField
                  value={accessKey}
                  onChange={onAccessKeyChange}
                  problem={accessProblem}
                  accepted={accessAccepted}
                  showLabel={false}
                />
              </div>
            </Callout>
          )}

          {readiness === 'incomplete' && healthData && (
            <Callout tone="warn" icon={<AlertTriangle className="w-4 h-4 text-amber-300" />} data-testid="banner-incomplete">
              <div className="font-semibold text-amber-200 mb-1">Setup is incomplete, so tasks will fail</div>
              <ul className="list-disc pl-5 space-y-0.5">
                {missing.map((item) => (
                  <li key={item}>Missing: {item}</li>
                ))}
              </ul>
              <div className="flex flex-wrap items-center gap-2 mt-2.5">
                {modelKeyMissing && (
                  <a href={routeHref('settings', 'models')} className={buttonClass('primary')} data-testid="add-provider-key">
                    Add a model provider key
                  </a>
                )}
                <a href={routeHref('settings', 'about')} className={buttonClass('secondary')}>
                  See what the server is configured with
                </a>
              </div>
            </Callout>
          )}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
        {/* On a narrow screen: repositories, the task, then the history. */}
        <aside className="contents lg:block lg:col-span-4 lg:space-y-6">
          <div className="order-1 min-w-0">
            <RepositoryPicker
              listing={listing}
              loading={listingLoading}
              error={listingError}
              selected={form.repository}
              onSelect={(fullName) => setForm({ repository: fullName, branch: '' })}
              onRefresh={() => void loadRepositories()}
              unavailable={listUnavailable}
            />
          </div>
          <div className="order-3 min-w-0">
            <TaskHistory
              sessions={sessions}
              activeId={activeId}
              onSelect={selectSession}
              onNew={startNewTask}
              onClear={clearHistory}
            />
          </div>
        </aside>

        <div className="order-2 lg:order-none lg:col-span-8 space-y-6 min-w-0">
          <TaskForm
            values={form}
            onChange={setForm}
            onSubmit={submitForm}
            isRunning={!!runningSession}
            blockedReason={blockedReason}
            blockedHref={blockedHref}
            repositories={repositories}
            repoCheck={repoCheck}
            chain={chain}
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
            onContinue={() => {
              if (activeSession && isUnfinished(activeSession)) continueWork(activeSession);
            }}
            onFixChecks={() => {
              if (activeSession) fixChecks(activeSession);
            }}
            onRequestChanges={() => {
              if (activeSession) requestChanges(activeSession);
            }}
            onRetry={() => {
              if (activeSession) retry(activeSession);
            }}
            onRefreshStatus={() => {
              if (activeSession) refreshNow(activeSession.id);
            }}
            refreshing={!!activeSession && refreshingId === activeSession.id}
            startBlocked={startBlocked}
            singleProvider={chain.length <= 1}
          />
        </div>
      </div>
    </div>
  );
}
