'use client';

import React from 'react';
import {
  AlertTriangle,
  ArrowRight,
  CheckCircle2,
  GitBranch,
  GitFork,
  GitPullRequest,
  Loader2,
  PackagePlus,
  Play,
  Sparkles,
  Wrench,
} from 'lucide-react';
import { looksLikeRepository } from '@/lib/api';
import type { RepoSummary } from '@/lib/api';
import type { Readiness } from '@/lib/health';
import type { ChainEntry } from '@/lib/providers';
import { routeHref } from '@/lib/route';
import type { Draft } from '@/lib/settings';
import { Linkified } from './Linkified';
import { buttonClass, INPUT, LABEL, Panel, TEXTAREA } from './ui';

/** What is in the form: where to work, what to deliver to, what to do. */
export type TaskFormValues = Draft;

export type RepoCheckState =
  | { phase: 'idle' }
  | { phase: 'checking' }
  | {
      phase: 'ok';
      repository: string;
      defaultBranch: string | null;
      isPrivate: boolean | null;
      authMode: string;
      /** The repository has no commits: the task creates the project and its first commit. */
      empty: boolean;
    }
  | { phase: 'error'; message: string };

interface TaskFormProps {
  values: TaskFormValues;
  onChange: (patch: Partial<TaskFormValues>) => void;
  onSubmit: () => void;
  isRunning: boolean;
  /** When set, tasks cannot be started and this explains why. */
  blockedReason: string | null;
  /** Where the user can fix what blocks them (a settings tab), if anywhere. */
  blockedHref?: string | null;
  repositories: RepoSummary[];
  repoCheck: RepoCheckState;
  /** The model providers a task will use, first choice first. */
  chain: ChainEntry[];
  readiness: Readiness;
}

const READINESS: Record<Readiness, { dot: string; text: string }> = {
  loading: { dot: 'bg-slate-500', text: 'Checking the backend…' },
  ready: { dot: 'bg-emerald-400', text: 'Ready: model, sandbox and GitHub credentials are in place' },
  incomplete: { dot: 'bg-amber-400', text: 'Setup is incomplete (see above)' },
  locked: { dot: 'bg-amber-400', text: 'Task execution is locked on this deployment (see above)' },
  unreachable: { dot: 'bg-rose-400', text: 'The backend is not reachable' },
};

const PULL_NUMBER = /^[1-9]\d{0,8}$/;

/** Whether what is in the form is enough to start a task. */
export function formIsComplete(values: TaskFormValues, emptyRepository: boolean): boolean {
  if (!looksLikeRepository(values.repository)) return false;
  if (!emptyRepository && values.target === 'pull_request' && !PULL_NUMBER.test(values.pullRequest.trim())) return false;
  // Failed checks say what there is to do; otherwise the task has to be described.
  if (values.fixChecks && !emptyRepository) return true;
  return values.prompt.trim().length > 0;
}

export function TaskForm({
  values,
  onChange,
  onSubmit,
  isRunning,
  blockedReason,
  blockedHref,
  repositories,
  repoCheck,
  chain,
  readiness,
}: TaskFormProps) {
  const repositoryValid = looksLikeRepository(values.repository);
  const knownRepo = repositories.find(
    (repo) => repo.full_name.toLowerCase() === values.repository.trim().toLowerCase()
  );
  const defaultBranch =
    (repoCheck.phase === 'ok' && repoCheck.defaultBranch) || (knownRepo && knownRepo.default_branch) || null;
  // Nothing to base a change on: the agent builds the project, and its result
  // becomes the first commit on the default branch.
  const emptyRepository = repositoryValid && repoCheck.phase === 'ok' && repoCheck.empty;

  // An empty repository has no pull requests and no checks: the task builds it.
  const toPull = !emptyRepository && values.target === 'pull_request';
  const fixChecks = values.fixChecks && !emptyRepository;
  const canSubmit = !isRunning && !blockedReason && formIsComplete(values, emptyRepository);

  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    if (canSubmit) onSubmit();
  };

  const onPromptKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === 'Enter' && (event.metaKey || event.ctrlKey) && canSubmit) {
      event.preventDefault();
      onSubmit();
    }
  };

  const status = READINESS[readiness];
  const first = chain[0];
  const via = repoCheck.phase === 'ok' ? (repoCheck.authMode === 'app' ? ' · via GitHub App' : repoCheck.authMode === 'token' ? ' · via token' : '') : '';
  const visibility = repoCheck.phase === 'ok' ? (repoCheck.isPrivate === null ? '' : repoCheck.isPrivate ? ' (private)' : ' (public)') : '';

  const promptPlaceholder = emptyRepository
    ? 'Describe what to build, e.g. “A to-do list web app that saves to the browser, ready to deploy on Vercel”'
    : fixChecks
      ? 'Optional: anything the agent should know about the failure'
      : toPull
        ? 'Describe what to change on the pull request, e.g. “Also handle empty input, and add a test for it”'
        : 'Describe one concrete change, e.g. “Fix the failing date parsing test and add a regression test”';

  return (
    <Panel
      icon={<Sparkles className="w-4 h-4 text-blue-400" />}
      title="New task"
      aside={
        first ? (
          <>
            {chain.length === 1 && (
              <a
                href={routeHref('settings', 'models')}
                className="hidden md:inline text-xs text-blue-400 hover:text-blue-300 whitespace-nowrap"
                title="With a key for a second provider the agent carries on with it when the first reaches a rate limit, instead of stopping."
                data-testid="add-fallback"
              >
                + Add a fallback
              </a>
            )}
            <a
              href={routeHref('settings', 'models')}
              className="inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md bg-slate-800/70 hover:bg-slate-800 border border-slate-700/70 text-xs text-slate-300 max-w-[220px] sm:max-w-[420px] min-w-0"
              title={`Model providers for a task, first choice first: ${chain.map((item) => `${item.label} (${item.model})`).join(' → ')}. Click to change.`}
              data-testid="model-label"
              data-providers={chain.map((item) => item.id).join(',')}
            >
              <span className="font-medium text-slate-100 flex-shrink-0">{first.label}</span>
              <span className="font-mono text-slate-400 truncate">{first.model}</span>
              {chain.slice(1).map((item) => (
                <span key={item.id} className="hidden sm:inline-flex items-center gap-1.5 flex-shrink-0 text-slate-400">
                  <ArrowRight className="w-3 h-3 text-slate-600" />
                  {item.label}
                </span>
              ))}
            </a>
          </>
        ) : undefined
      }
    >
      <form className="space-y-4" onSubmit={submit}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className="md:col-span-2">
            <label htmlFor="task-repository" className={LABEL}>
              <GitFork className="w-3 h-3 inline -mt-0.5 mr-1.5 text-slate-500" />
              Repository
            </label>
            <input
              id="task-repository"
              type="text"
              value={values.repository}
              onChange={(e) => onChange({ repository: e.target.value })}
              placeholder="owner/name or https://github.com/owner/name"
              list="task-repository-options"
              autoComplete="off"
              spellCheck={false}
              className={`${INPUT} font-mono`}
              required
            />
            <datalist id="task-repository-options">
              {repositories.map((repo) => (
                <option key={repo.full_name} value={repo.full_name} />
              ))}
            </datalist>
          </div>

          {toPull ? (
            <div>
              <label htmlFor="task-pull-request" className={LABEL}>
                <GitPullRequest className="w-3 h-3 inline -mt-0.5 mr-1.5 text-slate-500" />
                Pull request number
              </label>
              <input
                id="task-pull-request"
                type="text"
                inputMode="numeric"
                value={values.pullRequest}
                onChange={(e) => onChange({ pullRequest: e.target.value.replace(/[^\d]/g, '').slice(0, 9) })}
                placeholder="e.g. 12"
                autoComplete="off"
                className={`${INPUT} font-mono`}
              />
            </div>
          ) : (
            <div>
              <label htmlFor="task-branch" className={LABEL}>
                <GitBranch className="w-3 h-3 inline -mt-0.5 mr-1.5 text-slate-500" />
                Base branch
              </label>
              <input
                id="task-branch"
                type="text"
                value={values.branch}
                onChange={(e) => onChange({ branch: e.target.value })}
                placeholder={
                  emptyRepository
                    ? `${defaultBranch || 'default branch'} · first commit`
                    : defaultBranch
                      ? `default (${defaultBranch})`
                      : 'default branch'
                }
                title={
                  emptyRepository
                    ? 'An empty repository has no branches yet: the first commit is made on its default branch.'
                    : undefined
                }
                autoComplete="off"
                spellCheck={false}
                className={`${INPUT} font-mono`}
              />
            </div>
          )}
        </div>

        {/* What the backend says about this repository, before anything is run. */}
        <div
          className="min-h-[18px] text-xs -mt-1"
          data-testid="repo-check"
          data-phase={repoCheck.phase}
          data-empty={emptyRepository ? 'true' : undefined}
        >
          {values.repository.trim() !== '' && !repositoryValid && (
            <span className="text-amber-400">Enter the repository as owner/name or paste its GitHub URL.</span>
          )}
          {repositoryValid && repoCheck.phase === 'checking' && (
            <span className="text-slate-500 inline-flex items-center gap-1.5">
              <Loader2 className="w-3.5 h-3.5 animate-spin" /> Checking access to the repository…
            </span>
          )}
          {emptyRepository && repoCheck.phase === 'ok' && (
            <span className="text-sky-300 inline-flex items-start gap-1.5">
              <PackagePlus className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>
                {repoCheck.repository} is empty{visibility}. Jules will build the project from your instructions and
                commit it to {repoCheck.defaultBranch || 'the default branch'} as the first commit{via}
              </span>
            </span>
          )}
          {repositoryValid && repoCheck.phase === 'ok' && !repoCheck.empty && (
            <span className="text-emerald-400 inline-flex items-start gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <span>
                Jules can open pull requests on {repoCheck.repository}
                {visibility}
                {repoCheck.defaultBranch ? ` · default branch ${repoCheck.defaultBranch}` : ''}
                {via}
              </span>
            </span>
          )}
          {repositoryValid && repoCheck.phase === 'error' && (
            <span className="text-amber-400 inline-flex items-start gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
              <Linkified text={repoCheck.message} />
            </span>
          )}
        </div>

        {/* Where the result goes, and whether the task is the failed checks. */}
        {!emptyRepository && (
          <div className="flex flex-wrap items-center gap-x-5 gap-y-3" data-testid="task-target" data-target={toPull ? 'pull_request' : 'new'}>
            <div className="inline-flex rounded-lg border border-slate-700/70 bg-[#0b1220] p-0.5" role="radiogroup" aria-label="Deliver the result as">
              <button
                type="button"
                role="radio"
                aria-checked={!toPull}
                onClick={() => onChange({ target: 'new' })}
                data-testid="target-new"
                className={`h-8 px-3 rounded-md text-xs font-medium transition-colors ${
                  !toPull ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                New pull request
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={toPull}
                onClick={() => onChange({ target: 'pull_request' })}
                data-testid="target-pull-request"
                className={`h-8 px-3 rounded-md text-xs font-medium transition-colors ${
                  toPull ? 'bg-slate-700 text-white' : 'text-slate-400 hover:text-white'
                }`}
              >
                Add to a pull request
              </button>
            </div>

            <label className="inline-flex items-center gap-2 text-xs text-slate-300 cursor-pointer select-none">
              <input
                type="checkbox"
                checked={values.fixChecks}
                onChange={(e) => onChange({ fixChecks: e.target.checked })}
                data-testid="fix-checks"
                className="h-4 w-4 rounded border-slate-600 bg-[#0b1220] accent-blue-600"
              />
              <Wrench className="w-3.5 h-3.5 text-slate-500" />
              <span title="The agent reads what failed on GitHub - test runs, builds, deployments - and repairs it.">
                {toPull ? 'Fix its failed checks' : 'Fix the failed checks of the branch'}
              </span>
            </label>
          </div>
        )}

        <div>
          <div className="flex items-baseline justify-between">
            <label htmlFor="task-prompt" className={LABEL}>
              Instructions{fixChecks ? ' (optional)' : ''}
            </label>
            <span className="text-[11px] text-slate-600 tabular-nums">{values.prompt.length.toLocaleString('en-US')} / 8,000</span>
          </div>
          <textarea
            id="task-prompt"
            value={values.prompt}
            onChange={(e) => onChange({ prompt: e.target.value })}
            onKeyDown={onPromptKeyDown}
            rows={4}
            maxLength={8000}
            placeholder={promptPlaceholder}
            className={`${TEXTAREA} font-mono`}
            required={!fixChecks}
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-4 border-t border-slate-800">
          <div className="flex items-center gap-2 text-xs text-slate-400 min-w-0" data-testid="readiness" data-readiness={readiness}>
            <span className={`w-2 h-2 rounded-full flex-shrink-0 ${blockedReason ? 'bg-amber-400' : status.dot}`}></span>
            <span>
              {blockedReason || status.text}
              {blockedReason && blockedHref && (
                <>
                  {' '}
                  <a href={blockedHref} className="text-blue-400 hover:text-blue-300 underline underline-offset-2" data-testid="blocked-link">
                    Open settings
                  </a>
                </>
              )}
            </span>
          </div>

          <div className="flex items-center gap-3">
            <span className="hidden sm:inline text-[11px] text-slate-600">Ctrl / ⌘ + Enter</span>
            <button type="submit" disabled={!canSubmit} data-testid="run-task" className={buttonClass('primary', 'px-5')}>
              {isRunning ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Running…</span>
                </>
              ) : (
                <>
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{fixChecks ? 'Fix checks' : toPull ? 'Run follow-up' : 'Run task'}</span>
                </>
              )}
            </button>
          </div>
        </div>
      </form>
    </Panel>
  );
}
