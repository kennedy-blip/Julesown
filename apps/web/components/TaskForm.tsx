'use client';

import React from 'react';
import { GitBranch, GitFork, Play, Sparkles, CheckCircle2, AlertTriangle, Loader2 } from 'lucide-react';
import { looksLikeRepository } from '@/lib/api';
import type { RepoSummary } from '@/lib/api';
import type { Readiness } from '@/lib/health';
import { Linkified } from './Linkified';

export interface TaskFormValues {
  repository: string;
  branch: string;
  prompt: string;
}

export type RepoCheckState =
  | { phase: 'idle' }
  | { phase: 'checking' }
  | { phase: 'ok'; repository: string; defaultBranch: string | null; isPrivate: boolean | null; authMode: string }
  | { phase: 'error'; message: string };

interface TaskFormProps {
  values: TaskFormValues;
  onChange: (patch: Partial<TaskFormValues>) => void;
  onSubmit: () => void;
  isRunning: boolean;
  /** When set, tasks cannot be started and this explains why. */
  blockedReason: string | null;
  repositories: RepoSummary[];
  repoCheck: RepoCheckState;
  /** Provider and model the backend will use, as reported by the backend. */
  modelLabel: string | null;
  readiness: Readiness;
}

const READINESS: Record<Readiness, { dot: string; text: string }> = {
  loading: { dot: 'bg-slate-500', text: 'Checking the backend…' },
  ready: { dot: 'bg-emerald-400', text: 'Backend configured: LLM, sandbox and GitHub credentials are set' },
  incomplete: { dot: 'bg-amber-400', text: 'Backend setup is incomplete (see above)' },
  locked: { dot: 'bg-amber-400', text: 'Task execution is locked on this deployment (see above)' },
  unreachable: { dot: 'bg-rose-400', text: 'The backend is not reachable' },
};

export function TaskForm({
  values,
  onChange,
  onSubmit,
  isRunning,
  blockedReason,
  repositories,
  repoCheck,
  modelLabel,
  readiness,
}: TaskFormProps) {
  const repositoryValid = looksLikeRepository(values.repository);
  const canSubmit = !isRunning && !blockedReason && repositoryValid && values.prompt.trim().length > 0;

  const knownRepo = repositories.find(
    (repo) => repo.full_name.toLowerCase() === values.repository.trim().toLowerCase()
  );
  const defaultBranch =
    (repoCheck.phase === 'ok' && repoCheck.defaultBranch) || (knownRepo && knownRepo.default_branch) || null;

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

  return (
    <div className="bg-[#111827] border border-slate-800 rounded-xl p-5 shadow-2xl relative">
      <div className="flex items-center justify-between mb-4 gap-3">
        <div className="flex items-center space-x-2">
          <Sparkles className="w-4 h-4 text-blue-400" />
          <h2 className="text-sm font-semibold text-slate-200 uppercase tracking-wider">Launch Coding Session</h2>
        </div>
        {modelLabel && (
          <span
            className="text-xs font-mono text-slate-400 bg-slate-800/80 px-2 py-0.5 rounded border border-slate-700/60 truncate max-w-[55%]"
            title="LLM provider and model configured on the backend"
            data-testid="model-label"
          >
            {modelLabel}
          </span>
        )}
      </div>

      <form className="space-y-4" onSubmit={submit}>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
          <div className="md:col-span-2">
            <label htmlFor="task-repository" className="block text-xs font-medium text-slate-400 mb-1 flex items-center gap-1.5">
              <GitFork className="w-3 h-3 text-slate-500" />
              Target GitHub Repository
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
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors font-mono"
              required
            />
            <datalist id="task-repository-options">
              {repositories.map((repo) => (
                <option key={repo.full_name} value={repo.full_name} />
              ))}
            </datalist>
          </div>

          <div>
            <label htmlFor="task-branch" className="block text-xs font-medium text-slate-400 mb-1 flex items-center gap-1.5">
              <GitBranch className="w-3 h-3 text-slate-500" />
              Base Branch
            </label>
            <input
              id="task-branch"
              type="text"
              value={values.branch}
              onChange={(e) => onChange({ branch: e.target.value })}
              placeholder={defaultBranch ? `default (${defaultBranch})` : 'default branch'}
              autoComplete="off"
              spellCheck={false}
              className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors font-mono"
            />
          </div>
        </div>

        {/* What the backend says about this repository, before anything is run. */}
        <div className="min-h-[18px] text-[11px] -mt-2" data-testid="repo-check" data-phase={repoCheck.phase}>
          {values.repository.trim() !== '' && !repositoryValid && (
            <span className="text-amber-400">Enter the repository as owner/name or paste its GitHub URL.</span>
          )}
          {repositoryValid && repoCheck.phase === 'checking' && (
            <span className="text-slate-500 inline-flex items-center gap-1.5">
              <Loader2 className="w-3 h-3 animate-spin" /> Checking access to the repository…
            </span>
          )}
          {repositoryValid && repoCheck.phase === 'ok' && (
            <span className="text-emerald-400 inline-flex items-center gap-1.5">
              <CheckCircle2 className="w-3 h-3 flex-shrink-0" />
              <span>
                Jules can open pull requests on {repoCheck.repository}
                {repoCheck.isPrivate === null ? '' : repoCheck.isPrivate ? ' (private)' : ' (public)'}
                {repoCheck.defaultBranch ? ` · default branch ${repoCheck.defaultBranch}` : ''}
                {repoCheck.authMode === 'app' ? ' · via GitHub App' : repoCheck.authMode === 'token' ? ' · via token' : ''}
              </span>
            </span>
          )}
          {repositoryValid && repoCheck.phase === 'error' && (
            <span className="text-amber-400 inline-flex items-start gap-1.5">
              <AlertTriangle className="w-3 h-3 flex-shrink-0 mt-0.5" />
              <Linkified text={repoCheck.message} />
            </span>
          )}
        </div>

        <div>
          <label htmlFor="task-prompt" className="block text-xs font-medium text-slate-400 mb-1">
            Task Instructions &amp; Objective
          </label>
          <textarea
            id="task-prompt"
            value={values.prompt}
            onChange={(e) => onChange({ prompt: e.target.value })}
            onKeyDown={onPromptKeyDown}
            rows={3}
            maxLength={8000}
            placeholder="Describe one concrete change, e.g. “Fix the failing date parsing test and add a regression test”"
            className="w-full bg-[#090d16] border border-slate-800 focus:border-blue-500 rounded-lg px-3 py-2 text-xs text-slate-100 placeholder-slate-500 outline-none transition-colors resize-y font-mono"
            required
          />
        </div>

        <div className="flex flex-wrap items-center justify-between gap-3 pt-3 border-t border-slate-800/80">
          <div className="flex items-center space-x-2 text-xs text-slate-400 min-w-0" data-testid="readiness" data-readiness={readiness}>
            <span className={`w-1.5 h-1.5 rounded-full flex-shrink-0 ${status.dot}`}></span>
            <span>{blockedReason || status.text}</span>
          </div>

          <button
            type="submit"
            disabled={!canSubmit}
            data-testid="run-task"
            className={`inline-flex items-center space-x-1.5 px-4 py-1.5 rounded-lg text-xs font-medium transition-all ${
              canSubmit
                ? 'bg-blue-600 hover:bg-blue-500 text-white shadow-md shadow-blue-600/20 active:scale-95'
                : 'bg-blue-600/40 text-blue-200/70 cursor-not-allowed'
            }`}
          >
            {isRunning ? (
              <>
                <Loader2 className="w-3.5 h-3.5 animate-spin" />
                <span>Running…</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 fill-current" />
                <span>Run task</span>
              </>
            )}
          </button>
        </div>
      </form>
    </div>
  );
}
