'use client';

import React from 'react';
import {
  GitPullRequest,
  CheckCircle2,
  XCircle,
  Clock,
  ExternalLink,
  AlertTriangle,
  RefreshCw,
  MinusCircle,
  HelpCircle,
} from 'lucide-react';
import { pullRequestUrl } from '@/lib/api';
import type { Session } from '@/lib/sessions';
import { Linkified } from './Linkified';
import { StageProgress } from './StageProgress';

interface PRStatusTrackerProps {
  session: Session | null;
  onRecheck: () => void;
  rechecking: boolean;
}

const FILE_STATUS: Record<string, { label: string; className: string }> = {
  A: { label: 'added', className: 'text-emerald-400' },
  M: { label: 'modified', className: 'text-amber-300' },
  D: { label: 'deleted', className: 'text-rose-400' },
  T: { label: 'type changed', className: 'text-slate-300' },
};

const MAX_FILES_SHOWN = 12;

export function PRStatusTracker({ session, onRecheck, rechecking }: PRStatusTrackerProps) {
  if (!session) {
    return (
      <div className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 text-center text-slate-500 text-sm">
        <GitPullRequest className="w-8 h-8 mx-auto mb-2 text-slate-600 opacity-60" />
        No task selected. When a task finishes with changes, the pull request GitHub created for it is
        linked here.
      </div>
    );
  }

  const task = session.task;
  const prUrl = session.state === 'completed' ? pullRequestUrl(task) : null;
  const noChanges = session.state === 'completed' && !prUrl;
  const testsStatus = task ? task.tests_status : null;
  const failureText = (task && task.error) || session.note || 'The task failed without an error message.';

  const statusPill = () => {
    if (session.state === 'running') {
      return (
        <span className="inline-flex items-center text-xs text-yellow-400 bg-yellow-400/10 px-2.5 py-1 rounded-full border border-yellow-400/20">
          <Clock className="w-3.5 h-3.5 mr-1 animate-spin" />
          Agent working
        </span>
      );
    }
    if (prUrl) {
      return (
        <span className="inline-flex items-center text-xs text-emerald-400 bg-emerald-400/10 px-2.5 py-1 rounded-full border border-emerald-400/20">
          <CheckCircle2 className="w-3.5 h-3.5 mr-1" />
          Pull request opened
        </span>
      );
    }
    if (noChanges) {
      return (
        <span className="inline-flex items-center text-xs text-slate-300 bg-slate-700/40 px-2.5 py-1 rounded-full border border-slate-600/50">
          <MinusCircle className="w-3.5 h-3.5 mr-1" />
          Finished · no changes
        </span>
      );
    }
    if (session.state === 'failed') {
      return (
        <span className="inline-flex items-center text-xs text-rose-400 bg-rose-400/10 px-2.5 py-1 rounded-full border border-rose-400/20">
          <XCircle className="w-3.5 h-3.5 mr-1" />
          Task failed
        </span>
      );
    }
    return (
      <span className="inline-flex items-center text-xs text-amber-400 bg-amber-400/10 px-2.5 py-1 rounded-full border border-amber-400/20">
        <HelpCircle className="w-3.5 h-3.5 mr-1" />
        Result unknown
      </span>
    );
  };

  const testsCell = () => {
    if (testsStatus === 'passed') {
      return (
        <>
          <CheckCircle2 className="w-4 h-4 text-emerald-400 flex-shrink-0" />
          <span className="text-emerald-400">Passed in the sandbox</span>
        </>
      );
    }
    if (testsStatus === 'failed') {
      return (
        <>
          <XCircle className="w-4 h-4 text-rose-400 flex-shrink-0" />
          <span className="text-rose-400">Last run failed</span>
        </>
      );
    }
    if (testsStatus === 'not_run') {
      return (
        <>
          <AlertTriangle className="w-4 h-4 text-amber-400 flex-shrink-0" />
          <span className="text-amber-400">No tests were run</span>
        </>
      );
    }
    // Without a report from the server, say so instead of implying nothing was run.
    const unknown = session.state === 'interrupted' || (task !== null && task.recovered);
    return (
      <>
        <Clock className="w-4 h-4 text-slate-500 flex-shrink-0" />
        <span className="text-slate-500">
          {session.state === 'running' ? 'Not run yet' : unknown ? 'Unknown' : 'Not run'}
        </span>
      </>
    );
  };

  const changedFiles = task && Array.isArray(task.changed_files) ? task.changed_files : [];
  const verification = task && Array.isArray(task.verification) ? task.verification : [];

  return (
    <div
      className="bg-[#0f172a] border border-slate-800 rounded-xl p-6 shadow-xl relative overflow-hidden"
      data-testid="result-panel"
      data-state={session.state}
    >
      <div className="flex items-center justify-between mb-4 pb-3 border-b border-slate-800 gap-3">
        <div className="flex items-center space-x-2">
          <GitPullRequest className="w-5 h-5 text-indigo-400" />
          <h3 className="font-semibold text-white text-sm">Result</h3>
        </div>
        <div data-testid="result-status">{statusPill()}</div>
      </div>

      <div className="space-y-4">
        <StageProgress task={task} state={session.state} />

        <div>
          <div className="text-xs text-slate-400 mb-1">Task</div>
          <p className="text-xs text-slate-200 bg-[#090d16] p-2.5 rounded-lg border border-slate-800/80 font-mono line-clamp-3 whitespace-pre-wrap">
            {session.prompt || (task && task.recovered ? '(not available: this result was recovered from GitHub)' : '—')}
          </p>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
          <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800/80">
            <span className="text-xs text-slate-400 block mb-1">Tests</span>
            <div className="flex items-center space-x-1.5 text-xs font-medium" data-testid="tests-status">
              {testsCell()}
            </div>
          </div>

          <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800/80 min-w-0">
            <span className="text-xs text-slate-400 block mb-1">Branch</span>
            <span className="text-xs font-mono text-indigo-300 truncate block" data-testid="branch-info">
              {task && task.head_branch
                ? `${task.head_branch} → ${task.base_branch || '?'}`
                : session.branch
                  ? `base: ${session.branch}`
                  : '—'}
            </span>
          </div>

          <div className="bg-[#090d16] p-3 rounded-lg border border-slate-800/80 min-w-0">
            <span className="text-xs text-slate-400 block mb-1">Model</span>
            <span className="text-xs font-mono text-slate-300 truncate block">
              {task && task.llm_model ? `${task.llm_provider || ''} · ${task.llm_model}` : '—'}
            </span>
          </div>
        </div>

        {verification.length > 0 && (
          <div>
            <div className="text-xs text-slate-400 mb-1">Verification runs</div>
            <ul className="bg-[#090d16] rounded-lg border border-slate-800/80 divide-y divide-slate-800/80">
              {verification.slice(-4).map((run, index) => (
                <li key={index} className="flex items-center justify-between gap-3 px-2.5 py-1.5 text-[11px] font-mono">
                  <span className="truncate text-slate-300">$ {run.command}</span>
                  <span className={`flex-shrink-0 ${run.passed ? 'text-emerald-400' : 'text-rose-400'}`}>
                    {run.timed_out ? 'timed out' : run.passed ? 'passed' : `exit ${run.exit_code}`}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}

        {changedFiles.length > 0 && (
          <div>
            <div className="text-xs text-slate-400 mb-1">
              {changedFiles.length} file{changedFiles.length === 1 ? '' : 's'} changed
            </div>
            <ul className="bg-[#090d16] rounded-lg border border-slate-800/80 divide-y divide-slate-800/80">
              {changedFiles.slice(0, MAX_FILES_SHOWN).map((file) => {
                const style = FILE_STATUS[file.status] || { label: file.status, className: 'text-slate-300' };
                return (
                  <li key={file.path} className="flex items-center justify-between gap-3 px-2.5 py-1.5 text-[11px] font-mono">
                    <span className="truncate text-slate-200">{file.path}</span>
                    <span className="flex-shrink-0 space-x-2">
                      <span className={style.className}>{style.label}</span>
                      {file.additions !== null && file.deletions !== null && (
                        <span className="text-slate-500">
                          <span className="text-emerald-500">+{file.additions}</span>{' '}
                          <span className="text-rose-500">−{file.deletions}</span>
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
              {changedFiles.length > MAX_FILES_SHOWN && (
                <li className="px-2.5 py-1.5 text-[11px] font-mono text-slate-500">
                  … and {changedFiles.length - MAX_FILES_SHOWN} more
                </li>
              )}
            </ul>
          </div>
        )}

        {task && task.summary && session.state !== 'failed' && (
          <div>
            <div className="text-xs text-slate-400 mb-1">{task.recovered ? 'Summary' : 'Agent summary'}</div>
            <p className="text-xs text-slate-200 bg-[#090d16] p-2.5 rounded-lg border border-slate-800/80 whitespace-pre-wrap break-words">
              <Linkified text={task.summary} />
            </p>
          </div>
        )}

        {/* ---- Outcome ---- */}
        {prUrl && task && (
          <div className="pt-1 space-y-2">
            <a
              href={prUrl}
              target="_blank"
              rel="noreferrer noopener"
              data-testid="pr-link"
              className="w-full flex items-center justify-center space-x-2 bg-emerald-600 hover:bg-emerald-500 text-white font-medium text-xs py-2.5 px-4 rounded-lg transition-colors shadow-lg shadow-emerald-600/20"
            >
              <GitPullRequest className="w-4 h-4" />
              <span>
                Review pull request{task.pr_number ? ` #${task.pr_number}` : ''} on GitHub
              </span>
              <ExternalLink className="w-3.5 h-3.5 ml-1" />
            </a>
            <div className="text-[11px] font-mono text-slate-500 break-all text-center">{prUrl}</div>
            {testsStatus === 'failed' && (
              <p className="text-xs text-amber-400 flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                The agent&apos;s last test run failed. The pull request says so too. Review it carefully before merging.
              </p>
            )}
            {testsStatus === 'not_run' && (
              <p className="text-xs text-amber-400 flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                The agent did not run any tests for this change.
              </p>
            )}
            {task.recovered && (
              <p className="text-xs text-slate-400">
                This result was looked up on GitHub because the server instance that ran the task is no longer
                available. The run&apos;s log and details could not be recovered, only the pull request itself.
              </p>
            )}
          </div>
        )}

        {noChanges && (
          <div className="bg-slate-800/40 border border-slate-700/60 rounded-lg p-3 text-xs text-slate-300" data-testid="no-changes">
            The agent finished without changing any files, so no pull request was opened.
          </div>
        )}

        {session.state === 'failed' && (
          <div className="bg-rose-500/10 border border-rose-500/30 rounded-lg p-3" data-testid="failure">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-rose-300 mb-1">
              <XCircle className="w-3.5 h-3.5" />
              No pull request was opened
            </div>
            <p className="text-xs text-rose-100/90 whitespace-pre-wrap break-words">
              <Linkified text={failureText} />
            </p>
          </div>
        )}

        {session.state === 'interrupted' && (
          <div className="bg-amber-500/10 border border-amber-500/30 rounded-lg p-3 space-y-2" data-testid="interrupted">
            <div className="flex items-center gap-1.5 text-xs font-semibold text-amber-300">
              <AlertTriangle className="w-3.5 h-3.5" />
              The connection to this run was lost
            </div>
            <p className="text-xs text-amber-100/90 whitespace-pre-wrap break-words">
              <Linkified
                text={
                  session.note ||
                  'The result did not arrive. If the run finished, its pull request can be found by checking again.'
                }
              />
            </p>
            <button
              type="button"
              onClick={onRecheck}
              disabled={rechecking}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-slate-800 hover:bg-slate-700 text-slate-100 border border-slate-700 transition-colors disabled:opacity-60"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${rechecking ? 'animate-spin' : ''}`} />
              <span>{rechecking ? 'Checking…' : 'Check again'}</span>
            </button>
          </div>
        )}

        {session.state === 'running' && (
          <p className="text-xs text-slate-500 italic text-center">
            A pull request link appears here only once GitHub has created the pull request.
          </p>
        )}
      </div>
    </div>
  );
}
