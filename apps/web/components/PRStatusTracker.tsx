'use client';

import React from 'react';
import {
  AlertTriangle,
  CheckCircle2,
  Clock,
  ExternalLink,
  GitCommitHorizontal,
  GitMerge,
  GitPullRequest,
  GitPullRequestClosed,
  HelpCircle,
  Loader2,
  MessageSquarePlus,
  MinusCircle,
  Play,
  Plus,
  RefreshCw,
  RotateCcw,
  Wrench,
  XCircle,
} from 'lucide-react';
import { addedCommitUrl, firstCommitUrl, pullRequestUrl } from '@/lib/api';
import type { ChecksSummary, ProviderUsage } from '@/lib/api';
import { plainProse, runLabel, runOutcome, summarizeTests } from '@/lib/results';
import type { Tone } from '@/lib/results';
import { routeHref } from '@/lib/route';
import { isUnfinished, pullRequestState } from '@/lib/sessions';
import type { Session } from '@/lib/sessions';
import { providerName } from '@/lib/settings';
import { Linkified } from './Linkified';
import { StageProgress } from './StageProgress';
import { Badge, Button, buttonClass, Callout, CARD, INSET, Panel } from './ui';
import type { BadgeTone } from './ui';

interface PRStatusTrackerProps {
  session: Session | null;
  onRecheck: () => void;
  rechecking: boolean;
  /** Have the agent finish work it left unfinished. */
  onContinue: () => void;
  /** Have the agent repair the failed checks of the result. */
  onFixChecks: () => void;
  /** Put the pull request into the task form, to ask for more changes on it. */
  onRequestChanges: () => void;
  /** Run the task again as it was. */
  onRetry: () => void;
  /** Ask GitHub again what became of the result and what its checks say. */
  onRefreshStatus: () => void;
  refreshing: boolean;
  /** Why a task cannot be started right now (one is running, access is missing ...), or null. */
  startBlocked: string | null;
  /** Only one model provider is set up: nothing to carry on with when it gives out. */
  singleProvider: boolean;
}

const FILE_STATUS: Record<string, { label: string; className: string }> = {
  A: { label: 'added', className: 'text-emerald-400' },
  M: { label: 'modified', className: 'text-amber-300' },
  D: { label: 'deleted', className: 'text-rose-400' },
  T: { label: 'type changed', className: 'text-slate-300' },
};

const MAX_FILES_SHOWN = 12;
const MAX_READ_FILES_SHOWN = 8;
const MAX_CHECKS_SHOWN = 6;

const TONE_TEXT: Record<Tone, string> = {
  good: 'text-emerald-400',
  bad: 'text-rose-400',
  caution: 'text-amber-400',
  neutral: 'text-slate-300',
  pending: 'text-slate-500',
};

const AUTHORSHIP: Record<string, string> = {
  bot: 'Jules',
  coauthor: 'Jules, with you as co-author',
  user: 'You',
};

function ToneIcon({ tone }: { tone: Tone }) {
  const className = `w-4 h-4 flex-shrink-0 ${TONE_TEXT[tone]}`;
  switch (tone) {
    case 'good':
      return <CheckCircle2 className={className} />;
    case 'bad':
      return <XCircle className={className} />;
    case 'caution':
      return <AlertTriangle className={className} />;
    case 'neutral':
      return <MinusCircle className={className} />;
    default:
      return <Clock className={className} />;
  }
}

function SectionLabel({ children }: { children: React.ReactNode }) {
  return <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-1.5">{children}</div>;
}

function Fact({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className={`${INSET} px-3 py-2.5 min-w-0`}>
      <div className="text-[11px] text-slate-500 mb-1">{label}</div>
      {children}
    </div>
  );
}

function compact(count: number | undefined): string {
  const value = typeof count === 'number' && count > 0 ? count : 0;
  if (value < 1000) return String(value);
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}k`;
}

function plural(count: number, one: string, many: string): string {
  return `${count} ${count === 1 ? one : many}`;
}

/** What CI says about the result, with the way to have a failure repaired. */
function ChecksBlock({
  checks,
  subject,
  onFix,
  onRefresh,
  refreshing,
  startBlocked,
}: {
  checks: ChecksSummary;
  subject: string;
  onFix: () => void;
  onRefresh: () => void;
  refreshing: boolean;
  startBlocked: string | null;
}) {
  const counts = [
    checks.failed > 0 ? `${checks.failed} failed` : '',
    checks.pending > 0 ? `${checks.pending} running` : '',
    checks.passed > 0 ? `${checks.passed} passed` : '',
  ].filter(Boolean);
  let line: React.ReactNode;
  switch (checks.state) {
    case 'failure':
      line = (
        <span className="text-rose-400 inline-flex items-center gap-1.5">
          <XCircle className="w-4 h-4 flex-shrink-0" /> {counts.join(' · ')}
        </span>
      );
      break;
    case 'pending':
      line = (
        <span className="text-amber-300 inline-flex items-center gap-1.5">
          <Clock className="w-4 h-4 flex-shrink-0" /> {counts.join(' · ')}
        </span>
      );
      break;
    case 'success':
      line = (
        <span className="text-emerald-400 inline-flex items-center gap-1.5">
          <CheckCircle2 className="w-4 h-4 flex-shrink-0" />{' '}
          {checks.passed === 1 ? 'The check passed' : `All ${checks.passed} checks passed`}
        </span>
      );
      break;
    case 'none':
      line = (
        <span className="text-slate-400 inline-flex items-center gap-1.5">
          <MinusCircle className="w-4 h-4 flex-shrink-0" /> No checks have run on {subject}
        </span>
      );
      break;
    default:
      line = (
        <span className="text-amber-400 inline-flex items-center gap-1.5">
          <HelpCircle className="w-4 h-4 flex-shrink-0" /> The checks could not be read
        </span>
      );
  }
  return (
    <div data-testid="checks" data-state={checks.state}>
      <div className="flex items-center justify-between">
        <SectionLabel>Checks on GitHub</SectionLabel>
        <button
          type="button"
          onClick={onRefresh}
          disabled={refreshing}
          className="text-slate-500 hover:text-slate-200 disabled:opacity-50 -mt-1.5 p-1 rounded"
          aria-label="Ask GitHub again"
          title="Ask GitHub again"
          data-testid="checks-refresh"
        >
          <RefreshCw className={`w-3.5 h-3.5 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>
      <div className={`${INSET} px-3 py-2.5 space-y-2.5`}>
        <div className="text-[13px] font-medium" data-testid="checks-state">
          {line}
        </div>
        {checks.failing.length > 0 && (
          <ul className="space-y-1">
            {checks.failing.slice(0, MAX_CHECKS_SHOWN).map((check, index) => (
              <li key={`${check.name}-${index}`} className="flex items-center gap-2 text-xs min-w-0" data-testid="failing-check">
                <XCircle className="w-3.5 h-3.5 text-rose-400 flex-shrink-0" />
                {check.url && /^https:\/\//.test(check.url) ? (
                  <a href={check.url} target="_blank" rel="noreferrer noopener" className="text-slate-200 hover:text-white underline underline-offset-2 truncate">
                    {check.name}
                  </a>
                ) : (
                  <span className="text-slate-200 truncate">{check.name}</span>
                )}
                {check.app && (
                  <span className="text-slate-500 flex-shrink-0">{check.app === 'github-actions' ? 'GitHub Actions' : check.app}</span>
                )}
              </li>
            ))}
            {checks.failing.length > MAX_CHECKS_SHOWN && (
              <li className="text-xs text-slate-500">… and {checks.failing.length - MAX_CHECKS_SHOWN} more</li>
            )}
          </ul>
        )}
        {checks.notes.length > 0 && (
          <p className="text-xs text-amber-400/90" data-testid="checks-notes">
            {checks.notes.map((note) => note.charAt(0).toUpperCase() + note.slice(1)).join('. ')}.{' '}
            <a href={routeHref('settings', 'automation')} className="underline underline-offset-2 hover:text-amber-300">
              What the GitHub App needs
            </a>
          </p>
        )}
        {checks.state === 'failure' && (
          <div className="flex flex-wrap items-center gap-3">
            <Button variant="primary" onClick={onFix} disabled={!!startBlocked} title={startBlocked || undefined} data-testid="fix-checks-action">
              <Wrench className="w-4 h-4" />
              Fix failed checks
            </Button>
            <span className="text-xs text-slate-400">
              The agent reads what failed and pushes a fix{subject === 'this commit' ? ' as a pull request' : ' to the pull request'}.
            </span>
          </div>
        )}
      </div>
    </div>
  );
}

function ModelsUsed({ usage }: { usage: ProviderUsage[] }) {
  return (
    <div data-testid="models-used">
      <SectionLabel>Model providers</SectionLabel>
      <ul className={`${INSET} divide-y divide-slate-800`}>
        {usage.map((item, index) => {
          const requests = item.requests || 0;
          return (
            <li key={`${item.provider}-${index}`} className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 px-3 py-2 text-xs" data-provider={item.provider}>
              <span className="flex items-center gap-2 min-w-0">
                <span className={`font-medium ${requests > 0 ? 'text-slate-100' : 'text-slate-500'}`}>{providerName(item.provider)}</span>
                <span className="font-mono text-slate-500 truncate">{item.model}</span>
              </span>
              <span className="flex items-center gap-2 text-slate-400 tabular-nums">
                {requests > 0 ? (
                  <span>
                    {plural(requests, 'request', 'requests')} · {compact(item.input_tokens)} in / {compact(item.output_tokens)} out
                  </span>
                ) : (
                  <span className="text-slate-500">{item.failures ? 'did not answer' : 'not needed'}</span>
                )}
                {item.out && <Badge tone="warn">{item.out.replace(/^its? /, '')}</Badge>}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

export function PRStatusTracker({
  session,
  onRecheck,
  rechecking,
  onContinue,
  onFixChecks,
  onRequestChanges,
  onRetry,
  onRefreshStatus,
  refreshing,
  startBlocked,
  singleProvider,
}: PRStatusTrackerProps) {
  if (!session) {
    return (
      <div className={`${CARD} px-6 py-10 text-center`}>
        <GitPullRequest className="w-8 h-8 mx-auto mb-3 text-slate-600" />
        <p className="text-[13px] text-slate-400 max-w-md mx-auto leading-relaxed">
          No task selected. When a task finishes with changes, the pull request GitHub created for it is linked
          here, together with what its checks say.
        </p>
      </div>
    );
  }

  const task = session.task;
  const completed = session.state === 'completed';
  const outcome = completed && task ? task.outcome : null;
  // The pull request this task is about. For a follow-up it is known from the start.
  const targetUrl = pullRequestUrl(task);
  // The pull request that holds the result (a follow-up that changed nothing has none).
  const prUrl = completed && outcome !== 'no_changes' ? targetUrl : null;
  const updated = outcome === 'pr_updated';
  // The repository was empty: there is no pull request, the result is its first commit.
  const commitUrl = completed ? firstCommitUrl(task) : null;
  const addedUrl = completed ? addedCommitUrl(task) : null;
  const firstCommit = completed && !!task && task.outcome === 'initial_commit';
  const emptyRepository = !!task && (task.empty_repository === true || task.outcome === 'initial_commit');
  const noChanges = completed && !prUrl && !firstCommit;
  // What GitHub said the last time it was asked (null: not asked yet).
  const prState = prUrl ? pullRequestState(session) : null;
  const open = !!prUrl && prState !== 'merged' && prState !== 'closed';
  const unfinished = isUnfinished(session) && (firstCommit || open);
  const tests = summarizeTests(task, session.state);
  const failureText = (task && task.error) || session.note || 'The task failed without an error message.';

  // What kind of task this is: more work on a pull request, a repair of failed checks ...
  const target = session.target || null;
  const pullNumber = (task && task.pull_request) || (target && target.pullRequest) || null;
  const fixingChecks = !!(task && task.fix_checks) || !!(target && target.fixChecks);
  const continuing = !!(task && task.continue_work) || !!(target && target.continueWork);
  const kind = fixingChecks
    ? `Fix the failed checks of ${pullNumber ? `pull request #${pullNumber}` : 'the branch'}`
    : continuing
      ? pullNumber
        ? `Continue the unfinished work on pull request #${pullNumber}`
        : 'Continue the unfinished work'
      : pullNumber
        ? `Follow-up on pull request #${pullNumber}`
        : 'Task';

  const checks: ChecksSummary | null = open
    ? (session.pullRequest && session.pullRequest.checks) || null
    : firstCommit
      ? (session.commitChecks && session.commitChecks.checks) || null
      : null;

  const usage = task && Array.isArray(task.llm_usage) ? task.llm_usage.filter((item) => item && typeof item.provider === 'string') : [];
  const answered = usage.filter((item) => (item.requests || 0) > 0);
  const failedChecks = task && Array.isArray(task.failed_checks) ? task.failed_checks : [];

  const pill = (): { tone: BadgeTone; icon: React.ReactNode; label: string } => {
    if (session.state === 'running') return { tone: 'warn', icon: <Loader2 className="w-3.5 h-3.5 animate-spin" />, label: 'Agent working' };
    if (prUrl && prState === 'merged') return { tone: 'accent', icon: <GitMerge className="w-3.5 h-3.5" />, label: 'Pull request merged' };
    if (prUrl && prState === 'closed') return { tone: 'muted', icon: <GitPullRequestClosed className="w-3.5 h-3.5" />, label: 'Pull request closed' };
    if (unfinished && prUrl) return { tone: 'warn', icon: <AlertTriangle className="w-3.5 h-3.5" />, label: 'Unfinished · pull request ' + (updated ? 'updated' : 'opened') };
    if (unfinished && firstCommit) return { tone: 'warn', icon: <AlertTriangle className="w-3.5 h-3.5" />, label: 'Unfinished · first commit pushed' };
    if (prUrl && updated) return { tone: 'good', icon: <CheckCircle2 className="w-3.5 h-3.5" />, label: 'Pull request updated' };
    if (prUrl) return { tone: 'good', icon: <CheckCircle2 className="w-3.5 h-3.5" />, label: 'Pull request opened' };
    if (firstCommit) return { tone: 'info', icon: <GitCommitHorizontal className="w-3.5 h-3.5" />, label: 'First commit pushed' };
    if (noChanges) return { tone: 'muted', icon: <MinusCircle className="w-3.5 h-3.5" />, label: emptyRepository ? 'Finished · nothing created' : 'Finished · no changes' };
    if (session.state === 'failed') return { tone: 'bad', icon: <XCircle className="w-3.5 h-3.5" />, label: 'Task failed' };
    return { tone: 'warn', icon: <HelpCircle className="w-3.5 h-3.5" />, label: 'Result unknown' };
  };
  const status = pill();

  const changedFiles = task && Array.isArray(task.changed_files) ? task.changed_files : [];
  const verification = task && Array.isArray(task.verification) ? task.verification : [];
  // Results saved by earlier versions (and ones recovered from GitHub) do not say what was read.
  const filesRead = task && !task.recovered && Array.isArray(task.files_read) ? task.files_read : null;

  // A failure the provider's limits caused, with nothing to carry on with.
  const limitFailure = session.state === 'failed' && singleProvider && /rate limit|\b429\b|quota|limit of|time budget|tokens per/i.test(failureText);

  return (
    <Panel
      icon={<GitPullRequest className="w-4 h-4 text-indigo-400" />}
      title="Result"
      aside={
        <span data-testid="result-status">
          <Badge tone={status.tone} className="h-7 px-2.5 rounded-full text-xs">
            {status.icon}
            {status.label}
          </Badge>
        </span>
      }
      data-testid="result-panel"
      data-state={session.state}
      data-pr-state={prState || undefined}
      data-outcome={outcome || undefined}
      data-unfinished={unfinished ? 'true' : undefined}
    >
      <div className="space-y-5">
        <StageProgress task={task} state={session.state} followUp={!!pullNumber} />

        <div>
          <SectionLabel>
            <span data-testid="task-kind">{kind}</span>
          </SectionLabel>
          <p className={`${INSET} text-xs text-slate-200 px-3 py-2.5 font-mono line-clamp-4 whitespace-pre-wrap break-words`}>
            {session.prompt ||
              (task && task.recovered
                ? '(not available: this result was recovered from GitHub)'
                : fixingChecks
                  ? '(no further instructions: the failed checks are the task)'
                  : '—')}
          </p>
          {failedChecks.length > 0 && (
            <p className="text-xs text-slate-400 mt-1.5" data-testid="failed-checks">
              Failed when the task started:{' '}
              {failedChecks.slice(0, MAX_CHECKS_SHOWN).map((check, index) => (
                <span key={`${check.name}-${index}`} className="text-slate-200">
                  {index > 0 ? ', ' : ''}
                  {check.name}
                </span>
              ))}
              {failedChecks.length > MAX_CHECKS_SHOWN ? ` and ${failedChecks.length - MAX_CHECKS_SHOWN} more` : ''}
            </p>
          )}
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-4 gap-3">
          <Fact label="Tests">
            <div className="flex items-center gap-1.5 text-xs font-medium" data-testid="tests-status" data-tone={tests.tone}>
              <ToneIcon tone={tests.tone} />
              <span className={TONE_TEXT[tests.tone]}>{tests.label}</span>
            </div>
          </Fact>

          <Fact label="Branch">
            <span className="text-xs font-mono text-indigo-300 truncate block" data-testid="branch-info">
              {task && task.head_branch
                ? `${task.head_branch} → ${task.base_branch || '?'}`
                : emptyRepository && task && task.base_branch
                  ? `${task.base_branch} · first commit`
                  : session.branch
                    ? `base: ${session.branch}`
                    : '—'}
            </span>
          </Fact>

          <Fact label="Model">
            <span
              className="text-xs font-mono text-slate-300 truncate block"
              data-testid="model-used"
              title={answered.map((item) => `${item.provider} / ${item.model}`).join(' → ') || undefined}
            >
              {answered.length > 1
                ? answered.map((item) => providerName(item.provider)).join(' → ')
                : task && task.llm_model
                  ? `${providerName(task.llm_provider)} · ${task.llm_model}`
                  : '—'}
            </span>
          </Fact>

          <Fact label="Commit author">
            <span className="text-xs text-slate-300 truncate block" data-testid="commit-authorship">
              {task && task.commit_authorship ? AUTHORSHIP[task.commit_authorship] || task.commit_authorship : '—'}
            </span>
          </Fact>
        </div>

        {/* ---- Unfinished work ---- */}
        {unfinished && task && (
          <Callout tone="warn" icon={<AlertTriangle className="w-4 h-4 text-amber-300" />} data-testid="unfinished">
            <div className="font-semibold text-amber-200 mb-1">The work is unfinished</div>
            <p>
              {firstCommit
                ? 'What the agent had built was pushed so that nothing is lost; the project may not build yet. Continuing delivers the rest as a pull request.'
                : 'What the agent had changed is in the pull request so that nothing is lost. Do not merge it as it is: continuing adds the rest to it.'}
            </p>
            {task.unfinished_reason && (
              <p className="mt-1.5 text-amber-100/70 break-words" data-testid="unfinished-reason">
                Why it stopped: <Linkified text={task.unfinished_reason} />
              </p>
            )}
            <div className="mt-2.5">
              <Button variant="primary" onClick={onContinue} disabled={!!startBlocked} title={startBlocked || undefined} data-testid="continue-work">
                <Play className="w-3.5 h-3.5 fill-current" />
                Continue the work
              </Button>
            </div>
          </Callout>
        )}

        {/* ---- What CI says ---- */}
        {checks && (
          <ChecksBlock
            checks={checks}
            subject={firstCommit ? 'this commit' : 'the pull request'}
            onFix={onFixChecks}
            onRefresh={onRefreshStatus}
            refreshing={refreshing}
            startBlocked={startBlocked}
          />
        )}

        {usage.length > 1 && <ModelsUsed usage={usage} />}

        {verification.length > 0 && (
          <div>
            <SectionLabel>Verification runs</SectionLabel>
            <ul className={`${INSET} divide-y divide-slate-800`}>
              {verification.slice(-4).map((run, index) => {
                const result = runOutcome(run);
                const tone = result === 'passed' ? 'text-emerald-400' : result === 'no_tests' ? 'text-slate-400' : 'text-rose-400';
                return (
                  <li
                    key={index}
                    className="flex items-center justify-between gap-3 px-3 py-2 text-xs font-mono"
                    title={result === 'no_tests' && run.note ? `Nothing to test: ${run.note}` : undefined}
                  >
                    <span className="truncate text-slate-300">$ {run.command}</span>
                    <span className={`flex-shrink-0 ${tone}`} data-testid="verification-result">
                      {runLabel(run)}
                    </span>
                  </li>
                );
              })}
            </ul>
            {verification.some((run) => runOutcome(run) === 'no_tests' && run.note) && (
              <p className="text-xs text-slate-500 mt-1.5" data-testid="no-tests-note">
                Nothing to test: {verification.filter((run) => runOutcome(run) === 'no_tests' && run.note).slice(-1)[0].note}.
                This is not a failure of the change.
              </p>
            )}
          </div>
        )}

        {/* Only next to a change someone will review; an unfinished run has not read "nothing". */}
        {filesRead && prUrl && (
          <div data-testid="files-read">
            <SectionLabel>Files the agent looked at</SectionLabel>
            {filesRead.length > 0 ? (
              <p className={`${INSET} text-xs font-mono text-slate-300 px-3 py-2 break-words`}>
                {filesRead.slice(0, MAX_READ_FILES_SHOWN).join(', ')}
                {filesRead.length > MAX_READ_FILES_SHOWN && (
                  <span className="text-slate-500"> and {filesRead.length - MAX_READ_FILES_SHOWN} more</span>
                )}
              </p>
            ) : task && task.new_project ? (
              <p className="text-xs text-slate-400">None: the repository had no code yet, so the project was created from scratch.</p>
            ) : (
              <p className="text-xs text-amber-400 flex items-start gap-1.5">
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                None. The agent did not open any file of the repository, so check what it wrote against the code.
              </p>
            )}
          </div>
        )}

        {changedFiles.length > 0 && (
          <div>
            <SectionLabel>
              {changedFiles.length} file{changedFiles.length === 1 ? '' : 's'} {emptyRepository ? 'created' : 'changed'}
            </SectionLabel>
            <ul className={`${INSET} divide-y divide-slate-800`}>
              {changedFiles.slice(0, MAX_FILES_SHOWN).map((file) => {
                const style = FILE_STATUS[file.status] || { label: file.status, className: 'text-slate-300' };
                return (
                  <li key={file.path} className="flex items-center justify-between gap-3 px-3 py-2 text-xs font-mono">
                    <span className="truncate text-slate-200">{file.path}</span>
                    <span className="flex-shrink-0 space-x-2">
                      <span className={style.className}>{style.label}</span>
                      {file.additions !== null && file.deletions !== null && (
                        <span className="text-slate-500 tabular-nums">
                          <span className="text-emerald-500">+{file.additions}</span>{' '}
                          <span className="text-rose-500">−{file.deletions}</span>
                        </span>
                      )}
                    </span>
                  </li>
                );
              })}
              {changedFiles.length > MAX_FILES_SHOWN && (
                <li className="px-3 py-2 text-xs font-mono text-slate-500">… and {changedFiles.length - MAX_FILES_SHOWN} more</li>
              )}
            </ul>
          </div>
        )}

        {task && task.summary && session.state !== 'failed' && (
          <div>
            <SectionLabel>{task.recovered ? 'Summary' : 'Agent summary'}</SectionLabel>
            <p className={`${INSET} text-[13px] leading-relaxed text-slate-200 px-3 py-2.5 whitespace-pre-wrap break-words`}>
              <Linkified text={plainProse(task.summary)} />
            </p>
          </div>
        )}

        {/* ---- Outcome ---- */}
        {prUrl && task && (
          <div className="space-y-2.5">
            <a
              href={prUrl}
              target="_blank"
              rel="noreferrer noopener"
              data-testid="pr-link"
              className={`w-full flex items-center justify-center gap-2 text-white font-medium text-[13px] h-10 px-4 rounded-lg transition-colors shadow-lg ${
                prState === 'merged'
                  ? 'bg-purple-600 hover:bg-purple-500 shadow-purple-600/20'
                  : prState === 'closed'
                    ? 'bg-slate-700 hover:bg-slate-600 shadow-slate-900/20'
                    : 'bg-emerald-600 hover:bg-emerald-500 shadow-emerald-600/20'
              }`}
            >
              {prState === 'merged' ? (
                <GitMerge className="w-4 h-4" />
              ) : prState === 'closed' ? (
                <GitPullRequestClosed className="w-4 h-4" />
              ) : (
                <GitPullRequest className="w-4 h-4" />
              )}
              <span>
                {prState === 'merged' ? 'View merged' : prState === 'closed' ? 'View closed' : updated ? 'View' : 'Review'} pull request
                {task.pr_number ? ` #${task.pr_number}` : ''} on GitHub
              </span>
              <ExternalLink className="w-3.5 h-3.5" />
            </a>
            <div className="text-[11px] font-mono text-slate-500 break-all text-center">{prUrl}</div>

            {(addedUrl || open) && (
              <div className="flex flex-wrap items-center justify-center gap-2">
                {addedUrl && (
                  <a href={addedUrl} target="_blank" rel="noreferrer noopener" className={buttonClass('secondary')} data-testid="added-commit-link">
                    <GitCommitHorizontal className="w-4 h-4" />
                    The commit that was added{task.commit_sha ? ` · ${task.commit_sha.slice(0, 7)}` : ''}
                    <ExternalLink className="w-3.5 h-3.5" />
                  </a>
                )}
                {open && (
                  <Button onClick={onRequestChanges} data-testid="request-changes" title="Put this pull request into the task form and describe what else it needs">
                    <MessageSquarePlus className="w-4 h-4" />
                    Ask for more changes
                  </Button>
                )}
              </div>
            )}

            {prState === 'merged' && (
              <p className="text-xs text-purple-200/80 text-center" data-testid="pr-state-note">
                This pull request has been merged into {task.base_branch || 'the base branch'}.
              </p>
            )}
            {prState === 'closed' && (
              <p className="text-xs text-slate-400 text-center" data-testid="pr-state-note">
                This pull request was closed without being merged.
              </p>
            )}
            {/* Advice for the reviewer: pointless once the pull request is merged or closed. */}
            {tests.advice && open && !unfinished && (
              <p
                className={`text-xs flex items-start gap-1.5 ${tests.tone === 'neutral' ? 'text-slate-400' : 'text-amber-400'}`}
                data-testid="tests-advice"
              >
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                {tests.advice}
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

        {firstCommit && task && (
          <div className="space-y-2.5" data-testid="first-commit">
            {commitUrl ? (
              <a
                href={commitUrl}
                target="_blank"
                rel="noreferrer noopener"
                data-testid="commit-link"
                className="w-full flex items-center justify-center gap-2 bg-sky-600 hover:bg-sky-500 text-white font-medium text-[13px] h-10 px-4 rounded-lg transition-colors shadow-lg shadow-sky-600/20"
              >
                <GitCommitHorizontal className="w-4 h-4" />
                <span>View the first commit{task.commit_sha ? ` ${task.commit_sha.slice(0, 7)}` : ''} on GitHub</span>
                <ExternalLink className="w-3.5 h-3.5" />
              </a>
            ) : (
              <p className="text-xs text-slate-300 text-center font-mono">
                First commit {task.commit_sha ? task.commit_sha.slice(0, 10) : ''} on {task.base_branch || 'the default branch'}
              </p>
            )}
            {commitUrl && <div className="text-[11px] font-mono text-slate-500 break-all text-center">{commitUrl}</div>}
            <p className="text-xs leading-relaxed text-sky-200/80" data-testid="first-commit-note">
              The repository was empty, so there was no branch a pull request could be opened against. The files
              were committed directly to {task.base_branch || 'the default branch'}: the repository now has its
              first commit and can be cloned, changed by further tasks (as pull requests) or imported into a
              hosting platform.
            </p>
            {tests.advice && !unfinished && (
              <p
                className={`text-xs flex items-start gap-1.5 ${tests.tone === 'bad' ? 'text-amber-400' : 'text-slate-400'}`}
                data-testid="tests-advice"
              >
                <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
                {tests.advice}
              </p>
            )}
            {task.recovered && (
              <p className="text-xs text-slate-400">
                This result was looked up on GitHub because the server instance that ran the task is no longer
                available. The run&apos;s log and details could not be recovered, only the commit itself.
              </p>
            )}
          </div>
        )}

        {noChanges && (
          <div className={`${INSET} px-3.5 py-3 text-xs leading-relaxed text-slate-300`} data-testid="no-changes">
            {emptyRepository
              ? 'The agent finished without creating any files, so nothing was pushed. The repository is still empty.'
              : fixingChecks && task && !task.sandbox
                ? `Nothing to fix: no check has failed on ${pullNumber ? `pull request #${pullNumber}` : 'the branch'}.`
                : pullNumber
                  ? `The agent finished without changing any files, so nothing was added to pull request #${pullNumber}.`
                  : 'The agent finished without changing any files, so no pull request was opened.'}
            {targetUrl && (
              <>
                {' '}
                <a href={targetUrl} target="_blank" rel="noreferrer noopener" className="text-blue-400 hover:text-blue-300 underline underline-offset-2">
                  Open the pull request
                </a>
              </>
            )}
          </div>
        )}

        {session.state === 'failed' && (
          <Callout tone="bad" icon={<XCircle className="w-4 h-4 text-rose-300" />} data-testid="failure">
            <div className="font-semibold text-rose-200 mb-1">
              {emptyRepository ? 'The task did not complete' : pullNumber ? `Pull request #${pullNumber} was not changed` : 'No pull request was opened'}
            </div>
            <p className="whitespace-pre-wrap break-words">
              <Linkified text={failureText} />
            </p>
            <div className="mt-2.5 flex flex-wrap items-center gap-2">
              <Button onClick={onRetry} disabled={!!startBlocked} title={startBlocked || undefined} data-testid="retry">
                <RotateCcw className="w-4 h-4" />
                Run again
              </Button>
              {limitFailure && (
                <a
                  href={routeHref('settings', 'models')}
                  className={buttonClass('secondary')}
                  title="Only one model provider is set up. With a key for a second one the agent carries on with it instead of waiting."
                  data-testid="fallback-hint"
                >
                  <Plus className="w-4 h-4" />
                  Add a second model provider
                </a>
              )}
            </div>
          </Callout>
        )}

        {session.state === 'interrupted' && (
          <Callout tone="warn" icon={<AlertTriangle className="w-4 h-4 text-amber-300" />} data-testid="interrupted">
            <div className="font-semibold text-amber-200 mb-1">The connection to this run was lost</div>
            <p className="whitespace-pre-wrap break-words">
              <Linkified
                text={
                  session.note ||
                  'The result did not arrive. If the run finished, its pull request (or first commit) can be found by checking again.'
                }
              />
            </p>
            <div className="mt-2.5">
              <Button onClick={onRecheck} disabled={rechecking}>
                <RefreshCw className={`w-4 h-4 ${rechecking ? 'animate-spin' : ''}`} />
                {rechecking ? 'Checking…' : 'Check again'}
              </Button>
            </div>
          </Callout>
        )}

        {session.state === 'running' && (
          <p className="text-xs text-slate-500 italic text-center">
            {emptyRepository
              ? 'The repository is empty: a link to its first commit appears here once GitHub has stored it.'
              : pullNumber
                ? `The commit is added to pull request #${pullNumber} once the agent is done.`
                : 'A pull request link appears here only once GitHub has created the pull request.'}
          </p>
        )}
      </div>
    </Panel>
  );
}
