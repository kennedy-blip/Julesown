'use client';

import React from 'react';
import { Check, Loader2, Minus, X, HelpCircle } from 'lucide-react';
import type { TaskRecord } from '@/lib/api';
import type { SessionState } from '@/lib/sessions';

interface Stage {
  id: string;
  label: string;
  title: string;
}

/** The steps the backend really performs, in order (see services/runner.py). */
const STAGES: Stage[] = [
  { id: 'configuration', label: 'Config', title: 'Check that the server has the keys it needs' },
  { id: 'github', label: 'GitHub', title: 'Confirm write access to the repository and find the base branch' },
  { id: 'sandbox', label: 'Sandbox', title: 'Start an isolated sandbox' },
  { id: 'clone', label: 'Clone', title: 'Clone the repository into the sandbox' },
  { id: 'agent', label: 'Agent', title: 'The model reads, edits and tests the code' },
  { id: 'commit', label: 'Commit', title: 'Collect the changed files and push them to a new branch' },
  { id: 'pull_request', label: 'Pull request', title: 'Open the pull request on GitHub' },
];

/** An empty repository: there is nothing to clone and no branch for a pull request. */
const EMPTY_REPOSITORY_STAGES: Stage[] = [
  STAGES[0],
  { id: 'github', label: 'GitHub', title: 'Confirm write access to the repository and see that it is empty' },
  STAGES[2],
  { id: 'clone', label: 'Work tree', title: 'Set up an empty work tree in the sandbox' },
  { id: 'agent', label: 'Agent', title: 'The model creates the project and checks it' },
  { id: 'commit', label: 'Collect', title: 'Collect the files that were created' },
  { id: 'push', label: 'First commit', title: "Commit the files to the repository's default branch" },
];

/** More work on a pull request that exists: the result is a commit added to it. */
const FOLLOW_UP_STAGES: Stage[] = [
  STAGES[0],
  { id: 'github', label: 'GitHub', title: 'Confirm write access, read the pull request and what its checks say' },
  STAGES[2],
  { id: 'clone', label: 'Clone', title: "Clone the pull request's branch into the sandbox" },
  STAGES[4],
  { id: 'commit', label: 'Commit', title: 'Collect the changed files' },
  { id: 'pull_request', label: 'Update PR', title: 'Add the commit to the pull request' },
];

type StepState = 'pending' | 'active' | 'done' | 'failed' | 'skipped' | 'unknown';

function stagesFor(task: TaskRecord | null, followUp: boolean): Stage[] {
  const emptyRepository = !!task && (task.empty_repository === true || task.outcome === 'initial_commit');
  if (emptyRepository) return EMPTY_REPOSITORY_STAGES;
  return followUp || (!!task && (!!task.pull_request || task.outcome === 'pr_updated')) ? FOLLOW_UP_STAGES : STAGES;
}

function stepStates(stages: Stage[], task: TaskRecord | null, state: SessionState): StepState[] {
  const stage = task && task.stage ? task.stage : 'queued';
  const current = stages.findIndex((item) => item.id === stage);
  const last = stages[stages.length - 1].id;

  // A task about failed checks ends right after GitHub was asked when nothing has failed.
  const endedEarly = !!task && task.outcome === 'no_changes' && !task.sandbox && !task.recovered;
  const githubStep = stages.findIndex((item) => item.id === 'github');

  return stages.map((item, index) => {
    if (state === 'completed' || stage === 'done') {
      const noChanges = task ? task.outcome === 'no_changes' : false;
      if (endedEarly && index > githubStep) return 'skipped';
      if (noChanges && (item.id === 'commit' || item.id === last)) return 'skipped';
      // A record recovered from GitHub only proves that the result exists there.
      if (task && task.recovered && item.id !== last) return 'skipped';
      return 'done';
    }
    if (current < 0) return 'pending';
    if (index < current) return 'done';
    if (index > current) return 'pending';
    if (state === 'failed') return 'failed';
    if (state === 'interrupted') return 'unknown';
    return 'active';
  });
}

const STEP_STYLES: Record<StepState, string> = {
  pending: 'border-slate-800 text-slate-500 bg-transparent',
  active: 'border-yellow-400/40 text-yellow-300 bg-yellow-400/10',
  done: 'border-emerald-500/30 text-emerald-400 bg-emerald-500/10',
  failed: 'border-rose-500/40 text-rose-400 bg-rose-500/10',
  skipped: 'border-slate-800 text-slate-500 bg-slate-800/40',
  unknown: 'border-amber-500/40 text-amber-400 bg-amber-500/10',
};

function StepIcon({ state }: { state: StepState }) {
  switch (state) {
    case 'active':
      return <Loader2 className="w-3 h-3 animate-spin" />;
    case 'done':
      return <Check className="w-3 h-3" />;
    case 'failed':
      return <X className="w-3 h-3" />;
    case 'skipped':
      return <Minus className="w-3 h-3" />;
    case 'unknown':
      return <HelpCircle className="w-3 h-3" />;
    default:
      return <span className="w-1.5 h-1.5 rounded-full bg-slate-600 mx-[3px]" />;
  }
}

export function StageProgress({
  task,
  state,
  followUp = false,
}: {
  task: TaskRecord | null;
  state: SessionState;
  /** The task adds to an existing pull request (known before the server says so). */
  followUp?: boolean;
}) {
  const stages = stagesFor(task, followUp);
  const states = stepStates(stages, task, state);
  return (
    <ol className="flex flex-wrap gap-1.5" aria-label="Task progress" data-testid="stage-progress">
      {stages.map((item, index) => (
        <li
          key={item.id}
          title={item.title}
          data-stage={item.id}
          data-state={states[index]}
          className={`inline-flex items-center gap-1.5 h-7 px-2.5 rounded-md border text-[11px] font-medium ${STEP_STYLES[states[index]]}`}
        >
          <StepIcon state={states[index]} />
          <span>{item.label}</span>
        </li>
      ))}
    </ol>
  );
}
