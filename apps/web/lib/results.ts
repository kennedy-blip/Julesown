import type { TaskRecord, VerificationOutcome, VerificationRun } from './api';
import type { SessionState } from './sessions';

/**
 * How a task's testing is presented.
 *
 * good     - tests ran and passed
 * bad      - the last real test run failed
 * caution  - code was changed and nothing verified it
 * neutral  - nothing to worry about: there was nothing to test
 * pending  - not known (yet)
 */
export type Tone = 'good' | 'bad' | 'caution' | 'neutral' | 'pending';

export interface TestsSummary {
  tone: Tone;
  label: string;
  /** What a reviewer should know before merging; null when there is nothing to add. */
  advice: string | null;
}

export function runOutcome(run: VerificationRun): VerificationOutcome {
  if (run.outcome) return run.outcome;
  if (run.timed_out) return 'timed_out';
  return run.passed ? 'passed' : 'failed';
}

export function runLabel(run: VerificationRun): string {
  switch (runOutcome(run)) {
    case 'passed':
      return 'passed';
    case 'timed_out':
      return 'timed out';
    case 'no_tests':
      return 'no tests to run';
    default:
      return `exit ${run.exit_code}`;
  }
}

export function summarizeTests(task: TaskRecord | null, state: SessionState): TestsSummary {
  const status = task ? task.tests_status : null;
  const docsOnly = !!task && task.docs_only === true;
  // A warning about missing tests is about a change someone is going to review.
  // A task that failed, changed nothing or is still running has no such change (yet).
  const hasChange = state === 'completed' && !!task && task.outcome !== 'no_changes';

  if (!hasChange && (status === 'not_run' || status === 'no_tests')) {
    return {
      tone: 'pending',
      label: status === 'no_tests' ? 'No tests in this repository' : 'Not run',
      advice: null,
    };
  }

  switch (status) {
    case 'passed':
      return { tone: 'good', label: 'Passed in the sandbox', advice: null };
    case 'failed':
      return {
        tone: 'bad',
        label: 'Last run failed',
        advice:
          "The agent's last test run failed. The pull request says so too. Review it carefully before merging.",
      };
    case 'no_tests':
      // A test command ran but had nothing to test (for example npm's placeholder
      // "test" script). That is not a failure of the change.
      return docsOnly
        ? { tone: 'neutral', label: 'None needed · docs only', advice: null }
        : {
            tone: 'neutral',
            label: 'No tests in this repository',
            advice: 'This repository has no automated tests, so the change is untested. Review it before merging.',
          };
    case 'not_run':
      return docsOnly
        ? { tone: 'neutral', label: 'None needed · docs only', advice: null }
        : { tone: 'caution', label: 'No tests were run', advice: 'The agent did not run any tests for this change.' };
    default: {
      // Without a report from the server, say so instead of implying nothing was run.
      const unknown = state === 'interrupted' || (task !== null && task.recovered);
      return {
        tone: 'pending',
        label: state === 'running' ? 'Not run yet' : unknown ? 'Unknown' : 'Not run',
        advice: null,
      };
    }
  }
}

/** Model prose for a plain-text panel: Markdown bold markers and inline-code ticks are dropped. */
export function plainProse(text: string): string {
  // No look-behind here: Safari before 16.4 cannot even parse it.
  return text
    .replace(/\*\*(\S(?:[^\n]*?\S)?)\*\*/g, '$1')
    .replace(/`([^`\n]+)`/g, '$1')
    .replace(/^\s{0,3}#{1,6}\s+/gm, '');
}
