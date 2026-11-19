import { PYTHON, expect, finishedResult, openWorkspace, startTask, test } from './standin';
import type { Step } from './standin';

const CHECK = `${PYTHON} -c 'from calc import add; assert add(2, 3) == 5 and add(-1, 1) == 0'`;

// The second half of the work: a regression test for what the first run fixed.
const ADD_THE_TEST: Step[] = [
  [['read_file', { path: 'tests/test_calc.py' }]],
  [
    [
      'write_file',
      {
        path: 'tests/test_calc.py',
        content:
          'from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n\n\ndef test_add_negative():\n    assert add(-1, 1) == 0\n',
      },
    ],
  ],
  [['run_tests', { command: CHECK }]],
  [['finish', { summary: 'Added a regression test for negative numbers.', pr_title: 'Fix add() and test it' }]],
];

test.describe('work that needs more than one run', () => {
  test('is delivered marked as unfinished, and continued on the same pull request', async ({ page, standin }) => {
    // Two turns are enough to fix the bug, not to check the fix.
    await standin.reset({ env: { JULES_MAX_AGENT_TURNS: '2' } });
    await openWorkspace(page);
    await startTask(page, 'acme/demo', 'Fix add() and add a regression test');

    const result = await finishedResult(page);
    await expect(result).toHaveAttribute('data-outcome', 'pull_request');
    await expect(page.getByTestId('unfinished')).toBeVisible();
    await expect(page.getByTestId('unfinished-reason')).toContainText(/turn/i);
    let state = await standin.state();
    expect(state.pulls['acme/demo']).toHaveLength(1);
    expect(state.pulls['acme/demo'][0].title).toMatch(/^\[Unfinished\]/);

    await standin.update({ env: { JULES_MAX_AGENT_TURNS: null }, script: ADD_THE_TEST });
    await page.getByTestId('continue-work').click();

    // A second task, and its result: a commit added to the pull request that was there.
    await expect(page.getByTestId('history-item')).toHaveCount(2);
    const continued = await finishedResult(page);
    await expect(continued).toHaveAttribute('data-outcome', 'pr_updated');
    await expect(page.getByTestId('unfinished')).toHaveCount(0);
    await expect(page.getByTestId('tests-status')).toContainText('Passed in the sandbox');
    state = await standin.state();
    expect(state.pulls['acme/demo']).toHaveLength(1);
    expect(state.pulls['acme/demo'][0].title).not.toMatch(/Unfinished/);
  });

  test('builds a project in an empty repository as its first commit', async ({ page, standin }) => {
    await standin.reset({
      script: [
        [
          ['write_file', { path: 'index.html', content: '<!doctype html>\n<title>Hello</title>\n<h1>Hello</h1>\n' }],
          ['write_file', { path: 'README.md', content: '# Hello\n\nOpen index.html in a browser. Any static host can serve it.\n' }],
        ],
        [['run_command', { command: 'test -s index.html && echo ok' }]],
        [['finish', { summary: 'A one-page static site.', pr_title: 'Start the site' }]],
      ],
    });
    await openWorkspace(page);
    await startTask(page, 'acme/empty', 'Create a one-page site that says hello');

    const result = await finishedResult(page);
    await expect(result).toHaveAttribute('data-state', 'completed');
    await expect(result).toHaveAttribute('data-outcome', 'initial_commit');
    await expect(page.getByTestId('first-commit')).toBeVisible();
    await expect(page.getByTestId('commit-link')).toHaveAttribute('href', /^https:\/\/github\.com\/acme\/empty\/commit\/[0-9a-f]{40}$/);
    // Nothing to open a pull request against: there is none.
    expect((await standin.state()).pulls['acme/empty']).toBeUndefined();
  });
});
