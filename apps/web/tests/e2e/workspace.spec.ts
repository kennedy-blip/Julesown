import { expect, finishedResult, openWorkspace, startTask, test } from './standin';

test.describe('the workspace', () => {
  test('says what the server is ready for and lists the repositories', async ({ page }) => {
    await openWorkspace(page);

    await expect(page.getByTestId('system-status')).toHaveText('Ready');
    await expect(page.getByTestId('readiness')).toHaveAttribute('data-readiness', 'ready');
    await expect(page.getByTestId('repo-option')).toHaveCount(5);
    await expect(page.getByTestId('repo-summary')).toContainText('5 repositories');
    // Nothing can be started before a repository and a task are named.
    await expect(page.getByTestId('run-task')).toBeDisabled();

    await page.getByTestId('repo-search').fill('web');
    await expect(page.getByTestId('repo-option')).toHaveCount(1);
    await expect(page.getByTestId('repo-option')).toContainText('acme/website');
  });

  test('runs a task and shows the pull request it opened', async ({ page, standin }) => {
    await openWorkspace(page);
    await startTask(page, 'acme/demo', 'Fix the failing test');

    const result = await finishedResult(page);
    await expect(result).toHaveAttribute('data-state', 'completed');
    await expect(result).toHaveAttribute('data-outcome', 'pull_request');
    await expect(page.getByTestId('result-status')).toHaveText('Pull request opened');
    await expect(page.getByTestId('pr-link')).toHaveAttribute('href', 'https://github.com/acme/demo/pull/1');
    await expect(page.getByTestId('tests-status')).toContainText('Passed in the sandbox');
    await expect(page.getByTestId('files-read')).toContainText('calc.py');
    // The log is the server's, line by line: what it checked, what the agent did.
    const log = page.getByTestId('terminal-body');
    await expect(log).toContainText('GitHub access confirmed for acme/demo');
    await expect(log).toContainText('replace_in_file calc.py');
    await expect(page.getByTestId('terminal-state')).toHaveText('Finished');

    // The pull request exists where the link points, and the server's own key paid for it.
    const state = await standin.state();
    expect(state.pulls['acme/demo']).toEqual([
      expect.objectContaining({ number: 1, title: 'Fix add()', state: 'open', base: 'main' }),
    ]);
    expect(state.requests.map((request) => `${request.provider}:${request.key}`)).toEqual(
      Array(4).fill('groq:standin-groq-key')
    );
  });

  test('shows what the agent does while it is still doing it', async ({ page, standin }) => {
    await standin.reset({
      script: [
        [['read_file', { path: 'calc.py' }]],
        // (The log shows the command as well as its output: what is waited for is only in the output.)
        [['run_command', { command: 'sleep 3 && echo "waited $((6 * 7)) ticks"', timeout: 30 }]],
        [['finish', { summary: 'Looked at the calculator and changed nothing.', pr_title: 'Nothing' }]],
      ],
    });
    await openWorkspace(page);
    await startTask(page, 'acme/demo', 'Look at the calculator');

    // The first step is on screen while the second one is still running: the
    // log is streamed, not delivered in one piece at the end.
    const log = page.getByTestId('terminal-body');
    await expect(log).toContainText('read_file calc.py');
    await expect(log).not.toContainText('waited 42 ticks');
    await expect(page.getByTestId('terminal-state')).toHaveText('Running');
    await expect(page.getByTestId('run-task')).toBeDisabled();

    const result = await finishedResult(page);
    await expect(log).toContainText('waited 42 ticks');
    // Nothing was changed, so there is nothing to open a pull request for - and none is claimed.
    await expect(result).toHaveAttribute('data-state', 'completed');
    await expect(page.getByTestId('no-changes')).toBeVisible();
    await expect(page.getByTestId('pr-link')).toHaveCount(0);
    expect((await standin.state()).pulls['acme/demo']).toBeUndefined();
  });

  test('keeps a task in the history across a reload', async ({ page }) => {
    await openWorkspace(page);
    await startTask(page, 'acme/demo', 'Fix the failing test');
    await finishedResult(page);
    await expect(page.getByTestId('history-item')).toHaveCount(1);

    await page.reload();
    const item = page.getByTestId('history-item');
    await expect(item).toHaveCount(1);
    await expect(item).toContainText('acme/demo');
    await expect(item).toContainText('PR opened');

    await item.click();
    await expect(page.getByTestId('pr-link')).toHaveAttribute('href', 'https://github.com/acme/demo/pull/1');
  });

  test('says what is missing when the server has no model key', async ({ page, standin }) => {
    await standin.reset({ env: { GROQ_API_KEY: null } });
    await openWorkspace(page);

    await expect(page.getByTestId('readiness')).toHaveAttribute('data-readiness', 'incomplete');
    await expect(page.getByTestId('banner-incomplete')).toContainText('GROQ_API_KEY');
    await expect(page.getByTestId('add-provider-key')).toBeVisible();
  });

  test('carries on with the next provider when the first one has used up its day', async ({ page, standin }) => {
    await standin.reset({
      env: { SAMBANOVA_API_KEY: 'standin-sambanova-key' },
      providers: { groq: { mode: 'day_used', after: 2 } },
    });
    await openWorkspace(page);
    await startTask(page, 'acme/demo', 'Fix the failing test');

    const result = await finishedResult(page);
    await expect(result).toHaveAttribute('data-outcome', 'pull_request');
    await expect(page.getByTestId('terminal-body')).toContainText(/daily limit.*switching to/i);
    // The result names both, in the order they worked, and says why the first one stopped.
    await expect(page.getByTestId('model-used')).toHaveText('Groq → SambaNova');
    const models = page.getByTestId('models-used');
    await expect(models.locator('li[data-provider="groq"]')).toContainText('daily limit is used up');
    await expect(models.locator('li[data-provider="sambanova"]')).toContainText('2 requests');

    const state = await standin.state();
    const answering = state.requests.filter((request) => request.outcome === 'ok').map((request) => request.provider);
    expect(answering).toEqual(['groq', 'groq', 'sambanova', 'sambanova']);
  });
});
