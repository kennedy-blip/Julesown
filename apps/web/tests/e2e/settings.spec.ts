import { expect, finishedResult, openSettings, openWorkspace, providerCard, startTask, test } from './standin';

const ACCOUNT = '0123456789abcdef0123456789abcdef';

test.describe('settings: model providers', () => {
  test('lists the providers by what they cost and shows which the server has a key for', async ({ page }) => {
    await openSettings(page, 'models');

    await expect(page.getByTestId('providers-free')).toContainText('Free within limits');
    await expect(page.getByTestId('providers-prepaid')).toContainText('Paid from a balance');
    const cards = page.getByTestId('provider-card');
    await expect(cards).toHaveCount(9);
    // The free ones first, in the order the server tries them; then the two that spend a balance.
    expect(await cards.evaluateAll((all) => all.map((card) => card.getAttribute('data-provider')))).toEqual([
      'groq',
      'gemini',
      'nvidia',
      'zai',
      'cloudflare',
      'sambanova',
      'openrouter',
      'pollinations',
      'deepseek',
    ]);
    await expect(providerCard(page, 'groq').getByTestId('provider-key-state')).toContainText(/server/i);

    // The two that cost the most stay out of the way until they are asked for.
    await expect(providerCard(page, 'openai')).toHaveCount(0);
    await page.getByTestId('more-providers').click();
    await expect(providerCard(page, 'openai')).toBeVisible();
    await expect(providerCard(page, 'anthropic')).toBeVisible();
  });

  test('tries out a key that is typed in, and shows the refusal of a wrong one', async ({ page, standin }) => {
    await openSettings(page, 'models');
    const card = providerCard(page, 'deepseek');

    await card.getByTestId('provider-key').fill('standin-deepseek-key');
    await card.getByTestId('provider-test').click();
    await expect(card.getByTestId('provider-test-result')).toHaveAttribute('data-phase', 'done');
    await expect(card.getByTestId('provider-test-result')).toContainText(
      'DeepSeek accepted the key and deepseek-flash called the test tool'
    );
    // The key travelled with that one request, and to that provider only.
    const state = await standin.state();
    expect(state.requests).toEqual([
      expect.objectContaining({ provider: 'deepseek', key: 'standin-deepseek-key', outcome: 'ok', tools: ['confirm'] }),
    ]);

    await card.getByTestId('provider-key').fill('a-bad-key');
    await card.getByTestId('provider-test').click();
    await expect(card.getByTestId('provider-test-result')).toContainText('Invalid API Key');
  });

  test('asks for the account ID that Cloudflare needs beside its token', async ({ page, standin }) => {
    await openSettings(page, 'models');
    const card = providerCard(page, 'cloudflare');

    await card.getByTestId('provider-key').fill('standin-cloudflare-token');
    await expect(card.getByTestId('provider-account-missing')).toBeVisible();
    await expect(card.getByTestId('provider-test')).toBeDisabled();

    await card.getByTestId('provider-account').fill('not an account id');
    await expect(card.getByTestId('provider-account-problem')).toBeVisible();
    await expect(card.getByTestId('provider-test')).toBeDisabled();

    await card.getByTestId('provider-account').fill(ACCOUNT);
    await expect(card.getByTestId('provider-account-missing')).toHaveCount(0);
    await card.getByTestId('provider-test').click();
    // It has two models: each one is tried, and each has a line of its own.
    const tried = card.getByTestId('provider-test-model');
    await expect(tried).toHaveCount(2);
    await expect(tried.first()).toContainText('@cf/zai-org/glm-4.7-flash called the test tool');
    const state = await standin.state();
    expect(state.requests[0]).toEqual(
      expect.objectContaining({ provider: 'cloudflare', key: 'standin-cloudflare-token', account: ACCOUNT })
    );
  });

  test('sends a key that is kept in the browser with the task that needs it', async ({ page, standin }) => {
    await openSettings(page, 'models');
    await providerCard(page, 'deepseek').getByTestId('provider-key').fill('standin-deepseek-key');
    await page.getByTestId('primary-provider').selectOption('deepseek');

    await openWorkspace(page);
    await expect(page.getByTestId('model-label')).toContainText('DeepSeek');
    await startTask(page, 'acme/demo', 'Fix the failing test');
    const result = await finishedResult(page);
    await expect(result).toHaveAttribute('data-outcome', 'pull_request');
    await expect(page.getByTestId('model-used')).toContainText('DeepSeek');

    const state = await standin.state();
    expect(new Set(state.requests.map((request) => `${request.provider}:${request.key}`))).toEqual(
      new Set(['deepseek:standin-deepseek-key'])
    );
  });

  test('shows what the server is configured with, without the values', async ({ page }) => {
    await openSettings(page, 'about');

    const variables = page.getByTestId('variables');
    await expect(variables).toContainText('GROQ_API_KEY');
    await expect(variables).toContainText('SAMBANOVA_API_KEY');
    await expect(variables).not.toContainText('standin-groq-key');
  });
});
