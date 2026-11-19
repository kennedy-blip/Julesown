import { expect, openWorkspace, test } from './standin';

test.describe('a deployment with an access key', () => {
  test('asks for the key, refuses a wrong one and remembers the right one', async ({ page, standin }) => {
    await standin.reset({ env: { JULES_ACCESS_KEY: 'open-sesame' } });
    await openWorkspace(page);

    await expect(page.getByTestId('banner-access')).toBeVisible();
    await expect(page.getByTestId('repo-option')).toHaveCount(0);

    const field = page.getByLabel('Access key');
    await field.fill('not-the-key');
    await expect(page.getByTestId('access-key-status')).not.toHaveAttribute('data-accepted', 'true');
    await expect(page.getByTestId('repo-option')).toHaveCount(0);

    await field.fill('open-sesame');
    await expect(page.getByTestId('repo-option')).toHaveCount(5);
    await expect(page.getByTestId('banner-access')).toHaveCount(0);

    // It is kept in this browser: a reload does not ask again.
    await page.reload();
    await expect(page.getByTestId('repo-option')).toHaveCount(5);
    await expect(page.getByTestId('banner-access')).toHaveCount(0);
  });
});
