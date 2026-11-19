import { test as base, expect } from '@playwright/test';
import type { APIRequestContext, Locator, Page } from '@playwright/test';

/**
 * The backend these tests run against: the real API on local stand-ins
 * (apps/api/tests/standin_server.py). A test says how the world should be -
 * which keys the server has, how a provider behaves, what the model does -
 * and then uses the dashboard like a person would.
 */
const API = `http://127.0.0.1:${process.env.E2E_API_PORT || 8011}`;

export type Behaviour = 'ok' | 'bad_key' | 'day_used' | 'no_credit' | 'down';
/** What the model does in one turn: a reply in words, or calls of the agent's tools. */
export type Step = string | Array<[tool: string, args: Record<string, unknown>]>;

export interface World {
  /** Settings of the server to add - or, with null, to take away. */
  env?: Record<string, string | null>;
  providers?: Record<string, Behaviour | { mode: Behaviour; after?: number }>;
  script?: Step[];
}

export interface ProviderRequest {
  provider: string;
  model: string;
  key: string;
  account: string | null;
  turn: number;
  tools: string[];
  outcome: Behaviour;
}

export interface State {
  requests: ProviderRequest[];
  answered: Record<string, number>;
  pulls: Record<string, Array<{ number: number; title: string; state: string; head: string; base: string }>>;
  comments: number;
}

export class StandIn {
  constructor(private readonly request: APIRequestContext) {}

  /** Start from scratch: the demo repositories, no task, no pull request. */
  async reset(world: World = {}): Promise<void> {
    await this.send('reset', world);
  }

  /** Change only what is named, for the next step of a test. */
  async update(world: World): Promise<void> {
    await this.send('update', world);
  }

  async state(): Promise<State> {
    const response = await this.request.get(`${API}/__standin/state`);
    expect(response.ok(), 'the stand-in server answers').toBeTruthy();
    return (await response.json()) as State;
  }

  private async send(what: 'reset' | 'update', world: World): Promise<void> {
    const response = await this.request.post(`${API}/__standin/${what}`, { data: world });
    expect(response.ok(), `${what}: ${await response.text()}`).toBeTruthy();
  }
}

export const test = base.extend<{ standin: StandIn }>({
  // For every test, whether it asks for the stand-in or not: none of them
  // starts with what another one left behind.
  standin: [
    async ({ request }, use) => {
      const standin = new StandIn(request);
      await standin.reset();
      await use(standin);
    },
    { auto: true },
  ],
});

export { expect };

/** The interpreter the stand-in server runs on, for commands in a script. */
export const PYTHON = '{python}';

export async function openWorkspace(page: Page): Promise<void> {
  await page.goto('/');
  await expect(page.getByTestId('system-status')).not.toHaveText(/Checking/i);
}

export async function openSettings(page: Page, tab: 'models' | 'instructions' | 'commits' | 'automation' | 'access' | 'about'): Promise<void> {
  await page.goto(`/#/settings/${tab}`);
  await expect(page.getByTestId(`settings-${tab}`)).toBeVisible();
}

export function providerCard(page: Page, provider: string): Locator {
  return page.locator(`[data-testid="provider-card"][data-provider="${provider}"]`);
}

/** Pick a repository from the list, describe the task and start it. */
export async function startTask(page: Page, repository: string, prompt: string): Promise<void> {
  await page.getByTestId('repo-option').filter({ hasText: repository }).click();
  await expect(page.locator('#task-repository')).toHaveValue(repository);
  await page.locator('#task-prompt').fill(prompt);
  await page.getByTestId('run-task').click();
}

/** The result panel once the run is over, whichever way it went. */
export async function finishedResult(page: Page): Promise<Locator> {
  const panel = page.getByTestId('result-panel');
  await expect(panel).toHaveAttribute('data-state', /completed|failed/, { timeout: 45_000 });
  return panel;
}
