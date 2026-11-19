import { expect, test } from '@playwright/test';

import type { HealthEnv, ProviderInfo } from '../../lib/api';
import { effectiveChain, heldHere, onServer, providerOrder, providersIn, providersOf, usableHere } from '../../lib/providers';
import { DEFAULT_SETTINGS, PROVIDER_IDS } from '../../lib/settings';
import type { ProviderId, Settings } from '../../lib/settings';

const ACCOUNT = '0123456789abcdef0123456789abcdef';
const ORDER = ['groq', 'gemini', 'nvidia', 'zai', 'cloudflare', 'sambanova', 'openrouter', 'pollinations', 'deepseek', 'openai', 'anthropic'];

function settings(change: Partial<Settings> = {}): Settings {
  return { ...DEFAULT_SETTINGS, ...change };
}

/** What the server reports when it has keys for `configured`. */
function health(configured: ProviderId[], extra: Partial<NonNullable<HealthEnv['llm_preferences']>> = {}, cloudflare: Partial<ProviderInfo> = {}): HealthEnv {
  const providers = providersOf(null).map((info) => ({
    ...info,
    configured: configured.indexOf(info.id as ProviderId) >= 0,
    ...(info.id === 'cloudflare' ? cloudflare : {}),
  }));
  return { providers, llm_preferences: { order: ORDER, first_choice: null, fallback: true, ...extra } } as unknown as HealthEnv;
}

function labels(chain: Array<{ id: string; model: string }>): string[] {
  return chain.map((entry) => `${entry.id}/${entry.model}`);
}

test.describe('the providers a task will use', () => {
  test('are the ones with a key, in the server\'s order', () => {
    expect(providerOrder(health(['openrouter', 'groq', 'deepseek']), settings())).toEqual(['groq', 'openrouter', 'deepseek']);
    expect(providerOrder(health([]), settings())).toEqual([]);
    // Before the server has answered there is nothing to go by but this browser's keys.
    expect(providerOrder(null, settings({ keys: { zai: 'key' } }))).toEqual(['zai']);
  });

  test('include a provider whose key is kept in this browser', () => {
    expect(providerOrder(health(['groq']), settings({ keys: { sambanova: 'key' } }))).toEqual(['groq', 'sambanova']);
  });

  test('start with the first choice made here, if there is a key for it', () => {
    const server = health(['groq', 'gemini', 'deepseek']);
    expect(providerOrder(server, settings({ primary: 'deepseek' }))).toEqual(['deepseek', 'groq', 'gemini']);
    // A first choice without a key is no choice.
    expect(providerOrder(server, settings({ primary: 'openai' }))).toEqual(['groq', 'gemini', 'deepseek']);
  });

  test('follow the server\'s first choice, which runs nothing while it has no key', () => {
    expect(providerOrder(health(['groq', 'gemini'], { first_choice: 'gemini' }), settings())).toEqual(['gemini', 'groq']);
    expect(providerOrder(health(['groq'], { first_choice: 'openai' }), settings())).toEqual([]);
    // ... unless this browser brings a key of its own.
    expect(providerOrder(health(['groq'], { first_choice: 'openai' }), settings({ keys: { zai: 'key' } }))).toEqual(['groq', 'zai']);
  });

  test('are one only when falling back is switched off, here or on the server', () => {
    const server = health(['groq', 'gemini']);
    expect(providerOrder(server, settings({ fallback: false }))).toEqual(['groq']);
    expect(providerOrder(health(['groq', 'gemini'], { fallback: false }), settings())).toEqual(['groq']);
  });
});

test.describe('the models of a task', () => {
  test('are every provider\'s first model before anybody\'s second', () => {
    expect(labels(effectiveChain(health(['groq', 'deepseek', 'sambanova']), settings()))).toEqual([
      'groq/openai/gpt-oss-120b',
      'sambanova/DeepSeek-V3.1',
      'deepseek/deepseek-flash',
      'groq/openai/gpt-oss-20b',
      'sambanova/gpt-oss-120b',
    ]);
  });

  test('are the ones entered here instead of the server\'s', () => {
    const chain = effectiveChain(health(['groq']), settings({ models: { groq: 'mine-a, mine-b, mine-c' } }));
    expect(labels(chain)).toEqual(['groq/mine-a', 'groq/mine-b', 'groq/mine-c']);
  });

  test('are exactly one without fallback', () => {
    expect(labels(effectiveChain(health(['groq', 'gemini']), settings({ fallback: false })))).toEqual(['groq/openai/gpt-oss-120b']);
    expect(effectiveChain(health([]), settings())).toEqual([]);
  });

  test('name each provider once, in the order it is first used', () => {
    const chain = effectiveChain(health(['groq', 'sambanova']), settings());
    expect(providersIn(chain).map((entry) => entry.id)).toEqual(['groq', 'sambanova']);
  });
});

test.describe('a provider that needs an account ID', () => {
  const cloudflare = (server: Partial<ProviderInfo>): ProviderInfo => {
    const info = providersOf(health([], {}, server)).find((item) => item.id === 'cloudflare');
    if (!info) throw new Error('cloudflare is a known provider');
    return info;
  };

  test('can be used when token and ID are both there, from either side', () => {
    const nothing = cloudflare({ key_set: false, account_set: false });
    expect(usableHere(nothing, settings())).toBe(false);
    expect(usableHere(nothing, settings({ keys: { cloudflare: 'token' } }))).toBe(false);
    expect(usableHere(nothing, settings({ keys: { cloudflare: 'token' }, accounts: { cloudflare: ACCOUNT } }))).toBe(true);
    expect(usableHere(nothing, settings({ keys: { cloudflare: `${ACCOUNT}:token` } }))).toBe(true);

    // The server has the token and no ID - as when CLOUDFLARE_ACCOUNT_ID was forgotten.
    const tokenOnly = cloudflare({ key_set: true, account_set: false });
    expect(onServer(tokenOnly)).toEqual({ key: true, account: false });
    expect(usableHere(tokenOnly, settings())).toBe(false);
    expect(usableHere(tokenOnly, settings({ accounts: { cloudflare: ACCOUNT } }))).toBe(true);
  });

  test('does not count an ID that cannot be one', () => {
    const nothing = cloudflare({ key_set: false, account_set: false });
    const typed = settings({ keys: { cloudflare: 'token' }, accounts: { cloudflare: 'my account' } });
    expect(heldHere(nothing, typed)).toEqual({ key: true, account: false });
    expect(usableHere(nothing, typed)).toBe(false);
  });

  test('is like any other provider behind a gateway, which needs no ID', () => {
    const gateway = cloudflare({ configured: true, account_variable: undefined, key_set: undefined, account_set: undefined });
    expect(onServer(gateway)).toEqual({ key: true, account: true });
    expect(usableHere(gateway, settings())).toBe(true);
  });
});

test('the dashboard and the server know the same providers', () => {
  expect(PROVIDER_IDS).toEqual(ORDER);
  expect(providersOf(null).map((info) => info.id)).toEqual(ORDER);
});
