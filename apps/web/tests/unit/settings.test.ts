import { expect, test } from '@playwright/test';

import {
  DEFAULT_SETTINGS,
  cleanAccountId,
  identityComplete,
  keyHeaderValue,
  providerKeyHeaders,
  reviveSettings,
  splitAccountKey,
  splitModels,
  taskOptions,
} from '../../lib/settings';
import type { Settings } from '../../lib/settings';

const ACCOUNT = '0123456789abcdef0123456789abcdef';

function settings(change: Partial<Settings>): Settings {
  return { ...DEFAULT_SETTINGS, ...change };
}

test.describe('model names', () => {
  test('are split at commas, semicolons and line breaks, without repeats', () => {
    expect(splitModels('a, b;c\nb')).toEqual(['a', 'b', 'c']);
    expect(splitModels('')).toEqual([]);
    expect(splitModels(null)).toEqual([]);
  });

  test('keep the characters real names have, and nothing that is not a name', () => {
    expect(splitModels('@cf/zai-org/glm-4.7-flash, openai/gpt-oss-120b, meta/llama-3.3:free')).toEqual([
      '@cf/zai-org/glm-4.7-flash',
      'openai/gpt-oss-120b',
      'meta/llama-3.3:free',
    ]);
    expect(splitModels('two words, <script>, ok-model')).toEqual(['ok-model']);
  });

  test('stop at four', () => {
    expect(splitModels('a,b,c,d,e,f')).toEqual(['a', 'b', 'c', 'd']);
  });
});

test.describe('an account ID', () => {
  test('is 32 hexadecimal characters, whatever the case and the space around it', () => {
    expect(cleanAccountId(`  ${ACCOUNT.toUpperCase()} `)).toBe(ACCOUNT);
    expect(cleanAccountId(ACCOUNT.slice(1))).toBe('');
    expect(cleanAccountId(`${ACCOUNT}/../other`)).toBe('');
    expect(cleanAccountId(undefined)).toBe('');
  });

  test('can stand in front of the key', () => {
    expect(splitAccountKey(`${ACCOUNT}:token`)).toEqual({ account: ACCOUNT, key: 'token' });
    expect(splitAccountKey('token')).toEqual({ account: '', key: 'token' });
    // What stands in front is no account ID: it is not passed on as one.
    expect(splitAccountKey('nonsense:token')).toEqual({ account: '', key: 'token' });
  });
});

test.describe('the header a key travels in', () => {
  test('carries the key alone for most providers', () => {
    expect(keyHeaderValue('groq', '  gsk_abc  ')).toBe('gsk_abc');
    expect(keyHeaderValue('groq', '')).toBe('');
    // A header cannot carry spaces or anything outside ASCII: that is not a key.
    expect(keyHeaderValue('groq', 'two words')).toBe('');
    expect(keyHeaderValue('groq', 'schlüssel')).toBe('');
  });

  test('carries the account ID as well for Cloudflare, or either half alone', () => {
    expect(keyHeaderValue('cloudflare', 'token', ACCOUNT)).toBe(`${ACCOUNT}:token`);
    expect(keyHeaderValue('cloudflare', 'token', '')).toBe('token');
    expect(keyHeaderValue('cloudflare', '', ACCOUNT)).toBe(`${ACCOUNT}:`);
    expect(keyHeaderValue('cloudflare', `${ACCOUNT}:token`, '')).toBe(`${ACCOUNT}:token`);
    // The field for the ID counts before an ID pasted in front of the token.
    const other = 'f'.repeat(32);
    expect(keyHeaderValue('cloudflare', `${ACCOUNT}:token`, other)).toBe(`${other}:token`);
  });

  test('is named after the provider, and there is none for a provider without a key', () => {
    const headers = providerKeyHeaders(
      settings({ keys: { groq: 'gsk_abc', deepseek: '  ', cloudflare: 'token' }, accounts: { cloudflare: ACCOUNT } })
    );
    expect(headers).toEqual({ 'X-groq-Key': 'gsk_abc', 'X-cloudflare-Key': `${ACCOUNT}:token` });
  });
});

test.describe('what is sent with a task', () => {
  test('is nothing at all with the default settings', () => {
    expect(taskOptions(DEFAULT_SETTINGS)).toEqual({});
  });

  test('names the first choice, the models and whether to fall back - never a key', () => {
    const options = taskOptions(
      settings({
        keys: { groq: 'gsk_secret' },
        primary: 'gemini',
        fallback: false,
        models: { gemini: 'gemini-a, gemini-b', groq: '   ' },
      })
    );
    expect(options).toEqual({ llm: { primary: 'gemini', fallback: false, models: { gemini: 'gemini-a,gemini-b' } } });
    expect(JSON.stringify(options)).not.toContain('gsk_secret');
  });

  test('attributes commits to a person only when name and address are both there', () => {
    const half = settings({ commit: { mode: 'user', name: 'Ada', email: '', login: '' } });
    expect(taskOptions(half).commit).toBeUndefined();
    const whole = settings({ commit: { mode: 'coauthor', name: ' Ada ', email: ' ada@example.com ', login: 'ada' } });
    expect(taskOptions(whole).commit).toEqual({ mode: 'coauthor', name: 'Ada', email: 'ada@example.com' });
    expect(identityComplete({ mode: 'user', name: 'Ada', email: 'not an address', login: '' })).toBe(false);
  });

  test('includes standing instructions, cut to what the server accepts', () => {
    expect(taskOptions(settings({ guidelines: '  Use tabs.  ' })).guidelines).toBe('Use tabs.');
    expect((taskOptions(settings({ guidelines: 'x'.repeat(5000) })).guidelines || '').length).toBe(4000);
  });
});

test.describe('settings read back from storage', () => {
  test('are made valid, whatever was stored', () => {
    for (const stored of [null, 'text', 42, [], { keys: 'no', primary: 'nobody', fallback: 'yes' }]) {
      const revived = reviveSettings(stored);
      expect(revived.primary).toBe('auto');
      expect(revived.keys).toEqual({});
      expect(typeof revived.fallback).toBe('boolean');
    }
  });

  test('keep what is valid and drop a provider nobody knows', () => {
    const revived = reviveSettings({
      keys: { groq: 'gsk_abc', madeup: 'x' },
      accounts: { cloudflare: ACCOUNT },
      primary: 'groq',
      fallback: false,
    });
    expect(revived.keys).toEqual({ groq: 'gsk_abc' });
    expect(revived.accounts).toEqual({ cloudflare: ACCOUNT });
    expect(revived.primary).toBe('groq');
    expect(revived.fallback).toBe(false);
  });
});
