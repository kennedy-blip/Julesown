import { expect, test } from '@playwright/test';

import type { RepoSummary } from '../../lib/api';
import { searchRepositories, timeAgo } from '../../lib/repoSearch';
import { parseRoute, routeHref } from '../../lib/route';

function repo(full_name: string, extra: Partial<RepoSummary> = {}): RepoSummary {
  return { full_name, private: false, default_branch: 'main', html_url: `https://github.com/${full_name}`, ...extra };
}

const REPOSITORIES = [
  repo('acme/task-tracker', { description: 'Tracks tasks', language: 'TypeScript' }),
  repo('acme/agent-pulse', { language: 'Python' }),
  repo('pulse-labs/website'),
  repo('acme/impulse'),
];

function found(query: string): string[] {
  return searchRepositories(REPOSITORIES, query).map((item) => item.full_name);
}

test.describe('the address', () => {
  test('names the part of the dashboard that is on screen', () => {
    expect(parseRoute('')).toEqual({ view: 'workspace', tab: null });
    expect(parseRoute('#/')).toEqual({ view: 'workspace', tab: null });
    expect(parseRoute('#/settings')).toEqual({ view: 'settings', tab: null });
    expect(parseRoute('#/settings/models')).toEqual({ view: 'settings', tab: 'models' });
    // A tab nobody knows is the settings without a tab, not an error.
    expect(parseRoute('#/settings/nonsense')).toEqual({ view: 'settings', tab: null });
    expect(parseRoute('#/elsewhere')).toEqual({ view: 'workspace', tab: null });
  });

  test('is read back the way it is written', () => {
    expect(routeHref('workspace')).toBe('#/');
    expect(routeHref('settings')).toBe('#/settings');
    expect(parseRoute(routeHref('settings', 'about'))).toEqual({ view: 'settings', tab: 'about' });
  });
});

test.describe('searching the repositories', () => {
  test('returns the whole list for an empty query, in its order', () => {
    expect(found('  ')).toEqual(REPOSITORIES.map((item) => item.full_name));
  });

  test('ranks the start of a name above the middle of one, and names above owners', () => {
    expect(found('pulse')).toEqual(['acme/agent-pulse', 'acme/impulse', 'pulse-labs/website']);
  });

  test('needs every word to match, and looks into description and language', () => {
    expect(found('acme python')).toEqual(['acme/agent-pulse']);
    expect(found('acme rust')).toEqual([]);
  });

  test('finds what a pasted address names', () => {
    expect(found('https://github.com/acme/task-tracker.git')).toEqual(['acme/task-tracker']);
  });

  test('forgives a typo only when nothing matches properly', () => {
    expect(found('tsktr')).toEqual(['acme/task-tracker']);
  });
});

test('the time since the last push is short and never negative', () => {
  const now = Date.parse('2026-06-01T12:00:00Z');
  expect(timeAgo('2026-06-01T11:59:30Z', now)).toBe('now');
  expect(timeAgo('2026-06-01T11:15:00Z', now)).toBe('45m');
  expect(timeAgo('2026-06-01T07:00:00Z', now)).toBe('5h');
  expect(timeAgo('2026-05-20T12:00:00Z', now)).toBe('12d');
  expect(timeAgo('2026-01-01T12:00:00Z', now)).toBe('5mo');
  expect(timeAgo('2023-06-01T12:00:00Z', now)).toBe('3y');
  expect(timeAgo('2026-06-02T12:00:00Z', now)).toBe('now');
  expect(timeAgo(null, now)).toBe('');
  expect(timeAgo('not a date', now)).toBe('');
});
