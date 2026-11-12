'use client';

import { useMemo, useSyncExternalStore } from 'react';
import { SETTINGS_TABS } from './settings';
import type { SettingsTab } from './settings';

/**
 * Which part of the dashboard is on screen. It lives in the address
 * (`#/settings/models`), so reloading keeps the place, the back button works
 * and a tab can be linked to.
 */
export type View = 'workspace' | 'settings';

export interface Route {
  view: View;
  /** The settings tab named in the address, if any. */
  tab: SettingsTab | null;
}

export function parseRoute(hash: string): Route {
  const parts = hash.replace(/^#\/?/, '').split('/').filter(Boolean);
  if (parts[0] !== 'settings') return { view: 'workspace', tab: null };
  const tab = SETTINGS_TABS.indexOf(parts[1] as SettingsTab) >= 0 ? (parts[1] as SettingsTab) : null;
  return { view: 'settings', tab };
}

export function routeHref(view: View, tab?: SettingsTab | null): string {
  if (view === 'workspace') return '#/';
  return tab ? `#/settings/${tab}` : '#/settings';
}

export function navigate(view: View, tab?: SettingsTab | null): void {
  const href = routeHref(view, tab);
  if (window.location.hash === href) return;
  if (view === 'workspace' && (window.location.hash === '' || window.location.hash === '#')) return;
  window.location.hash = href;
}

function subscribe(listener: () => void): () => void {
  window.addEventListener('hashchange', listener);
  return () => window.removeEventListener('hashchange', listener);
}

export function useRoute(): Route {
  const hash = useSyncExternalStore(
    subscribe,
    () => window.location.hash,
    () => ''
  );
  return useMemo(() => parseRoute(hash), [hash]);
}
