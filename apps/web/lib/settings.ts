'use client';

import { useEffect, useSyncExternalStore } from 'react';

/**
 * What the dashboard remembers between visits, kept in this browser only:
 * API keys for model providers, which provider to start with, how commits are
 * attributed, the settings tab last used and the task being written.
 *
 * Keys never leave the browser except in the requests that need them: they
 * travel to the dashboard's own backend in headers, are used for that request
 * and are not stored there.
 */
export type ProviderId = 'groq' | 'gemini' | 'openrouter' | 'openai' | 'anthropic';
export const PROVIDER_IDS: ProviderId[] = ['groq', 'gemini', 'openrouter', 'openai', 'anthropic'];

const PROVIDER_NAMES: Record<string, string> = {
  groq: 'Groq',
  gemini: 'Gemini',
  openrouter: 'OpenRouter',
  openai: 'OpenAI',
  anthropic: 'Anthropic',
};

/** How a provider is called on screen. */
export function providerName(id: string | null | undefined): string {
  return (id && PROVIDER_NAMES[id]) || id || '';
}

export type CommitMode = 'bot' | 'coauthor' | 'user';
export type SettingsTab = 'access' | 'models' | 'commits' | 'automation' | 'about';
export const SETTINGS_TABS: SettingsTab[] = ['access', 'models', 'commits', 'automation', 'about'];

export interface CommitIdentity {
  mode: CommitMode;
  name: string;
  email: string;
  /** GitHub user name the name and address were looked up for (a convenience, not sent). */
  login: string;
}

/** The task form as it was left: nothing typed is lost by a reload. */
export interface Draft {
  repository: string;
  branch: string;
  prompt: string;
  /** Where the result goes: a new pull request, or one that exists. */
  target: 'new' | 'pull_request';
  /** Number of the pull request to add to (used when the target is one). */
  pullRequest: string;
  /** The task is to repair the failed checks. */
  fixChecks: boolean;
}

export const EMPTY_DRAFT: Draft = {
  repository: '',
  branch: '',
  prompt: '',
  target: 'new',
  pullRequest: '',
  fixChecks: false,
};

export interface Settings {
  keys: Partial<Record<ProviderId, string>>;
  models: Partial<Record<ProviderId, string>>;
  /** Provider a task starts with; 'auto' leaves the order to the server. */
  primary: ProviderId | 'auto';
  /** Carry on with the next provider when the one in use reaches a limit or fails. */
  fallback: boolean;
  commit: CommitIdentity;
  settingsTab: SettingsTab;
  draft: Draft;
}

const STORAGE_KEY = 'jules_settings_v1';

export const DEFAULT_SETTINGS: Settings = {
  keys: {},
  models: {},
  primary: 'auto',
  fallback: true,
  commit: { mode: 'bot', name: '', email: '', login: '' },
  settingsTab: 'access',
  draft: EMPTY_DRAFT,
};

function text(value: unknown, limit = 600): string {
  return typeof value === 'string' ? value.slice(0, limit) : '';
}

function providerMap(value: unknown): Partial<Record<ProviderId, string>> {
  const result: Partial<Record<ProviderId, string>> = {};
  if (!value || typeof value !== 'object') return result;
  const raw = value as Record<string, unknown>;
  PROVIDER_IDS.forEach((id) => {
    const item = text(raw[id]).trim();
    if (item) result[id] = item;
  });
  return result;
}

/** Whatever was stored, made into valid settings (nothing in storage is trusted blindly). */
export function reviveSettings(value: unknown): Settings {
  if (!value || typeof value !== 'object') return DEFAULT_SETTINGS;
  const raw = value as Record<string, unknown>;
  const commit = (raw.commit && typeof raw.commit === 'object' ? raw.commit : {}) as Record<string, unknown>;
  const draft = (raw.draft && typeof raw.draft === 'object' ? raw.draft : {}) as Record<string, unknown>;
  const modes: CommitMode[] = ['bot', 'coauthor', 'user'];
  return {
    keys: providerMap(raw.keys),
    models: providerMap(raw.models),
    primary: PROVIDER_IDS.indexOf(raw.primary as ProviderId) >= 0 ? (raw.primary as ProviderId) : 'auto',
    fallback: raw.fallback !== false,
    commit: {
      mode: modes.indexOf(commit.mode as CommitMode) >= 0 ? (commit.mode as CommitMode) : 'bot',
      name: text(commit.name, 100),
      email: text(commit.email, 200),
      login: text(commit.login, 60),
    },
    settingsTab: SETTINGS_TABS.indexOf(raw.settingsTab as SettingsTab) >= 0 ? (raw.settingsTab as SettingsTab) : 'access',
    draft: {
      repository: text(draft.repository, 200),
      branch: text(draft.branch, 200),
      prompt: text(draft.prompt, 8000),
      target: draft.target === 'pull_request' ? 'pull_request' : 'new',
      pullRequest: /^[1-9]\d{0,8}$/.test(text(draft.pullRequest, 12)) ? text(draft.pullRequest, 12) : '',
      fixChecks: draft.fixChecks === true,
    },
  };
}

let current: Settings = DEFAULT_SETTINGS;
let loaded = false;
const listeners: Array<() => void> = [];

function persist(): void {
  try {
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(current));
  } catch {
    /* storage full or unavailable (private mode): settings last for this visit only */
  }
}

/** Read the stored settings once. Safe to call repeatedly. */
export function loadSettings(): Settings {
  if (loaded || typeof window === 'undefined') return current;
  loaded = true;
  try {
    const saved = window.localStorage.getItem(STORAGE_KEY);
    if (saved) current = reviveSettings(JSON.parse(saved));
  } catch {
    current = DEFAULT_SETTINGS;
  }
  listeners.slice().forEach((listener) => listener());
  return current;
}

export function getSettings(): Settings {
  return current;
}

export function updateSettings(change: Partial<Settings> | ((settings: Settings) => Partial<Settings>)): void {
  loadSettings(); // never write over stored settings that were not read yet
  const patch = typeof change === 'function' ? change(current) : change;
  current = { ...current, ...patch };
  persist();
  listeners.slice().forEach((listener) => listener());
}

function subscribe(listener: () => void): () => void {
  listeners.push(listener);
  return () => {
    const index = listeners.indexOf(listener);
    if (index >= 0) listeners.splice(index, 1);
  };
}

export function useSettings(): Settings {
  const settings = useSyncExternalStore(
    subscribe,
    () => current,
    () => DEFAULT_SETTINGS
  );
  useEffect(() => {
    loadSettings();
  }, []);
  return settings;
}

/** Whether this browser holds a key for at least one model provider. */
export function hasProviderKey(settings: Settings): boolean {
  return PROVIDER_IDS.some((id) => (settings.keys[id] || '').trim() !== '');
}

const EMAIL = /^[^\s<>@,;]+@[^\s<>@,;]+\.[^\s<>@,;]+$/;

/** Whether a name and address are there to attribute a commit to a person. */
export function identityComplete(identity: CommitIdentity): boolean {
  return identity.name.trim().length > 0 && EMAIL.test(identity.email.trim());
}

/** Headers that carry this browser's provider keys to the backend. */
export function providerKeyHeaders(settings: Settings = current): Record<string, string> {
  const headers: Record<string, string> = {};
  PROVIDER_IDS.forEach((id) => {
    const key = (settings.keys[id] || '').trim();
    // Header values must be plain ASCII without spaces; anything else is not a key.
    if (key && /^[\x21-\x7e]+$/.test(key)) headers[`X-${id}-Key`] = key;
  });
  return headers;
}

export interface TaskOptions {
  llm?: { primary?: string; fallback?: boolean; models?: Record<string, string> };
  commit?: { mode: CommitMode; name?: string; email?: string };
}

/** The choices sent along with every task (never keys: those travel in headers). */
export function taskOptions(settings: Settings = current): TaskOptions {
  const options: TaskOptions = {};
  const models: Record<string, string> = {};
  PROVIDER_IDS.forEach((id) => {
    const model = (settings.models[id] || '').trim();
    if (model) models[id] = model;
  });
  const llm: NonNullable<TaskOptions['llm']> = {};
  if (settings.primary !== 'auto') llm.primary = settings.primary;
  if (!settings.fallback) llm.fallback = false;
  if (Object.keys(models).length > 0) llm.models = models;
  if (Object.keys(llm).length > 0) options.llm = llm;
  if (settings.commit.mode !== 'bot' && identityComplete(settings.commit)) {
    options.commit = {
      mode: settings.commit.mode,
      name: settings.commit.name.trim(),
      email: settings.commit.email.trim(),
    };
  }
  return options;
}
