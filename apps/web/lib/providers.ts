import type { HealthEnv, ProviderInfo } from './api';
import { PROVIDER_IDS, splitModels } from './settings';
import type { ProviderId, Settings } from './settings';

/** What is known about the providers before the backend has answered. */
const KNOWN_PROVIDERS: ProviderInfo[] = [
  {
    id: 'groq',
    label: 'Groq',
    configured: false,
    model: 'openai/gpt-oss-120b',
    models: ['openai/gpt-oss-120b', 'openai/gpt-oss-20b'],
    key_variable: 'GROQ_API_KEY',
    model_variable: 'GROQ_MODEL',
  },
  {
    id: 'gemini',
    label: 'Google Gemini',
    configured: false,
    model: 'gemini-flash-latest',
    models: ['gemini-flash-latest', 'gemini-3.5-flash-lite'],
    key_variable: 'GEMINI_API_KEY',
    model_variable: 'GEMINI_MODEL',
  },
  {
    id: 'nvidia',
    label: 'NVIDIA NIM',
    configured: false,
    model: 'deepseek-ai/deepseek-v4.1-flash',
    models: ['deepseek-ai/deepseek-v4.1-flash', 'nvidia/nemotron-3-super-120b-a12b'],
    key_variable: 'NVIDIA_API_KEY',
    model_variable: 'NVIDIA_MODEL',
  },
  {
    id: 'zai',
    label: 'Z.AI (GLM)',
    configured: false,
    model: 'glm-4.7-flash',
    models: ['glm-4.7-flash', 'glm-4.5-flash'],
    key_variable: 'ZAI_API_KEY',
    model_variable: 'ZAI_MODEL',
  },
  {
    id: 'openrouter',
    label: 'OpenRouter',
    configured: false,
    model: 'openrouter/free',
    models: ['openrouter/free'],
    key_variable: 'OPENROUTER_API_KEY',
    model_variable: 'OPENROUTER_MODEL',
  },
  { id: 'openai', label: 'OpenAI', configured: false, model: '', models: [], key_variable: 'OPENAI_API_KEY', model_variable: 'OPENAI_MODEL' },
  { id: 'anthropic', label: 'Anthropic', configured: false, model: '', models: [], key_variable: 'ANTHROPIC_API_KEY', model_variable: 'ANTHROPIC_MODEL' },
];

/** Every provider, with what the server says about it (its default models, whether it has a key). */
export function providersOf(health: HealthEnv | null): ProviderInfo[] {
  const reported = health && Array.isArray(health.providers) ? health.providers : [];
  return KNOWN_PROVIDERS.map((known) => reported.find((item) => item.id === known.id) || known);
}

/** The models the server gives a provider, in the order they are tried. */
export function serverModels(info: ProviderInfo): string[] {
  if (Array.isArray(info.models) && info.models.length > 0) return info.models;
  return info.model ? [info.model] : [];
}

/**
 * The models a task started from this browser uses with a provider: the ones
 * entered here ("a, b") replace the server's list, as they do on the server.
 */
export function modelsFor(info: ProviderInfo, settings: Settings): string[] {
  const mine = splitModels(settings.models[info.id]);
  return mine.length > 0 ? mine : serverModels(info);
}

/** One model a task can use. A provider appears once for each of its models. */
export interface ChainEntry {
  id: ProviderId;
  label: string;
  model: string;
}

/**
 * The providers a task started from this browser will use, first choice first.
 *
 * This repeats what the server does with a request (`Settings.with_overrides`
 * in apps/api/config.py): every provider that has a key - the server's or one
 * kept here - in the server's order of preference, with the first choice in
 * front. The first choice made here counts only if there is a key for it; the
 * server's own (JULES_LLM_PROVIDER) is set aside when it has no key but this
 * browser brings one for another provider.
 */
export function providerOrder(health: HealthEnv | null, settings: Settings): ProviderId[] {
  const providers = providersOf(health);
  const preferences = health && health.llm_preferences ? health.llm_preferences : null;
  const hasKeyHere = (id: ProviderId) => (settings.keys[id] || '').trim() !== '';

  const order: ProviderId[] = [];
  ((preferences && preferences.order) || []).concat(PROVIDER_IDS).forEach((id) => {
    if (PROVIDER_IDS.indexOf(id as ProviderId) >= 0 && order.indexOf(id as ProviderId) < 0) order.push(id as ProviderId);
  });
  const usable = order.filter((id) => hasKeyHere(id) || providers.some((item) => item.id === id && item.configured));

  const chosenHere = settings.primary !== 'auto' && usable.indexOf(settings.primary) >= 0 ? settings.primary : null;
  const choice = chosenHere || ((preferences && preferences.first_choice) as ProviderId | null) || null;
  let chain: ProviderId[];
  if (!choice) chain = usable;
  else if (usable.indexOf(choice) >= 0) chain = [choice].concat(usable.filter((id) => id !== choice));
  else chain = PROVIDER_IDS.some(hasKeyHere) ? usable : [];

  const fallback = settings.fallback && !(preferences && preferences.fallback === false);
  return fallback ? chain : chain.slice(0, 1);
}

/**
 * Every model a task will use, in the order they are tried - the server's
 * `Settings.llm_slots`: each provider with its first model, then each
 * provider's second model, and so on, so that every provider's best model is
 * tried before anybody's stand-in. Without fallback there is exactly one.
 */
export function effectiveChain(health: HealthEnv | null, settings: Settings): ChainEntry[] {
  const providers = providersOf(health);
  const preferences = health && health.llm_preferences ? health.llm_preferences : null;
  const fallback = settings.fallback && !(preferences && preferences.fallback === false);
  const lists = providerOrder(health, settings).map((id) => {
    const info = providers.find((item) => item.id === id) as ProviderInfo;
    return { id, label: info.label, models: modelsFor(info, settings) };
  });
  if (lists.length === 0) return [];
  if (!fallback) return [{ id: lists[0].id, label: lists[0].label, model: lists[0].models[0] || '' }];
  const chain: ChainEntry[] = [];
  const depth = Math.max(...lists.map((item) => item.models.length), 1);
  for (let level = 0; level < depth; level += 1) {
    lists.forEach((item) => {
      // A provider without a known model still takes its place: the server names the model.
      if (level < item.models.length) chain.push({ id: item.id, label: item.label, model: item.models[level] });
      else if (level === 0) chain.push({ id: item.id, label: item.label, model: '' });
    });
  }
  return chain;
}

/** The providers of a chain, each once, in the order they are first used. */
export function providersIn(chain: ChainEntry[]): ChainEntry[] {
  const seen: ChainEntry[] = [];
  chain.forEach((entry) => {
    if (!seen.some((item) => item.id === entry.id)) seen.push(entry);
  });
  return seen;
}
