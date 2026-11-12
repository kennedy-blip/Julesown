import type { HealthEnv, ProviderInfo } from './api';
import { PROVIDER_IDS } from './settings';
import type { ProviderId, Settings } from './settings';

/** What is known about the providers before the backend has answered. */
const KNOWN_PROVIDERS: ProviderInfo[] = [
  { id: 'groq', label: 'Groq', configured: false, model: 'openai/gpt-oss-120b', key_variable: 'GROQ_API_KEY', model_variable: 'GROQ_MODEL' },
  { id: 'gemini', label: 'Google Gemini', configured: false, model: 'gemini-flash-latest', key_variable: 'GEMINI_API_KEY', model_variable: 'GEMINI_MODEL' },
  { id: 'openrouter', label: 'OpenRouter', configured: false, model: 'baidu/ernie-5.5-flash-preview:free', key_variable: 'OPENROUTER_API_KEY', model_variable: 'OPENROUTER_MODEL' },
  { id: 'openai', label: 'OpenAI', configured: false, model: '', key_variable: 'OPENAI_API_KEY', model_variable: 'OPENAI_MODEL' },
  { id: 'anthropic', label: 'Anthropic', configured: false, model: '', key_variable: 'ANTHROPIC_API_KEY', model_variable: 'ANTHROPIC_MODEL' },
];

/** Every provider, with what the server says about it (its default model, whether it has a key). */
export function providersOf(health: HealthEnv | null): ProviderInfo[] {
  const reported = health && Array.isArray(health.providers) ? health.providers : [];
  return KNOWN_PROVIDERS.map((known) => reported.find((item) => item.id === known.id) || known);
}

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
export function effectiveChain(health: HealthEnv | null, settings: Settings): ChainEntry[] {
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
  return (fallback ? chain : chain.slice(0, 1)).map((id) => {
    const info = providers.find((item) => item.id === id) as ProviderInfo;
    return { id, label: info.label, model: (settings.models[id] || '').trim() || info.model };
  });
}
