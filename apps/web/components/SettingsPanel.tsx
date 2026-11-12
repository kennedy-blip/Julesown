'use client';

import React, { useEffect, useState } from 'react';
import {
  AlertTriangle,
  ArrowRight,
  Bot,
  Check,
  CheckCircle2,
  Copy,
  Cpu,
  ExternalLink,
  Eye,
  EyeOff,
  GitCommitHorizontal,
  Info,
  KeyRound,
  Loader2,
  RefreshCw,
  Search,
  Trash2,
  User,
  Users,
  Wrench,
  XCircle,
  Zap,
} from 'lucide-react';
import { ApiError, describeError, fetchAutomation, fetchIdentity, testProvider } from '@/lib/api';
import type { AppInstallation, AutomationStatus, HealthEnv, ProviderInfo, ProviderTest } from '@/lib/api';
import { effectiveChain, providersOf } from '@/lib/providers';
import { identityComplete, SETTINGS_TABS, updateSettings, useSettings } from '@/lib/settings';
import type { CommitMode, ProviderId, Settings, SettingsTab } from '@/lib/settings';
import { AccessKeyField } from './AccessKeyField';
import { Badge, Button, buttonClass, Callout, CARD, Hint, INPUT, INSET, LABEL, Toggle } from './ui';

interface SettingsPanelProps {
  health: HealthEnv | null;
  accessMode: 'open' | 'key' | 'locked';
  accessKey: string;
  onAccessKeyChange: (value: string) => void;
  accessProblem: string | null;
  /** The backend has accepted a request made with the key that is entered. */
  accessAccepted: boolean;
  /** Requests that need the access key can be made right now. */
  canCallApi: boolean;
  installations: AppInstallation[];
  tab: SettingsTab;
  onTab: (tab: SettingsTab) => void;
}

const TAB_META: Record<SettingsTab, { label: string; description: string; icon: React.ReactNode }> = {
  access: { label: 'Access', description: 'The key that unlocks this deployment', icon: <KeyRound className="w-4 h-4" /> },
  models: { label: 'AI models', description: 'Provider keys, models and fallbacks', icon: <Cpu className="w-4 h-4" /> },
  commits: { label: 'Commits', description: 'Who the commits are attributed to', icon: <GitCommitHorizontal className="w-4 h-4" /> },
  automation: { label: 'Automation', description: 'Failed checks and unfinished work', icon: <Wrench className="w-4 h-4" /> },
  about: { label: 'About', description: 'What the server is configured with', icon: <Info className="w-4 h-4" /> },
};

/** Where a key for each provider is created, and what it looks like. */
const PROVIDER_HELP: Record<ProviderId, { keysUrl: string; keysLabel: string; placeholder: string; blurb: string }> = {
  groq: {
    keysUrl: 'https://console.groq.com/keys',
    keysLabel: 'console.groq.com',
    placeholder: 'gsk_…',
    blurb: 'Very fast. Its free plan allows few tokens per minute, which is what makes long tasks wait.',
  },
  gemini: {
    keysUrl: 'https://aistudio.google.com/apikey',
    keysLabel: 'Google AI Studio',
    placeholder: 'AQ.… or AIza…',
    blurb: 'Generous limits on the free tier. A good second provider to carry on with when Groq is exhausted.',
  },
  openrouter: {
    keysUrl: 'https://openrouter.ai/keys',
    keysLabel: 'openrouter.ai',
    placeholder: 'sk-or-…',
    blurb: 'One key for many models, including free ones. Free models are limited by requests per day.',
  },
  openai: {
    keysUrl: 'https://platform.openai.com/api-keys',
    keysLabel: 'platform.openai.com',
    placeholder: 'sk-…',
    blurb: 'Pay per token.',
  },
  anthropic: {
    keysUrl: 'https://console.anthropic.com/settings/keys',
    keysLabel: 'console.anthropic.com',
    placeholder: 'sk-ant-…',
    blurb: 'Pay per token.',
  },
};

function ExternalAnchor({ href, children }: { href: string; children: React.ReactNode }) {
  return (
    <a href={href} target="_blank" rel="noreferrer noopener" className="inline-flex items-center gap-1 text-blue-400 hover:text-blue-300 underline underline-offset-2">
      {children}
      <ExternalLink className="w-3 h-3" />
    </a>
  );
}

function SectionTitle({ title, children }: { title: string; children?: React.ReactNode }) {
  return (
    <div className="mb-4">
      <h3 className="text-[15px] font-semibold text-white">{title}</h3>
      {children && <Hint className="mt-1 max-w-2xl">{children}</Hint>}
    </div>
  );
}

// --------------------------------------------------------------------------
// Access
// --------------------------------------------------------------------------
function AccessTab({ accessMode, accessKey, onAccessKeyChange, accessProblem, accessAccepted }: SettingsPanelProps) {
  return (
    <div data-testid="settings-access">
      <SectionTitle title="Access key">
        Running a task spends this deployment&apos;s model and sandbox quota and writes to GitHub with its
        credentials, so the backend only does it for someone who knows its access key.
      </SectionTitle>

      {accessMode === 'open' && (
        <Callout tone="info" icon={<Info className="w-4 h-4 text-sky-300" />} data-testid="access-open">
          This server does not ask for an access key: it runs locally, or it was opened up with{' '}
          <code className="font-mono">JULES_ALLOW_PUBLIC_ACCESS=true</code>.
        </Callout>
      )}
      {accessMode === 'locked' && (
        <Callout tone="warn" icon={<AlertTriangle className="w-4 h-4 text-amber-300" />}>
          Task execution is locked: this deployment is reachable from the internet and has no access key. Add the
          environment variable <code className="font-mono">JULES_ACCESS_KEY</code> (any long random value) to the
          deployment and redeploy, then enter the same value here.
        </Callout>
      )}
      {accessMode === 'key' && (
        <div className="max-w-xl">
          <AccessKeyField value={accessKey} onChange={onAccessKeyChange} problem={accessProblem} accepted={accessAccepted} />
          <Hint className="mt-3">
            It is the value of the <code className="font-mono">JULES_ACCESS_KEY</code> variable of the deployment. The
            browser sends it to your backend with every request and keeps it until you remove it here.
          </Hint>
        </div>
      )}
    </div>
  );
}

// --------------------------------------------------------------------------
// AI models
// --------------------------------------------------------------------------
type TestState = { phase: 'idle' } | { phase: 'testing' } | { phase: 'done'; result: ProviderTest } | { phase: 'error'; message: string };

function ProviderCard({
  info,
  settings,
  position,
  canCallApi,
}: {
  info: ProviderInfo;
  settings: Settings;
  /** Place in the chain a task uses (1 = first choice), or 0 when it is not in it. */
  position: number;
  canCallApi: boolean;
}) {
  const help = PROVIDER_HELP[info.id];
  const key = settings.keys[info.id] || '';
  const model = settings.models[info.id] || '';
  const [shown, setShown] = useState(false);
  const [test, setTest] = useState<TestState>({ phase: 'idle' });
  const hasKey = key.trim() !== '';

  const setKey = (value: string) => {
    updateSettings((current) => {
      const keys = { ...current.keys };
      if (value.trim()) keys[info.id] = value.trim();
      else delete keys[info.id];
      // A provider without any key cannot stay the first choice.
      const primary = !value.trim() && !info.configured && current.primary === info.id ? 'auto' : current.primary;
      return { keys, primary };
    });
    setTest({ phase: 'idle' });
  };
  const setModel = (value: string) => {
    updateSettings((current) => {
      const models = { ...current.models };
      if (value.trim()) models[info.id] = value.trim();
      else delete models[info.id];
      return { models };
    });
    setTest({ phase: 'idle' });
  };

  const runTest = async () => {
    setTest({ phase: 'testing' });
    try {
      setTest({ phase: 'done', result: await testProvider(info.id, key, model) });
    } catch (error) {
      setTest({
        phase: 'error',
        message:
          error instanceof ApiError && error.code === 'access_key_required'
            ? 'Enter the access key first (Access tab).'
            : describeError(error),
      });
    }
  };

  return (
    <div className={`${CARD} p-4`} data-testid="provider-card" data-provider={info.id} data-position={position || undefined}>
      <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
        <div className="flex items-center gap-2 min-w-0">
          <span className="text-sm font-semibold text-white">{info.label}</span>
          {position > 0 && (
            <Badge tone={position === 1 ? 'info' : 'muted'} data-testid="provider-position">
              {position === 1 ? 'First choice' : `Fallback ${position - 1}`}
            </Badge>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-1.5" data-testid="provider-key-state">
          {hasKey && <Badge tone="good">Key in this browser</Badge>}
          {info.configured && <Badge tone={hasKey ? 'muted' : 'good'}>Key on the server</Badge>}
          {!hasKey && !info.configured && <Badge tone="muted">No key</Badge>}
        </div>
      </div>
      <Hint className="mb-3">
        {help.blurb} Get a key at <ExternalAnchor href={help.keysUrl}>{help.keysLabel}</ExternalAnchor>.
      </Hint>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
        <div>
          <label htmlFor={`key-${info.id}`} className={LABEL}>
            API key
          </label>
          <div className="flex gap-2">
            <input
              id={`key-${info.id}`}
              type={shown ? 'text' : 'password'}
              value={key}
              onChange={(event) => setKey(event.target.value)}
              placeholder={info.configured ? 'Using the server\u2019s key' : help.placeholder}
              autoComplete="off"
              spellCheck={false}
              data-testid="provider-key"
              className={`${INPUT} font-mono`}
            />
            <Button onClick={() => setShown(!shown)} aria-label={shown ? 'Hide the key' : 'Show the key'} title={shown ? 'Hide' : 'Show'}>
              {shown ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
            </Button>
            {hasKey && (
              <Button variant="danger" onClick={() => setKey('')} aria-label="Remove the key from this browser" title="Remove from this browser">
                <Trash2 className="w-4 h-4" />
              </Button>
            )}
          </div>
        </div>
        <div>
          <label htmlFor={`model-${info.id}`} className={LABEL}>
            Model
          </label>
          <input
            id={`model-${info.id}`}
            type="text"
            value={model}
            onChange={(event) => setModel(event.target.value)}
            placeholder={info.model || 'the server\u2019s default'}
            autoComplete="off"
            spellCheck={false}
            data-testid="provider-model"
            className={`${INPUT} font-mono`}
          />
        </div>
      </div>

      <div className="flex flex-wrap items-center gap-3 mt-3">
        <Button
          onClick={() => void runTest()}
          disabled={!canCallApi || test.phase === 'testing' || (!hasKey && !info.configured)}
          data-testid="provider-test"
          title={
            !hasKey && !info.configured
              ? 'Enter a key first'
              : 'Sends one small request with a tool to the provider - the kind of request the agent makes'
          }
        >
          {test.phase === 'testing' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Zap className="w-4 h-4" />}
          {test.phase === 'testing' ? 'Testing…' : 'Test'}
        </Button>
        <div className="text-xs min-w-0 flex-1" data-testid="provider-test-result" data-phase={test.phase}>
          {test.phase === 'done' && test.result.ok && test.result.tool_call && (
            <span className="text-emerald-400 inline-flex items-start gap-1.5">
              <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span>
                {test.result.message}
                {typeof test.result.latency_ms === 'number' ? ` (${(test.result.latency_ms / 1000).toFixed(1)} s` : ' ('}
                {test.result.key_source === 'server' ? ', the server\u2019s key)' : ', the key entered here)'}
              </span>
            </span>
          )}
          {test.phase === 'done' && test.result.ok && !test.result.tool_call && (
            <span className="text-amber-400 inline-flex items-start gap-1.5">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span>{test.result.message}</span>
            </span>
          )}
          {test.phase === 'done' && !test.result.ok && (
            <span className="text-rose-400 inline-flex items-start gap-1.5">
              <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span className="break-words">{test.result.message}</span>
            </span>
          )}
          {test.phase === 'error' && (
            <span className="text-rose-400 inline-flex items-start gap-1.5">
              <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              <span className="break-words">{test.message}</span>
            </span>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelsTab({ health, canCallApi }: SettingsPanelProps) {
  const settings = useSettings();
  const providers = providersOf(health);
  const chain = effectiveChain(health, settings);
  const [showMore, setShowMore] = useState(false);
  const usable = providers.filter((item) => item.configured || !!(settings.keys[item.id] || '').trim());
  const main = providers.filter((item) => ['groq', 'gemini', 'openrouter'].indexOf(item.id) >= 0);
  const more = providers.filter((item) => main.indexOf(item) < 0);
  const moreInUse = more.some((item) => usable.indexOf(item) >= 0);
  const serverFallbackOff =
    !!health &&
    ((!!health.llm_preferences && health.llm_preferences.fallback === false) || (!!health.llm && health.llm.fallback === false));

  return (
    <div data-testid="settings-models">
      <SectionTitle title="AI models">
        The agent works with one provider at a time. With keys for more than one, it does not wait when a
        provider reaches its rate limit, rejects a request or is down: it carries on with the next one, in
        the same conversation.
      </SectionTitle>

      <div className={`${INSET} p-4 mb-5`}>
        <div className={LABEL}>A task uses</div>
        {chain.length === 0 ? (
          <p className="text-[13px] text-amber-300 flex items-center gap-2" data-testid="chain-empty">
            <AlertTriangle className="w-4 h-4" /> No provider has a key yet. Add one below.
          </p>
        ) : (
          <ol className="flex flex-wrap items-center gap-2" data-testid="provider-chain">
            {chain.map((item, index) => (
              <li key={item.id} className="flex items-center gap-2" data-provider={item.id}>
                {index > 0 && <ArrowRight className="w-3.5 h-3.5 text-slate-500" />}
                <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md bg-slate-800/80 border border-slate-700/70 text-xs">
                  <span className="font-medium text-slate-100">{item.label}</span>
                  <span className="font-mono text-slate-400 truncate max-w-[220px]">{item.model}</span>
                </span>
              </li>
            ))}
          </ol>
        )}
        {chain.length === 1 && (
          <Hint className="mt-2">
            Only one provider: when it reaches its limit the task has to wait, and fails if the wait is longer
            than its time budget. Add a key for a second one.
          </Hint>
        )}

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4 pt-4 border-t border-slate-800">
          <div>
            <label htmlFor="primary-provider" className={LABEL}>
              Start with
            </label>
            <select
              id="primary-provider"
              value={settings.primary}
              onChange={(event) => updateSettings({ primary: event.target.value as Settings['primary'] })}
              className={INPUT}
              data-testid="primary-provider"
            >
              <option value="auto">Automatic (the server&apos;s order)</option>
              {usable.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.label}
                </option>
              ))}
            </select>
          </div>
          <div className="flex items-end">
            <Toggle
              checked={settings.fallback && !serverFallbackOff}
              onChange={(value) => updateSettings({ fallback: value })}
              label="Carry on with the next provider"
              description={
                serverFallbackOff
                  ? 'Switched off on the server (JULES_LLM_FALLBACK).'
                  : 'When the one in use is rate limited, rejects the key or fails.'
              }
              testId="fallback-toggle"
            />
          </div>
        </div>
      </div>

      <div className="space-y-4">
        {main.map((item) => (
          <ProviderCard
            key={item.id}
            info={item}
            settings={settings}
            position={chain.findIndex((entry) => entry.id === item.id) + 1}
            canCallApi={canCallApi}
          />
        ))}
        {(showMore || moreInUse) &&
          more.map((item) => (
            <ProviderCard
              key={item.id}
              info={item}
              settings={settings}
              position={chain.findIndex((entry) => entry.id === item.id) + 1}
              canCallApi={canCallApi}
            />
          ))}
        {!moreInUse && (
          <button type="button" onClick={() => setShowMore(!showMore)} className="text-xs text-blue-400 hover:text-blue-300 underline underline-offset-2" data-testid="more-providers">
            {showMore ? 'Hide OpenAI and Anthropic' : 'More providers: OpenAI, Anthropic'}
          </button>
        )}
      </div>

      <Callout tone="info" className="mt-5" icon={<Info className="w-4 h-4 text-sky-300" />}>
        <strong className="font-semibold">Where keys live.</strong> A key entered here is kept in this browser and
        sent to your own backend with each task you start from it; the backend uses it for that task and does not
        store it. Tasks that GitHub starts (a <code className="font-mono">/jules</code> comment, a failed check) do
        not pass through a browser: for those, set the key as an environment variable of the deployment (
        {main.map((item) => item.key_variable).join(', ')}).
      </Callout>
    </div>
  );
}

// --------------------------------------------------------------------------
// Commits
// --------------------------------------------------------------------------
const COMMIT_MODES: Array<{ id: CommitMode; title: string; icon: React.ReactNode; text: string }> = [
  {
    id: 'bot',
    title: 'Jules',
    icon: <Bot className="w-4 h-4" />,
    text: 'The GitHub App (or the owner of the token) is the author. GitHub marks such commits as Verified.',
  },
  {
    id: 'coauthor',
    title: 'Co-authored',
    icon: <Users className="w-4 h-4" />,
    text: 'The App is the author and you are named as co-author, so the commit also counts as yours on GitHub.',
  },
  {
    id: 'user',
    title: 'You',
    icon: <User className="w-4 h-4" />,
    text: 'You are the author; the App appears only as the committer. Not marked Verified: GitHub signs only what the App authors itself.',
  },
];

function CommitsTab({ health, canCallApi, installations }: SettingsPanelProps) {
  const settings = useSettings();
  const identity = settings.commit;
  const [lookup, setLookup] = useState<{ phase: 'idle' | 'busy' | 'error' | 'found'; message?: string }>({ phase: 'idle' });
  const complete = identityComplete(identity);
  const accounts = installations.map((item) => item.account).filter((item): item is string => !!item);
  const server = health && health.commits ? health.commits : null;

  const patch = (change: Partial<Settings['commit']>) => updateSettings((current) => ({ commit: { ...current.commit, ...change } }));

  const find = async (login: string) => {
    const wanted = login.trim().replace(/^@/, '');
    if (!wanted) return;
    setLookup({ phase: 'busy' });
    try {
      const found = await fetchIdentity(wanted);
      if (!found.ok || !found.email) {
        setLookup({ phase: 'error', message: found.error || 'That account could not be looked up.' });
        return;
      }
      patch({ login: found.login, name: found.name || found.login, email: found.email });
      setLookup({ phase: 'found', message: `Filled in from github.com/${found.login}.` });
    } catch (error) {
      setLookup({ phase: 'error', message: describeError(error) });
    }
  };

  return (
    <div data-testid="settings-commits">
      <SectionTitle title="Commit authorship">
        Who a commit made by the agent is attributed to. This applies to the tasks you start from this browser.
      </SectionTitle>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5" role="radiogroup" aria-label="Commit authorship">
        {COMMIT_MODES.map((mode) => {
          const active = identity.mode === mode.id;
          return (
            <button
              key={mode.id}
              type="button"
              role="radio"
              aria-checked={active}
              onClick={() => patch({ mode: mode.id })}
              data-testid="commit-mode"
              data-mode={mode.id}
              data-active={active ? 'true' : undefined}
              className={`flex flex-col justify-start text-left rounded-xl border p-4 transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ${
                active ? 'border-blue-500/70 bg-blue-500/10' : 'border-slate-800 bg-[#0b1220] hover:border-slate-600'
              }`}
            >
              <span className={`flex items-center gap-2 w-full text-sm font-semibold ${active ? 'text-white' : 'text-slate-200'}`}>
                {mode.icon}
                {mode.title}
                {active && <Check className="w-4 h-4 text-blue-400 ml-auto" />}
              </span>
              <span className="block text-xs leading-relaxed text-slate-400 mt-1.5">{mode.text}</span>
            </button>
          );
        })}
      </div>

      {identity.mode !== 'bot' && (
        <div className={`${INSET} p-4 space-y-4`} data-testid="commit-identity">
          <div>
            <label htmlFor="commit-login" className={LABEL}>
              Your GitHub user name
            </label>
            <div className="flex gap-2 max-w-xl">
              <input
                id="commit-login"
                type="text"
                value={identity.login}
                onChange={(event) => patch({ login: event.target.value })}
                onKeyDown={(event) => {
                  if (event.key === 'Enter') void find(identity.login);
                }}
                placeholder="octocat"
                autoComplete="off"
                spellCheck={false}
                className={`${INPUT} font-mono`}
              />
              <Button onClick={() => void find(identity.login)} disabled={!canCallApi || !identity.login.trim() || lookup.phase === 'busy'} data-testid="commit-lookup">
                {lookup.phase === 'busy' ? <Loader2 className="w-4 h-4 animate-spin" /> : <Search className="w-4 h-4" />}
                Look up
              </Button>
            </div>
            {accounts.length > 0 && (
              <div className="flex flex-wrap items-center gap-1.5 mt-2 text-xs text-slate-400">
                <span>The GitHub App is installed on:</span>
                {accounts.slice(0, 6).map((account) => (
                  <button
                    key={account}
                    type="button"
                    onClick={() => void find(account)}
                    data-testid="commit-account"
                    className="px-2 py-0.5 rounded-md bg-slate-800 hover:bg-slate-700 border border-slate-700 font-mono text-slate-200"
                  >
                    {account}
                  </button>
                ))}
              </div>
            )}
            {lookup.message && (
              <p className={`text-xs mt-2 ${lookup.phase === 'error' ? 'text-rose-400' : 'text-emerald-400'}`} data-testid="commit-lookup-result">
                {lookup.message}
              </p>
            )}
            <Hint className="mt-2">
              Looking the account up fills in its name and its <code className="font-mono">noreply</code> address:
              GitHub attributes commits with that address to the account without publishing a private one.
            </Hint>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label htmlFor="commit-name" className={LABEL}>
                Name
              </label>
              <input id="commit-name" type="text" value={identity.name} onChange={(event) => patch({ name: event.target.value })} placeholder="Your name" autoComplete="off" className={INPUT} />
            </div>
            <div>
              <label htmlFor="commit-email" className={LABEL}>
                E-mail
              </label>
              <input
                id="commit-email"
                type="email"
                value={identity.email}
                onChange={(event) => patch({ email: event.target.value })}
                placeholder="12345+you@users.noreply.github.com"
                autoComplete="off"
                spellCheck={false}
                className={`${INPUT} font-mono`}
              />
            </div>
          </div>

          {complete ? (
            <p className="text-xs text-slate-300 font-mono break-all" data-testid="commit-preview">
              {identity.mode === 'user' ? (
                <>
                  <span className="text-slate-500">Author:</span> {identity.name.trim()} &lt;{identity.email.trim()}&gt;{' '}
                  <span className="text-slate-500">· Committer:</span> the GitHub App
                </>
              ) : (
                <>
                  <span className="text-slate-500">Author:</span> the GitHub App{' '}
                  <span className="text-slate-500">· last line of the message:</span> Co-authored-by: {identity.name.trim()} &lt;
                  {identity.email.trim()}&gt;
                </>
              )}
            </p>
          ) : (
            <p className="text-xs text-amber-400 flex items-start gap-1.5" data-testid="commit-incomplete">
              <AlertTriangle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0" />
              A name and a valid e-mail address are needed. Until both are there, commits are made as Jules.
            </p>
          )}
        </div>
      )}

      {server && server.authorship !== 'bot' && (
        <Hint className="mt-4">
          The server&apos;s own default (for tasks started from GitHub) is{' '}
          <strong className="text-slate-200">{server.authorship === 'user' ? 'You' : 'Co-authored'}</strong>
          {server.author_name ? ` as ${server.author_name}` : ', taking the person who asked for the task'}.
        </Hint>
      )}
      <Hint className="mt-4">
        The pull request itself is always opened by the GitHub App: only commits can carry another author.
      </Hint>
    </div>
  );
}

// --------------------------------------------------------------------------
// Automation
// --------------------------------------------------------------------------
function CheckRow({ ok, children }: { ok: boolean | null; children: React.ReactNode }) {
  return (
    <li className="flex items-start gap-2 text-xs leading-relaxed" data-ok={ok === null ? 'unknown' : ok ? 'true' : 'false'}>
      {ok === null ? (
        <Info className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-slate-500" />
      ) : ok ? (
        <CheckCircle2 className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-emerald-400" />
      ) : (
        <XCircle className="w-3.5 h-3.5 mt-0.5 flex-shrink-0 text-slate-500" />
      )}
      <span className={ok === false ? 'text-slate-400' : 'text-slate-200'}>{children}</span>
    </li>
  );
}

function AutomationTab({ health, canCallApi }: SettingsPanelProps) {
  const [status, setStatus] = useState<AutomationStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const load = async () => {
    setLoading(true);
    setError(null);
    try {
      setStatus(await fetchAutomation());
    } catch (problem) {
      setError(describeError(problem));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => {
    if (canCallApi) void load();
  }, [canCallApi]);

  const webhookUrl = typeof window !== 'undefined' ? `${window.location.origin}${status ? status.webhook_path : '/api/v1/webhooks/github'}` : '';
  const copy = async () => {
    try {
      await navigator.clipboard.writeText(webhookUrl);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    } catch {
      /* clipboard not available: the URL can be selected by hand */
    }
  };
  const automation = health && health.automation ? health.automation : null;

  return (
    <div data-testid="settings-automation">
      <SectionTitle title="Failed checks">
        When the checks of a pull request fail - a test run on GitHub Actions, a deployment on Vercel - the agent
        can read what failed, fix it and push the fix to the same pull request.
      </SectionTitle>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3 mb-5">
        <div className={`${INSET} p-4`}>
          <div className="text-[13px] font-semibold text-slate-100 mb-1">From here</div>
          <Hint>
            The result of a task shows the checks of its pull request. If they fail, <strong className="text-slate-200">Fix failed checks</strong>{' '}
            starts the repair. Needs nothing set up.
          </Hint>
        </div>
        <div className={`${INSET} p-4`}>
          <div className="text-[13px] font-semibold text-slate-100 mb-1">From GitHub, on request</div>
          <Hint>
            Comment <code className="font-mono text-slate-200">/jules fix the checks</code> on a pull request - or{' '}
            <code className="font-mono text-slate-200">/jules</code> with any instruction to have more work added to it.
          </Hint>
        </div>
        <div className={`${INSET} p-4`}>
          <div className="text-[13px] font-semibold text-slate-100 mb-1">Unasked</div>
          <Hint>
            For pull requests Jules opened itself, as soon as GitHub reports the failure - at most{' '}
            {status ? status.attempts : automation ? automation.ci_autofix_attempts : 2} times per pull request.
          </Hint>
        </div>
      </div>

      <div className={`${INSET} p-4`} data-testid="automation-status">
        <div className="flex items-center justify-between gap-3 mb-3">
          <div className="flex items-center gap-2">
            <span className="text-[13px] font-semibold text-slate-100">Fixing unasked</span>
            {status && (
              <Badge tone={status.ready ? 'good' : 'warn'} data-testid="automation-ready">
                {status.ready ? 'Ready' : 'Not set up yet'}
              </Badge>
            )}
          </div>
          <Button variant="ghost" onClick={() => void load()} disabled={!canCallApi || loading} aria-label="Check again" title="Check again">
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          </Button>
        </div>

        {!canCallApi && <Hint>Enter the access key (Access tab) to see what is set up.</Hint>}
        {error && <p className="text-xs text-rose-400">{error}</p>}
        {status && (
          <>
            {status.error && <p className="text-xs text-amber-400 mb-2">{status.error}</p>}
            <ul className="space-y-1.5">
              <CheckRow ok={status.ci_autofix && status.attempts > 0}>
                Switched on{' '}
                <span className="text-slate-500">
                  (JULES_CI_AUTOFIX, on unless set to false; JULES_CI_AUTOFIX_ATTEMPTS = {status.attempts})
                </span>
              </CheckRow>
              <CheckRow ok={status.webhook_secret}>
                A webhook secret is set on the server <span className="text-slate-500">(GITHUB_WEBHOOK_SECRET)</span> - without
                it nothing GitHub sends is acted on
              </CheckRow>
              {status.events
                .filter((event) => event.id !== 'issue_comment')
                .map((event) => (
                  <CheckRow key={event.id} ok={event.subscribed}>
                    The GitHub App is subscribed to <strong className="font-semibold">{event.label}</strong> events
                  </CheckRow>
                ))}
              {status.permissions.map((permission) => (
                <CheckRow key={permission.id} ok={permission.granted}>
                  The GitHub App has the permission <strong className="font-semibold">{permission.label}: Read-only</strong>
                  {permission.id === 'actions' && <span className="text-slate-500"> (to read the log of a failed job)</span>}
                </CheckRow>
              ))}
              <CheckRow ok={status.events.some((event) => event.id === 'issue_comment' && event.subscribed === true) ? true : null}>
                For <code className="font-mono">/jules</code> comments: subscribed to <strong className="font-semibold">Issue comment</strong> events
              </CheckRow>
            </ul>
            {status.app && (
              <Hint className="mt-2.5">
                One of the three events is enough to hear about a failure. Check suite covers GitHub Actions and other
                check runs; Status covers what is reported as a commit status, such as a Vercel deployment.
              </Hint>
            )}

            <div className="mt-4 pt-4 border-t border-slate-800 space-y-2">
              <div className={LABEL}>Webhook URL of the GitHub App</div>
              <div className="flex gap-2">
                <input readOnly value={webhookUrl} className={`${INPUT} font-mono`} onFocus={(event) => event.target.select()} data-testid="webhook-url" />
                <Button onClick={() => void copy()} aria-label="Copy the webhook URL">
                  {copied ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                  {copied ? 'Copied' : 'Copy'}
                </Button>
              </div>
              {status.app && status.app.settings_url && (
                <Hint>
                  Events, permissions and the webhook are set in the App&apos;s settings:{' '}
                  <ExternalAnchor href={status.app.permissions_url || status.app.settings_url}>
                    {status.app.name || status.app.slug} on GitHub
                  </ExternalAnchor>
                  . After adding permissions, approve them on the installation (GitHub asks the account owner).
                </Hint>
              )}
              {!status.app && (
                <Hint>
                  The server uses a personal token, not a GitHub App: add this URL as a webhook of the repository
                  (events: Check suites, Statuses, Workflow runs, Issue comments) with the same secret.
                </Hint>
              )}
            </div>
          </>
        )}
      </div>

      <div className="mt-6">
        <SectionTitle title="Unfinished work">
          {automation && automation.keep_unfinished_work === false
            ? 'Switched off on the server (JULES_KEEP_UNFINISHED_WORK=false): when a run is cut short, what the agent had changed is discarded.'
            : 'When a run is cut short - the time budget is used up, or every model provider is exhausted - what the agent had changed is not thrown away. It is delivered marked as unfinished (a pull request titled "[Unfinished] …", or the first commit of an empty repository), and Continue on its result picks it up.'}
        </SectionTitle>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// About
// --------------------------------------------------------------------------
function AboutTab({ health }: SettingsPanelProps) {
  if (!health) {
    return (
      <div data-testid="settings-about">
        <SectionTitle title="About this server" />
        <Hint>The backend has not answered yet.</Hint>
      </div>
    );
  }
  const githubAuth = health.github ? health.github.auth_mode : 'none';
  const facts: Array<[string, string]> = [
    ['Status', health.status],
    ['GitHub', githubAuth === 'app' ? 'GitHub App' : githubAuth === 'token' ? 'Personal token' : 'not configured'],
    ['Access', health.access ? health.access.mode : 'open'],
    ['Time budget per task', health.limits ? `${health.limits.task_budget_seconds} s` : 'unknown'],
    ['Model turns per task', health.limits ? String(health.limits.max_agent_turns) : 'unknown'],
    ['Environment', [health.environment, health.region].filter(Boolean).join(' · ') || 'unknown'],
  ];
  return (
    <div data-testid="settings-about">
      <SectionTitle title="About this server">
        Reported by the backend. It shows whether each setting is present, never its value, and it cannot tell
        whether a key is valid until something uses it (the Test buttons under AI models do).
      </SectionTitle>

      <dl className="grid grid-cols-2 md:grid-cols-3 gap-3 mb-5">
        {facts.map(([name, value]) => (
          <div key={name} className={`${INSET} px-3 py-2.5`}>
            <dt className="text-[11px] text-slate-500">{name}</dt>
            <dd className="text-[13px] font-mono text-slate-200 break-words" data-testid={name === 'Status' ? 'backend-status' : undefined}>
              {value}
            </dd>
          </div>
        ))}
      </dl>

      {(health.missing || []).length > 0 && (
        <Callout tone="warn" className="mb-5" icon={<AlertTriangle className="w-4 h-4 text-amber-300" />}>
          <div className="font-semibold mb-1">Still needed to run tasks</div>
          <ul className="list-disc pl-5 space-y-0.5">
            {(health.missing || []).map((item) => (
              <li key={item}>{item}</li>
            ))}
          </ul>
        </Callout>
      )}

      <div className={LABEL}>Environment variables of the server</div>
      <ul className="grid grid-cols-1 md:grid-cols-2 gap-2" data-testid="variables">
        {Object.keys(health.variables || {}).map((name) => {
          const present = health.variables[name];
          return (
            <li key={name} className={`${INSET} flex items-center justify-between gap-3 px-3 h-9 text-xs font-mono`}>
              <span className="text-slate-300 truncate">{name}</span>
              <Badge tone={present ? 'good' : 'muted'}>{present ? 'SET' : 'NOT SET'}</Badge>
            </li>
          );
        })}
      </ul>

      <div className="mt-5 flex flex-wrap gap-2">
        <a href="/api/docs" target="_blank" rel="noreferrer" className={buttonClass('secondary')}>
          API documentation <ExternalLink className="w-3.5 h-3.5" />
        </a>
      </div>
    </div>
  );
}

// --------------------------------------------------------------------------
// Panel
// --------------------------------------------------------------------------
export function SettingsPanel(props: SettingsPanelProps) {
  const { tab, onTab } = props;
  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start" data-testid="settings-panel">
      <nav className={`${CARD} lg:col-span-3 p-2 lg:sticky lg:top-20`} aria-label="Settings sections">
        <ul className="flex lg:flex-col gap-1 overflow-x-auto" role="tablist">
          {SETTINGS_TABS.map((id) => {
            const meta = TAB_META[id];
            const active = id === tab;
            return (
              <li key={id} className="flex-shrink-0">
                <button
                  type="button"
                  role="tab"
                  aria-selected={active}
                  onClick={() => onTab(id)}
                  data-testid="settings-tab"
                  data-tab={id}
                  data-active={active ? 'true' : undefined}
                  className={`w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-left transition-colors focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500/60 ${
                    active ? 'bg-blue-500/15 text-white' : 'text-slate-300 hover:bg-slate-800/70 hover:text-white'
                  }`}
                >
                  <span className={active ? 'text-blue-400' : 'text-slate-500'}>{meta.icon}</span>
                  <span className="min-w-0">
                    <span className="block text-[13px] font-medium whitespace-nowrap">{meta.label}</span>
                    <span className="hidden lg:block text-[11px] text-slate-500 truncate">{meta.description}</span>
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      </nav>

      <div className={`${CARD} lg:col-span-9 p-5 sm:p-6 min-w-0`} role="tabpanel">
        {tab === 'access' && <AccessTab {...props} />}
        {tab === 'models' && <ModelsTab {...props} />}
        {tab === 'commits' && <CommitsTab {...props} />}
        {tab === 'automation' && <AutomationTab {...props} />}
        {tab === 'about' && <AboutTab {...props} />}
      </div>
    </div>
  );
}
